# predict.py
import os, argparse, random, sys
import numpy as np
import joblib
from utils import _flex_read_csv, build_text, ensure_single_label
import pandas as pd

# Compatibility: some saved joblib bundles reference LossCallback under __main__
# (e.g. when model was saved from train.py run as __main__). To allow
# unpickling here in predict.py, inject train.LossCallback into this module's
# namespace under the common names pickle may look for.
try:
    import train as _train_module
    _mod = sys.modules.get("__main__")
    if _mod is not None and not hasattr(_mod, "LossCallback"):
        setattr(_mod, "LossCallback", _train_module.LossCallback)
except Exception:
    pass

# 兼容从 train_bert.py 序列化的 BERT 包装器（BERTModelWrapper、LossRecorder 等）
try:
    import train_bert as _train_bert_module  # noqa: F401
    _mod = sys.modules.get("__main__")
    if _mod is not None:
        # 某些情况下，joblib 在保存时记录的模块为 __main__，需要在当前 __main__ 注入同名类
        for _name in ("BERTModelWrapper", "LossRecorder"):
            if hasattr(_train_bert_module, _name) and not hasattr(_mod, _name):
                setattr(_mod, _name, getattr(_train_bert_module, _name))
except Exception:
    # 若未找到也不致命；仅在反序列化 BERT 模型 bundle 时需要
    pass

def _ooc_prob_from_detector(ooc_detector, p_max_scalar: float, override_tau: float | None = None) -> float:
    """从保存的 ooc_detector 计算 not-in-train 概率。
    支持两种形式：
    - {kind:'logreg', estimator: sklearn model}
    - {kind:'threshold', tau: float, temperature: float}
    也兼容直接保存的 sklearn 模型。
    """
    if ooc_detector is None:
        return 0.0
    # dict 格式
    if isinstance(ooc_detector, dict):
        kind = ooc_detector.get("kind")
        if kind == "logreg" and "estimator" in ooc_detector:
            est = ooc_detector["estimator"]
            try:
                return float(est.predict_proba(np.array([[p_max_scalar]]))[:, 1][0])
            except Exception:
                return 0.0
        if kind == "threshold":
            tau = float(override_tau if override_tau is not None else ooc_detector.get("tau", 0.5))
            T = float(ooc_detector.get("temperature", 20.0))
            # 低于阈值越像 OOC：σ((tau - p_max) * T)
            return float(1.0 / (1.0 + np.exp((p_max_scalar - tau) * T)))
    # 直接当作 sklearn 模型
    try:
        return float(ooc_detector.predict_proba(np.array([[p_max_scalar]]))[:, 1][0])
    except Exception:
        return 0.0

def main(args):
    # 解析模型路径
    modelsdir = args.modelsdir
    model_name = args.model
    bundle_path = os.path.join(modelsdir, model_name)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Model not found: {bundle_path}")
    bundle = joblib.load(bundle_path)
    model = bundle.get("model")
    le = bundle.get("label_encoder")
    ooc_detector = bundle.get("ooc_detector")
    model_type = bundle.get("model_type", None)

    # 若为 BERT 训练生成的 bundle
    if model_type == "bert":
        # 尝试对设备进行校正（防止在GPU环境保存而在CPU环境加载导致的设备不匹配）
        try:
            import torch  # 延迟导入
            if model is not None and hasattr(model, "device"):
                desired = torch.device(
                    "cuda" if torch.cuda.is_available() else (
                        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
                    )
                )
                # 有些情况下 device 可能保存为字符串
                try:
                    saved = str(model.device)
                except Exception:
                    saved = str(model.device) if model.device is not None else ""
                if (("cuda" in saved or "mps" in saved) and desired.type == "cpu") or ("cpu" in saved and desired.type in ("cuda", "mps")):
                    model.device = desired
        except Exception:
            pass

        # 若未能直接反序列化出包装器，则尝试基于保存的 HF 目录重建
        if model is None or not hasattr(model, "predict_proba"):
            try:
                # 从 bundle 读取保存的 HF 目录与标签编码器
                model_dir = bundle.get("model_dir")
                if model_dir is None or le is None:
                    raise RuntimeError("BERT bundle 缺少 model_dir 或 label_encoder")
                # 动态获取包装器类
                from train_bert import BERTModelWrapper  # type: ignore
                # 选择设备
                import torch
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else (
                        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
                    )
                )
                # 分词器：直接从 HF 目录加载（BERTModelWrapper.fit 内部会加载模型）
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
                model = BERTModelWrapper(model_dir, tokenizer, le, device)
            except Exception as e:
                raise RuntimeError(f"无法重建 BERT 模型：{e}")

    # 读取输入样本（支持 *_X.csv + *_y.csv 分离，或单表）
    outdir = args.outdir
    infile = args.infile if args.infile else args.path  # 兼容旧参数 --path

    def _read_split_or_combined(base_dir: str, base: str) -> pd.DataFrame:
        # 若是绝对路径且存在且为 *_X.csv，优先寻找旁边的 *_y.csv
        if base and os.path.isabs(base) and os.path.exists(base):
            stem = os.path.splitext(os.path.basename(base))[0]
            if stem.endswith("_X"):
                y_abs = base[:-6] + "_y.csv"
                if os.path.exists(y_abs):
                    X_df = pd.read_csv(base)
                    y_df = pd.read_csv(y_abs)
                    if "linked_items" in y_df.columns:
                        return pd.concat([X_df.reset_index(drop=True), y_df[["linked_items"]].reset_index(drop=True)], axis=1)
                    else:
                        # 没有标签也允许，仅用于展示文本
                        return X_df.reset_index(drop=True)
            # 单表绝对路径
            return pd.read_csv(base)

        name = os.path.basename(base)
        stem = os.path.splitext(name)[0]
        if stem.endswith("_X"):
            stem = stem[:-2]
        if stem.endswith("_y"):
            stem = stem[:-2]
        x_path = os.path.join(base_dir, f"{stem}_X.csv")
        y_path = os.path.join(base_dir, f"{stem}_y.csv")
        if os.path.exists(x_path):
            X_df = pd.read_csv(x_path)
            if os.path.exists(y_path):
                y_df = pd.read_csv(y_path)
                if "linked_items" in y_df.columns:
                    return pd.concat([X_df.reset_index(drop=True), y_df[["linked_items"]].reset_index(drop=True)], axis=1)
            return X_df.reset_index(drop=True)
        # 回退到单表
        return _flex_read_csv(base_dir, base)

    df = _read_split_or_combined(outdir, infile)
    texts = build_text(df).tolist()
    if len(texts) == 0:
        raise ValueError("输入文件为空或无法构造文本。")

    # 选择索引
    idx = args.index
    if idx is None or idx < 0 or idx >= len(texts):
        idx = random.randint(0, len(texts) - 1)
    text = texts[idx]

    # 预测已知类别概率
    probs = model.predict_proba([text])[0]
    sorted_idx = np.argsort(-probs)
    top_k = min(10, probs.shape[0])
    preds = le.inverse_transform(sorted_idx[:top_k])
    scores = probs[sorted_idx[:top_k]]

    # 计算 not-in-train 判定（MSP: 用 p_max 阈值；LogReg: 用概率阈值）
    p_max = float(probs.max())
    if isinstance(ooc_detector, dict) and ooc_detector.get("kind") == "logreg":
        ooc_proba = _ooc_prob_from_detector(ooc_detector, p_max)
        prob_thr = float(getattr(args, "ooc_decision_threshold", 0.5))
        is_ooc = (ooc_proba >= prob_thr)
        decision_info = f"(prob_thr={prob_thr:.3f})"
    else:
        # MSP/threshold-based
        if getattr(args, "reject_threshold", None) is not None:
            thr = float(args.reject_threshold)
        elif isinstance(ooc_detector, dict) and ("tau" in ooc_detector):
            thr = float(ooc_detector["tau"])
        else:
            thr = 0.5
        is_ooc = (p_max < thr)
        ooc_proba = _ooc_prob_from_detector(ooc_detector, p_max, override_tau=thr)
        decision_info = f"(p_max={p_max:.4f}, thr={thr:.4f})"

    print(f"\n[预测结果] 样本 #{idx}\n")
    print(f"Text: {text[:200]} ...")
    # 打印真实标签及其是否在训练集中出现，并同时打印对应的 item_title（若存在）
    true_label = None
    if "linked_items" in df.columns:
        true_label = str(ensure_single_label(df.iloc[idx]["linked_items"]))
        in_train = true_label in set(le.classes_)
        if "item_title" in df.columns:
            item_title_val = df.iloc[idx].get("item_title")
            try:
                item_title_str = "" if pd.isna(item_title_val) else str(item_title_val)
            except Exception:
                item_title_str = str(item_title_val)
            print(f"True label: {true_label}  (in_train={in_train}) | item_title: {item_title_str}")
        else:
            print(f"True label: {true_label}  (in_train={in_train}) | item_title: [列 item_title 缺失]")
    else:
        if "item_title" in df.columns:
            item_title_val = df.iloc[idx].get("item_title")
            try:
                item_title_str = "" if pd.isna(item_title_val) else str(item_title_val)
            except Exception:
                item_title_str = str(item_title_val)
            print(f"True label: [列 linked_items 缺失] | item_title: {item_title_str}")
        else:
            print("True label: [列 linked_items 缺失] | item_title: [列 item_title 缺失]")
    print("\nTop-10 predictions:")
    for lbl, sc in zip(preds, scores):
        mark = " ✅" if (true_label is not None and str(lbl) == true_label) else ""
        print(f"{lbl:<10}\t{sc:.4f}{mark}")
    mark = " ✅" if is_ooc else ""
    print(f"\nNot-in-train probability: {ooc_proba:.4f} {decision_info} {mark}".rstrip())

    # 最终决策：若 ooc 概率高于阈值 => Not-in-train；否则选择 Top-1 已知标签
    if is_ooc:
        print("Final: Not-in-train")
    else:
        top1 = str(preds[0]) if len(preds) > 0 else ""
        print(f"Final: {top1}（Known）")


def predict(texts, model_path=None, top_k=10):
    """Predict top-k labels for given texts.

    texts: str | list[str] | pandas.DataFrame
    model_path: path to a joblib bundle (defaults to ./output/model_best.joblib)
    Returns a list of dicts: {'preds': [...], 'scores': [...]} per input.
    """
    if model_path is None:
        model_path = os.path.join("../output", "model_best.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    bundle = joblib.load(model_path)
    model = bundle.get("model")
    le = bundle.get("label_encoder")
    model_type = bundle.get("model_type", None)
    if model_type == "bert" and (model is None or not hasattr(model, "predict_proba")):
        # 与 main 中一致的回退：基于保存的 HF 目录重建包装器
        model_dir = bundle.get("model_dir")
        if model_dir is None or le is None:
            raise ValueError("BERT bundle 缺少 model_dir 或 label_encoder。")
        from train_bert import BERTModelWrapper  # type: ignore
        from transformers import AutoTokenizer
        import torch
        device = torch.device(
            "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = BERTModelWrapper(model_dir, tokenizer, le, device)
    if model is None or le is None:
        raise ValueError("Loaded bundle does not contain 'model' and 'label_encoder'.")

    # Normalize input to list of texts
    if isinstance(texts, pd.DataFrame):
        X = build_text(texts).tolist()
    elif isinstance(texts, str):
        X = [texts]
    else:
        # assume iterable of strings
        X = list(texts)

    probs = model.predict_proba(X)
    out = []
    for p in probs:
        idxs = np.argsort(-p)[:top_k]
        preds = le.inverse_transform(idxs)
        scores = p[idxs]
        out.append({"preds": list(map(str, preds)), "scores": list(map(float, scores))})
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelsdir", type=str, default="./models")
    parser.add_argument("--model", type=str, default="7.joblib")
    parser.add_argument("--outdir", type=str, default="./output/2025_up_to_month_7")
    parser.add_argument("--infile", type=str, default="eval.csv", help="输入文件名（在 outdir 下，或提供绝对路径）")
    parser.add_argument("--index", type=int, default=-1, help="样本下标；<0 则随机")
    # MSP: p_max 阈值（与 eval 对齐）；LogReg: 概率阈值
    parser.add_argument("--reject-threshold", type=float, default=None, help="MSP：最大类别概率 p_max 的拒判阈值")
    parser.add_argument("--ooc-decision-threshold", type=float, default=0.5, help="LogReg 检测器下的概率阈值")
    # 兼容旧参数
    parser.add_argument("--path", type=str, default=None)
    args = parser.parse_args()
    main(args)
