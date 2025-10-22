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

def _ooc_prob_from_detector(ooc_detector, p_max_scalar: float) -> float:
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
            tau = float(ooc_detector.get("tau", 0.5))
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
    model = bundle["model"]
    le = bundle["label_encoder"]
    ooc_detector = bundle.get("ooc_detector")

    # 读取输入样本
    outdir = args.outdir
    infile = args.infile if args.infile else args.path  # 兼容旧参数 --path
    df = _flex_read_csv(outdir, infile)
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

    # 计算 not-in-train 概率（独立输出）
    p_max = float(probs.max())
    ooc_proba = _ooc_prob_from_detector(ooc_detector, p_max)
    is_ooc = ooc_proba >= float(getattr(args, "ooc_decision_threshold", 0.5))

    print(f"\n[预测结果] 样本 #{idx}\n")
    print(f"Text: {text[:200]} ...")
    # 打印真实标签及其是否在训练集中出现
    true_label = None
    if "linked_items" in df.columns:
        true_label = str(ensure_single_label(df.iloc[idx]["linked_items"]))
        in_train = true_label in set(le.classes_)
        print(f"True label: {true_label}  (in_train={in_train})")
    else:
        print("True label: [列 linked_items 缺失]")
    print("\nTop-10 predictions:")
    for lbl, sc in zip(preds, scores):
        mark = " ✅" if (true_label is not None and str(lbl) == true_label) else ""
        print(f"{lbl:<10}\t{sc:.4f}{mark}")
    mark = " ✅" if is_ooc else ""
    print(f"\nNot-in-train probability: {ooc_proba:.4f}{mark}")

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
    parser.add_argument("--model", type=str, default="model_best.joblib")
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--infile", type=str, default="eval.csv", help="输入文件名（在 outdir 下，或提供绝对路径）")
    parser.add_argument("--index", type=int, default=-1, help="样本下标；<0 则随机")
    parser.add_argument("--ooc-decision-threshold", type=float, default=0.5, help="将 not-in-train 概率转为判定的阈值")
    # 兼容旧参数
    parser.add_argument("--path", type=str, default=None)
    args = parser.parse_args()
    main(args)
