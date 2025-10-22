# predict.py
import os, json, argparse, random, ast, sys
import pandas as pd
import numpy as np
import joblib

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
    # if import/inject fails, let joblib raise the original error when loading
    pass

def ensure_single_label(s):
    """将可能是列表或字符串列表的字段转为单标签字符串。"""
    if isinstance(s, list):
        return str(s[0]) if s else ""
    if isinstance(s, str):
        t = s.strip()
        if (t.startswith("[") and t.endswith("]")) or (t.startswith("(") and t.endswith(")")):
            try:
                v = ast.literal_eval(t)
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return str(v[0])
            except Exception:
                pass
        return t
    return str(s)

def build_text(df):
    """组合 case_title + performed_work"""
    return (df["case_title"].fillna("") + " " + df["performed_work"].fillna("")).astype(str)

def main(args):
    path = args.path
    outdir = args.outdir
    # 加载数据
    df = pd.read_csv(os.path.join(outdir, path))
    # 加载模型
    best_model_path = os.path.join(outdir, "model_best.joblib")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found: {best_model_path}")
    bundle = joblib.load(best_model_path)
    model = bundle["model"]
    le = bundle["label_encoder"]
    # 直接从文件中抽取一条样本进行预测
    # 规范标签字段
    df["linked_items"] = df["linked_items"].apply(ensure_single_label).astype(str)
    texts = build_text(df).tolist()
    labels = df["linked_items"].tolist()
    # 过滤未知标签
    valid = [lbl in set(le.classes_) for lbl in labels]
    if not any(valid):
        print("无可用的已知标签样本，无法预测。")
        return
    texts_f = [t for t, ok in zip(texts, valid) if ok]
    labels_f = [l for l, ok in zip(labels, valid) if ok]
    idx = random.randint(0, len(texts_f) - 1)
    text = texts_f[idx]
    true_label = labels_f[idx]
    probs = model.predict_proba([text])[0]
    sorted_idx = np.argsort(-probs)
    preds = le.inverse_transform(sorted_idx)
    scores = probs[sorted_idx]
    print(f"\n[预测结果] 样本 #{idx}\n")
    print(f"Text: {text[:200]} ...")
    print(f"True label: {true_label}")
    hits = {f"hit@{k}": int(true_label in preds[:k]) for k in [1, 3, 5, 10]}
    print("Hit stats:", "  ".join([f"{k}={v}" for k,v in hits.items()]))
    print("\nTop-10 predictions:")
    for lbl, sc in zip(preds[:10], scores[:10]):
        mark = " ✅" if lbl == true_label else ""
        print(f"{lbl:<10}\t{sc:.4f}{mark}")


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
    parser.add_argument("--path", type=str,default="m_test.csv")
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    main(args)
