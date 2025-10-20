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
    outdir = args.outdir
    # 加载数据
    df = pd.read_csv(os.path.join(outdir, "df_filtered.csv"))
    with open(os.path.join(outdir, "folds.json"), "r", encoding="utf-8") as f:
        folds = json.load(f)
    # 只用最优 fold 的 test 集
    best_model_path = os.path.join(outdir, "model_best.joblib")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found: {best_model_path}")
    bundle = joblib.load(best_model_path)
    model = bundle["model"]
    le = bundle["label_encoder"]
    # 选 test 集
    # 找到最优 fold（与 train.py 输出一致）
    test_accs = []
    X_text = build_text(df).tolist()
    y_raw = df["linked_items"].astype(str).tolist()
    for k, fold in enumerate(folds):
        test_idx = fold["test"]
        X_test = [X_text[i] for i in test_idx]
        y_test = [y_raw[i] for i in test_idx]
        y_pred = model.predict(X_test)
        acc = (y_pred == le.transform(y_test)).mean() if len(y_test) > 0 else 0
        test_accs.append(acc)
    best_fold = int(np.argmax(test_accs))
    test_idx = folds[best_fold]["test"]
    X_test = [X_text[i] for i in test_idx]
    y_test = [y_raw[i] for i in test_idx]
    # 随机抽样 test 集
    if len(X_test) == 0:
        print("No test samples found in best fold.")
        return
    idx = random.randint(0, len(X_test) - 1)
    text = X_test[idx]
    true_label = y_test[idx]
    probs = model.predict_proba([text])[0]
    sorted_idx = np.argsort(-probs)
    preds = le.inverse_transform(sorted_idx)
    scores = probs[sorted_idx]
    print(f"\n[Best Fold {best_fold}] Random test sample #{idx}\n")
    print(f"Text: {text[:200]} ...")
    print(f"True label: {true_label}")
    hits = {f"hit@{k}": int(true_label in preds[:k]) for k in [1, 3, 5, 10]}
    print("Hit stats:", "  ".join([f"{k}={v}" for k,v in hits.items()]))
    print("\nTop-10 predictions:")
    for lbl, sc in zip(preds[:10], scores[:10]):
        mark = " ✅" if lbl == true_label else ""
        print(f"{lbl:<10}\t{sc:.4f}{mark}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    main(args)
