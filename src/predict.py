# predict.py
import os, json, argparse, random, ast
import pandas as pd
import numpy as np
import joblib

def ensure_single_label(s):
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
    return (df["case_title"].fillna("") + " " + df["performed_work"].fillna("")).astype(str)

def main(args):
    outdir = args.outdir
    fold = args.fold

    df = pd.read_csv(os.path.join(outdir, "df_filtered.csv"))
    for col in ["case_title", "performed_work", "linked_items"]:
        if col not in df.columns:
            raise KeyError(f"df_filtered.csv 缺少列：{col}")

    df["linked_items"] = df["linked_items"].apply(ensure_single_label).astype(str)
    X = build_text(df).tolist()
    y = df["linked_items"].astype(str).tolist()

    bundle = joblib.load(os.path.join(outdir, f"model_fold{fold}.joblib"))
    model = bundle["model"]; le = bundle["label_encoder"]

    idx = random.randint(0, len(X) - 1)
    text = X[idx]; true_label = y[idx]

    probs = model.predict_proba([text])[0]
    labels_sorted = np.argsort(-probs)
    preds = le.inverse_transform(labels_sorted)
    scores = probs[labels_sorted]

    print(f"\n[Fold {fold}] 随机样本 #{idx}")
    print("真实标签：", true_label)
    print("文本片段：", text[:200].replace("\n", " ") + ("..." if len(text) > 200 else ""))

    # 计算 hit@k
    hits = {}
    for k in [1, 3, 5, 10]:
        hits[f"hit@{k}"] = int(true_label in preds[:k])
    print("\nHit@K:")
    for k, v in hits.items():
        print(f"{k}: {v}")

    print("\nTop-10 预测：")
    for lbl, sc in zip(preds[:10], scores[:10]):
        mark = "✅" if lbl == true_label else ""
        print(f"{lbl:<12}\t{sc:.4f} {mark}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="../output")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    main(args)
