# predict.py
import os, json, argparse, random, ast
import pandas as pd
import numpy as np
import joblib

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
    fold = args.fold

    # df_filtered 原始样本数(展开): 2362 -> 过滤后样本数: 2072
    df = pd.read_csv(os.path.join(outdir, "df_filtered.csv"))

    required_cols = ['extern_id', 'linked_items', 'itemcreationdate',
                     'item_title', 'case_id', 'case_title', 'performed_work']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"df_filtered.csv 缺少字段: {missing}")

    df["linked_items_norm"] = df["linked_items"].apply(ensure_single_label).astype(str)
    X = build_text(df).tolist()
    y = df["linked_items_norm"].astype(str).tolist()

    # 加载模型
    model_path = os.path.join(outdir, f"model_fold{fold}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型: {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]; le = bundle["label_encoder"]

    # 随机抽样
    idx = random.randint(0, len(df) - 1)
    row = df.iloc[idx]
    text = X[idx]
    true_label = y[idx]

    # 模型预测
    probs = model.predict_proba([text])[0]
    sorted_idx = np.argsort(-probs)
    preds = le.inverse_transform(sorted_idx)
    scores = probs[sorted_idx]

    # 打印基础字段信息
    print(f"\n[Fold {fold}] 随机样本 #{idx}\n")
    for col in required_cols:
        val = str(row[col])[:500].replace("\n", " ")
        print(f"{col:20s}: {val}")

    print(f"{'-'*80}\n真实标签（标准化）: {true_label}\n")

    # 命中情况
    hits = {f"hit@{k}": int(true_label in preds[:k]) for k in [1, 3, 5, 10]}
    print("命中统计：", "  ".join([f"{k}={v}" for k,v in hits.items()]))

    # 打印前10预测
    print("\nTop-10 预测结果：")
    for lbl, sc in zip(preds[:10], scores[:10]):
        mark = "✅" if lbl == true_label else ""
        print(f"{lbl:<10}\t{sc:.4f} {mark}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    main(args)
