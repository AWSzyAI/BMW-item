# eval.py
import os, json, argparse, ast
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, log_loss, roc_auc_score
)

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

def hit_at_k(y_true_idx: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """
    y_true_idx: shape (N,)  — 已经是 label encoder 之后的整数索引
    y_proba   : shape (N, C)
    返回: 命中率（float）
    """
    # 取每行 top-k 的类别索引（二维：N x k）
    # 用 argsort(-proba) 更直观；若追求更快可用 argpartition
    topk_idx = np.argsort(-y_proba, axis=1)[:, :k]
    # 判断每行是否命中：真实索引是否在该行的 top-k 索引中
    # 利用广播 (N,1) vs (N,k)
    hits = (topk_idx == y_true_idx.reshape(-1, 1)).any(axis=1)
    return float(hits.mean())

def eval_split(model, le, X_text, y_raw, indices):
    # 真实标签转成索引（1D）
    y_true = le.transform([y_raw[i] for i in indices])
    Xs = [X_text[i] for i in indices]

    # 预测标签索引
    y_pred = model.predict(Xs)
    if isinstance(y_pred[0], str):  # 兼容某些模型返回原始标签名
        y_pred = le.transform(y_pred)
    y_pred = np.asarray(y_pred)

    # 概率
    y_proba = model.predict_proba(Xs)

    m = {}
    m["acc"] = accuracy_score(y_true, y_pred)
    m["bal_acc"] = balanced_accuracy_score(y_true, y_pred)
    m["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    m["prec_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    m["rec_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # logloss / AUC
    try:
        m["logloss"] = log_loss(y_true, y_proba, labels=np.arange(len(le.classes_)))
    except Exception:
        m["logloss"] = np.nan
    try:
        m["auc_macro"] = roc_auc_score(
            y_true, y_proba, multi_class="ovo", average="macro", labels=np.arange(len(le.classes_))
        )
    except Exception:
        m["auc_macro"] = np.nan

    # hit@k
    for k in [1, 3, 5, 10]:
        m[f"hit@{k}"] = hit_at_k(y_true, y_proba, k)
    return m

def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(os.path.join(outdir, "df_filtered.csv"))
    for col in ["case_title", "performed_work", "linked_items"]:
        if col not in df.columns:
            raise KeyError(f"df_filtered.csv 缺少列：{col}")
    df["linked_items"] = df["linked_items"].apply(ensure_single_label).astype(str)
    X_text = build_text(df).tolist()
    y_raw  = df["linked_items"].astype(str).tolist()

    with open(os.path.join(outdir, "folds.json"), "r", encoding="utf-8") as f:
        folds = json.load(f)

    rows = []
    for k, fold in enumerate(folds):
        bundle = joblib.load(os.path.join(outdir, f"model_fold{k}.joblib"))
        model = bundle["model"]; le = bundle["label_encoder"]

        for split_name in ["val", "test"]:
            m = eval_split(model, le, X_text, y_raw, fold[split_name])
            m["fold"] = k; m["split"] = split_name
            rows.append(m)
            print(
                f"[Fold {k}][{split_name}] "
                f"acc={m['acc']:.3f} | f1_macro={m['f1_macro']:.3f} | "
                f"hit@1={m['hit@1']:.3f} | hit@3={m['hit@3']:.3f} | hit@5={m['hit@5']:.3f} | hit@10={m['hit@10']:.3f}"
            )

    dfm = pd.DataFrame(rows)
    out_path = os.path.join(outdir, "metrics_hitk.csv")
    dfm.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n总体平均：")
    print(dfm.groupby("split")[["acc","f1_macro","hit@1","hit@3","hit@5","hit@10"]].mean().round(3))
    print(f"\n明细保存到 {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./output")
    args = parser.parse_args()
    main(args)
