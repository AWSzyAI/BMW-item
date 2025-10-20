# inspect_miss_hit10.py
import os, json, argparse, ast, sys
import numpy as np
import pandas as pd
import joblib

# Compatibility for loading older joblib model bundles that referenced
# LossCallback under __main__ (e.g. when model was saved from train.py as __main__).
try:
    import train as _train_module
    _mod = sys.modules.get("__main__")
    if _mod is not None and not hasattr(_mod, "LossCallback"):
        setattr(_mod, "LossCallback", _train_module.LossCallback)
except Exception:
    pass

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

def topk_indices(proba_row: np.ndarray, k: int):
    """返回按概率降序的前k个类别索引"""
    k = min(k, proba_row.shape[0])
    part = np.argpartition(-proba_row, kth=k-1)[:k]
    # 再按概率排序
    order = np.argsort(-proba_row[part])
    return part[order]

def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # 读取数据
    df = pd.read_csv(os.path.join(outdir, "df_filtered.csv"))
    for col in ["case_title", "performed_work", "linked_items"]:
        if col not in df.columns:
            raise KeyError(f"df_filtered.csv 缺少列：{col}")
    has_case_id = "case_id" in df.columns

    df["linked_items"] = df["linked_items"].apply(ensure_single_label).astype(str)
    texts = build_text(df).tolist()
    y_raw = df["linked_items"].astype(str).tolist()

    with open(os.path.join(outdir, "folds.json"), "r", encoding="utf-8") as f:
        folds = json.load(f)

    miss_rows = []
    for k, fold in enumerate(folds):
        # 加载该折模型
        bundle = joblib.load(os.path.join(outdir, f"model_fold-{k}.joblib"))
        model = bundle["model"]; le = bundle["label_encoder"]
        C = len(le.classes_)
        K = min(10, C)

        for split_name in ["val", "test"]:
            idxs = fold[split_name]
            Xs = [texts[i] for i in idxs]
            y_true_lbl = [y_raw[i] for i in idxs]
            y_true_idx = le.transform(y_true_lbl)

            y_proba = model.predict_proba(Xs)  # (N, C)

            for row_pos, (gidx, true_idx, proba) in enumerate(zip(idxs, y_true_idx, y_proba)):
                top_idx = topk_indices(proba, K)
                hit10 = int(true_idx in top_idx)
                if hit10 == 0:
                    top_labels = le.inverse_transform(top_idx).tolist()
                    top_scores = proba[top_idx].tolist()
                    row = {
                        "fold": k,
                        "split": split_name,
                        "global_idx": gidx,
                        "true_label": le.inverse_transform([true_idx])[0],
                        "top1_label": top_labels[0],
                        "top1_prob": round(top_scores[0], 6),
                        "topk_labels": "|".join(map(str, top_labels)),
                        "topk_probs": "|".join(f"{s:.6f}" for s in top_scores),
                        "text_snippet": (Xs[row_pos][:300].replace("\n", " ") + ("..." if len(Xs[row_pos]) > 300 else "")),
                    }
                    if has_case_id:
                        row["case_id"] = df.loc[gidx, "case_id"]
                    # 可选也保存原始标题/描述，便于排查
                    row["case_title"] = df.loc[gidx, "case_title"] if "case_title" in df.columns else ""
                    row["performed_work_head"] = str(df.loc[gidx, "performed_work"])[:200].replace("\n", " ")

                    miss_rows.append(row)

    miss_df = pd.DataFrame(miss_rows)
    save_path = os.path.join(outdir, "miss_hit10_samples.csv")
    miss_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Saved hit@10=0 samples to: {save_path}")
    print(f"Total rows: {len(miss_df)} (val+test across all folds)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./output")
    args = parser.parse_args()
    main(args)
