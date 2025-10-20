#!/usr/bin/env python3
import os, json, argparse, ast, sys
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
    """计算 hit@k 命中率"""
    topk_idx = np.argsort(-y_proba, axis=1)[:, :k]
    hits = (topk_idx == y_true_idx.reshape(-1, 1)).any(axis=1)
    return float(hits.mean())

def eval_split(model, le, X_text, y_raw, indices):
    y_true = le.transform([y_raw[i] for i in indices])
    Xs = [X_text[i] for i in indices]
    y_pred = model.predict(Xs)
    if isinstance(y_pred[0], str):
        y_pred = le.transform(y_pred)
    y_pred = np.asarray(y_pred)
    y_proba = model.predict_proba(Xs)

    m = {}
    m["acc"] = accuracy_score(y_true, y_pred)
    m["bal_acc"] = balanced_accuracy_score(y_true, y_pred)
    m["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    m["prec_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    m["rec_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

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

    for k in [1, 3, 5, 10]:
        m[f"hit@{k}"] = hit_at_k(y_true, y_proba, k)
    return m

def main(args):
    path = args.path
    outdir = args.outdir
    mode = args.mode
    os.makedirs(outdir, exist_ok=True)

    # 读取数据文件：优先按给定路径，其次在 outdir/path
    read_candidates = [path, os.path.join(outdir, path)]
    file_to_read = None
    for p in read_candidates:
        if p and os.path.exists(p):
            file_to_read = p
            break
    if file_to_read is None:
        # 回退到旧逻辑（按 outdir 拼接），保留原有行为的错误信息
        file_to_read = os.path.join(outdir, path)
        if not os.path.exists(file_to_read):
            raise FileNotFoundError(f"找不到评估数据文件：{path} 或 {file_to_read}")

    df = pd.read_csv(file_to_read)
    for col in ["case_title", "performed_work", "linked_items"]:
        if col not in df.columns:
            raise KeyError(f"{file_to_read} 缺少列：{col}")
    df["linked_items"] = df["linked_items"].apply(ensure_single_label).astype(str)
    X_text = build_text(df).tolist()
    y_raw  = df["linked_items"].astype(str).tolist()

    # 加载最优模型
    model = args.model
    best_model_path = os.path.join(outdir, model)
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found: {best_model_path}")
    bundle = joblib.load(best_model_path)
    model = bundle["model"]
    le = bundle["label_encoder"]

    results = {}

    if mode == "new":
        # new 模式：对整文件进行评估。
        # 仅对标签在训练类别中的样本计算指标，避免未知标签导致 transform 失败。
        valid_mask = [lbl in set(le.classes_) for lbl in y_raw]
        valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
        unknown_cnt = len(y_raw) - len(valid_indices)
        if len(valid_indices) == 0:
            raise ValueError("新文件中没有任何标签出现在已训练的类别中，无法计算分类指标。")
        if unknown_cnt > 0:
            print(f"[new] 注意：共有 {unknown_cnt} 条样本的标签不在训练类别中，这些样本被排除在指标计算外。")

        m = eval_split(model, le, X_text, y_raw, valid_indices)
        results["new"] = m
        print(
            f"[new]\nacc={m['acc']:.3f} | f1_macro={m['f1_macro']:.3f} | hit@1={m['hit@1']:.3f} | hit@3={m['hit@3']:.3f} | hit@5={m['hit@5']:.3f} | hit@10={m['hit@10']:.3f}"
        )

        # 生成逐样本预测并保存（包含所有行，包括未知标签行）
        Xs_all = X_text
        y_proba_all = model.predict_proba(Xs_all)
        cls = le.classes_
        topk = min(10, len(cls))
        topk_idx = np.argsort(-y_proba_all, axis=1)[:, :topk]
        topk_labels = [[str(cls[j]) for j in row] for row in topk_idx]
        topk_scores = [[float(y_proba_all[i, j]) for j in topk_idx[i]] for i in range(len(df))]
        pred_top1 = [labels[0] if labels else "" for labels in topk_labels]

        # 命中率按是否为已知标签决定，未知标签命中置 NaN
        cls_set = set(cls)
        true_labels = [str(t) for t in y_raw]
        def _hit_at(preds, true, k):
            if true not in cls_set:
                return np.nan
            return 1 if true in preds[:k] else 0

        rows = []
        for i in range(len(df)):
            preds_i = topk_labels[i]
            scores_i = topk_scores[i]
            true_i = true_labels[i]
            rows.append({
                "index": i,
                "case_id": df.iloc[i].get("case_id"),
                "true_label": true_i,
                "pred_top1": pred_top1[i],
                "preds_top10": "|".join(preds_i),
                "scores_top10": "|".join(f"{s:.6f}" for s in scores_i),
                "hit@1": _hit_at(preds_i, true_i, 1),
                "hit@3": _hit_at(preds_i, true_i, 3),
                "hit@5": _hit_at(preds_i, true_i, 5),
                "hit@10": _hit_at(preds_i, true_i, 10),
            })

        pred_df = pd.DataFrame(rows)
        base = os.path.splitext(os.path.basename(file_to_read))[0]
        pred_out = os.path.join(outdir, f"predictions_{base}.csv")
        pred_df.to_csv(pred_out, index=False, encoding="utf-8-sig")
        print(f"逐样本预测已保存：{pred_out}")

        # 保存指标
        dfm = pd.DataFrame([dict(split=split_name, **metrics) for split_name, metrics in results.items()])
        out_path = os.path.join(outdir, "metrics_best_model_all_splits.csv")
        dfm.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nBest model metrics saved to {out_path}")

        return

    # 非 new 模式需要 folds.json
    with open(os.path.join(outdir, "folds.json"), "r", encoding="utf-8") as f:
        folds = json.load(f)

    # 聚合各折的索引
    all_splits = {"train": [], "val": [], "test": []}
    for k, fold in enumerate(folds):
        for split_name in ["train", "val", "test"]:
            all_splits[split_name].extend(fold[split_name])

    # 根据模式选择需要评估的切分
    target_splits = ["val", "test"] if mode == "clean" else ["train", "val", "test"]

    for split_name in target_splits:
        idxs = all_splits.get(split_name, [])
        if not idxs:
            print(f"No indices for split {split_name}, skipping")
            continue
        m = eval_split(model, le, X_text, y_raw, idxs)
        results[split_name] = m
        print(
            f"[All {split_name}]\nacc={m['acc']:.3f} | f1_macro={m['f1_macro']:.3f} | hit@1={m['hit@1']:.3f} | hit@3={m['hit@3']:.3f} | hit@5={m['hit@5']:.3f} | hit@10={m['hit@10']:.3f}"
        )

    # 保存结果
    if results:
        dfm = pd.DataFrame([dict(split=split_name, **metrics) for split_name, metrics in results.items()])
        out_path = os.path.join(outdir, "metrics_best_model_all_splits.csv")
        dfm.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nBest model metrics saved to {out_path}")
    else:
        print("没有可保存的评估结果。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default="model_best.joblib",
                        help="用于评估的模型文件名，位于 outdir 下")
    parser.add_argument("--path", type=str,default="common_34_56.csv")
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--mode", type=str, default="new", choices=["clean", "dirty", "new"],
                        help="评估模式：clean=val+test，dirty=train+val+test，new=对传入文件整体评估")
    args = parser.parse_args()
    main(args)
