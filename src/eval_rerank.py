#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线评估 Qwen2 rerank 效果。

支持两种来源：
- `predictions_eval.csv`（原先 eval.py 的输出），包含 `preds_top10` / `scores_top10`；
- `eval.csv`（原始逐样本评估文件），包含 `true_label` / `pred_top10` / `score_top10` 等列。

脚本会从指定文件中抽取 top-k 候选和文本，构造 record，
调用 `rerank.qwen_rerank_single` 做重排，并对比 base 与 rerank 的 hit@1/3/5/10，
同时写出逐样本详细结果 CSV 以便后续分析。
"""

import argparse
import csv
from pathlib import Path
from typing import List

import pandas as pd

from rerank import qwen_rerank_single


def _extract_topk_and_scores_from_predictions(row: pd.Series, top_k: int) -> tuple[list[str], list[float]]:
    """从 predictions_eval.csv 风格行中抽取 labels/scores。"""

    labels = [p.strip() for p in str(row.get("preds_top10", "")).split("|") if p.strip()]
    scores_raw = [s.strip() for s in str(row.get("scores_top10", "")).split("|") if s.strip()]

    labels = labels[:top_k]
    scores: List[float] = []
    for s in scores_raw[:top_k]:
        try:
            scores.append(float(s))
        except ValueError:
            scores.append(0.0)

    while len(scores) < len(labels):
        scores.append(0.0)

    return labels, scores


def _extract_topk_and_scores_from_eval(row: pd.Series, top_k: int) -> tuple[list[str], list[float]]:
    """从 eval.csv 风格行中抽取 labels/scores。

    约定：
    - 预测标签列名为 `pred_top10`（"|" 分隔）；
    - 预测分数列名为 `score_top10`（"|" 分隔）。
    """

    labels = [p.strip() for p in str(row.get("pred_top10", "")).split("|") if p.strip()]
    scores_raw = [s.strip() for s in str(row.get("score_top10", "")).split("|") if s.strip()]

    labels = labels[:top_k]
    scores: List[float] = []
    for s in scores_raw[:top_k]:
        try:
            scores.append(float(s))
        except ValueError:
            scores.append(0.0)

    while len(scores) < len(labels):
        scores.append(0.0)

    return labels, scores


def _build_text(row: pd.Series) -> str:
    parts: List[str] = []
    # 统一尽量利用已有字段，按优先级拼接
    for col in ["case_title", "performed_work", "item_title", "text"]:
        if col in row.index:
            v = str(row.get(col, "")).strip()
            if v:
                parts.append(v)
    return "\n".join(parts)


def _build_record(
    row: pd.Series,
    text: str,
    labels: List[str],
    scores: List[float],
    record_id: str,
) -> dict:
    """构造传给 Qwen 的 record。

    这里尽量从源 DataFrame 中补齐可用信息：
    - 对于 predictions_eval.csv：目前只能提供 label/score；
    - 对于 eval.csv：linked_items 对应 true_label，我们没有单独的 item 标题列，
      因此这里仍然只提供 label/score，文本信息全部放在 record["text"] 中。

    若后续在 eval 结果中增加 item_title/extern_id 相关列，可在此处一并补入。
    """

    topk = []
    for lbl, sc in zip(labels, scores):
        topk.append(
            {
                "label": str(lbl),
                "score": float(sc),
                "extern_id": "",  # 当前结果文件里没有对应列，保持为空
                "item_title": "",  # 同上，如后续有列可在此填充
            }
        )

    return {"record_id": record_id, "text": text, "topk": topk}


def _hit_at_k(true_label: str, preds: List[str], k: int) -> int:
    return 1 if str(true_label) in preds[:k] else 0


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.results_file)
    cols = set(df.columns)

    # 三种模式：
    # 1) predictions_eval.csv 直接评估（含 true_label + preds_top10/scores_top10）；
    # 2) eval.csv + predictions_eval.csv 联合评估（eval 提供文本，predictions_eval 提供 top10 与 true_label）；
    # 3) 将来可能的 eval 扩展格式（含 pred_top10/score_top10），当前暂不使用。

    use_predictions_style = {"preds_top10", "scores_top10"}.issubset(cols)
    use_eval_style = {"pred_top10", "score_top10"}.issubset(cols)

    # 纯 eval.csv（只有 case_title/performed_work/month/linked_items）时，
    # 我们需要 fallback 到 predictions_eval.csv 读取 top10 信息。
    is_plain_eval = (cols == {"case_title", "performed_work", "month", "linked_items"})

    if not (use_predictions_style or use_eval_style or is_plain_eval):
        raise ValueError(
            "results_file 需为以下格式之一：\n"
            "- predictions_eval 风格: 含 preds_top10, scores_top10, true_label 等列\n"
            "- eval 扩展风格: 含 pred_top10, score_top10\n"
            "- 纯 eval.csv: 仅含 case_title, performed_work, month, linked_items（将自动联表 predictions_eval.csv）"
        )

    # 联表模式：eval.csv + predictions_eval.csv
    if is_plain_eval:
        # 从 predictions_eval.csv 中取 true_label + top10
        pred_path = Path(args.results_file).with_name("predictions_eval.csv")
        if not pred_path.exists():
            raise FileNotFoundError(f"联表模式需要 {pred_path} 存在以提供 top10 预测结果")

        df_pred = pd.read_csv(pred_path)
        # eval.csv 的顺序与 predictions_eval.csv 的 index 一致（同一评估批次）
        if len(df) != len(df_pred):
            raise ValueError("eval.csv 与 predictions_eval.csv 行数不一致，无法一一对应联表")

        df = df.copy()
        df["true_label"] = df_pred["true_label"].astype(str)
        df["preds_top10"] = df_pred["preds_top10"].astype(str)
        df["scores_top10"] = df_pred["scores_top10"].astype(str)
        cols = set(df.columns)
        use_predictions_style = True

    # 真实标签列：优先 true_label，其次 open_true，再次 true，再次 linked_items（纯 eval.csv 场景）
    if "true_label" in df.columns:
        label_col = "true_label"
    elif "open_true" in df.columns:
        label_col = "open_true"
    elif "true" in df.columns:
        label_col = "true"
    elif "linked_items" in df.columns:
        label_col = "linked_items"
    else:
        raise KeyError("结果文件缺少 true_label / open_true / true / linked_items 任一列")

    if args.sample_size and 0 < args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        print(f"[eval_rerank] 使用子样本评估: n={len(df)}")

    top_k = args.top_k
    y_true = df[label_col].astype(str).tolist()

    base_topk: List[List[str]] = []
    rerank_topk: List[List[str]] = []

    for idx, row in df.iterrows():
        if use_predictions_style:
            labels, scores = _extract_topk_and_scores_from_predictions(row, top_k)
        else:
            labels, scores = _extract_topk_and_scores_from_eval(row, top_k)
        base_topk.append(labels)

        text = _build_text(row)
        record = _build_record(row, text, labels, scores, record_id=str(idx))
        reranked = qwen_rerank_single(record, top_k=top_k)
        reranked_labels = [str(c.get("label")) for c in reranked.get("reranked_topk", record["topk"])]
        rerank_topk.append(reranked_labels)

    n = len(df)
    def _aggregate(topk_lists: List[List[str]], name: str) -> dict:
        hits = {1: 0, 3: 0, 5: 0, 10: 0}
        for yt, preds in zip(y_true, topk_lists):
            for k in hits.keys():
                if _hit_at_k(yt, preds, k):
                    hits[k] += 1
        metrics = {f"hit@{k}": hits[k] / n for k in hits}
        metrics["accuracy"] = metrics["hit@1"]
        print(f"[{name}] metrics:")
        for k in ("hit@1", "hit@3", "hit@5", "hit@10", "accuracy"):
            print(f"  - {k} = {metrics[k]:.4f}")
        return metrics

    base_metrics = _aggregate(base_topk, "base")
    rerank_metrics = _aggregate(rerank_topk, "rerank")

    print("\n[eval_rerank] 差值 (rerank - base):")
    for k in ("hit@1", "hit@3", "hit@5", "hit@10", "accuracy"):
        b = base_metrics[k]
        r = rerank_metrics[k]
        print(f"  - {k}: Δ={r - b:+.4f} (base={b:.4f} -> rerank={r:.4f})")

    # 写逐样本详细结果
    out_path = Path(args.outfile or "eval_rerank_detailed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "true_label",
            "base_top10", "rerank_top10",
            "base_hit1", "rerank_hit1",
            "base_hit3", "rerank_hit3",
            "base_hit5", "rerank_hit5",
            "base_hit10", "rerank_hit10",
        ])

        for i, (yt, b_labels, r_labels) in enumerate(zip(y_true, base_topk, rerank_topk)):
            b1 = _hit_at_k(yt, b_labels, 1)
            r1 = _hit_at_k(yt, r_labels, 1)
            b3 = _hit_at_k(yt, b_labels, 3)
            r3 = _hit_at_k(yt, r_labels, 3)
            b5 = _hit_at_k(yt, b_labels, 5)
            r5 = _hit_at_k(yt, r_labels, 5)
            b10 = _hit_at_k(yt, b_labels, 10)
            r10 = _hit_at_k(yt, r_labels, 10)

            writer.writerow([
                i, yt,
                "|".join(b_labels),
                "|".join(r_labels),
                b1, r1, b3, r3, b5, r5, b10, r10,
            ])

    print(f"[eval_rerank] 逐样本详细结果已写入: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=str, required=True, help="predictions_eval.csv 路径")
    parser.add_argument("--top-k", type=int, default=10, help="参与 rerank 的 top_k 候选数，默认 10")
    parser.add_argument("--sample-size", type=int, default=0, help="可选，从结果中随机抽样的样本数（默认 0 表示使用全量）")
    parser.add_argument("--outfile", type=str, default="", help="保存逐样本详细结果的 CSV 路径")
    args = parser.parse_args()
    main(args)
