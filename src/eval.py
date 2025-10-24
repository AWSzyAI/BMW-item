#!/usr/bin/env python3
import os, json, argparse, ast, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score
)

from utils import ensure_single_label, build_text, hit_at_k, eval_split
# def ensure_single_label(s):
#     if isinstance(s, list):
#         return str(s[0]) if s else ""
#     if isinstance(s, str):
#         t = s.strip()
#         if (t.startswith("[") and t.endswith("]")) or (t.startswith("(") and t.endswith(")")):
#             try:
#                 v = ast.literal_eval(t)
#                 if isinstance(v, (list, tuple)) and len(v) > 0:
#                     return str(v[0])
#             except Exception:
#                 pass
#         return t
#     return str(s)
# def build_text(df):
#     parts = [df.get("case_title", "").fillna("").astype(str),
#              df.get("performed_work", "").fillna("").astype(str)]
#     if "item_title" in df.columns:
#         parts.append(df["item_title"].fillna("").astype(str))
#     return (parts[0] + " " + parts[1] + (" " + parts[2] if len(parts) > 2 else "")).astype(str)

# def hit_at_k(y_true_idx: np.ndarray, y_proba: np.ndarray, k: int) -> float:
#     """计算 hit@k 命中率"""
#     topk_idx = np.argsort(-y_proba, axis=1)[:, :k]
#     hits = (topk_idx == y_true_idx.reshape(-1, 1)).any(axis=1)
#     return float(hits.mean())



def main(args):

    # test.csv to be evaluated
    path = args.path

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # mode: new,dirty,clean
    mode = args.mode
    
    # path 参数有歧义，有时是在outdir之下，有时是完整路径
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
    # check X,y
    # X: case_title + performed_work, (case_submitted_date)//data有什么用呢？预测未来的故障趋势？根据季节/型号发售时间来检测集中爆发的故障？
    # y: linked_items, (item_title)//目前为止item_title还没有被用起来

    for col in ["case_title", "performed_work", "linked_items"]:
        if col not in df.columns:
            raise KeyError(f"{file_to_read} 缺少列：{col}")
    df["linked_items"] = df["linked_items"].apply(ensure_single_label).astype(str)
    X_text = build_text(df).tolist()
    y_raw  = df["linked_items"].astype(str).tolist()

    # 加载最优模型
    modeldir = args.modeldir
    model = args.model
    best_model_path = os.path.join(modeldir, model)
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found: {best_model_path}")
    bundle = joblib.load(best_model_path)
    model = bundle["model"]
    le = bundle["label_encoder"]
    ooc_detector = bundle.get("ooc_detector")

    results = {}

    if mode == "new":
        # new 模式：对整文件进行评估。
        # 先给出数据与训练类别的对照描述：训练集类别集合 vs 本次评估文件（视作测试集）的标签集合
        try:
            train_cls = list(map(str, le.classes_))
            train_cls_set = set(train_cls)
            test_labels_series = pd.Series([str(t) for t in y_raw])
            N_test = len(test_labels_series)
            uniq_test_labels = set(test_labels_series.unique().tolist())
            # 样本级统计
            known_mask = test_labels_series.isin(train_cls_set)
            n_known_samples = int(known_mask.sum())
            n_unknown_samples = N_test - n_known_samples
            # 标签级统计
            uniq_known_labels = sorted([l for l in uniq_test_labels if l in train_cls_set])
            uniq_unknown_labels = sorted([l for l in uniq_test_labels if l not in train_cls_set])

            print("\n[data] 训练类别集合与当前文件（测试集视角）对照")
            print("- Train: #classes=%d" % (len(train_cls)))
            # 只展示前若干个，避免过长
            if len(train_cls) > 0:
                preview = ", ".join(train_cls[:10]) + (" ..." if len(train_cls) > 10 else "")
                print("  classes preview: %s" % preview)
            print("- Test(file): N=%d | unique_labels=%d" % (N_test, len(uniq_test_labels)))
            print("  samples: known=%d | not_in_train=%d" % (n_known_samples, n_unknown_samples))
            print("  unique labels: known=%d | not_in_train=%d" % (len(uniq_known_labels), len(uniq_unknown_labels)))
            # 测试集中未在训练出现过的标签 Top-N（按样本数）
            if len(uniq_unknown_labels) > 0:
                cnt_unknown = (
                    test_labels_series[~known_mask]
                    .value_counts()  # type: ignore
                    .sort_values(ascending=False)
                )
                topn = cnt_unknown.head(10)
                print("  unseen-in-train labels in test (top by count):")
                for lbl, cnt in topn.items():
                    print(f"    {lbl}: {int(cnt)}")
        except Exception as _e:
            # 描述性输出失败不影响后续评估
            print(f"[data] 训练/测试标签描述时出现问题：{_e}")

        # 优先：若设置 --reject-threshold 或 --sweep-thresholds，则启用开放集（n+1）评估，基于最大概率阈值拒判为 not_in_train。
        def _compute_probs(Xs_all):
            try:
                return model.predict_proba(Xs_all)
            except Exception:
                try:
                    dec = model.decision_function(Xs_all)
                    if dec.ndim == 1:
                        probs_pos = 1 / (1 + np.exp(-dec))
                        return np.vstack([1 - probs_pos, probs_pos]).T
                    else:
                        e = np.exp(dec - np.max(dec, axis=1, keepdims=True))
                        return e / e.sum(axis=1, keepdims=True)
                except Exception:
                    raise RuntimeError("无法获得预测概率用于开放集评估")

        def _eval_open_set(y_proba_all: np.ndarray, thresholds):
            """对给定的一组阈值进行开放集评估，返回逐阈值的指标列表。
            指标包含：
            - overall: acc, f1_macro, f1_weighted, hit@1/3/5/10（将 not_in_train 视为一个标签，拒判视为 not_in_train）
            - known-only: known_hit@1/3/5/10（仅统计 true_in_train=True 的样本）
            - binary(not_in_train检测): not_in_train_precision/recall/f1/accuracy + 混淆计数
            - 其他：reject_rate, correct_reject, false_reject, false_accept
            """
            cls = le.classes_
            cls_set = set(cls)
            NOT_IN_TRAIN = args.not_in_train_label
            n = len(df)
            topk = min(10, len(cls))
            topk_idx = np.argsort(-y_proba_all, axis=1)[:, :topk]
            topk_labels = [[str(cls[j]) for j in row] for row in topk_idx]
            topk_scores = [[float(y_proba_all[i, j]) for j in topk_idx[i]] for i in range(n)]
            pred_top1 = [labels[0] if labels else "" for labels in topk_labels]
            max_prob = np.max(y_proba_all, axis=1)

            true_labels_orig = [str(t) for t in y_raw]
            true_in_train = np.array([t in cls_set for t in true_labels_orig], dtype=bool)
            open_true = np.array([t if it else NOT_IN_TRAIN for t, it in zip(true_labels_orig, true_in_train)], dtype=object)

            def _hit_k_idx(i, k, reject_mask_i):
                if open_true[i] == NOT_IN_TRAIN:
                    return 1 if reject_mask_i else 0
                if reject_mask_i:
                    return 0
                return 1 if open_true[i] in topk_labels[i][:k] else 0

            rows_by_thr = []
            for thr in thresholds:
                thr = float(thr)
                reject_mask = (max_prob < thr)
                open_pred = np.array([NOT_IN_TRAIN if reject_mask[i] else pred_top1[i] for i in range(n)], dtype=object)

                # overall metrics
                acc = accuracy_score(open_true, open_pred)
                f1_macro = f1_score(open_true, open_pred, average="macro", zero_division=0)
                f1_weighted = f1_score(open_true, open_pred, average="weighted", zero_division=0)
                hit1 = float(np.mean([_hit_k_idx(i, 1, reject_mask[i]) for i in range(n)]))
                hit3 = float(np.mean([_hit_k_idx(i, 3, reject_mask[i]) for i in range(n)]))
                hit5 = float(np.mean([_hit_k_idx(i, 5, reject_mask[i]) for i in range(n)]))
                hit10 = float(np.mean([_hit_k_idx(i, 10, reject_mask[i]) for i in range(n)]))

                # known-only hit@k
                if true_in_train.any():
                    known_idx = np.where(true_in_train)[0]
                    known_hit = lambda k: float(np.mean([
                        (0 if reject_mask[i] else (1 if (open_true[i] in topk_labels[i][:k]) else 0)) for i in known_idx
                    ]))
                    known_hit1 = known_hit(1)
                    known_hit3 = known_hit(3)
                    known_hit5 = known_hit(5)
                    known_hit10 = known_hit(10)
                else:
                    known_hit1 = known_hit3 = known_hit5 = known_hit10 = float("nan")

                # binary detection for not_in_train
                y_true_bin = ~true_in_train  # True 表示 not_in_train
                y_pred_bin = reject_mask
                TP = int(np.sum(y_pred_bin & y_true_bin))
                TN = int(np.sum((~y_pred_bin) & (~y_true_bin)))
                FP = int(np.sum(y_pred_bin & (~y_true_bin)))
                FN = int(np.sum((~y_pred_bin) & y_true_bin))
                prec_bin = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
                rec_bin = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
                f1_bin = float(2 * prec_bin * rec_bin / (prec_bin + rec_bin)) if (prec_bin + rec_bin) > 0 else 0.0
                acc_bin = float((TP + TN) / n) if n > 0 else float("nan")

                n_reject = int(np.sum(reject_mask))
                n_known = int(np.sum(true_in_train))
                n_unknown = n - n_known
                correct_reject = TP
                false_reject = FP
                false_accept = FN

                rows_by_thr.append({
                    "threshold": round(thr, 6),
                    "N": n,
                    "known": n_known,
                    "not_in_train": n_unknown,
                    "reject_rate": round(n_reject / n, 6),
                    # overall
                    "acc": round(float(acc), 6),
                    "f1_macro": round(float(f1_macro), 6),
                    "f1_weighted": round(float(f1_weighted), 6),
                    "hit@1": round(hit1, 6),
                    "hit@3": round(hit3, 6),
                    "hit@5": round(hit5, 6),
                    "hit@10": round(hit10, 6),
                    # known-only
                    "known_hit@1": round(known_hit1, 6) if not np.isnan(known_hit1) else known_hit1,
                    "known_hit@3": round(known_hit3, 6) if not np.isnan(known_hit3) else known_hit3,
                    "known_hit@5": round(known_hit5, 6) if not np.isnan(known_hit5) else known_hit5,
                    "known_hit@10": round(known_hit10, 6) if not np.isnan(known_hit10) else known_hit10,
                    # binary not_in_train detection
                    "not_in_train_precision": round(prec_bin, 6),
                    "not_in_train_recall": round(rec_bin, 6),
                    "not_in_train_f1": round(f1_bin, 6),
                    "not_in_train_accuracy": round(acc_bin, 6),
                    "correct_reject": correct_reject,
                    "false_reject": false_reject,
                    "false_accept": false_accept,
                    # for sample export
                    "_pred_top1": pred_top1,
                    "_topk_labels": topk_labels,
                    "_topk_scores": topk_scores,
                    "_max_prob": max_prob,
                    "_reject_mask": reject_mask,
                    "_open_true": open_true,
                    "_true_labels_orig": true_labels_orig,
                    "_true_in_train": true_in_train,
                })
            return rows_by_thr

        sweep_spec = getattr(args, "sweep_thresholds", None)
        # 若未显式提供阈值/扫描，且模型保存了基于MSP的阈值检测器，则默认采用其 tau 作为阈值
        if (getattr(args, "reject_threshold", None) is None) and (sweep_spec is None or len(str(sweep_spec).strip()) == 0):
            if isinstance(ooc_detector, dict) and ooc_detector.get("kind") == "threshold" and ("tau" in ooc_detector):
                args.reject_threshold = float(ooc_detector["tau"])  # MSP: p_max < tau => reject
                print(f"[new-open-set] 使用模型内置 MSP 阈值 tau={args.reject_threshold:.4f} 进行开放集评估")

        if (getattr(args, "reject_threshold", None) is not None) or (sweep_spec is not None and len(str(sweep_spec).strip()) > 0):
            Xs_all = X_text
            y_proba_all = _compute_probs(Xs_all)

            thresholds = []
            if sweep_spec is not None and len(str(sweep_spec).strip()) > 0:
                # 支持两种形式："0.1,0.2,0.3" 或 "0.1:0.9:0.02"
                spec = str(sweep_spec).strip()
                if ":" in spec:
                    try:
                        start, stop, step = [float(x) for x in spec.split(":")]
                        cur = start
                        # 包含 stop
                        while cur <= stop + 1e-12:
                            thresholds.append(round(cur, 6))
                            cur += step
                    except Exception:
                        raise ValueError("--sweep-thresholds 使用 start:stop:step 或 逗号分隔列表，如 0.1:0.9:0.02 或 0.2,0.25,0.3")
                else:
                    try:
                        thresholds = [float(x) for x in spec.split(",") if x.strip()]
                    except Exception:
                        raise ValueError("--sweep-thresholds 使用逗号分隔数值，如 0.2,0.25,0.3")
            else:
                thresholds = [float(args.reject_threshold)]

            rows_by_thr = _eval_open_set(y_proba_all, thresholds)

            # 如果是 sweep，保存 sweep 表；若是单阈值，同时导出逐样本预测
            sweep_df = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows_by_thr])
            sweep_path = os.path.join(outdir, "threshold_sweep.csv")
            sweep_df.to_csv(sweep_path, index=False, encoding="utf-8-sig")
            print(f"开放集阈值扫描结果已保存：{sweep_path}")

            # 控制台打印：
            if len(thresholds) == 1:
                # 单阈值：全面打印（包含/不包含 not-in-train、not-in-train 二分类、整体 hit@1/3/5/10）
                r = rows_by_thr[0]
                print("\n[new-open-set] Summary at threshold=%.4f" % r["threshold"]) 
                print("- Counts: N=%d | known=%d | not_in_train=%d | reject_rate=%.3f" % (
                    r["N"], r["known"], r["not_in_train"], r["reject_rate"]
                ))
                # overall（包含 not-in-train 作为一个类）
                print("- Overall: acc=%.3f | f1_macro=%.3f | f1_weighted=%.3f" % (
                    r["acc"], r["f1_macro"], r["f1_weighted"]
                ))
                print("- Overall hit@k: hit@1=%.3f | hit@3=%.3f | hit@5=%.3f | hit@10=%.3f" % (
                    r["hit@1"], r["hit@3"], r["hit@5"], r["hit@10"]
                ))
                # known-only（不包含 not-in-train）
                print("- Known-only hit@k: hit@1=%.3f | hit@3=%.3f | hit@5=%.3f | hit@10=%.3f" % (
                    r["known_hit@1"], r["known_hit@3"], r["known_hit@5"], r["known_hit@10"]
                ))
                # not-in-train 二分类
                print("- Not-in-train detection: precision=%.3f | recall=%.3f | f1=%.3f | acc=%.3f" % (
                    r["not_in_train_precision"], r["not_in_train_recall"], r["not_in_train_f1"], r["not_in_train_accuracy"]
                ))
                # Final 决策评测（与 predict 的 Final 等价）：
                # - final_acc = Overall acc（n+1 分类准确率）
                # - final_known_acc = Known-only 的 hit@1（拒判计为错误）
                # - final_unknown_recall/precision/f1 = 上述 not-in-train 二分类指标
                print("- Final decision: final_acc=%.3f | final_known_acc=%.3f | final_unknown_precision=%.3f | final_unknown_recall=%.3f | final_unknown_f1=%.3f" % (
                    r["acc"], r["known_hit@1"], r["not_in_train_precision"], r["not_in_train_recall"], r["not_in_train_f1"]
                ))
            else:
                # 多阈值扫：打印按 not_in_train_f1 排名前三的阈值摘要，完整见 CSV
                rows_sorted = sorted(rows_by_thr, key=lambda r: r["not_in_train_f1"], reverse=True)
                topn = rows_sorted[:min(3, len(rows_sorted))]
                print("\n[new-open-set] Top thresholds by not_in_train_f1 (see threshold_sweep.csv for full):")
                for r in topn:
                    print(
                        "  thr=%.4f | overall hit@1=%.3f hit@3=%.3f hit@5=%.3f hit@10=%.3f | known_hit@3=%.3f | "
                        "OOD: prec=%.3f rec=%.3f f1=%.3f acc=%.3f | reject_rate=%.3f | "
                        "Final: acc=%.3f known_acc=%.3f" % (
                            r["threshold"], r["hit@1"], r["hit@3"], r["hit@5"], r["hit@10"], r["known_hit@3"],
                            r["not_in_train_precision"], r["not_in_train_recall"], r["not_in_train_f1"], r["not_in_train_accuracy"], r["reject_rate"],
                            r["acc"], r["known_hit@1"]
                        )
                    )
                # 同时标注最佳行
                best_row = rows_sorted[0]
                print(
                    f"[new-open-set] best_by_f1 threshold={best_row['threshold']:.3f} | known_hit@3={best_row['known_hit@3']:.3f} | "
                    f"not_in_train_recall={best_row['not_in_train_recall']:.3f} | not_in_train_f1={best_row['not_in_train_f1']:.3f}"
                )

            # 若仅单阈值，导出逐样本
            if len(thresholds) == 1:
                row = rows_by_thr[0]
                pred_top1 = row["_pred_top1"]
                topk_labels = row["_topk_labels"]
                topk_scores = row["_topk_scores"]
                max_prob = row["_max_prob"]
                reject_mask = row["_reject_mask"]
                open_true = row["_open_true"]
                true_labels_orig = row["_true_labels_orig"]
                true_in_train = row["_true_in_train"]
                NOT_IN_TRAIN = args.not_in_train_label

                def _hit_k(i, k):
                    if open_true[i] == NOT_IN_TRAIN:
                        return 1 if reject_mask[i] else 0
                    if reject_mask[i]:
                        return 0
                    return 1 if open_true[i] in topk_labels[i][:k] else 0

                n = len(df)
                rows = []
                for i in range(n):
                    rows.append({
                        "index": i,
                        "case_id": df.iloc[i].get("case_id"),
                        "true_label": true_labels_orig[i],
                        "true_in_train": bool(true_in_train[i]),
                        "open_true": open_true[i],
                        "open_pred": (args.not_in_train_label if reject_mask[i] else pred_top1[i]),
                        "pred_top1": pred_top1[i],
                        "preds_top10": "|".join(topk_labels[i]),
                        "scores_top10": "|".join(f"{s:.6f}" for s in topk_scores[i]),
                        "max_prob": float(max_prob[i]),
                        "rejected_by_threshold": bool(reject_mask[i]),
                        "hit@1": _hit_k(i, 1),
                        "hit@3": _hit_k(i, 3),
                        "hit@5": _hit_k(i, 5),
                        "hit@10": _hit_k(i, 10),
                    })
                pred_df = pd.DataFrame(rows)
                base = os.path.splitext(os.path.basename(file_to_read))[0]
                pred_out = os.path.join(outdir, f"predictions_{base}.csv")
                pred_df.to_csv(pred_out, index=False, encoding="utf-8-sig")
                print(f"逐样本预测已保存：{pred_out}")

            # 同步写一份 metrics_best_model_all_splits.csv（首行作为代表）
            head = {k: v for k, v in rows_by_thr[0].items() if not k.startswith("_")}
            results["new-open-set"] = head
            dfm = pd.DataFrame([dict(split=k, **v) for k, v in results.items()])
            out_path = os.path.join(outdir, "metrics_best_model_all_splits.csv")
            dfm.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"\nBest model metrics saved to {out_path}")
            return

        # 未设置拒判阈值：沿用未知标签策略（exclude | map-to-other | tag-not-in-train）
        # unknown 处理策略：exclude | map-to-other | tag-not-in-train
        unknown_policy = getattr(args, "unknown_policy", "tag-not-in-train")
        # 兼容旧开关：--map-unknown-to-other 优先
        cls_set = set(le.classes_)
        use_mapping = bool(getattr(args, "map_unknown_to_other", False) and (args.other_label in cls_set))
        if args.map_unknown_to_other and not (args.other_label in cls_set):
            print(f"[new] 提示：要求映射未知标签为 '{args.other_label}'，但模型未包含该类。将回退为排除未知标签的评估方式。")

        # 统计已知/未知数量
        known_mask = [lbl in cls_set for lbl in y_raw]
        known_indices = [i for i, ok in enumerate(known_mask) if ok]
        unknown_indices = [i for i, ok in enumerate(known_mask) if not ok]
        print(f"[new] 样本总数={len(y_raw)}，in_train={len(known_indices)}，not_in_train={len(unknown_indices)}")

        if use_mapping or (unknown_policy == "map-to-other" and (args.other_label in cls_set)):
            y_raw_mapped = [lbl if lbl in cls_set else args.other_label for lbl in y_raw]
            mapped_cnt = sum(1 for o, m in zip(y_raw, y_raw_mapped) if o != m)
            if mapped_cnt > 0:
                print(f"[new] 已将 {mapped_cnt} 条未知标签映射为 '{args.other_label}' 并纳入指标计算。")
            all_indices = list(range(len(y_raw_mapped)))
            m = eval_split(model, le, X_text, y_raw_mapped, all_indices)
        else:
            # tag-not-in-train 或 exclude：仅对已知标签样本计算指标
            if len(known_indices) == 0:
                raise ValueError("新文件中没有任何标签出现在已训练的类别中，无法计算分类指标。")
            if len(unknown_indices) > 0:
                print(f"[new] 注意：共有 {len(unknown_indices)} 条样本的标签不在训练类别中（记为 not_in_train）")
            m = eval_split(model, le, X_text, y_raw, known_indices)
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

        # 命中率：若启用映射/策略为 map-to-other，则使用映射后的标签；否则，未知标签命中置 NaN
        true_labels_orig = [str(t) for t in y_raw]
        if use_mapping:
            true_labels_mapped = [t if t in cls_set else args.other_label for t in true_labels_orig]
        else:
            true_labels_mapped = true_labels_orig

        def _hit_at(preds, true, k):
            if (not use_mapping) and (true not in cls_set):
                return np.nan
            return 1 if true in preds[:k] else 0

        rows = []
        for i in range(len(df)):
            preds_i = topk_labels[i]
            scores_i = topk_scores[i]
            true_i = true_labels_orig[i]
            true_i_m = true_labels_mapped[i]
            rows.append({
                "index": i,
                "case_id": df.iloc[i].get("case_id"),
                "true_label": true_i,
                "true_label_mapped": true_i_m if true_i_m != true_i else "",
                "true_in_train": (true_i in cls_set),
                "pred_top1": pred_top1[i],
                "preds_top10": "|".join(preds_i),
                "scores_top10": "|".join(f"{s:.6f}" for s in scores_i),
                "hit@1": _hit_at(preds_i, true_i_m, 1),
                "hit@3": _hit_at(preds_i, true_i_m, 3),
                "hit@5": _hit_at(preds_i, true_i_m, 5),
                "hit@10": _hit_at(preds_i, true_i_m, 10),
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

    # 在非 new 模式下，增加对 train/test 标签集合的描述，以及 test 中未见过的标签情况
    try:
        if len(all_splits.get("train", [])) > 0 and len(all_splits.get("test", [])) > 0:
            train_idx = all_splits["train"]
            test_idx = all_splits["test"]
            y_train = [str(ensure_single_label(df.iloc[i]["linked_items"])) for i in train_idx]
            y_test = [str(ensure_single_label(df.iloc[i]["linked_items"])) for i in test_idx]
            train_set = set(y_train)
            test_set = set(y_test)
            unseen_labels = sorted(list(test_set - train_set))
            # 样本级数量
            test_unknown_samples = int(sum(1 for t in y_test if t not in train_set))
            print("\n[data] Split label summary (from folds.json)")
            print("- Train: N=%d | unique_labels=%d" % (len(train_idx), len(train_set)))
            print("- Test:  N=%d | unique_labels=%d" % (len(test_idx), len(test_set)))
            print("  test samples: known=%d | not_in_train=%d" % (len(test_idx) - test_unknown_samples, test_unknown_samples))
            print("  test unique labels: known=%d | not_in_train=%d" % (len(test_set & train_set), len(unseen_labels)))
            if len(unseen_labels) > 0:
                # 统计 test 中未见过标签的样本数，输出 top-N
                from collections import Counter
                cnt = Counter([t for t in y_test if t not in train_set])
                print("  unseen-in-train labels in test (top by count):")
                for lbl, c in cnt.most_common(10):
                    print(f"    {lbl}: {c}")
    except Exception as _e:
        print(f"[data] 基于折划分的标签描述时出现问题：{_e}")

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
    
    parser.add_argument("--modeldir",type=str,default="./models")
    parser.add_argument("--model", type=str, default="8.joblib",
                        help="用于评估的模型文件名，位于 outdir 下")
    parser.add_argument("--path", type=str,default="test.csv")
    parser.add_argument("--outdir", type=str, default="./output/2025_up_to_month_8")
    parser.add_argument("--mode", type=str, default="new", choices=["clean", "dirty", "new"],
                        help="评估模式：clean=val+test，dirty=train+val+test，new=对传入文件整体评估")
    parser.add_argument("--map-unknown-to-other", action="store_true", help="在 new 模式下，将未知标签映射为 __OTHER__ 并纳入指标计算（模型需包含该类）")
    parser.add_argument("--other-label", type=str, default="__OTHER__", help="用于接收未知/稀有标签的合并标签名")
    parser.add_argument("--unknown-policy", type=str, default="tag-not-in-train", choices=["exclude", "map-to-other", "tag-not-in-train"],
                        help="未知标签处理策略：exclude=排除；map-to-other=映射到 other_label；tag-not-in-train=仅标注但不纳入指标（仅在未设置阈值时生效）")
    # 开放集（n+1）拒判阈值：设置后将启用开放集评估，并覆盖上述 unknown 策略
    parser.add_argument("--reject-threshold", type=float, default=None,
                        help="当最大类别概率 < 阈值时拒判为 not_in_train（开启开放集 n+1 评估）")
    parser.add_argument("--not-in-train-label", type=str, default="__NOT_IN_TRAIN__",
                        help="开放集评估中用于拒判的标签名")
    parser.add_argument("--sweep-thresholds", type=str, default=None,
                        help="开放集阈值扫描（覆盖 --reject-threshold）。格式：'0.1,0.2,0.3' 或 '0.1:0.9:0.02'")
    args = parser.parse_args()
    main(args)
