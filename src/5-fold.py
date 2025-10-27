#!/usr/bin/env python3
"""
分层切分工具：支持 5-fold（或 K-fold）与按比例的分层切分。

主要用途：
- 对输入 CSV（或 outdir/path）进行标签规范化后，执行 StratifiedKFold 切分，
    选择某一个 fold 作为 eval，其余作为 train；
- 或者按比例（train_ratio）进行分层切分（兼容旧行为）。

输出：在 --outdir 下生成 train.csv 与 eval.csv（可选 test.csv 见旧 3-way 模式）。

用法示例：
    # 使用 5 折并选择第 0 折为 eval
    uv run python ./src/5-fold.py --path ./output/2025_up_to_month_9/train_all.csv \
            --outdir ./output/2025_up_to_month_9 --method kfold --k 5 --fold-index 0 --seed 42
"""
import os
import ast
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict


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


def read_input(path: str, outdir: str) -> pd.DataFrame:
    candidates = [path, os.path.join(outdir, path)]
    for p in candidates:
        if p and os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"找不到输入文件：{path} 或 {os.path.join(outdir, path)}")


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    df = read_input(args.path, args.outdir)
    for col in ["case_title", "performed_work", "linked_items"]:
        if col not in df.columns:
            raise KeyError(f"输入缺少列：{col}")

    # 规范标签
    df = df.copy()
    df["linked_items"] = df["linked_items"].apply(ensure_single_label).astype(str)

    # 可选：将少见标签并入 __OTHER__
    min_count = int(args.min_count)
    if getattr(args, "map_rare_to_other", False):
        counts0 = df["linked_items"].value_counts()
        rare_labels = set(counts0[counts0 < min_count].index)
        if rare_labels:
            df.loc[df["linked_items"].isin(rare_labels), "linked_items"] = "__OTHER__"
            print(f"[info] 将 {len(rare_labels)} 个少见标签(频次<{min_count}) 映射为 __OTHER__")

    # 要求：每类样本数 >= min_count（映射后再次检查）
    counts = df["linked_items"].value_counts()
    valid_labels = set(counts[counts >= min_count].index)
    df = df[df["linked_items"].isin(valid_labels)].reset_index(drop=True)
    if df.empty:
        raise ValueError("过滤/映射后数据为空，请检查标签分布或降低阈值。")

    # 方法选择：kfold 或 比例切分
    method = str(getattr(args, "method", "kfold")).lower()
    if method == "kfold":
        k = int(getattr(args, "k", 5))
        fold_index = int(getattr(args, "fold_index", 0))
        if not (0 <= fold_index < k):
            raise ValueError(f"fold_index 必须在 [0,{k-1}] 区间")
        # 检查每类样本数 >= k
        counts2 = df["linked_items"].value_counts()
        insufficient = counts2[counts2 < k]
        if len(insufficient) > 0:
            print(f"[警告] 以下类别样本数 < k={k}，将被移除：{list(insufficient.index)[:10]} ...")
            df = df[df["linked_items"].isin(counts2[counts2 >= k].index)].reset_index(drop=True)
            if df.empty:
                raise ValueError("移除稀有类别后数据为空，请降低 k 或放宽 min_count。")

        X_idx = pd.Series(range(len(df)))
        y = df["linked_items"].astype(str).values
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
        # 选择指定折为 eval，其余为 train
        splits = list(skf.split(X_idx, y))
        tr_idx, ev_idx = splits[fold_index]
        df_train = df.iloc[tr_idx].reset_index(drop=True)
        df_eval = df.iloc[ev_idx].reset_index(drop=True)
        out_eval = os.path.join(args.outdir, "eval.csv")
        out_train = os.path.join(args.outdir, "train.csv")
        df_train.to_csv(out_train, index=False, encoding="utf-8-sig")
        df_eval.to_csv(out_eval, index=False, encoding="utf-8-sig")
        print(f"[kfold] Saved: {out_train} ({len(df_train)}) | {out_eval} ({len(df_eval)}) | k={k}, fold={fold_index}")
    else:
        # 兼容旧逻辑：按比例分层 + 可选 3 路切分
        rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=args.seed).index  # for stable shuffle order
        df = df.loc[rng].reset_index(drop=True)
        label_to_rows = defaultdict(list)
        for i, yv in enumerate(df["linked_items"].tolist()):
            label_to_rows[yv].append(i)

        test_idx, eval_idx, train_idx = [], [], []
        split_type = str(getattr(args, "split_type", "2")).strip()
        train_ratio = float(getattr(args, "train_ratio", 0.8))
        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio 必须在 (0,1) 区间，例如 0.8 代表 4:1 切分")
        for yv, idxs in label_to_rows.items():
            if len(idxs) < min_count:
                continue
            if split_type == "3":
                t = idxs[0]
                rem = idxs[1:]
                n = len(rem)
                if n <= 1:
                    test_idx.append(t)
                    train_idx.extend(rem)
                    continue
                n_eval = max(1, int(round(n * (1 - train_ratio))))
                n_eval = min(n_eval, n - 1)
                e = rem[:n_eval]
                tr = rem[n_eval:]
                test_idx.append(t)
                eval_idx.extend(e)
                train_idx.extend(tr)
            else:
                n = len(idxs)
                if n <= 1:
                    train_idx.extend(idxs)
                    continue
                n_eval = max(1, int(round(n * (1 - train_ratio))))
                n_eval = min(n_eval, n - 1)
                e = idxs[:n_eval]
                tr = idxs[n_eval:]
                eval_idx.extend(e)
                train_idx.extend(tr)

        df_train = df.iloc[sorted(train_idx)].reset_index(drop=True)
        df_eval  = df.iloc[sorted(eval_idx)].reset_index(drop=True)
        out_eval = os.path.join(args.outdir, "eval.csv")
        out_train = os.path.join(args.outdir, "train.csv")
        df_train.to_csv(out_train, index=False, encoding="utf-8-sig")
        df_eval.to_csv(out_eval, index=False, encoding="utf-8-sig")
        print(f"[ratio] Saved: {out_train} ({len(df_train)}) | {out_eval} ({len(df_eval)}) | train_ratio={train_ratio}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./output/2025_up_to_month_7/train_eval.csv", help="输入 CSV 文件路径（绝对或相对）")
    parser.add_argument("--outdir", type=str, default="./output", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--min_count", type=int, default=5, help="少见标签阈值与最低样本数要求（建议与 k 相同）")
    parser.add_argument("--map_rare_to_other", action="store_true", help="将少见标签映射为 __OTHER__")
    # 新增：kfold 参数
    parser.add_argument("--method", type=str, default="kfold", choices=["kfold", "ratio"], help="切分方法：kfold 或 ratio")
    parser.add_argument("--k", type=int, default=5, help="kfold 的折数")
    parser.add_argument("--fold-index", dest="fold_index", type=int, default=0, help="作为 eval 的折编号 [0..k-1]")
    # 兼容旧的 2/3 分法
    parser.add_argument("--split_type", type=str, default="2", help="ratio 模式下：3=每类先取1条test，剩余按比例切成train/eval；2=仅按比例切成train/eval")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="ratio 模式下：train:eval 的比例（默认 0.8，即 4:1）")
    args = parser.parse_args()
    main(args)
