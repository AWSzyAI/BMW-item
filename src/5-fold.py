#!/usr/bin/env python3
"""
按标签分层拆分数据，输出 train.csv / eval.csv / test.csv。
规则：
- 先将标签列 linked_items 规范为单标签字符串；
- 过滤掉样本数 < 5 的类别（确保每个 y 至少 5 个样本）；
- 对于每个类别，随机选取 1 个样本作为 test，另选 1 个不同样本作为 eval，其余归为 train；
- 输出到 --outdir 目录下：train.csv, eval.csv, test.csv。

用法示例：
  python ./src/5-fold.py --path ./output/df_filtered_xxx.csv --outdir ./output --seed 42
"""
import os
import ast
import argparse
import pandas as pd
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

    # 分层拆分
    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=args.seed).index  # for stable shuffle order
    df = df.loc[rng].reset_index(drop=True)

    label_to_rows = defaultdict(list)
    for i, y in enumerate(df["linked_items"].tolist()):
        label_to_rows[y].append(i)

    test_idx, eval_idx, train_idx = [], [], []
    split_type = str(getattr(args, "split_type", "2")).strip()
    train_ratio = float(getattr(args, "train_ratio", 0.8))
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio 必须在 (0,1) 区间，例如 0.8 代表 4:1 切分")
    for y, idxs in label_to_rows.items():
        if len(idxs) < min_count:
            # 理论上不会发生（已过滤），防御性判断
            continue
        if split_type == "3":
            # 先取 1 个样本作为 test，剩余按 4:1（train:eval）比例切分
            t = idxs[0]
            rem = idxs[1:]
            n = len(rem)
            if n <= 1:
                # 极端情况兜底
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
            # 2 分法：全量按 4:1（train:eval）比例切分
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

    # 构造并保存
    
    df_train = df.iloc[sorted(train_idx)].reset_index(drop=True)
    df_eval  = df.iloc[sorted(eval_idx)].reset_index(drop=True) 
    if split_type == "3":
        df_test = df.iloc[sorted(test_idx)].reset_index(drop=True)
    out_eval = os.path.join(args.outdir, "eval.csv") 
    out_train = os.path.join(args.outdir, "train.csv")
    out_test = os.path.join(args.outdir, "test.csv") if split_type == "3" else None
    

    df_train.to_csv(out_train, index=False, encoding="utf-8-sig")
    if out_eval is not None and df_eval is not None:
        df_eval.to_csv(out_eval, index=False, encoding="utf-8-sig")
    if out_test is not None and df_test is not None:
        df_test.to_csv(out_test, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_train} ({len(df_train)} rows)")
    if out_eval is not None and df_eval is not None:
        print(f"Saved: {out_eval} ({len(df_eval)} rows)")
    if out_test is not None and df_test is not None:
        print(f"Saved: {out_test} ({len(df_test)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./output/m_train_eval.csv", help="输入 CSV 文件路径（绝对或相对）")
    parser.add_argument("--outdir", type=str, default="./output", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--min_count", type=int, default=5, help="少见标签阈值与最低样本数要求")
    parser.add_argument("--map_rare_to_other", action="store_true", help="将少见标签映射为 __OTHER__")
    parser.add_argument("--split_type", type=str, default="2", help="3=每类先取1条test，剩余按比例切成train/eval；2=仅按比例切成train/eval")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train:eval 的比例（默认 0.8，即 4:1）")
    args = parser.parse_args()
    main(args)
