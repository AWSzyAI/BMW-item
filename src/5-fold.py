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

    # 过滤样本数 <5 的类别
    counts = df["linked_items"].value_counts()
    valid_labels = set(counts[counts >= 5].index)
    df = df[df["linked_items"].isin(valid_labels)].reset_index(drop=True)
    if df.empty:
        raise ValueError("过滤后数据为空，检查标签分布或降低阈值。")

    # 分层拆分
    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=args.seed).index  # for stable shuffle order
    df = df.loc[rng].reset_index(drop=True)

    label_to_rows = defaultdict(list)
    for i, y in enumerate(df["linked_items"].tolist()):
        label_to_rows[y].append(i)

    test_idx, eval_idx, train_idx = [], [], []
    for y, idxs in label_to_rows.items():
        if len(idxs) < 5:
            # 理论上不会发生（已过滤），防御性判断
            continue
        # 取两个不同样本：一个给 test，一个给 eval
        t = idxs[0]
        v = idxs[1]
        rest = idxs[2:]
        test_idx.append(t)
        eval_idx.append(v)
        train_idx.extend(rest)

    # 构造并保存
    df_test = df.iloc[sorted(test_idx)].reset_index(drop=True)
    df_eval = df.iloc[sorted(eval_idx)].reset_index(drop=True)
    df_train = df.iloc[sorted(train_idx)].reset_index(drop=True)

    out_train = os.path.join(args.outdir, "train.csv")
    out_eval = os.path.join(args.outdir, "eval.csv")
    out_test = os.path.join(args.outdir, "test.csv")

    df_train.to_csv(out_train, index=False, encoding="utf-8-sig")
    df_eval.to_csv(out_eval, index=False, encoding="utf-8-sig")
    df_test.to_csv(out_test, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_train} ({len(df_train)} rows)")
    print(f"Saved: {out_eval} ({len(df_eval)} rows)")
    print(f"Saved: {out_test} ({len(df_test)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./data/m_train.csv", help="输入 CSV 文件路径（绝对或相对）")
    parser.add_argument("--outdir", type=str, default="./output", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    main(args)
