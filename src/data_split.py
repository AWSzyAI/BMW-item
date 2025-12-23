#!/usr/bin/env python3
"""
统一的数据切分脚本：按月份划分 train/test，并在 train 内做 eval 分层切分（kfold 或比例），
整合了原 2025.py 与 5-fold.py 的能力，保证：
1) train = 1..(month-1)，test = month（保留 test 中未见过的类别，供开放集评估）；
2) eval 从 train 内部分层切分，默认 kfold；
3) 产出 train/eval/test 单表与 X/y 分表、label_mapping.csv、train_eval_raw/test_raw。
"""
import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import ensure_single_label


def _fmt_no_decimal(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s == "":
        return ""
    try:
        v = float(s)
        if v.is_integer():
            return str(int(v))
    except Exception:
        pass
    return s


def _mode_or_empty(series: pd.Series) -> str:
    """返回非空众数，若全为空或序列为空则返回空串。"""
    if series is None or len(series) == 0:
        return ""
    s = series.astype(str).str.strip()
    s = s[~s.isin(["", "nan", "NaN", "NAN", "None", "none"])]
    if len(s) == 0:
        return ""
    try:
        return s.value_counts().index[0]
    except Exception:
        return ""


def _read_csv_with_fallback(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="gbk")
        except UnicodeDecodeError:
            return pd.read_csv(path)


def _feature_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in ["case_id", "case_title", "performed_work", "month"]:
        if c in df.columns:
            cols.append(c)
    return cols


def _encode_with_le(le: LabelEncoder, labels: List[str]) -> np.ndarray:
    classes = set(le.classes_)
    out = []
    for lbl in labels:
        if lbl in classes:
            out.append(int(le.transform([lbl])[0]))
        else:
            out.append(-1)
    return np.array(out, dtype=int)


def _save_split(df_split: pd.DataFrame, name: str, outdir: Path, le: LabelEncoder) -> tuple[str, str, str]:
    out_single = outdir / f"{name}.csv"
    df_split.to_csv(out_single, index=False, encoding="utf-8-sig")

    feat_cols = _feature_cols(df_split)
    X_df = df_split[feat_cols] if feat_cols else pd.DataFrame(index=df_split.index)
    X_path = outdir / f"{name}_X.csv"
    X_df.to_csv(X_path, index=False, encoding="utf-8-sig")

    y_enc = _encode_with_le(le, df_split["linked_items"].astype(str).tolist())
    y_df = pd.DataFrame({"linked_items": y_enc})
    Y_path = outdir / f"{name}_y.csv"
    y_df.to_csv(Y_path, index=False, encoding="utf-8-sig")
    return str(out_single), str(X_path), str(Y_path)


def _build_label_mapping(train_df: pd.DataFrame, raw_df: pd.DataFrame, label_col: str, outdir: Path) -> None:
    le = LabelEncoder()
    le.fit(train_df["linked_items"].astype(str))

    item_title_map = {}
    extern_id_map = {}
    orig_linked_map = {}
    itemcreationdate_map = {}
    case_id_map = {}

    all_label_series = raw_df[label_col].astype(str)
    for raw_label in le.classes_:
        mask = all_label_series.astype(str) == str(raw_label)
        sub = raw_df[mask]
        if "item_title" in sub.columns:
            item_title_map[raw_label] = _mode_or_empty(sub["item_title"])
        else:
            item_title_map[raw_label] = ""
        if "case_id" in sub.columns:
            case_id_map[raw_label] = _mode_or_empty(sub["case_id"])
        else:
            case_id_map[raw_label] = ""
        if label_col == "extern_id":
            extern_id_map[raw_label] = str(raw_label)
        else:
            if "extern_id" in sub.columns:
                extern_id_map[raw_label] = _mode_or_empty(sub["extern_id"])
            else:
                extern_id_map[raw_label] = ""
        if "linked_items" in sub.columns:
            orig_linked_map[raw_label] = _mode_or_empty(sub["linked_items"])
        else:
            orig_linked_map[raw_label] = str(raw_label) if label_col == "linked_items" else ""
        if "itemcreationdate" in sub.columns:
            itemcreationdate_map[raw_label] = _mode_or_empty(sub["itemcreationdate"])
        else:
            itemcreationdate_map[raw_label] = ""

    display_vals = [str(li) for li in le.classes_]
    mapping_df = pd.DataFrame(
        {
            "linked_items": display_vals,
            "label": range(len(le.classes_)),
            "item_title": [item_title_map.get(li, "") for li in le.classes_],
            "extern_id": [extern_id_map.get(li, "") for li in le.classes_],
            "case_id": [case_id_map.get(li, "") for li in le.classes_],
            "orig_linked_items": [orig_linked_map.get(li, str(li)) for li in le.classes_],
            "itemcreationdate": [itemcreationdate_map.get(li, "") for li in le.classes_],
        }
    )
    mapping_df = mapping_df.map(_fmt_no_decimal)
    mapping_df.to_csv(outdir / "label_mapping.csv", index=False, encoding="utf-8-sig")


def split_data(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = _read_csv_with_fallback(args.src_path)
    df.columns = df.columns.str.strip()

    required_cols = ["case_title", "performed_work", "case_submitted_date", args.label_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"输入缺少列：{missing}")

    # 计算 month 列
    df = df.copy()
    df["month"] = pd.to_datetime(df["case_submitted_date"]).dt.to_period("M")

    if args.month <= 1:
        raise ValueError("--month 必须大于 1，才能使用 1..month-1 作为训练集")

    # 训练月 = 1..month-1，测试月 = month
    month_groups = df.groupby("month")
    train_months = list(range(1, args.month))

    train_parts = []
    for m in month_groups.groups:
        if m.month in train_months:
            train_parts.append(month_groups.get_group(m))
    if not train_parts:
        raise ValueError(f"未找到训练月份数据：{train_months}")
    train_df_raw = pd.concat(train_parts, axis=0)

    try:
        test_group = next(m for m in month_groups.groups if m.month == args.month)
    except StopIteration:
        raise ValueError(f"未找到测试月份 {args.month} 的数据")
    test_df = month_groups.get_group(test_group)

    # 标准化标签列
    label_col = args.label_col
    for d in (train_df_raw, test_df):
        d["linked_items"] = d[label_col].apply(ensure_single_label).astype(str)

    # 仅在训练集上处理少样本/映射
    work_train = train_df_raw.copy()
    label_counts = work_train["linked_items"].value_counts()
    if args.map_rare_to_other:
        rare = set(label_counts[label_counts < args.min_count].index)
        if rare:
            work_train.loc[work_train["linked_items"].isin(rare), "linked_items"] = args.other_label
            print(f"[info] 训练集中 {len(rare)} 个少见标签映射为 {args.other_label}")
    label_counts = work_train["linked_items"].value_counts()
    valid_labels = set(label_counts[label_counts >= args.min_count].index)
    work_train = work_train[work_train["linked_items"].isin(valid_labels)].reset_index(drop=True)
    if work_train.empty:
        raise ValueError("过滤/映射后训练集为空，请检查标签分布或降低阈值/折数。")

    # month 列转字符串，避免 period 类型带来的不一致
    for d in (work_train, test_df):
        d.loc[:, "month"] = d["month"].astype(str)
    train_df_raw.loc[:, "month"] = train_df_raw["month"].astype(str)

    # 在训练集内部做 eval 切分
    if args.method == "kfold":
        k = int(args.k)
        fold_index = int(args.fold_index)
        if not (0 <= fold_index < k):
            raise ValueError(f"fold_index 必须在 [0,{k-1}] 区间")
        counts = work_train["linked_items"].value_counts()
        insufficient = counts[counts < k]
        if len(insufficient) > 0:
            print(f"[警告] 以下类别样本数 < k={k}，将被移除：{list(insufficient.index)[:10]} ...")
            work_train = work_train[work_train["linked_items"].isin(counts[counts >= k].index)].reset_index(drop=True)
            if work_train.empty:
                raise ValueError("移除稀有类别后训练集为空，请降低 k 或放宽 min_count。")

        X_idx = pd.Series(range(len(work_train)))
        y = work_train["linked_items"].astype(str).values
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
        splits = list(skf.split(X_idx, y))
        tr_idx, ev_idx = splits[fold_index]
        df_train = work_train.iloc[tr_idx].reset_index(drop=True)
        df_eval = work_train.iloc[ev_idx].reset_index(drop=True)
    else:
        tr_df, ev_df = train_test_split(
            work_train,
            test_size=(1 - args.train_ratio),
            random_state=args.seed,
            stratify=work_train["linked_items"].astype(str),
        )
        df_train = tr_df.reset_index(drop=True)
        df_eval = ev_df.reset_index(drop=True)

    # LabelEncoder 基于训练集
    le = LabelEncoder()
    le.fit(df_train["linked_items"].astype(str))

    # 写出分割文件
    tr_single, tr_X, tr_y = _save_split(df_train, "train", outdir, le)
    ev_single, ev_X, ev_y = _save_split(df_eval, "eval", outdir, le)
    ts_single, ts_X, ts_y = _save_split(test_df.reset_index(drop=True), "test", outdir, le)

    print(f"[data_split] Saved: train({len(df_train)}) -> {tr_X}, {tr_y} | eval({len(df_eval)}) -> {ev_X}, {ev_y} | test({len(test_df)}) -> {ts_X}, {ts_y}")
    print(f"[data_split] also single: {tr_single}, {ev_single}, {ts_single}")

    # label_mapping
    _build_label_mapping(df_train, df, label_col, outdir)

    # 兼容 raw
    train_eval_raw = pd.concat([df_train, df_eval], axis=0).reset_index(drop=True)
    train_eval_raw.to_csv(outdir / "train_eval_raw.csv", index=False, encoding="utf-8-sig")
    test_df.drop(columns=["linked_items"]).to_csv(outdir / "test_raw.csv", index=False, encoding="utf-8-sig")
    print(f"[data_split] raw files saved to {outdir}")

    # data.csv（train+test）
    full_data = pd.concat([train_df_raw, test_df]).reset_index(drop=True)
    full_data.to_csv(outdir / "data.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", dest="src_path", type=str, default="./data/14609.csv", help="原始 CSV 路径")
    parser.add_argument("--outdir", type=str, default="./output/2025_up_to_month_2", help="输出目录")
    parser.add_argument("--month", type=int, default=2, help="测试月份（训练集使用 1..month-1）")
    parser.add_argument("--label-col", dest="label_col", type=str, default="extern_id", help="作为标签的列名：extern_id 或 linked_items（默认 extern_id）")
    parser.add_argument("--method", type=str, default="kfold", choices=["kfold", "ratio"], help="train 内 eval 切分方法：kfold 或 ratio")
    parser.add_argument("--k", type=int, default=5, help="kfold 折数")
    parser.add_argument("--fold-index", dest="fold_index", type=int, default=0, help="作为 eval 的折编号 [0..k-1]")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="ratio 模式下 train:eval 的比例（默认 0.8，即 4:1）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--min-count", type=int, default=5, help="训练集中类别最小样本数要求")
    parser.add_argument("--map-rare-to-other", action="store_true", help="将训练集中少见标签映射为 other_label")
    parser.add_argument("--other-label", type=str, default="__OTHER__", help="少见标签映射名称")
    args = parser.parse_args()
    split_data(args)


if __name__ == "__main__":
    main()
