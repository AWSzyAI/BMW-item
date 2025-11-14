#!/usr/bin/env python3
"""
分层切分工具：支持 5-fold（或 K-fold）与按比例的分层切分。

主要用途：
- 对输入 CSV（或 outdir/path）进行标签规范化后，执行 StratifiedKFold 切分，
    选择某一个 fold 作为 eval，其余作为 train；
- 或者按比例（train_ratio）进行分层切分（兼容旧行为）。

输出：在 --outdir 下生成分离文件：train_X.csv/train_y.csv、eval_X.csv/eval_y.csv（ratio=3 时额外 test_X.csv/test_y.csv）。
为保持兼容，也会同时写出 train.csv 与 eval.csv（以及需要时的 test.csv）。

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
    """读取输入数据。

    支持两类输入：
    1) 目录：目录下应包含 X.csv 与 y.csv（y.csv 至少包含列 linked_items），将二者横向合并后作为输入；
    2) 文件：按 CSV 读取（兼容旧单表）。
    同时支持相对路径（会在 outdir 下拼接）与绝对路径。
    """

    def _read_from_dir(dir_path: str) -> pd.DataFrame | None:
        if not os.path.isdir(dir_path):
            return None
        x_file = os.path.join(dir_path, "X_train.csv")
        y_file = os.path.join(dir_path, "y_train.csv")
        if os.path.exists(x_file) and os.path.exists(y_file):
            X_df = pd.read_csv(x_file)
            y_df = pd.read_csv(y_file)
            if "linked_items" not in y_df.columns:
                raise KeyError(f"{y_file} 缺少列：linked_items")
            df = pd.concat([X_df.reset_index(drop=True), y_df[["linked_items"]].reset_index(drop=True)], axis=1)
            return df
        # 目录存在但缺少文件
        raise FileNotFoundError(f"目录 {dir_path} 下未找到 X.csv 与 y.csv，请检查输入目录内容。")

    # 1) 绝对路径优先
    if path and os.path.isabs(path):
        if os.path.isdir(path):
            return _read_from_dir(path)
        if os.path.exists(path):
            return pd.read_csv(path)

    # 2) 原样路径（可能是相对路径，若直接存在则读取；若是目录则按目录处理）
    if path and os.path.exists(path):
        if os.path.isdir(path):
            return _read_from_dir(path)
        return pd.read_csv(path)

    # 3) 尝试在 outdir 下拼接
    joined = os.path.join(outdir, path)
    if os.path.isdir(joined):
        return _read_from_dir(joined)
    if os.path.exists(joined):
        return pd.read_csv(joined)

    raise FileNotFoundError(f"找不到输入：{path}，也无法在 {outdir} 下找到对应的文件或目录。")


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    # 若未提供 --path，则默认使用 --outdir 作为输入目录
    in_path = args.path if (hasattr(args, 'path') and args.path and str(args.path).strip()) else args.outdir
    df = read_input(in_path, args.outdir)
    for col in ["case_title", "performed_work"]:
        if col not in df.columns:
            raise KeyError(f"输入缺少列：{col}")
    # 标签列选择：linked_items 或 extern_id（默认 linked_items）
    label_col = str(getattr(args, "label_col", "linked_items"))

    # 规范标签：
    # 情况A：输入目录包含 y_train.csv -> 合并后已有 'linked_items' 工作列（数值或字符串均可）；直接使用
    # 情况B：单表/原始数据 -> 若存在 label_col，则写回工作列 'linked_items'
    df = df.copy()
    if "linked_items" in df.columns and (label_col == "linked_items" or label_col not in df.columns):
        df["linked_items"] = df["linked_items"].apply(ensure_single_label).astype(str)
    elif label_col in df.columns:
        df["linked_items"] = df[label_col].apply(ensure_single_label).astype(str)
    else:
        raise KeyError(f"输入缺少标签列：{label_col}，且未在输入中发现标准列 linked_items")

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
    # 小工具：写出分离文件 + 兼容的单表
    def _save_split(df_split: pd.DataFrame, name: str):
        out_single = os.path.join(args.outdir, f"{name}.csv")
        # 分离：X（去除 linked_items）和 y（仅 linked_items）
        if "linked_items" not in df_split.columns:
            raise KeyError("内部错误：分割后的数据缺少 linked_items 列")
        X_df = df_split.drop(columns=["linked_items"]) if df_split.shape[1] > 1 else pd.DataFrame(index=df_split.index)
        y_df = df_split[["linked_items"]].copy()
        X_path = os.path.join(args.outdir, f"{name}_X.csv")
        Y_path = os.path.join(args.outdir, f"{name}_y.csv")
        # 保存
        df_split.to_csv(out_single, index=False, encoding="utf-8-sig")
        X_df.to_csv(X_path, index=False, encoding="utf-8-sig")
        y_df.to_csv(Y_path, index=False, encoding="utf-8-sig")
        return out_single, X_path, Y_path

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
        tr_single, tr_X, tr_y = _save_split(df_train, "train")
        ev_single, ev_X, ev_y = _save_split(df_eval, "eval")
        print(
            f"[kfold] Saved: train({len(df_train)}) -> {tr_X}, {tr_y} | eval({len(df_eval)}) -> {ev_X}, {ev_y} | also single: {tr_single}, {ev_single} | k={k}, fold={fold_index}"
        )
        # 复用上游（2025.py）生成的 label_mapping.csv；若不存在则写出最小映射
        mapping_path = os.path.join(args.outdir, 'label_mapping.csv')
        if os.path.exists(mapping_path):
            print(f"[kfold] 复用已存在的映射：{mapping_path}")
        else:
            labels_order = sorted(df_train['linked_items'].unique())
            mapping_df = pd.DataFrame({
                'linked_items': labels_order,
                'label': range(len(labels_order)),
                'item_title': ["" for _ in labels_order],
                'extern_id': ["" for _ in labels_order],
                'orig_linked_items': labels_order,
            })
            # 统一转换为不带小数点的字符串
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
            mapping_df = mapping_df.applymap(_fmt_no_decimal)
            mapping_df.to_csv(mapping_path, index=False, encoding='utf-8-sig')
            print(f"[kfold] 缺少上游映射，已写出最小映射：{mapping_path}")
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
        tr_single, tr_X, tr_y = _save_split(df_train, "train")
        ev_single, ev_X, ev_y = _save_split(df_eval, "eval")
        msg = [
            f"train({len(df_train)}) -> {tr_X}, {tr_y}",
            f"eval({len(df_eval)}) -> {ev_X}, {ev_y}"
        ]
        # 若是 3 路切分，额外输出 test
        if str(getattr(args, "split_type", "2")).strip() == "3" and len(test_idx) > 0:
            df_test = df.iloc[sorted(test_idx)].reset_index(drop=True)
            ts_single, ts_X, ts_y = _save_split(df_test, "test")
            msg.append(f"test({len(df_test)}) -> {ts_X}, {ts_y}")
            msg.append(f"also single: {tr_single}, {ev_single}, {ts_single}")
        else:
            msg.append(f"also single: {tr_single}, {ev_single}")
        print(f"[ratio] Saved: {' | '.join(msg)} | train_ratio={train_ratio}")
        # 复用上游（2025.py）生成的 label_mapping.csv；若不存在则写出最小映射
        mapping_path = os.path.join(args.outdir, 'label_mapping.csv')
        if os.path.exists(mapping_path):
            print(f"[ratio] 复用已存在的映射：{mapping_path}")
        else:
            labels_order = sorted(df_train['linked_items'].unique())
            mapping_df = pd.DataFrame({
                'linked_items': labels_order,
                'label': range(len(labels_order)),
                'item_title': ["" for _ in labels_order],
                'extern_id': ["" for _ in labels_order],
                'orig_linked_items': labels_order,
            })
            # 统一转换为不带小数点的字符串
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
            mapping_df = mapping_df.applymap(_fmt_no_decimal)
            mapping_df.to_csv(mapping_path, index=False, encoding='utf-8-sig')
            print(f"[ratio] 缺少上游映射，已写出最小映射：{mapping_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="", help="输入路径：可为 CSV 文件，或包含 X.csv 与 y.csv 的文件夹；留空则使用 --outdir")
    parser.add_argument("--outdir", type=str, default="./output", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--min_count", type=int, default=5, help="少见标签阈值与最低样本数要求（建议与 k 相同）")
    parser.add_argument("--map_rare_to_other", action="store_true", help="将少见标签映射为 __OTHER__")
    parser.add_argument("--label-col", dest="label_col", type=str, default="linked_items", help="作为标签的列名：linked_items 或 extern_id（默认 linked_items）")
    # 新增：kfold 参数
    parser.add_argument("--method", type=str, default="kfold", choices=["kfold", "ratio"], help="切分方法：kfold 或 ratio")
    parser.add_argument("--k", type=int, default=5, help="kfold 的折数")
    parser.add_argument("--fold-index", dest="fold_index", type=int, default=0, help="作为 eval 的折编号 [0..k-1]")
    # 兼容旧的 2/3 分法
    parser.add_argument("--split_type", type=str, default="2", help="ratio 模式下：3=每类先取1条test，剩余按比例切成train/eval；2=仅按比例切成train/eval")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="ratio 模式下：train:eval 的比例（默认 0.8，即 4:1）")
    args = parser.parse_args()
    main(args)
