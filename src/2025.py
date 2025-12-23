import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log_info = logging.info


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


def main(args):
    label_col = args.label_col
    test_month = args.month
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_path = args.src_path
    
    # 尝试读取，优先 utf-8-sig 兼容 BOM
    try:
        df = pd.read_csv(src_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(src_path, encoding="gbk")
        except UnicodeDecodeError:
            df = pd.read_csv(src_path) # fallback to default

    # 清理列名首尾空格
    df.columns = df.columns.str.strip()
    print(df.columns)


    if label_col not in df.columns:
        raise ValueError(f"label_col={label_col} 不在原始数据列中")

    if test_month <= 1:
        raise ValueError("--month 必须大于 1，才能使用 1..month-1 作为训练集")

    feature_cols = ["case_id","case_title", "performed_work", "month"]

    df["month"] = pd.to_datetime(df["case_submitted_date"]).dt.to_period("M")
    month_groups = df.groupby("month")

    train_months = list(range(1, test_month))
    train_df = pd.concat(
        [
            month_groups.get_group(m)
            for m in month_groups.groups
            if m.month in train_months
        ]
    )
    test_df = month_groups.get_group(
        next(m for m in month_groups.groups if m.month == test_month)
    )

    full_data = pd.concat([train_df, test_df])
    full_data.to_csv(out_dir / "data.csv", index=False)

    eval_size = 0.2
    try:
        tr_df, ev_df = train_test_split(
            train_df,
            test_size=eval_size,
            random_state=42,
            stratify=train_df[label_col].astype(str),
        )
    except ValueError:
        tr_df, ev_df = train_test_split(
            train_df, test_size=eval_size, random_state=42, shuffle=True
        )

    for d in (tr_df, ev_df, test_df):
        d.loc[:, "month"] = d["month"].astype(str)

    le = LabelEncoder()
    tr_df = tr_df.copy()
    ev_df = ev_df.copy()
    test_df = test_df.copy()

    tr_df["label"] = le.fit_transform(tr_df[label_col].astype(str))

    ev_df = ev_df[ev_df[label_col].astype(str).isin(le.classes_)]
    test_df = test_df[test_df[label_col].astype(str).isin(le.classes_)]

    ev_df["label"] = le.transform(ev_df[label_col].astype(str))
    test_df["label"] = le.transform(test_df[label_col].astype(str))

    item_title_map = {}
    extern_id_map = {}
    orig_linked_map = {}
    itemcreationdate_map = {}
    case_id_map = {}

    all_label_series = df[label_col].astype(str)
    for raw_label in le.classes_:
        mask = all_label_series.astype(str) == str(raw_label)
        sub = df[mask]
        if "item_title" in sub.columns:
            item_title_map[raw_label] = _mode_or_empty(sub["item_title"])
        else:
            item_title_map[raw_label] = ""
        if "case_id" in le.classes_:
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
            orig_linked_map[raw_label] = (
                str(raw_label) if label_col == "linked_items" else ""
            )
        if "itemcreationdate" in sub.columns:
            itemcreationdate_map[raw_label] = _mode_or_empty(sub["itemcreationdate"])
        else:
            itemcreationdate_map[raw_label] = ""

    if label_col == "extern_id":
        display_vals = [extern_id_map.get(li, str(li)) for li in le.classes_]
    else:
        display_vals = [str(li) for li in le.classes_]

    mapping_df = pd.DataFrame(
        {
            "linked_items": display_vals,
            "label": range(len(le.classes_)),
            "item_title": [item_title_map.get(li, "") for li in le.classes_],
            "extern_id": [extern_id_map.get(li, "") for li in le.classes_],
            "case_id": [case_id_map.get(li,"")for li in le.classes_],
            "orig_linked_items": [
                orig_linked_map.get(li, str(li)) for li in le.classes_
            ],
            "itemcreationdate": [
                itemcreationdate_map.get(li, "") for li in le.classes_
            ],
        }
    )
    mapping_df = mapping_df.map(_fmt_no_decimal)
    mapping_df.to_csv(out_dir / "label_mapping.csv", index=False, encoding="utf-8-sig")

    tr_df[feature_cols].to_csv(out_dir / "X_train.csv", index=False)
    tr_df["label"].rename("linked_items").to_csv(out_dir / "y_train.csv", index=False)

    ev_df[feature_cols].to_csv(out_dir / "X_eval.csv", index=False)
    ev_df["label"].rename("linked_items").to_csv(out_dir / "y_eval.csv", index=False)

    test_df[feature_cols].to_csv(out_dir / "X_test.csv", index=False)
    test_df["label"].rename("linked_items").to_csv(out_dir / "y_test.csv", index=False)

    # 兼容 eval/predict：生成 train.csv / eval.csv / test.csv（含标签列）
    tr_df_full = tr_df.copy()
    tr_df_full["linked_items"] = le.inverse_transform(tr_df["label"])
    tr_df_full.drop(columns=["label"]).to_csv(out_dir / "train.csv", index=False)

    ev_df_full = ev_df.copy()
    ev_df_full["linked_items"] = le.inverse_transform(ev_df["label"])
    ev_df_full.drop(columns=["label"]).to_csv(out_dir / "eval.csv", index=False)

    test_df_full = test_df.copy()
    test_df_full["linked_items"] = le.inverse_transform(test_df["label"])
    test_df_full.drop(columns=["label"]).to_csv(out_dir / "test.csv", index=False)

    train_df.to_csv(out_dir / "train_eval_raw.csv", index=False)
    test_df.drop(columns=["label"]).to_csv(out_dir / "test_raw.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-col",
        dest="label_col",
        type=str,
        default="extern_id",
        help="作为标签的列：linked_items 或 extern_id（默认 linked_items）",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=2,
        help="测试月份（训练集使用 1..month-1；0 表示 small 随机样本）",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./output/2025_up_to_month_2",
        help="输出目录（写入生成的数据集文件）",
    )
    parser.add_argument(
        "--src-path",
        dest="src_path",
        type=str,
        default="./data/14609.csv",
    )
    main(parser.parse_args())