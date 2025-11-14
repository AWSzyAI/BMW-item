import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import argparse


def _mode_or_empty(series: pd.Series) -> str:
    """返回非空众数，若全为空或序列为空则返回空串。"""
    if series is None or len(series) == 0:
        return ""
    # 转成字符串并过滤空白
    s = series.astype(str).str.strip()
    s = s[s != ""]
    if len(s) == 0:
        return ""
    try:
        return s.value_counts().index[0]
    except Exception:
        return ""

def main(args):
    df = pd.read_csv("../data/case_item_1023(in).csv")
    df['month'] = pd.to_datetime(df['case_submitted_date']).dt.to_period('M')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    duplicate_rows = df[df.duplicated()]
    df['case_submitted_date'] = pd.to_datetime(df['case_submitted_date'])
    # 目标列：linked_items 或 extern_id
    label_col = str(getattr(args, 'label_col', 'linked_items'))
    if label_col not in df.columns:
        raise KeyError(f"输入缺少标签列：{label_col}")
    # 导出未知目标列样本
    unknown_mask = df[label_col].isnull()
    os.makedirs("../output", exist_ok=True)
    df[unknown_mask].to_csv("../output/2025_unknown.csv", index=False)
    df = df.drop(df[unknown_mask].index)

# 构造一批train,eval,test
# 拿123456训练，预测7
# 拿1234567训练，预测8
# 拿12345678训练，预测9
# 拿123456789训练，预测10

    # 仅生成指定月份的数据集（train=1..month-1，test=month）
    test_month = int(getattr(args, 'month', 2))
    if test_month <= 1:
        raise ValueError("--month 必须大于 1，才能使用 1..month-1 作为训练集")

# 特征列
    feature_cols = ["case_title", "performed_work", "month"]

# 生成月份列与分组
    df["month"] = pd.to_datetime(df["case_submitted_date"]).dt.to_period("M")
    month_groups = df.groupby("month")

    # （可选）如需按月合并可在此添加辅助函数

    train_months = list(range(1, test_month))

    train_df = pd.concat([month_groups.get_group(month) for month in month_groups.groups if month.month in train_months])
    test_df = month_groups.get_group(next(month for month in month_groups.groups if month.month == test_month))


    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 按标签分层切分 train -> train/eval（按选定标签列分层）
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

    # 统一 month 为字符串，避免 Period 写盘/读取差异
    for d in (tr_df, ev_df, test_df):
        d["month"] = d["month"].astype(str)

    # 标签编码：在训练集上拟合，eval/test 使用相同映射（基于选定标签列）
    le = LabelEncoder()
    tr_df = tr_df.copy()
    ev_df = ev_df.copy()
    test_df = test_df.copy()

    tr_df["label"] = le.fit_transform(tr_df[label_col].astype(str))

    # 过滤掉训练集中未见过的标签，避免 transform 报错
    ev_df = ev_df[ev_df[label_col].astype(str).isin(le.classes_)]
    test_df = test_df[test_df[label_col].astype(str).isin(le.classes_)]

    ev_df["label"] = le.transform(ev_df[label_col].astype(str))
    test_df["label"] = le.transform(test_df[label_col].astype(str))

    # 保存标签映射，附加 item_title / extern_id / orig_linked_items（均取训练集中该类的众数值）
    # 当 label_col == extern_id 时，extern_id_map 直接为身份映射，避免重复列导致 groupby 报错
    item_title_map = {}
    extern_id_map = {}
    orig_linked_map = {}

    # 针对每个原始类值（即 le.classes_ 中的字符串形式）构建映射
    # 注意：le.classes_ 的元素类型与训练时 tr_df[label_col].astype(str) 一致
    train_label_series = train_df[label_col].astype(str)
    for raw_label in le.classes_:
        # 选出该类的子集
        mask = (train_label_series.astype(str) == str(raw_label))
        sub = train_df[mask]
        if 'item_title' in sub.columns:
            item_title_map[raw_label] = _mode_or_empty(sub['item_title'])
        else:
            item_title_map[raw_label] = ""
        if label_col == 'extern_id':
            # 直接身份映射
            extern_id_map[raw_label] = str(raw_label)
        else:
            if 'extern_id' in sub.columns:
                extern_id_map[raw_label] = _mode_or_empty(sub['extern_id'])
            else:
                extern_id_map[raw_label] = ""
        if 'linked_items' in sub.columns:
            orig_linked_map[raw_label] = _mode_or_empty(sub['linked_items'])
        else:
            # 若当前就是 linked_items 作为标签，则保留; 否则为空
            orig_linked_map[raw_label] = (str(raw_label) if label_col == 'linked_items' else "")

    # 展示列：随 label_col 切换（用于下游兼容维持列名 linked_items）
    if label_col == 'extern_id':
        display_vals = [extern_id_map.get(li, str(li)) for li in le.classes_]
    else:
        display_vals = [str(li) for li in le.classes_]

    mapping_df = pd.DataFrame({
        'linked_items': display_vals,
        'label': range(len(le.classes_)),
        'item_title': [item_title_map.get(li, '') for li in le.classes_],
        'extern_id': [extern_id_map.get(li, '') for li in le.classes_],
        'orig_linked_items': [orig_linked_map.get(li, str(li)) for li in le.classes_]
    })
    # 将所有字段转换为字符串，并去掉数值型中的小数点（如 123.0 -> "123"）
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
    mapping_df.to_csv(out_dir / 'label_mapping.csv', index=False, encoding='utf-8-sig')

    # 保存 X/y（train/eval/test）
    tr_df[feature_cols].to_csv(out_dir / "X_train.csv", index=False)
    tr_df["label"].rename("linked_items").to_csv(out_dir / "y_train.csv", index=False)

    ev_df[feature_cols].to_csv(out_dir / "X_eval.csv", index=False)
    ev_df["label"].rename("linked_items").to_csv(out_dir / "y_eval.csv", index=False)

    test_df[feature_cols].to_csv(out_dir / "X_test.csv", index=False)
    test_df["label"].rename("linked_items").to_csv(out_dir / "y_test.csv", index=False)

    # 可选：同时保留原始明细，便于排查
    train_df.to_csv(out_dir / "train_eval_raw.csv", index=False)
    test_df.drop(columns=["label"]).to_csv(out_dir / "test_raw.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-col", dest="label_col", type=str, default="linked_items", help="作为标签的列：linked_items 或 extern_id（默认 linked_items）")
    parser.add_argument("--month", type=int, default=2, help="测试月份（训练集使用 1..month-1）")
    parser.add_argument("--outdir", type=str, default="./output/2025_up_to_month_2", help="输出目录（写入生成的数据集文件）")
    args = parser.parse_args()
    main(args)