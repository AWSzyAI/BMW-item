import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

df = pd.read_csv("../data/case_item_1023(in).csv")
df['month'] = pd.to_datetime(df['case_submitted_date']).dt.to_period('M')
df = df.drop(columns=['Unnamed: 0'])
duplicate_rows = df[df.duplicated()]
df['case_submitted_date'] = pd.to_datetime(df['case_submitted_date'])
unknown_item_df = df[df['linked_items'].isnull()]
os.makedirs("../output", exist_ok=True)
unknown_item_df.to_csv("../output/2025_unknown.csv", index=False)
df = df.drop(unknown_item_df.index)

# 构造一批train,eval,test
# 拿123456训练，预测7
# 拿1234567训练，预测8
# 拿12345678训练，预测9
# 拿123456789训练，预测10

test_points = [
    {"train": [1], "test": 2},
    {"train": [1, 2, 3], "test": 4},
    {"train": [1, 2, 3, 4], "test": 5},
    {"train": [1, 2, 3, 4, 5], "test": 6},
    {"train": [1, 2, 3, 4, 5, 6], "test": 7},
    {"train": [1, 2, 3, 4, 5, 6, 7], "test": 8},
    {"train": [1, 2, 3, 4, 5, 6, 7, 8], "test": 9},
    {"train": [1, 2, 3, 4, 5, 6, 7, 8, 9], "test": 10},
]

# 特征列
feature_cols = ["case_title", "performed_work", "month"]

# 生成月份列与分组
df["month"] = pd.to_datetime(df["case_submitted_date"]).dt.to_period("M")
month_groups = df.groupby("month")

def concat_by_months(month_int_list):
    keys = [k for k in month_groups.groups if k.month in set(month_int_list)]
    return pd.concat([month_groups.get_group(k) for k in keys], ignore_index=True)

for point in test_points:
    train_months = point["train"]
    test_month = point["test"]

    train_df = pd.concat([month_groups.get_group(month) for month in month_groups.groups if month.month in train_months])
    test_df = month_groups.get_group(next(month for month in month_groups.groups if month.month == test_month))


    out_dir = Path(f"../output/2025_up_to_month_{test_month}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 按标签分层切分 train -> train/eval
    eval_size = 0.2
    try:
        tr_df, ev_df = train_test_split(
            train_df,
            test_size=eval_size,
            random_state=42,
            stratify=train_df["linked_items"].astype(str),
        )
    except ValueError:
        tr_df, ev_df = train_test_split(
            train_df, test_size=eval_size, random_state=42, shuffle=True
        )

    # 统一 month 为字符串，避免 Period 写盘/读取差异
    for d in (tr_df, ev_df, test_df):
        d["month"] = d["month"].astype(str)

    # 标签编码：在训练集上拟合，eval/test 使用相同映射
    le = LabelEncoder()
    tr_df = tr_df.copy()
    ev_df = ev_df.copy()
    test_df = test_df.copy()

    tr_df["label"] = le.fit_transform(tr_df["linked_items"].astype(str))

    # 过滤掉训练集中未见过的标签，避免 transform 报错
    ev_df = ev_df[ev_df["linked_items"].astype(str).isin(le.classes_)]
    test_df = test_df[test_df["linked_items"].astype(str).isin(le.classes_)]

    ev_df["label"] = le.transform(ev_df["linked_items"].astype(str))
    test_df["label"] = le.transform(test_df["linked_items"].astype(str))

    # 保存标签映射
    pd.DataFrame(
        {"linked_items": le.classes_, "label": range(len(le.classes_))}
    ).to_csv(out_dir / "label_mapping.csv", index=False)

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