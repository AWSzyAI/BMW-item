import sys, os
root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
import importlib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import ast
import numpy as np
import argparse

def main(args):
    # 使用命令行参数中的目录与文件名
    dir_name = args.dir
    file_name = args.file

    df_1020 = pd.read_csv(os.path.join(dir_name, file_name))

    # drop accidental index column if present
    if 'Unnamed: 0' in df_1020.columns:
        df_1020 = df_1020.drop(columns=['Unnamed: 0'])
    print('raw rows:', len(df_1020))
    # Deduplicate by case_id (keep first occurrence)
    if 'case_id' in df_1020.columns:
        before = len(df_1020)
        df_1020 = df_1020.drop_duplicates(subset=['case_id'], keep='first')
        after = len(df_1020)
        print(f'dedup by case_id: {before} -> {after} rows')
    else:
        print('case_id column not found; skipping case_id dedup')
    # ---- 1) 统一把 linked_items 解析成 list ----
    def to_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        if isinstance(x, str):
            # 兼容 '["A","B"]' 或 "['A','B']" 等字符串
            try:
                v = ast.literal_eval(x)
                return v if isinstance(v, list) else [v]
            except Exception:
                return [x]
        return [x]

    df_1020["linked_items"] = df_1020["linked_items"].apply(to_list)

    # ---- 2) explode：一行一个标签（解决 list 不可哈希问题）----
    # 将列表"linked_items"展开，那不就相当于直接用extern_id吗？

    df_exp = df_1020.explode("linked_items").reset_index(drop=True)

    # 统一成字符串，避免 123 vs "123" 的类型不一致
    df_exp["linked_items"] = df_exp["linked_items"].astype(str).str.strip()

    # 现在可以安全统计
    total_samples = len(df_exp)
    num_classes = df_exp["linked_items"].nunique()
    print(f"展开后样本数: {total_samples}, 类别数: {num_classes}")

    # ---- 3) 过滤掉样本数 <5 的类别，保证 5-fold 可用 ----
    counts = df_exp["linked_items"].value_counts()
    valid_items = counts[counts >= 5].index

    # 不过滤
    df_filtered = df_exp
    # 确保输出目录存在
    os.makedirs("../output", exist_ok=True)
    df_filtered.to_csv(os.path.join("../output", file_name.replace('.csv', '_all.csv')))

    df_filtered_1020_5 = df_exp[df_exp["linked_items"].isin(valid_items)].reset_index(drop=True)
    df_filtered_1020_5.to_csv(os.path.join("../output", file_name.replace('.csv', '_filtered.csv')), index=False)

    print(f"原始类别数: {num_classes} -> 过滤后类别数: {df_filtered['linked_items'].nunique()}")
    print(f"原始样本数(展开): {total_samples} -> 过滤后样本数: {len(df_filtered)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dir_name = "./data"
    file_name = "m_train.csv"

    parser.add_argument("--dir", type=str, default=dir_name, help="数据文件目录")
    parser.add_argument("--file", type=str, default=file_name, help="数据文件名或绝对路径")
    args = parser.parse_args()
    
    # 在进入 main 前做一处健壮性修正：如果传入的是绝对路径，拆出 dir 与 file
    if os.path.isabs(args.file) and os.path.exists(args.file):
        args.dir = os.path.dirname(args.file)
        args.file = os.path.basename(args.file)
    
    # 在 main 内部进行读取前，构建多个候选路径用于读取
    # 注意：这里不直接读取，仅将解析逻辑交由 main 使用 args.dir/args.file
    main(args)
    