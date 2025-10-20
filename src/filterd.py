# %% [markdown]
# 拿到了新的5,6月份的数据`../data/test_data1020.csv`。
# 先那之前的7，8月份数据训练出来的模型用predict.py预测一下，跑出来所有指标都是0，怀疑是新的数据中的y和老的标签不一致。

# %%
# Make project root importable when running this notebook from the 'test' folder
import sys, os
# Add the parent directory (project root) to sys.path so 'from src...` imports work
root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
import importlib

import src.predict as _predict_mod
importlib.reload(_predict_mod)
predict = _predict_mod.predict
print('Imported predict from src.predict')

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import ast
import numpy as np


# %%
# Load CSV
df_1020 = pd.read_csv("../data/test_data1020_34.csv")
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


# %% [markdown]
# 新数据有7323行，但是根据case_id去掉重复的数据后其实只有2718行，这么大量的重复到底是怎么造成的？我们拿着去重后的数据分析一下标签extern_id 和 linked_items的情况

# %%
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
            #!/usr/bin/env python3
            """
            过滤与评估工具：
            参考 eval.py 的传参形式，支持：
            - --path: 输入 CSV 文件路径（绝对/相对）
            - --outdir: 输出目录（默认 ./output）
            - --mode: prepare | predict | new
              * prepare: 解析+展开+按最小样本阈值过滤，并将结果保存
              * predict: 对已过滤后的数据集做预测并评估（需要先有 prepare 的输出）
              * new: 一次性完成 prepare + predict（对整文件作为测试集评估）

            输出约定（设输入为 data/my.csv，其 basename 为 my）：
            - outdir/df_my.csv                     展开后的数据
            - outdir/df_filtered_my.csv            过滤后的数据（按 --min-count）
            - outdir/predictions_df_filtered_my.csv 过滤后数据上的预测结果
            """

            import os
            import sys
            import ast
            import glob
            import argparse
            import numpy as np
            import pandas as pd
            from typing import List, Tuple
            from sklearn.metrics import accuracy_score, f1_score


            def _ensure_project_root_on_path():
                """确保可以导入 src.predict"""
                root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                if root not in sys.path:
                    sys.path.insert(0, root)


            def parse_args():
                parser = argparse.ArgumentParser()
                parser.add_argument("--path", type=str, required=True, help="输入 CSV 文件路径（绝对或相对）")
                parser.add_argument("--outdir", type=str, default="./output", help="输出目录")
                parser.add_argument(
                    "--mode",
                    type=str,
                    default="prepare",
                    choices=["prepare", "predict", "new"],
                    help="prepare=仅生成过滤数据；predict=对过滤数据评估；new=prepare+predict"
                )
                parser.add_argument("--min-count", type=int, default=5, help="类别最小样本阈值（用于过滤）")
                parser.add_argument("--dedup-by-case-id", action="store_true", help="按 case_id 去重（可选）")
                return parser.parse_args()


            def read_input_csv(path: str, outdir: str) -> pd.DataFrame:
                candidates = [path, os.path.join(outdir, path)]
                file_to_read = None
                for p in candidates:
                    if p and os.path.exists(p):
                        file_to_read = p
                        break
                if file_to_read is None:
                    raise FileNotFoundError(f"找不到输入文件：{path} 或 {os.path.join(outdir, path)}")
                df = pd.read_csv(file_to_read)
                # drop accidental index column if present
                if "Unnamed: 0" in df.columns:
                    df = df.drop(columns=["Unnamed: 0"])
                return df


            def to_list(x):
                if isinstance(x, list):
                    return x
                if pd.isna(x):
                    return []
                if isinstance(x, str):
                    try:
                        v = ast.literal_eval(x)
                        return v if isinstance(v, list) else [v]
                    except Exception:
                        return [x]
                return [x]


            def prepare_dataset(df: pd.DataFrame, min_count: int, dedup_by_case_id: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
                if dedup_by_case_id and "case_id" in df.columns:
                    before = len(df)
                    df = df.drop_duplicates(subset=["case_id"], keep="first").reset_index(drop=True)
                    after = len(df)
                    print(f"dedup by case_id: {before} -> {after} rows")

                if "linked_items" not in df.columns:
                    raise KeyError("输入数据缺少列：linked_items")

                df = df.copy()
                df["linked_items"] = df["linked_items"].apply(to_list)
                df_exp = df.explode("linked_items").reset_index(drop=True)
                df_exp["linked_items"] = df_exp["linked_items"].astype(str).str.strip()

                counts = df_exp["linked_items"].value_counts()
                valid_items = counts[counts >= min_count].index
                df_filtered = df_exp[df_exp["linked_items"].isin(valid_items)].reset_index(drop=True)

                print(
                    f"展开后样本数: {len(df_exp)}, 类别数: {df_exp['linked_items'].nunique()} | "
                    f"过滤后样本数: {len(df_filtered)}, 类别数: {df_filtered['linked_items'].nunique()} (min_count={min_count})"
                )
                return df_exp, df_filtered


            def pick_true_label(row: pd.Series) -> str:
                if "extern_id" in row and pd.notna(row.get("extern_id")) and str(row.get("extern_id")).strip() != "":
                    return str(row.get("extern_id")).strip()
                if "linked_parsed" in row and isinstance(row.get("linked_parsed"), str):
                    try:
                        v = ast.literal_eval(row.get("linked_parsed"))
                        if isinstance(v, (list, tuple)) and len(v) > 0:
                            return str(v[0])
                    except Exception:
                        pass
                for col in ("linked_parsed", "linked_items"):
                    val = row.get(col)
                    if isinstance(val, list) and val:
                        return str(val[0])
                    if pd.notna(val) and isinstance(val, str):
                        try:
                            v = ast.literal_eval(val)
                            if isinstance(v, (list, tuple)) and v:
                                return str(v[0])
                        except Exception:
                            pass
                return ""


            def load_model_bundle(outdir: str):
                best_model_path = os.path.join(outdir, "model_best.joblib")
                if not os.path.exists(best_model_path):
                    # 回退到任意 model_*.joblib
                    cands = sorted(glob.glob(os.path.join(outdir, "model_*.joblib")))
                    if not cands:
                        raise FileNotFoundError("未找到模型文件 (output/model_best.joblib 或 output/model_*.joblib)")
                    best_model_path = cands[0]
                print("Using model:", best_model_path)
                # 仅在需要预测时导入 predict 模块
                _ensure_project_root_on_path()
                import src.predict as _predict_mod
                return _predict_mod.predict, best_model_path


            def batch_predict(predict_fn, model_path, X_df: pd.DataFrame, top_k: int = 10) -> List:
                from tqdm import tqdm
                results = []
                batch_size = 256
                n = len(X_df)
                for i in tqdm(range(0, n, batch_size), desc="Predict batches"):
                    batch = X_df.iloc[i: i + batch_size]
                    try:
                        res = predict_fn(batch, model_path=model_path, top_k=top_k)
                    except TypeError:
                        res = predict_fn(batch, top_k=top_k)
                    if isinstance(res, dict):
                        res = [res]
                    results.extend(res)
                return results


            def normalize_result(r) -> Tuple[List[str], List[float]]:
                if r is None:
                    return [], []
                if isinstance(r, dict):
                    preds = r.get("preds") or r.get("labels") or []
                    scores = r.get("scores") or r.get("probs") or []
                elif isinstance(r, (list, tuple)) and len(r) >= 1:
                    preds = r[0] if isinstance(r[0], list) else r
                    scores = r[1] if len(r) > 1 else []
                else:
                    preds, scores = [], []
                preds = [str(x) for x in (preds or [])]
                scores = [float(x) for x in (scores or [])[: len(preds)]]
                return preds, scores


            def run_predict_and_metrics(df_filtered: pd.DataFrame, outdir: str, base: str) -> str:
                # 准备 X_df
                X_cols = [c for c in ("case_title", "performed_work") if c in df_filtered.columns]
                X_df = df_filtered[X_cols].copy() if X_cols else df_filtered.copy()

                predict_fn, model_path = load_model_bundle(outdir)
                results = batch_predict(predict_fn, model_path, X_df, top_k=10)

                rows = []
                for idx, r in enumerate(results):
                    preds, scores = normalize_result(r)
                    true = str(df_filtered.iloc[idx].get("true_label")) if "true_label" in df_filtered.columns else ""
                    if true == "":
                        # 尝试从原始列推断
                        true = pick_true_label(df_filtered.iloc[idx])
                    top1 = preds[0] if preds else ""
                    rows.append({
                        "index": df_filtered.index[idx],
                        "extern_id": df_filtered.iloc[idx].get("extern_id") if "extern_id" in df_filtered.columns else None,
                        "case_id": df_filtered.iloc[idx].get("case_id") if "case_id" in df_filtered.columns else None,
                        "true_label": true,
                        "pred_top1": top1,
                        "preds_top10": "|".join(preds),
                        "scores_top10": "|".join(f"{s:.6f}" for s in scores),
                        "hit@1": int(true != "" and top1 == true),
                        "hit@3": int(true != "" and true in preds[:3]),
                        "hit@5": int(true != "" and true in preds[:5]),
                        "hit@10": int(true != "" and true in preds[:10]),
                    })

                pred_df = pd.DataFrame(rows)
                valid = pred_df[pred_df["true_label"].astype(str) != ""]
                y_true = valid["true_label"].tolist()
                y_pred_top1 = valid["pred_top1"].tolist()

                acc = accuracy_score(y_true, y_pred_top1) if len(y_true) > 0 else float("nan")
                try:
                    f1_w = f1_score(y_true, y_pred_top1, average="weighted") if len(y_true) > 0 else float("nan")
                except Exception as e:
                    print("计算 weighted F1 失败：", e)
                    f1_w = float("nan")

                print(
                    f"Metrics on filtered (n={len(pred_df)}, valid={len(valid)}): acc={acc:.6f}, f1_w={f1_w:.6f}, "
                    f"hit@1={pred_df['hit@1'].mean() if len(pred_df)>0 else float('nan'):.6f}, "
                    f"hit@3={pred_df['hit@3'].mean():.6f}, hit@5={pred_df['hit@5'].mean():.6f}, hit@10={pred_df['hit@10'].mean():.6f}"
                )

                os.makedirs(outdir, exist_ok=True)
                out_path = os.path.join(outdir, f"predictions_df_filtered_{base}.csv")
                pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")
                print("Predictions saved to:", out_path)
                return out_path


            def main():
                args = parse_args()
                outdir = args.outdir
                os.makedirs(outdir, exist_ok=True)

                df_raw = read_input_csv(args.path, outdir)
                base = os.path.splitext(os.path.basename(args.path))[0]

                if args.mode in ("prepare", "new"):
                    df_exp, df_filtered = prepare_dataset(df_raw, args.min_count, args.dedup_by_case_id)
                    # 补充 true_label，便于后续评估
                    if "true_label" not in df_filtered.columns:
                        df_filtered = df_filtered.copy()
                        df_filtered["true_label"] = df_filtered.apply(pick_true_label, axis=1)

                    df_exp_out = os.path.join(outdir, f"df_{base}.csv")
                    df_filtered_out = os.path.join(outdir, f"df_filtered_{base}.csv")
                    df_exp.to_csv(df_exp_out, index=False, encoding="utf-8-sig")
                    df_filtered.to_csv(df_filtered_out, index=False, encoding="utf-8-sig")
                    print("Saved:", df_exp_out)
                    print("Saved:", df_filtered_out)

                    if args.mode == "prepare":
                        return

                # predict 模式：需要已过滤数据
                if args.mode in ("predict", "new"):
                    # 若是 predict 且还未生成过滤文件，则尝试读取现成的 df_filtered_<base>.csv
                    if args.mode == "predict":
                        df_filtered_path = os.path.join(outdir, f"df_filtered_{base}.csv")
                        if not os.path.exists(df_filtered_path):
                            raise FileNotFoundError(f"找不到过滤后文件：{df_filtered_path}，请先运行 --mode prepare")
                        df_filtered = pd.read_csv(df_filtered_path)
                    # 执行预测与指标
                    run_predict_and_metrics(df_filtered, outdir, base)


            if __name__ == "__main__":
                main()
# 过滤出新数据中属于交集的行

def has_common(row):

    return any(str(x) in common for x in row.get('linked_items_parsed', []))


