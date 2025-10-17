# train.py
import os, json, ast, argparse, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

warnings.filterwarnings("ignore")

def ensure_single_label(s):
    """若列里偶有 '["a","b"]' 这类字符串，就取第一个；正常单标签直接返回字符串。"""
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

def build_text(df):
    return (df["case_title"].fillna("") + " " + df["performed_work"].fillna("")).astype(str)

def make_5fold_val_test_one_per_class(y):
    """
    自定义 5 折划分：
    - 对每个类，打乱索引并循环选取两个作为 (val, test)，其余为 train
    - 对于第 k 折：val 取 pos[k % m]，test 取 pos[(k+1) % m]，其余 train
    要求：每类样本数 m >= 2（建议 >=5 更稳）
    返回：folds = [{"train": [...], "val": [...], "test": [...]}, ...] (len=5)
    """
    label_to_idxs = defaultdict(list)
    for idx, lbl in enumerate(y):
        label_to_idxs[lbl].append(idx)

    # 确保每个类至少2条
    for lbl, idxs in label_to_idxs.items():
        if len(idxs) < 2:
            raise ValueError(f"类别 {lbl} 只有 {len(idxs)} 条，无法保证 val/test 各1条。请先过滤该类或合并类别。")

    # 打乱每个类索引
    rng = np.random.RandomState(42)
    for lbl in label_to_idxs:
        rng.shuffle(label_to_idxs[lbl])

    folds = []
    for k in range(5):
        tr, va, te = [], [], []
        for lbl, idxs in label_to_idxs.items():
            m = len(idxs)
            v = idxs[k % m]
            t = idxs[(k + 1) % m]
            va.append(v)
            te.append(t)
            # 其余进训练
            for j in range(m):
                if j != (k % m) and j != ((k + 1) % m):
                    tr.append(idxs[j])
        folds.append({"train": sorted(tr), "val": sorted(va), "test": sorted(te)})
    return folds

def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(os.path.join(outdir, "df_filtered.csv"))
    # 关键列检查
    for col in ["case_title", "performed_work", "linked_items"]:
        if col not in df.columns:
            raise KeyError(f"df_filtered.csv 缺少列：{col}")

    # 单标签 & 文本
    df["linked_items"] = df["linked_items"].apply(ensure_single_label).astype(str)
    X_text = build_text(df).tolist()
    y_raw  = df["linked_items"].astype(str).tolist()

    # 生成 5 折：每类 val/test 各1样本，其余 train
    folds = make_5fold_val_test_one_per_class(y_raw)
    with open(os.path.join(outdir, "folds.json"), "w", encoding="utf-8") as f:
        json.dump(folds, f, ensure_ascii=False, indent=2)
    print(f"已生成 5 折并保存到 {os.path.join(outdir,'folds.json')}")

    # 统一标签编码（所有折共享）
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    # 训练每个折（用该折的 train）
    for k, fold in enumerate(folds):
        tr_idx = fold["train"]
        X_tr = [X_text[i] for i in tr_idx]
        y_tr = y_enc[tr_idx]

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=100_000,
                ngram_range=(1, 2),
                sublinear_tf=True
                # 若英文缩写多可改 analyzer="char", ngram_range=(2,4)
            )),
            ("clf", LogisticRegression(
                max_iter=3000,
                n_jobs=-1,
                class_weight="balanced",
                solver="saga",
                C=2.0,
                multi_class="ovr",
                random_state=42
            ))
        ])

        print(f"[Fold {k}] 训练样本数：{len(X_tr)}，类别数：{len(le.classes_)}")
        pipe.fit(X_tr, y_tr)

        joblib.dump(
            {"model": pipe, "label_encoder": le},
            os.path.join(outdir, f"model_fold{k}.joblib")
        )
        print(f"[Fold {k}] 模型已保存：{os.path.join(outdir, f'model_fold{k}.joblib')}")

    # 验证每折类覆盖（每类 val/test 应各1样本）
    for k, fold in enumerate(folds):
        v_labels = [y_raw[i] for i in fold["val"]]
        t_labels = [y_raw[i] for i in fold["test"]]
        print(f"[Fold {k}] 验证集类数：{len(set(v_labels))}，测试集类数：{len(set(t_labels))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="../output")
    args = parser.parse_args()
    main(args)
