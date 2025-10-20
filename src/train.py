# train.py
import os, json, ast, argparse, warnings, time
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import joblib
from tqdm import tqdm

warnings.filterwarnings("ignore")


class LossCallback:
    """顶层定义的回调类，保证可被 pickle 序列化。
    原先定义在 main() 内部会导致 pickling 时找不到类路径，从而报错。
    """
    def __init__(self):
        self.losses = []

    def __call__(self, params):
        if hasattr(params, "score"):
            # LogisticRegression 的损失是负对数似然
            # sklearn 内部在训练过程中会设置 score_ 等属性
            # 这里保留与原实现一致的负对数似然记录方式
            try:
                self.losses.append(-params.score_)
            except Exception:
                # 若 score_ 不可用则跳过
                pass

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
    """将 case_title + performed_work 合并为输入文本"""
    return (df["case_title"].fillna("") + " " + df["performed_work"].fillna("")).astype(str)

def make_5fold_val_test_one_per_class(y):
    """
    自定义 5 折划分：
    - 对每个类，打乱索引并循环选取两个作为 (val, test)，其余为 train
    - 对于第 k 折：val 取 pos[k % m]，test 取 pos[(k+1) % m]，其余 train
    要求：每类样本数 m >= 2（建议 >=5 更稳）
    返回：folds = [{"train": [...], "val": [...], "test": [...]}, ...] (len=5)
    """
    from collections import defaultdict
    label_to_idxs = defaultdict(list)
    for idx, lbl in enumerate(y):
        label_to_idxs[lbl].append(idx)

    # 确保每个类至少2条
    for lbl, idxs in label_to_idxs.items():
        if len(idxs) < 2:
            raise ValueError(f"类别 {lbl} 只有 {len(idxs)} 条，无法保证 val/test 各1条。请先过滤该类或合并类别。")

    # 打乱每个类索引（固定随机种子）
    rng = np.random.RandomState(42)
    for lbl in label_to_idxs:
        rng.shuffle(label_to_idxs[lbl])

    folds = []
    for k in range(5):
        tr, va, te = [], [], []
        for _, idxs in label_to_idxs.items():
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

def fmt_sec(sec: float) -> str:
    """将秒格式化为 H:MM:SS"""
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def main(args):
    global_start = time.time()
    print("=== 模型训练开始 ===")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # 读取数据
    df = pd.read_csv(os.path.join(outdir, "df_filtered.csv"))
    for col in ["case_title", "performed_work", "linked_items"]:
        if col not in df.columns:
            raise KeyError(f"df_filtered.csv 缺少列：{col}")

    # 单标签标准化 & 文本构建
    df["linked_items"] = df["linked_items"].apply(ensure_single_label).astype(str)
    X_text = build_text(df).tolist()
    y_raw  = df["linked_items"].astype(str).tolist()

    # 生成 5 折（每类 val/test 各1样本，其余 train）
    folds = make_5fold_val_test_one_per_class(y_raw)
    with open(os.path.join(outdir, "folds.json"), "w", encoding="utf-8") as f:
        json.dump(folds, f, ensure_ascii=False, indent=2)
    print(f"已生成 5 折并保存到 {os.path.join(outdir,'folds.json')}")
    print("各折样本数：")
    for k, fold in enumerate(folds):
        print(f"  Fold {k}: Train={len(fold['train'])}, Val={len(fold['val'])}, Test={len(fold['test'])}")


    # 标签编码（所有折共享同一编码器）
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)  # ndarray shape (N,)

    # 定义模型（字符级 n-gram + 逻辑回归）

    def make_pipeline():
        pipe1 = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=50_000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                # 若英文缩写多可改 analyzer="char", ngram_range=(2,4)
                min_df=2

            )),
            ("clf", LogisticRegression(
                max_iter=3000,
                n_jobs=-1,
                class_weight="balanced",
                # solver="saga",
                solver="liblinear",
                C=2.0,
                multi_class="ovr",
                random_state=42
            ))
        ])
        pipe2 = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char",
                # 字词级
                # analyzer="word",
                ngram_range=(2, 5),
                max_features=100_000,
                min_df=1,
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                # we'll perform iterative fitting by calling fit multiple times with warm_start
                max_iter=10,
                warm_start=True,
                n_jobs=-1,
                class_weight="balanced",
                solver="saga", 
                # solver="liblinear",
                C=4.0,
                # multi_class="ovr",
                multi_class = "multinomial",
                random_state=42,
                # verbose=1,  # 启用详细输出
                tol=1e-4    # 收敛容差
            ))
        ])
        return pipe2

    # 训练每个折（train 集），并评估 test，保存模型
    print("\n开始 5 折训练：")
    test_scores = []
    for k, fold in enumerate(tqdm(folds, total=5, desc="Training folds", ncols=88)):
        fold_start = time.time()

        tr_idx = fold["train"]
        X_tr = [X_text[i] for i in tr_idx]
        y_tr = y_enc[tr_idx]

        pipe = make_pipeline()
        callback = LossCallback()

        print(f"\n[Fold {k}] Train samples: {len(X_tr)}, Classes: {len(le.classes_)}")
        pipe_fit_start = time.time()
        tfidf = pipe.named_steps["tfidf"]
        clf = pipe.named_steps["clf"]
        X_tr_vec = tfidf.fit_transform(X_tr)

        # iterative fitting using warm_start; record loss each iteration
        classes = np.arange(len(le.classes_))
        
        
        max_epochs = 100

        patience = 5
        best_loss = float('inf')
        wait = 0
        from tqdm import trange
        epoch_bar = trange(max_epochs, desc=f"Fold {k} epochs", ncols=88)
        for epoch in epoch_bar:
            clf.fit(X_tr_vec, y_tr)
            try:
                y_proba = clf.predict_proba(X_tr_vec)
                loss_val = float(log_loss(y_tr, y_proba, labels=classes))
            except Exception:
                try:
                    dec = clf.decision_function(X_tr_vec)
                    if dec.ndim == 1:
                        probs_pos = 1 / (1 + np.exp(-dec))
                        y_proba = np.vstack([1 - probs_pos, probs_pos]).T
                    else:
                        e = np.exp(dec - np.max(dec, axis=1, keepdims=True))
                        y_proba = e / e.sum(axis=1, keepdims=True)
                    loss_val = float(log_loss(y_tr, y_proba, labels=classes))
                except Exception:
                    continue
            callback.losses.append(loss_val)
            epoch_bar.set_postfix({"loss": f"{loss_val:.4f}"})
            if loss_val + 1e-9 < best_loss:
                best_loss = loss_val
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        pipe_fit_end = time.time()

        # plot loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(callback.losses, label=f'Fold {k}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Fold {k} Training Loss Curve')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(outdir, f"loss_curve_fold{k}.png")
        plt.savefig(loss_plot_path)
        plt.close()
        loss_data_path = os.path.join(outdir, f"loss_data_fold{k}.json")
        with open(loss_data_path, "w") as f:
            json.dump({"losses": callback.losses}, f, indent=2)
        model_path = os.path.join(outdir, f"model_fold-{k}.joblib")
        joblib.dump({
            "model": pipe,
            "label_encoder": le
        }, model_path)
        fold_end = time.time()
        print(f"[Fold {k}] Loss plot saved: {loss_plot_path}")
        print(f"[Fold {k}] Loss data saved: {loss_data_path}")
        print(f"[Fold {k}] Vectorize+train time: {fmt_sec(pipe_fit_end - pipe_fit_start)}")
        print(f"[Fold {k}] Total time (incl. save): {fmt_sec(fold_end - fold_start)}")
        print(f"[Fold {k}] Model saved: {model_path}")

        # 评估 test
        test_idx = fold["test"]
        X_test = [X_text[i] for i in test_idx]
        y_test = y_enc[test_idx]
        X_test_vec = tfidf.transform(X_test)
        y_pred = clf.predict(X_test_vec)
        acc = (y_pred == y_test).mean()
        test_scores.append(acc)
        print(f"[Fold {k}] Test accuracy: {acc:.4f}")

    # 选 test accuracy 最优的 fold
    best_fold = int(np.argmax(test_scores))
    print(f"\nBest fold: {best_fold}, Test accuracy: {test_scores[best_fold]:.4f}")

    # 用 best fold 的 train+val 数据训练全量模型
    best_idxs = folds[best_fold]["train"] + folds[best_fold]["val"]
    X_best = [X_text[i] for i in best_idxs]
    y_best = y_enc[best_idxs]
    pipe = make_pipeline()
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    X_best_vec = tfidf.fit_transform(X_best)
    clf.fit(X_best_vec, y_best)
    joblib.dump({
        "model": pipe,
        "label_encoder": le
    }, os.path.join(outdir, "model_best.joblib"))
    print(f"Best model trained on train+val of fold {best_fold} and saved to model_best.joblib")

    # 验证每折类覆盖（每类 val/test 应各1样本）
    print("\n折内类覆盖检查：")
    for k, fold in enumerate(folds):
        v_labels = [y_raw[i] for i in fold["val"]]
        t_labels = [y_raw[i] for i in fold["test"]]
        print(f"[Fold {k}] 验证集类数：{len(set(v_labels))}，测试集类数：{len(set(t_labels))}")

    total_sec = time.time() - global_start
    print(f"\n=== 所有 Fold 训练完成，总耗时：{fmt_sec(total_sec)} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./output")
    args = parser.parse_args()
    main(args)
