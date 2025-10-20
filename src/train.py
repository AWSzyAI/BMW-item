# train.py
import os, json, ast, argparse, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, f1_score

import joblib
from tqdm import trange

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
    """将 case_title + performed_work 合并为输入文本"""
    return (df["case_title"].fillna("") + " " + df["performed_work"].fillna("")).astype(str)

def _flex_read_csv(base_dir: str, filename: str) -> pd.DataFrame:
    """尝试读取绝对路径；否则从 base_dir/filename 读取。"""
    if not filename:
        raise ValueError("filename 不能为空")
    if os.path.isabs(filename) and os.path.exists(filename):
        return pd.read_csv(filename)
    path = os.path.join(base_dir, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    # 最后再尝试工作目录
    if os.path.exists(filename):
        return pd.read_csv(filename)
    raise FileNotFoundError(f"未找到文件：{filename} 或 {path}")

def fmt_sec(sec: float) -> str:
    """将秒格式化为 H:MM:SS"""
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def main(args):
    global_start = time.time()
    print("=== 模型训练开始（使用 train.csv / eval.csv）===")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(args.modelsdir, exist_ok=True)

    # 读取 train / eval 数据
    df_tr = _flex_read_csv(outdir, args.train_file)
    df_ev = _flex_read_csv(outdir, args.eval_file)
    for df_name, df in [("train", df_tr), ("eval", df_ev)]:
        for col in ["case_title", "performed_work", "linked_items"]:
            if col not in df.columns:
                raise KeyError(f"{df_name}.csv 缺少列：{col}")

    # 规范标签 & 文本
    df_tr["linked_items"] = df_tr["linked_items"].apply(ensure_single_label).astype(str)
    df_ev["linked_items"] = df_ev["linked_items"].apply(ensure_single_label).astype(str)

    X_tr_text = build_text(df_tr).tolist()
    y_tr_raw = df_tr["linked_items"].astype(str).tolist()
    X_ev_text = build_text(df_ev).tolist()
    y_ev_raw = df_ev["linked_items"].astype(str).tolist()

    # 标签编码（仅基于训练集）
    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr_raw)

    # 过滤 eval 中不在训练标签集的样本（理论上 5-fold.py 会保证都有）
    ev_mask = [lbl in set(le.classes_) for lbl in y_ev_raw]
    if not all(ev_mask):
        dropped = int(np.sum(~np.array(ev_mask)))
        print(f"[警告] eval 中有 {dropped} 条样本的标签不在训练集中，将从评估中排除。")
    X_ev_text_f = [t for t, m in zip(X_ev_text, ev_mask) if m]
    y_ev_raw_f = [l for l, m in zip(y_ev_raw, ev_mask) if m]
    y_ev = le.transform(y_ev_raw_f) if len(y_ev_raw_f) > 0 else np.array([])

    # 定义模型：字符级 n-gram + 逻辑回归（warm_start 以便记录多轮损失）
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=100_000,
            min_df=1,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=10,
            warm_start=True,
            n_jobs=-1,
            class_weight="balanced",
            solver="saga",
            C=4.0,
            multi_class="multinomial",
            random_state=42,
            tol=1e-4
        ))
    ])

    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    X_tr_vec = tfidf.fit_transform(X_tr_text)

    # 迭代训练并记录训练损失（对训练集）
    classes = np.arange(len(le.classes_))
    max_epochs = args.max_epochs
    patience = args.patience
    best_loss = float("inf")
    wait = 0
    losses = []
    epoch_bar = trange(max_epochs, desc="Epochs", ncols=88)
    for _ in epoch_bar:
        clf.fit(X_tr_vec, y_tr)
        # 计算训练损失
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
                # 如果无法计算损失，则跳过该轮的记录
                continue
        losses.append(loss_val)
        epoch_bar.set_postfix({"train_loss": f"{loss_val:.4f}"})
        if loss_val + 1e-9 < best_loss:
            best_loss = loss_val
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # 评估（eval 集）
    X_ev_vec = tfidf.transform(X_ev_text_f) if len(X_ev_text_f) > 0 else None
    eval_metrics = {}
    if X_ev_vec is not None and len(y_ev) > 0:
        y_pred = clf.predict(X_ev_vec)
        acc = accuracy_score(y_ev, y_pred)
        f1w = f1_score(y_ev, y_pred, average="weighted")
        f1m = f1_score(y_ev, y_pred, average="macro")
        eval_metrics = {"accuracy": round(float(acc), 6), "f1_weighted": round(float(f1w), 6), "f1_macro": round(float(f1m), 6)}
        print(f"Eval metrics: {eval_metrics}")
    else:
        print("[提示] eval 集为空或无可评估样本，跳过评估。")

    # 保存 loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='train')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(outdir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()
    with open(os.path.join(outdir, "loss_data.json"), "w", encoding="utf-8") as f:
        json.dump({"losses": losses}, f, ensure_ascii=False, indent=2)

    # 保存模型
    model_bundle = {"model": pipe, "label_encoder": le}
    model_path = os.path.join(args.modelsdir, args.outmodel)
    joblib.dump(model_bundle, model_path)
    print(f"模型已保存到: {model_path}")

    # 保存评估指标
    if eval_metrics:
        metrics_path = os.path.join(outdir, "metrics_eval.csv")
        pd.DataFrame([eval_metrics]).to_csv(metrics_path, index=False)
        print(f"评估指标已保存到: {metrics_path}")

    total_sec = time.time() - global_start
    print(f"=== 训练完成，总耗时：{fmt_sec(total_sec)} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, default="train.csv", help="训练集文件名（默认从 outdir 读取）")
    parser.add_argument("--eval-file", type=str, default="eval.csv", help="验证集文件名（默认从 outdir 读取）")
    parser.add_argument("--outdir", type=str, default="./output", help="输出目录（读取数据与保存训练曲线/指标）")
    parser.add_argument("--modelsdir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--outmodel", type=str, default="model_best.joblib", help="模型保存文件名")
    parser.add_argument("--max-epochs", type=int, default=100, help="最大迭代轮次")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值")
    args = parser.parse_args()
    main(args)
