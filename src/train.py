import os, json, argparse, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score, f1_score
import joblib
from tqdm import trange

warnings.filterwarnings("ignore")
from utils import ensure_single_label, build_text, hit_at_k, fmt_sec, _flex_read_csv

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
        print(f"[警告] eval 中有 {dropped} 条样本的标签未在训练集中出现（记为 not_in_train），这些样本将从指标计算中排除，仅供事后分析。")
    X_ev_text_f = [t for t, m in zip(X_ev_text, ev_mask) if m]
    y_ev_raw_f = [l for l, m in zip(y_ev_raw, ev_mask) if m]
    y_ev = le.transform(y_ev_raw_f) if len(y_ev_raw_f) > 0 else np.array([])

    # 定义模型：字符级 n-gram + SGDClassifier（对大规模稀疏特征更快）
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer=args.tfidf_analyzer,
            ngram_range=(args.ngram_min, args.ngram_max),
            max_features=args.tfidf_max_features,
            min_df=1,
            sublinear_tf=True,
        )),
        ("clf", SGDClassifier(
            loss="log_loss",           # 逻辑回归等价的对数损失
            max_iter=args.max_epochs,
            tol=1e-4,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=args.patience,
            learning_rate="optimal",
            class_weight=("balanced" if args.class_weight_balanced else None),
            penalty=args.sgd_penalty,
            alpha=args.sgd_alpha,
            average=args.sgd_average,
            n_jobs=-1,
        ))
    ])

    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    X_tr_vec = tfidf.fit_transform(X_tr_text)

    # 单次拟合（去掉外层 epoch 循环）
    classes = np.arange(len(le.classes_))
    fit_t0 = time.time()
    clf.fit(X_tr_vec, y_tr)
    fit_sec = time.time() - fit_t0
    # SGDClassifier 没有 max_iter 属性? 实际存在，但为参数；此处稳妥打印拟合耗时
    print(f"训练完成，耗时 {fmt_sec(fit_sec)}")

    # 计算训练集损失（仅一次）
    losses = []
    try:
        # SGDClassifier 在 loss='log_loss' 时支持 predict_proba
        y_proba = clf.predict_proba(X_tr_vec)
        loss_val = float(log_loss(y_tr, y_proba, labels=classes))
    except Exception:
        # 兜底：使用 decision_function 转换为概率（多分类使用 softmax）
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
            loss_val = float("nan")
    losses.append(loss_val)

    # 评估（eval 集）
    X_ev_vec = tfidf.transform(X_ev_text_f) if len(X_ev_text_f) > 0 else None
    eval_metrics = {}
    if X_ev_vec is not None and len(y_ev) > 0:
        # 可选：概率校准（提升阈值型/开放集判别的可靠性）
        calibrator = None
        if args.calibrate in {"sigmoid", "isotonic"}:
            try:
                calibrator = CalibratedClassifierCV(base_estimator=clf, method=args.calibrate, cv="prefit")
                calibrate_t0 = time.time()
                calibrator.fit(X_ev_vec, y_ev)
                print(f"已使用 {args.calibrate} 概率校准（耗时 {fmt_sec(time.time() - calibrate_t0)}）")
                # 替换管道中的分类器为校准后的版本
                pipe.named_steps["clf"] = calibrator
            except Exception as e:
                print(f"[警告] 概率校准失败，将继续使用未校准模型。原因：{e}")

        # 使用（可能校准的）分类器进行评估
        clf_for_eval = pipe.named_steps["clf"]
        y_pred = clf_for_eval.predict(X_ev_vec)
        acc = accuracy_score(y_ev, y_pred)
        f1w = f1_score(y_ev, y_pred, average="weighted")
        f1m = f1_score(y_ev, y_pred, average="macro")
        # 计算概率用于 hit@k
        y_proba_ev = None
        try:
            y_proba_ev = clf_for_eval.predict_proba(X_ev_vec)
        except Exception:
            try:
                dec = clf_for_eval.decision_function(X_ev_vec)
                if dec.ndim == 1:
                    probs_pos = 1 / (1 + np.exp(-dec))
                    y_proba_ev = np.vstack([1 - probs_pos, probs_pos]).T
                else:
                    e = np.exp(dec - np.max(dec, axis=1, keepdims=True))
                    y_proba_ev = e / e.sum(axis=1, keepdims=True)
            except Exception:
                y_proba_ev = None

        eval_metrics = {
            "accuracy": round(float(acc), 6),
            "f1_weighted": round(float(f1w), 6),
            "f1_macro": round(float(f1m), 6),
            "hit@1": round(hit_at_k(y_ev, y_proba_ev, 1), 6) if y_proba_ev is not None else float("nan"),
            "hit@3": round(hit_at_k(y_ev, y_proba_ev, 3), 6) if y_proba_ev is not None else float("nan"),
            "hit@5": round(hit_at_k(y_ev, y_proba_ev, 5), 6) if y_proba_ev is not None else float("nan"),
            "hit@10": round(hit_at_k(y_ev, y_proba_ev, 10), 6) if y_proba_ev is not None else float("nan"),
        }
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
    # 可调向量化/优化参数（为了冲击更高 hit@3 & 稳定概率）
    parser.add_argument("--tfidf-analyzer", type=str, default="char", choices=["char", "char_wb"], help="TF-IDF 分析粒度")
    parser.add_argument("--ngram-min", type=int, default=2, help="ngram 最小长度")
    parser.add_argument("--ngram-max", type=int, default=5, help="ngram 最大长度")
    parser.add_argument("--tfidf-max-features", type=int, default=100_000, help="TF-IDF 最大特征数（可调大以提升效果）")
    parser.add_argument("--sgd-penalty", type=str, default="l2", choices=["l2", "l1", "elasticnet"], help="SGD 正则项")
    parser.add_argument("--sgd-alpha", type=float, default=0.0001, help="SGD 正则强度 alpha")
    parser.add_argument("--sgd-average", action="store_true", help="启用参数平均（有助于泛化）")
    parser.add_argument("--class-weight-balanced", action="store_true", help="启用类别平衡权重")
    parser.add_argument("--calibrate", type=str, default="none", choices=["none", "sigmoid", "isotonic"], help="概率校准方式（基于 eval 进行预拟合）")
    args = parser.parse_args()
    main(args)
