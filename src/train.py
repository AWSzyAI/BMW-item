#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from tqdm import tqdm

warnings.filterwarnings("ignore")
from utils import ensure_single_label, build_text, hit_at_k, fmt_sec, _flex_read_csv

# ========== 可选依赖：imbalanced-learn ==========
# 正确的导入位置：SMOTE 在 over_sampling；SMOTEENN/SMOTETomek 在 combine
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    _HAS_IMBLEARN = True
except Exception:
    SMOTE = SMOTEENN = SMOTETomek = RandomOverSampler = None
    _HAS_IMBLEARN = False

# 稀疏检测可选（无则降级处理）
try:
    from scipy.sparse import issparse, vstack as sp_vstack  # noqa: F401
except Exception:
    issparse = lambda m: False  # type: ignore
    sp_vstack = None  # type: ignore

from collections import Counter


# def _read_split_or_combined(base_dir: str, base_filename: str) -> pd.DataFrame:
#     """读取数据集，优先支持 X/Y 分离文件，其次回退到单表 CSV。

#     约定：
#     - 若传入 train.csv/test.csv/eval.csv，则优先查找同目录下的 train_X.csv + train_y.csv 等；
#     - X 文件应包含文本特征列：case_title、performed_work（可选 item_title 等）；
#     - y 文件应至少包含 'linked_items' 列。
#     返回值为合并后的 DataFrame（X 列 + linked_items 列）。
#     """
#     # 绝对路径优先，其次 base_dir 拼接
#     def _cand(p: str):
#         return p if (p and os.path.isabs(p) and os.path.exists(p)) else None

#     # 计算 stem（去掉 .csv 和可能的 _X/_y 后缀）
#     name = os.path.basename(base_filename)
#     stem = os.path.splitext(name)[0]
#     if stem.endswith("_X"):
#         stem = stem[:-2]
#     if stem.endswith("_y"):
#         stem = stem[:-2]

#     # 可能的文件名
#     x_name = f"{stem}_X.csv"
#     y_name = f"{stem}_y.csv"

#     # 1) 绝对路径的 X/Y
#     x_abs = _cand(base_filename)  # 若本身就是一个存在的绝对路径（很少见）
#     if x_abs and x_abs.endswith("_X.csv"):
#         y_abs = x_abs[:-6] + "_y.csv"
#         if os.path.exists(y_abs):
#             X_df = pd.read_csv(x_abs)
#             y_df = pd.read_csv(y_abs)
#             if "linked_items" not in y_df.columns:
#                 raise KeyError(f"{y_abs} 缺少列：linked_items")
#             df = pd.concat([X_df.reset_index(drop=True), y_df[["linked_items"]].reset_index(drop=True)], axis=1)
#             return df

#     # 2) base_dir 下的 X/Y
#     x_path = os.path.join(base_dir, x_name)
#     y_path = os.path.join(base_dir, y_name)
#     if os.path.exists(x_path) and os.path.exists(y_path):
#         X_df = pd.read_csv(x_path)
#         y_df = pd.read_csv(y_path)
#         if "linked_items" not in y_df.columns:
#             raise KeyError(f"{y_path} 缺少列：linked_items")
#         df = pd.concat([X_df.reset_index(drop=True), y_df[["linked_items"]].reset_index(drop=True)], axis=1)
#         return df

#     # 3) 回退：读取单表 CSV（绝对路径优先，再 base_dir）
#     return _flex_read_csv(base_dir, base_filename)



def _read_split_or_combined(base_dir: str, base_filename: str) -> pd.DataFrame:
    """优先读取 X/Y 分离文件；若不存在则回退到单表 CSV。

    约定：
    - base_filename 可为 train.csv / eval.csv 或 train_X.csv / eval_X.csv；
    - 若为单表名，将尝试在 base_dir 下寻找 <stem>_X.csv 与 <stem>_y.csv；
    - X 文件应包含文本特征列（如 case_title、performed_work 等），
      y 文件至少包含 'linked_items'（若为 'label'/'y' 会自动重命名）。
    """
    base_dir = os.path.abspath(base_dir)
    name = os.path.basename(base_filename)
    stem, ext = os.path.splitext(name)
    # 兼容传入 *_X.csv 或 *_y.csv 的情况，统一回到公共 stem
    if stem.endswith("_X"):
        stem = stem[:-2]
    if stem.endswith("_y"):
        stem = stem[:-2]

    x_name = f"{stem}_X.csv"
    y_name = f"{stem}_y.csv"

    def _exists_in_dir(fname: str) -> str | None:
        p = os.path.join(base_dir, fname)
        return p if os.path.exists(p) else None

    # 1) 优先尝试分表
    x_path = _exists_in_dir(x_name)
    y_path = _exists_in_dir(y_name)
    if x_path and y_path:
        X = _flex_read_csv(base_dir, os.path.basename(x_path))
        y = _flex_read_csv(base_dir, os.path.basename(y_path))

        # 兼容 y 列名
        if "linked_items" not in y.columns:
            if "label" in y.columns:
                y = y.rename(columns={"label": "linked_items"})
            elif "y" in y.columns:
                y = y.rename(columns={"y": "linked_items"})
            else:
                # 若多列，取第一列作为标签
                first_label_col = y.columns[0]
                warnings.warn(f"未找到 'linked_items'，使用 '{first_label_col}' 作为标签列")
                y = y.rename(columns={first_label_col: "linked_items"})

        # 只保留标签列
        y = y[["linked_items"]]
        if len(X) != len(y):
            raise ValueError(f"X/Y 行数不一致：X={len(X)} Y={len(y)}（stem={stem}）")
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        return df

    # 2) 回退：读取单表（例如 train.csv / eval.csv）
    warnings.warn(
        f"未找到分表 {x_name}+{y_name}，回退到单表 {name}（目录：{base_dir}）"
    )
    return _flex_read_csv(base_dir, name)


class LossCallback:
    """记录并计算训练损失的简单回调（用于兼容旧模型反序列化）。

    - 使用 sklearn.metrics.log_loss 作为显式的 loss_fn
    - 保存每个 epoch 的损失值到 self.losses
    """

    def __init__(self, classes: np.ndarray):
        self.classes = np.asarray(classes)
        self.losses: list[float] = []

    def compute(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        try:
            return float(log_loss(y_true, y_proba, labels=self.classes))
        except Exception:
            return float("nan")

    def on_epoch_end(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        v = self.compute(y_true, y_proba)
        self.losses.append(v)
        return v


def main(args):
    global_start = time.time()
    print("=== 模型训练开始（使用 train.csv / eval.csv）===")

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.modelsdir, exist_ok=True)

    # 读取 train / eval 数据（优先 X/Y 分离，再回退单表）
    df_tr = _read_split_or_combined(args.outdir, args.train_file)
    df_ev = _read_split_or_combined(args.outdir, args.eval_file)
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

    # 如果某一标签下只有一个样本，那就把这个样本复制一份（极端少样本的兜底）
    vc = df_tr["linked_items"].value_counts()
    rare_labels = vc[vc == 1].index.tolist()
    if rare_labels:
        rare_samples = df_tr[df_tr["linked_items"].isin(rare_labels)]
        df_tr = pd.concat([df_tr, rare_samples], ignore_index=True)
        print(f"已复制 {len(rare_samples)} 个单样本类别，以平衡训练集。")

    # 更新文本与标签（复制后）
    X_tr_text = build_text(df_tr).tolist()
    y_tr_raw = df_tr["linked_items"].astype(str).tolist()

    # 标签编码（仅基于训练集）
    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr_raw)

    # 过滤 eval 中不在训练标签集的样本
    ev_mask = [lbl in set(le.classes_) for lbl in y_ev_raw]
    if not all(ev_mask):
        dropped = int(np.sum(~np.array(ev_mask)))
        print(f"[警告] eval 中有 {dropped} 条样本的标签未在训练集中出现（记为 not_in_train）")
    X_ev_text_f = [t for t, m in zip(X_ev_text, ev_mask) if m]
    y_ev_raw_f = [l for l, m in zip(y_ev_raw, ev_mask) if m]
    y_ev = le.transform(y_ev_raw_f) if len(y_ev_raw_f) > 0 else np.array([])

    # 向量化器与分类器（后续用 partial_fit 显式逐 epoch 训练，便于记录 loss 曲线）
    tfidf = TfidfVectorizer(
        analyzer=args.tfidf_analyzer,
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.tfidf_max_features,
        min_df=1,
        sublinear_tf=True,
    )
    clf = SGDClassifier(
        loss="log_loss",
        early_stopping=False,
        max_iter=1,
        tol=None,
        random_state=42,
        learning_rate="optimal",
        class_weight=("balanced" if args.class_weight_balanced else None),
        penalty=args.sgd_penalty,
        alpha=args.sgd_alpha,
        average=args.sgd_average,
        n_jobs=-1,
    )

    # === 进度条与剩余时间估计 ===
    do_eval = (len(X_ev_text_f) > 0 and len(y_ev) > 0)
    do_calib = (args.calibrate in {"sigmoid", "isotonic"}) and do_eval
    total_steps = 0
    total_steps += 1  # vectorize train
    total_steps += 1  # epoch training (aggregated)
    if do_eval:
        total_steps += 1  # vectorize eval
    if do_calib:
        total_steps += 1  # calibrate
    if do_eval:
        total_steps += 1  # eval metrics
    total_steps += 1      # ood detector
    total_steps += 1      # save loss
    total_steps += 1      # save model
    total_steps += 1      # save metrics (may be skipped)

    pbar = tqdm(total=total_steps, desc="Train pipeline", unit="step")
    step_durations = []

    def _update_pbar(t0, label):
        dt = time.time() - t0
        step_durations.append(dt)
        remaining_steps = pbar.total - pbar.n
        mean_dt = (sum(step_durations) / len(step_durations)) if step_durations else 0.0
        eta_sec = mean_dt * max(0, remaining_steps)
        pbar.set_postfix_str(f"{label}={fmt_sec(dt)} | ETA~{fmt_sec(eta_sec)}")
        pbar.update(1)

    # 向量化训练集
    t0 = time.time()
    X_tr_vec = tfidf.fit_transform(X_tr_text)
    _update_pbar(t0, "vectorize_train")

    # ========== 类别不平衡处理（修正版，含无依赖回退） ==========
    resample_method = getattr(args, "resample_method", "none")
    if resample_method != "none":
        print(f"[Info] 启用不平衡采样：{resample_method}")

        # 简易随机过采样（无依赖回退），支持稀疏/稠密
        def _simple_ros_sparse_or_dense(X, y):
            y = np.asarray(y)
            classes, counts = np.unique(y, return_counts=True)
            max_n = counts.max()
            idx_all = []
            rng = np.random.default_rng(42)
            for c in classes:
                idx_c = np.where(y == c)[0]
                if len(idx_c) == 0:
                    continue
                if len(idx_c) < max_n:
                    extra = rng.choice(idx_c, size=max_n - len(idx_c), replace=True)
                    idx_c = np.concatenate([idx_c, extra], axis=0)
                idx_all.append(idx_c)
            sel = np.concatenate(idx_all, axis=0)
            if issparse(X):
                return X[sel], y[sel]
            else:
                return X[sel], y[sel]

        requires_dense = False
        sampler = None

        if resample_method == "smote":
            if _HAS_IMBLEARN and SMOTE is not None:
                sampler = SMOTE(random_state=42)
                requires_dense = True
            else:
                print("[Warning] 未安装 imbalanced-learn 或导入失败，SMOTE 回退为随机过采样（简易实现）。")
                resample_method = "ros"

        if resample_method == "smoteenn":
            if _HAS_IMBLEARN and SMOTEENN is not None:
                sampler = SMOTEENN(random_state=42)
                requires_dense = True
            else:
                print("[Warning] 未安装 imbalanced-learn 或导入失败，SMOTEENN 回退为随机过采样（简易实现）。")
                resample_method = "ros"

        if resample_method == "smotetomek":
            if _HAS_IMBLEARN and SMOTETomek is not None:
                sampler = SMOTETomek(random_state=42)
                requires_dense = True
            else:
                print("[Warning] 未安装 imbalanced-learn 或导入失败，SMOTETomek 回退为随机过采样（简易实现）。")
                resample_method = "ros"

        if resample_method == "ros":
            if _HAS_IMBLEARN and RandomOverSampler is not None:
                sampler = RandomOverSampler(random_state=42)
                requires_dense = False
            else:
                sampler = None  # 使用简易 ROS

        # 执行采样（避免超大矩阵稠密化导致 OOM）
        if sampler is not None:
            if requires_dense:
                n_samples, n_features = X_tr_vec.shape
                DENSE_THRESHOLD = 50_000_000  # 50M 元素保守阈值
                if n_samples * n_features > DENSE_THRESHOLD:
                    print("[Warning] 稠密化内存开销过大，回退为简易随机过采样。")
                    X_tr_vec, y_tr = _simple_ros_sparse_or_dense(X_tr_vec, y_tr)
                else:
                    X_tr_vec = X_tr_vec.toarray()
                    X_tr_vec, y_tr = sampler.fit_resample(X_tr_vec, y_tr)
            else:
                try:
                    X_tr_vec, y_tr = sampler.fit_resample(X_tr_vec, y_tr)
                except Exception:
                    print("[Warning] 采样器不支持稀疏输入，回退为简易随机过采样。")
                    X_tr_vec, y_tr = _simple_ros_sparse_or_dense(X_tr_vec, y_tr)
        else:
            X_tr_vec, y_tr = _simple_ros_sparse_or_dense(X_tr_vec, y_tr)

        n_after = X_tr_vec.shape[0]  # 稀疏/稠密/ndarray 都有 shape
        print(f"[Info] 采样后训练集样本数: {n_after} | label分布: {Counter(y_tr)}")
    # ========== 不平衡处理结束 ==========

    # 逐 epoch 训练并记录 loss
    classes = np.arange(len(le.classes_))
    loss_cb = LossCallback(classes)
    losses: list[float] = []

    # 简单的早停逻辑
    best_loss = float("inf")
    no_improve = 0

    # 按批量进行增量更新
    n_samples = X_tr_vec.shape[0]
    batch_size = max(1, int(args.batch_size))
    n_batches = (n_samples + batch_size - 1) // batch_size

    fit_t0 = time.time()
    epoch_pbar = tqdm(total=int(args.max_epochs), desc="Train epochs", unit="epoch", leave=False)
    epoch_durations = []
    rng = np.random.default_rng(42)

    for epoch in range(int(args.max_epochs)):
        ep_t0 = time.time()
        idx = np.arange(n_samples)
        if args.shuffle:
            rng.shuffle(idx)

        # 增量学习：首个 partial_fit 需要提供 classes
        first = True
        for b in range(n_batches):
            sl = idx[b * batch_size : (b + 1) * batch_size]
            X_b = X_tr_vec[sl]
            y_b = np.asarray(y_tr)[sl]
            if first:
                clf.partial_fit(X_b, y_b, classes=classes)
                first = False
            else:
                clf.partial_fit(X_b, y_b)

        # 每个 epoch 结束后在训练集（或其子集）上计算 log loss
        loss_idx = np.arange(n_samples)
        loss_sample_size = int(getattr(args, "loss_sample_size", 20000))
        if loss_sample_size > 0 and n_samples > loss_sample_size:
            loss_idx = rng.choice(n_samples, size=loss_sample_size, replace=False)
        X_loss = X_tr_vec[loss_idx]
        y_loss = np.asarray(y_tr)[loss_idx]
        try:
            y_proba = clf.predict_proba(X_loss)
        except Exception:
            dec = clf.decision_function(X_loss)
            if dec.ndim == 1:
                probs_pos = 1 / (1 + np.exp(-dec))
                y_proba = np.vstack([1 - probs_pos, probs_pos]).T
            else:
                e = np.exp(dec - np.max(dec, axis=1, keepdims=True))
                y_proba = e / e.sum(axis=1, keepdims=True)
        loss_val = loss_cb.on_epoch_end(np.asarray(y_loss), y_proba)
        losses.append(loss_val)

        # tqdm 后缀打印 loss 与剩余时间 ETA
        try:
            epoch_durations.append(time.time() - ep_t0)
            rem = max(0, int(args.max_epochs) - (epoch + 1))
            mean_dt = (sum(epoch_durations) / len(epoch_durations)) if epoch_durations else 0.0
            eta_sec = mean_dt * rem
            epoch_pbar.set_postfix_str(f"loss={loss_val:.6f} | ETA~{fmt_sec(eta_sec)}")
            epoch_pbar.update(1)
        except Exception:
            pass

        # 打印并写入训练日志
        log_dir = os.path.join(os.getcwd(), "log")
        os.makedirs(log_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.outmodel))[0]
        train_log_path = os.path.join(log_dir, f"{stem}_train.txt")
        msg = f"epoch={epoch+1}/{args.max_epochs} loss={loss_val:.6f}"
        print(msg)
        try:
            with open(train_log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

        # 早停
        if loss_val < best_loss - 1e-8:
            best_loss = loss_val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(args.patience):
                print(f"Early stopping at epoch {epoch+1} (best_loss={best_loss:.6f})")
                break

    fit_sec = time.time() - fit_t0
    try:
        epoch_pbar.close()
    except Exception:
        pass

    print(f"训练完成，耗时 {fmt_sec(fit_sec)}")
    try:
        _update_pbar(fit_t0, "fit_epochs")
    except Exception:
        pass

    # 评估（eval 集）
    X_ev_vec = None
    if len(X_ev_text_f) > 0:
        t0 = time.time()
        X_ev_vec = tfidf.transform(X_ev_text_f)
        try:
            _update_pbar(t0, "vectorize_eval")
        except Exception:
            pass
    eval_metrics = {}
    if X_ev_vec is not None and len(y_ev) > 0:
        # 概率校准（可选）
        calibrator = None
        if args.calibrate in {"sigmoid", "isotonic"}:
            try:
                calibrator = CalibratedClassifierCV(base_estimator=clf, method=args.calibrate, cv="prefit")
                calibrate_t0 = time.time()
                calibrator.fit(X_ev_vec, y_ev)
                print(f"已使用 {args.calibrate} 概率校准（耗时 {fmt_sec(time.time() - calibrate_t0)}）")
                clf = calibrator
                try:
                    _update_pbar(calibrate_t0, "calibration")
                except Exception:
                    pass
            except Exception as e:
                print(f"[警告] 概率校准失败，将继续使用未校准模型。原因：{e}")

        clf_for_eval = clf
        t0 = time.time()
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
        try:
            _update_pbar(t0, "eval_metrics")
        except Exception:
            pass

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

    # 训练未知标签（not-in-train）检测器（MSP/LogReg）
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    clf_final = clf  # 可能已被校准替换

    # 1) 在 eval 全量上计算 p_max（若 eval 为空则跳过）
    y_ooc = None
    p_max_ev_all = None
    if len(X_ev_text) > 0:
        y_ooc = np.array([0 if lbl in set(le.classes_) else 1 for lbl in y_ev_raw])
        X_ev_vec_all = tfidf.transform(X_ev_text)
        try:
            y_proba_ev_all = clf_final.predict_proba(X_ev_vec_all)
        except Exception:
            dec_all = clf_final.decision_function(X_ev_vec_all)
            if dec_all.ndim == 1:
                probs_pos = 1 / (1 + np.exp(-dec_all))
                y_proba_ev_all = np.vstack([1 - probs_pos, probs_pos]).T
            else:
                e = np.exp(dec_all - np.max(dec_all, axis=1, keepdims=True))
                y_proba_ev_all = e / e.sum(axis=1, keepdims=True)
        p_max_ev_all = y_proba_ev_all.max(axis=1).reshape(-1, 1)

    # 2) 统计 ID 的 p_max 分布（优先 eval 中的 ID；否则回退到 train）
    if p_max_ev_all is not None and y_ooc is not None and np.any(y_ooc == 0):
        id_pmax_for_stats = p_max_ev_all[y_ooc == 0].ravel()
    else:
        # 回退使用训练集 p_max 作为 ID 分布
        try:
            y_proba_tr_all = clf_final.predict_proba(X_tr_vec)
        except Exception:
            dec_tr = clf_final.decision_function(X_tr_vec)
            if dec_tr.ndim == 1:
                probs_pos = 1 / (1 + np.exp(-dec_tr))
                y_proba_tr_all = np.vstack([1 - probs_pos, probs_pos]).T
            else:
                e = np.exp(dec_tr - np.max(dec_tr, axis=1, keepdims=True))
                y_proba_tr_all = e / e.sum(axis=1, keepdims=True)
        id_pmax_for_stats = y_proba_tr_all.max(axis=1)

    # 3) 若 eval 有 OOD 正样本，拟合 LogReg；否则使用MSP分位数阈值回退
    ooc_detector = None
    ood_t0 = time.time()
    if y_ooc is not None and len(np.unique(y_ooc)) >= 2:
        ooc_clf = LogisticRegression(random_state=42)
        ooc_clf.fit(p_max_ev_all, y_ooc)
        try:
            auc = roc_auc_score(y_ooc, ooc_clf.predict_proba(p_max_ev_all)[:, 1])
            print(f"OOC detection(LogReg) AUC: {auc:.4f}")
        except Exception:
            print("OOC detection(LogReg) fitted (AUC unavailable)")
        ooc_detector = {"kind": "logreg", "estimator": ooc_clf}
    else:
        tau = float(np.percentile(id_pmax_for_stats, getattr(args, "ooc_tau_percentile", 5.0)))
        temperature = float(getattr(args, "ooc_temperature", 20.0))
        ooc_detector = {"kind": "threshold", "tau": tau, "temperature": temperature}
        print(f"OOC detection(Threshold) used: tau(p_max)={tau:.4f} at {getattr(args, 'ooc_tau_percentile', 5.0)}th percentile (no positive OOD)")
    try:
        _update_pbar(ood_t0, "fit_ood_detector")
    except Exception:
        pass

    # 保存 loss 曲线
    loss_t0 = time.time()
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='train')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(args.outdir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()
    with open(os.path.join(args.outdir, "loss_data.json"), "w", encoding="utf-8") as f:
        json.dump({"losses": losses}, f, ensure_ascii=False, indent=2)
    try:
        _update_pbar(loss_t0, "save_loss")
    except Exception:
        pass

    # 保存模型（统一推理接口）
    model_t0 = time.time()
    pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
    model_bundle = {"model": pipe, "label_encoder": le, "ooc_detector": ooc_detector}
    model_path = os.path.join(args.modelsdir, args.outmodel)
    joblib.dump(model_bundle, model_path)
    print(f"模型已保存到: {model_path}")
    try:
        _update_pbar(model_t0, "save_model")
    except Exception:
        pass

    # 保存评估指标
    if eval_metrics:
        metrics_t0 = time.time()
        metrics_path = os.path.join(args.outdir, "metrics_eval.csv")
        pd.DataFrame([eval_metrics]).to_csv(metrics_path, index=False)
        print(f"评估指标已保存到: {metrics_path}")
        try:
            _update_pbar(metrics_t0, "save_metrics")
        except Exception:
            pass

    total_sec = time.time() - global_start
    print(f"=== 训练完成，总耗时：{fmt_sec(total_sec)} ===")
    try:
        pbar.close()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, default="train.csv", help="训练集文件名（默认从 outdir 读取）")
    parser.add_argument("--eval-file", type=str, default="eval.csv", help="验证集文件名（默认从 outdir 读取）")
    parser.add_argument("--outdir", type=str, default="./output/2025_up_to_month_2", help="输出目录（读取数据与保存训练曲线/指标）")
    parser.add_argument("--modelsdir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--outmodel", type=str, default="9.joblib", help="模型保存文件名")

    parser.add_argument("--max-epochs", type=int, default=1000, help="最大迭代轮次")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值（若连续 N 个 epoch 未提升则停止）")
    parser.add_argument("--batch-size", type=int, default=1024, help="增量训练的 batch 大小")
    parser.add_argument("--shuffle", action="store_true", help="每个 epoch 打乱样本")
    parser.add_argument("--loss-sample-size", type=int, default=20000, help="每个epoch用于计算loss的样本数（0或负数表示使用全量）")

    # 可调向量化/优化参数
    parser.add_argument("--tfidf-analyzer", type=str, default="char", choices=["char", "char_wb"], help="TF-IDF 分析粒度")
    parser.add_argument("--ngram-min", type=int, default=2, help="ngram 最小长度")
    parser.add_argument("--ngram-max", type=int, default=5, help="ngram 最大长度")
    parser.add_argument("--tfidf-max-features", type=int, default=100_000, help="TF-IDF 最大特征数")
    parser.add_argument("--sgd-penalty", type=str, default="l2", choices=["l2", "l1", "elasticnet"], help="SGD 正则项")
    parser.add_argument("--sgd-alpha", type=float, default=0.0001, help="SGD 正则强度 alpha")
    parser.add_argument("--sgd-average", action="store_true", help="启用参数平均（有助于泛化）")
    parser.add_argument("--class-weight-balanced", action="store_true", help="启用类别平衡权重")
    parser.add_argument("--calibrate", type=str, default="none", choices=["none", "sigmoid", "isotonic"], help="概率校准方式（基于 eval 进行预拟合）")

    # OOC/MSP 阈值回退参数
    parser.add_argument("--ooc-tau-percentile", type=float, default=5.0, help="无 OOD 正样本时，p_max 的分位数阈值（百分位）")
    parser.add_argument("--ooc-temperature", type=float, default=20.0, help="将 (tau - p_max) 经 sigmoid 映射为概率的温度系数")

    parser.add_argument(
        "--resample-method",
        type=str,
        default="none",
        choices=["none", "ros", "smote", "smoteenn", "smotetomek"],
        help="不平衡处理方法：none/ros/smote/smoteenn/smotetomek",
    )
    args = parser.parse_args()
    main(args)