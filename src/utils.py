
import ast
import numpy as np
import pandas as pd
import os

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, log_loss, roc_auc_score
)
def ensure_single_label(s):
    """
    处理linked_items字段。
    若列里偶有 '["a","b"]' 这类字符串，就取第一个；正常单标签直接返回字符串。
    """
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
    """合并文本字段：case_title + performed_work"""
    parts = [df.get("case_title", "").fillna("").astype(str),
             df.get("performed_work", "").fillna("").astype(str)]
    # if "item_title" in df.columns:
    #     parts.append(df["item_title"].fillna("").astype(str))
    return (parts[0] + " " + parts[1] + (" " + parts[2] if len(parts) > 2 else "")).astype(str)

def hit_at_k(y_true_idx: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """计算 hit@k 命中率，与 eval.py 保持一致实现。"""
    if y_proba is None or y_proba.size == 0:
        return float("nan")
    k = min(k, y_proba.shape[1])
    topk_idx = np.argsort(-y_proba, axis=1)[:, :k]
    hits = (topk_idx == y_true_idx.reshape(-1, 1)).any(axis=1)
    return float(hits.mean())

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
    """将秒格式化为 HH:MM:SS"""
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def eval_split(model, le, X_text, y_raw, indices):
    y_true = le.transform([y_raw[i] for i in indices])
    Xs = [X_text[i] for i in indices]
    y_pred = model.predict(Xs)
    if isinstance(y_pred[0], str):
        y_pred = le.transform(y_pred)
    y_pred = np.asarray(y_pred)
    y_proba = model.predict_proba(Xs)

    m = {}
    m["acc"] = accuracy_score(y_true, y_pred)
    m["bal_acc"] = balanced_accuracy_score(y_true, y_pred)
    m["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    m["prec_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    m["rec_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        m["logloss"] = log_loss(y_true, y_proba, labels=np.arange(len(le.classes_)))
    except Exception:
        m["logloss"] = np.nan

    try:
        m["auc_macro"] = roc_auc_score(
            y_true, y_proba, multi_class="ovo", average="macro", labels=np.arange(len(le.classes_))
        )
    except Exception:
        m["auc_macro"] = np.nan

    for k in [1, 3, 5, 10]:
        m[f"hit@{k}"] = hit_at_k(y_true, y_proba, k)
    return m