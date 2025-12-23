
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
    
    # 如果k大于类别数量，返回1.0（因为所有类别都会被包含）
    if k >= y_proba.shape[1]:
        return 1.0
    
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


def mean_reciprocal_rank(y_true_idx: np.ndarray, y_proba: np.ndarray) -> float:
    """
    计算平均倒数排名 (MRR)
    MRR = (1/|Q|) * Σ(1/rank_i)
    其中rank_i是正确答案在预测结果中的排名
    """
    if y_proba is None or y_proba.size == 0:
        return float("nan")
    
    # 获取排名（降序排列）
    ranks = np.argsort(-y_proba, axis=1)
    
    # 找到每个样本的正确答案排名
    mrr_sum = 0.0
    for i, true_idx in enumerate(y_true_idx):
        # 找到正确答案在排名中的位置（从1开始）
        rank_position = np.where(ranks[i] == true_idx)[0]
        if len(rank_position) > 0:
            # 排名从1开始，所以加1
            mrr_sum += 1.0 / (rank_position[0] + 1)
    
    return mrr_sum / len(y_true_idx)


def ndcg_at_k(y_true_idx: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """
    计算NDCG@K (Normalized Discounted Cumulative Gain at K)
    NDCG@K = DCG@K / IDCG@K
    """
    if y_proba is None or y_proba.size == 0:
        return float("nan")
    
    # 如果k大于类别数量，调整k值
    if k >= y_proba.shape[1]:
        k = y_proba.shape[1]
    
    # 获取前K个预测的索引
    topk_idx = np.argsort(-y_proba, axis=1)[:, :k]
    
    # 计算DCG@K
    dcg_sum = 0.0
    for i, true_idx in enumerate(y_true_idx):
        dcg = 0.0
        for j, pred_idx in enumerate(topk_idx[i]):
            if pred_idx == true_idx:
                # DCG使用log2(j+2)作为折扣因子（j从0开始）
                dcg += 1.0 / np.log2(j + 2)
                break
        dcg_sum += dcg
    
    # 计算IDCG@K（理想DCG，假设正确答案排在第1位）
    idcg_sum = 0.0
    for i in range(len(y_true_idx)):
        idcg = 1.0 / np.log2(1 + 1)  # j=0, log2(0+2)=1
        idcg_sum += idcg
    
    # 避免除零
    if idcg_sum == 0:
        return 0.0
    
    return dcg_sum / idcg_sum


def coverage_at_k(y_proba: np.ndarray, k: int) -> float:
    """
    计算覆盖率@K (Coverage@K)
    Coverage@K = |∪_{i=1}^{|Q|} TopK(q_i)| / |I|
    表示所有查询的TopK结果并集占总类别集合的比例
    """
    if y_proba is None or y_proba.size == 0:
        return float("nan")
    
    # 如果k大于类别数量，调整k值
    if k >= y_proba.shape[1]:
        k = y_proba.shape[1]
    
    # 获取所有样本的TopK预测
    topk_idx = np.argsort(-y_proba, axis=1)[:, :k]
    
    # 计算并集大小
    unique_items = set()
    for i in range(topk_idx.shape[0]):
        unique_items.update(topk_idx[i])
    
    # 计算覆盖率
    total_items = y_proba.shape[1]
    coverage = len(unique_items) / total_items
    
    return coverage


def calculate_error_analysis(y_true_labels: np.ndarray, y_pred_labels: np.ndarray,
                           X_texts: list, error_types: dict = None) -> dict:
    """
    计算错误类型分析
    分析不同类型的错误：文本噪声、描述模糊、过短/过长等
    """
    if error_types is None:
        error_types = {
            'text_noise': [],      # 文本噪声
            'vague_description': [], # 描述模糊
            'too_short': [],       # 过短
            'too_long': [],        # 过长
            'other': []            # 其他
        }
    
    # 找出错误预测的样本
    error_mask = y_true_labels != y_pred_labels
    error_indices = np.where(error_mask)[0]
    
    for idx in error_indices:
        text = X_texts[idx]
        text_length = len(text.split())
        
        # 根据文本特征分类错误类型
        if text_length < 5:
            error_types['too_short'].append(idx)
        elif text_length > 200:
            error_types['too_long'].append(idx)
        elif any(char.isdigit() for char in text) and any(char.isalpha() for char in text):
            error_types['text_noise'].append(idx)
        elif any(word in text.lower() for word in ['其他', '其它', '等等', '之类', '一些']):
            error_types['vague_description'].append(idx)
        else:
            error_types['other'].append(idx)
    
    # 计算错误类型分布
    total_errors = len(error_indices)
    error_distribution = {}
    for error_type, indices in error_types.items():
        error_distribution[error_type] = {
            'count': len(indices),
            'percentage': len(indices) / total_errors * 100 if total_errors > 0 else 0,
            'indices': indices
        }
    
    return error_distribution


def calculate_top_n_distribution(y_proba: np.ndarray, y_true_idx: np.ndarray,
                                 label_names: list = None) -> dict:
    """
    计算Top-N分布
    统计每个item在Top-N中的出现频率，分析热门偏向
    """
    if y_proba is None or y_proba.size == 0:
        return {}
    
    n_items = y_proba.shape[1]
    if label_names is None:
        label_names = [f"item_{i}" for i in range(n_items)]
    
    # 计算每个item在Top-1, Top-3, Top-5, Top-10中的出现次数
    top_k_values = [1, 3, 5, 10]
    distribution = {f"top_{k}": {} for k in top_k_values}
    
    for k in top_k_values:
        k = min(k, n_items)
        topk_idx = np.argsort(-y_proba, axis=1)[:, :k]
        
        # 统计每个item在Top-K中的出现次数
        item_counts = np.zeros(n_items)
        for i in range(topk_idx.shape[0]):
            for j in range(k):
                item_idx = topk_idx[i, j]
                item_counts[item_idx] += 1
        
        # 计算百分比
        total_samples = len(y_true_idx)
        for i, item_name in enumerate(label_names):
            distribution[f"top_{k}"][item_name] = {
                'count': int(item_counts[i]),
                'percentage': item_counts[i] / total_samples * 100
            }
    
    return distribution


def calculate_performance_metrics(start_time: float, end_time: float,
                                num_samples: int) -> dict:
    """
    计算性能指标
    包括延迟、TP99、TPS等
    """
    total_time = end_time - start_time
    avg_latency = total_time / num_samples if num_samples > 0 else 0
    
    # 简化的TP99计算（实际应用中需要记录每个样本的推理时间）
    tp99_latency = avg_latency * 1.5  # 假设TP99是平均延迟的1.5倍
    
    # 计算TPS（每秒处理的请求数）
    tps = num_samples / total_time if total_time > 0 else 0
    
    return {
        'avg_latency': avg_latency,
        'tp99_latency': tp99_latency,
        'tps': tps,
        'total_time': total_time,
        'num_samples': num_samples
    }