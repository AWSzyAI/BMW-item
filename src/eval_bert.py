#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一的评估脚本，支持BERT和TF-IDF模型
使用统一的模型管理器和配置管理系统
"""

import os
import json
import argparse
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from contextlib import nullcontext

# 导入我们的管理系统
from model_manager import ModelManager
from config_manager import get_config_manager
from error_handler import get_error_handler, log_info, log_warning, log_error, handle_exception, handle_oom, retry, log_metrics, log_experiment_summary
from utils import (
    ensure_single_label, build_text, hit_at_k, fmt_sec, _flex_read_csv,
    mean_reciprocal_rank, ndcg_at_k, coverage_at_k, calculate_error_analysis,
    calculate_top_n_distribution, calculate_performance_metrics
)

warnings.filterwarnings("ignore")

# TF-IDF相关导入
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.calibration import CalibratedClassifierCV
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# BERT相关导入
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )
    import torch
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


class UnifiedDataset(torch.utils.data.Dataset):
    """统一数据集类，支持BERT和TF-IDF"""
    def __init__(self, encodings=None, texts=None, labels=None):
        self.encodings = encodings  # BERT编码
        self.texts = texts  # TF-IDF文本
        self.labels = labels

    def __getitem__(self, idx: int):
        if self.encodings is not None:
            # BERT数据
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            if self.labels is not None:
                item["labels"] = torch.tensor(int(self.labels[idx]))
            return item
        else:
            # TF-IDF数据
            item = {"text": self.texts[idx]}
            if self.labels is not None:
                item["label"] = self.labels[idx]
            return item

    def __len__(self) -> int:
        if self.encodings is not None:
            return len(self.encodings["input_ids"])
        else:
            return len(self.texts)


def _compute_metrics(eval_pred, num_labels: int) -> dict:
    """计算评估指标"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1w = f1_score(labels, preds, average="weighted")
    f1m = f1_score(labels, preds, average="macro")
    
    # 计算概率
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    proba = e / e.sum(axis=1, keepdims=True)
    
    # 计算基础指标
    out = {
        "accuracy": float(acc),
        "f1_weighted": float(f1w),
        "f1_macro": float(f1m),
        "hit@1": hit_at_k(labels, proba, 1),
        "hit@3": hit_at_k(labels, proba, 3),
        "hit@5": hit_at_k(labels, proba, 5) if num_labels >= 5 else float("nan"),
        "hit@10": hit_at_k(labels, proba, 10) if num_labels >= 10 else float("nan"),
    }
    
    # 计算高级指标
    out["mrr"] = mean_reciprocal_rank(labels, proba)
    out["ndcg@3"] = ndcg_at_k(labels, proba, 3)
    out["ndcg@5"] = ndcg_at_k(labels, proba, 5)
    out["ndcg@10"] = ndcg_at_k(labels, proba, 10)
    out["coverage@3"] = coverage_at_k(proba, 3)
    out["coverage@5"] = coverage_at_k(proba, 5)
    out["coverage@10"] = coverage_at_k(proba, 10)
    
    return out


def _str2bool(v) -> bool:
    return str(v).lower() in {"1", "true", "t", "y", "yes"}


def _read_split_or_combined(base_dir: str, base_filename: str) -> pd.DataFrame:
    """优先读取 X/Y 分离文件；若不存在则回退到单表 CSV。"""
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
        y = _flex_read_csv(base_dir, os.path.basename(y_name))

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


def _choose_label_column(df: pd.DataFrame) -> str:
    """选择标签列"""
    # 优先级：extend_id > linked_items > item_title
    for col in ["extend_id", "linked_items", "item_title"]:
        if col in df.columns:
            return col
    raise KeyError("未找到可用标签列（extend_id/linked_items/item_title）")


def _export_open_set_predictions(
    df_ev: pd.DataFrame,
    outdir: str,
    y_raw: list[str],
    y_proba_all: np.ndarray,
    label_encoder: LabelEncoder,
    not_in_train_label: str = "__NOT_IN_TRAIN__",
    other_label: str = "__OTHER__",
    unknown_policy: str = "tag-not-in-train",
) -> None:
    """复用 eval.py 的逐样本预测导出逻辑，生成 predictions_eval.csv 风格文件。

    - 使用 BERT 的概率 y_proba_all 和 label_encoder
    - 结构对齐 eval.py 的 "new" 模式导出的 predictions_<base>.csv
    """

    cls = label_encoder.classes_
    cls_set = set(cls)
    n = len(df_ev)
    topk = min(10, len(cls))

    # Top-k 预测
    topk_idx = np.argsort(-y_proba_all, axis=1)[:, :topk]
    topk_labels = [[str(cls[j]) for j in row] for row in topk_idx]
    topk_scores = [[float(y_proba_all[i, j]) for j in topk_idx[i]] for i in range(n)]
    pred_top1 = [labels[0] if labels else "" for labels in topk_labels]

    # 命中率：若 unknown_policy=map-to-other，则映射未知标签；否则未知标签命中置 NaN
    true_labels_orig = [str(t) for t in y_raw]

    use_mapping = (unknown_policy == "map-to-other" and (other_label in cls_set))
    if use_mapping:
        true_labels_mapped = [t if t in cls_set else other_label for t in true_labels_orig]
    else:
        true_labels_mapped = true_labels_orig

    def _hit_at(preds, true, k):
        if (not use_mapping) and (true not in cls_set):
            return np.nan
        return 1 if true in preds[:k] else 0

    rows = []
    for i in range(n):
        preds_i = topk_labels[i]
        scores_i = topk_scores[i]
        true_i = true_labels_orig[i]
        true_i_m = true_labels_mapped[i]
        rows.append({
            "index": i,
            "case_id": df_ev.iloc[i].get("case_id"),
            "true_label": true_i,
            "true_label_mapped": true_i_m if true_i_m != true_i else "",
            "true_in_train": (true_i in cls_set),
            "pred_top1": pred_top1[i],
            "preds_top10": "|".join(preds_i),
            "scores_top10": "|".join(f"{s:.6f}" for s in scores_i),
            "hit@1": _hit_at(preds_i, true_i_m, 1),
            "hit@3": _hit_at(preds_i, true_i_m, 3),
            "hit@5": _hit_at(preds_i, true_i_m, 5),
            "hit@10": _hit_at(preds_i, true_i_m, 10),
        })

    pred_df = pd.DataFrame(rows)
    pred_out = os.path.join(outdir, "predictions_eval.csv")
    pred_df.to_csv(pred_out, index=False, encoding="utf-8-sig")
    log_info(f"逐样本预测已保存：{pred_out}")


@handle_exception
@retry(max_retries=3, delay=2.0)
def main(args):
    """主评估函数"""
    global_start = time.time()
    
    # 初始化配置管理器和错误处理器
    config_manager = get_config_manager()
    config_manager.update_from_args(vars(args))
    
    error_handler = get_error_handler(
        log_file=f"./logs/eval_{time.strftime('%Y%m%d_%H%M%S')}.log",
        log_level=config_manager.get_system_config().log_level
    )
    
    # 验证配置
    if not config_manager.validate_configs():
        raise RuntimeError("配置验证失败")
    
    # 获取配置
    bert_config = config_manager.get_bert_config()
    data_config = config_manager.get_data_config()
    
    # 确定模型类型 - 默认为BERT以保持向后兼容性
    model_type = getattr(args, 'model_type', 'bert')
    
    log_info(f"=== {model_type.upper()}模型评估开始 ===")
    
    # 初始化模型管理器
    model_manager = ModelManager()
    
    # 读取数据
    log_info("📖 正在读取评估数据...")
    log_info(f"   数据目录: {data_config.outdir}")
    log_info(f"   评估文件: {data_config.eval_file}")
    df_ev = _read_split_or_combined(data_config.outdir, data_config.eval_file)
    log_info(f"✓ 评估数据读取完成: {df_ev.shape}")
    
    label_col = _choose_label_column(df_ev)
    log_info(f"✓ 选择标签列: {label_col}")
    
    # 检查必要列
    for col in ["case_title", "performed_work", label_col]:
        if col not in df_ev.columns:
            raise KeyError(f"评估数据缺少列：{col}")
    
    # 清洗标签
    df_ev[label_col] = df_ev[label_col].apply(ensure_single_label).astype(str)
    
    X_ev = build_text(df_ev).tolist()
    y_ev_raw = df_ev[label_col].astype(str).tolist()
    
    # 加载模型
    log_info(f"🔧 正在加载{model_type.upper()}模型...")
    # 从命令行参数获取模型文件名，优先级最高
    model_name = getattr(args, 'model', None)
    if model_name is None:
        # 如果命令行没有提供，则使用配置中的默认值
        model_name = getattr(data_config, 'model', data_config.outmodel)
    
    model_path = os.path.join(data_config.modelsdir, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    if model_type == 'bert':
        # 加载BERT模型
        model_bundle = model_manager.load_model_bundle(model_path)
        
        if model_bundle["model_type"] != "bert":
            raise ValueError(f"模型类型不匹配，期望: bert, 实际: {model_bundle['model_type']}")
        
        # 设置标签编码器
        if "labels" in model_bundle:
            model_manager.setup_label_encoder(model_bundle["labels"])
        elif "label_encoder" in model_bundle:
            # 如果bundle中包含label_encoder对象，直接使用其classes_
            model_manager.setup_label_encoder(model_bundle["label_encoder"].classes_.tolist())
        else:
            raise KeyError("模型bundle中未找到labels或label_encoder键")
        
        # 过滤不在训练标签集的样本
        ev_mask = [lbl in set(model_bundle["labels"]) for lbl in y_ev_raw]
        if not all(ev_mask):
            dropped = int(np.sum(~np.array(ev_mask)))
            log_info(f"[警告] eval 中有 {dropped} 条样本的标签未在训练集中出现（将被过滤）")
        
        X_ev_f = [t for t, m in zip(X_ev, ev_mask) if m]
        y_ev_f = [l for l, m in zip(y_ev_raw, ev_mask) if m]
        y_ev = model_manager.label_encoder.transform(y_ev_f) if len(y_ev_f) > 0 else np.array([])
        
        # 设置模型和分词器
        model_manager.setup_tokenizer(model_bundle["model_dir"], local_files_only=True)
        model_manager.setup_model(model_bundle["model_dir"], len(model_bundle["labels"]), local_files_only=True)
        
        # 编码数据
        def _tokenize(batch_texts: list[str]):
            return model_manager.tokenizer(
                batch_texts,
                padding=False,
                truncation=True,
                max_length=bert_config.max_length,
            )
        
        log_info("🔤 正在编码评估数据...")
        log_info(f"   最大序列长度: {bert_config.max_length}")
        enc_ev = _tokenize(X_ev_f)
        log_info(f"✓ 评估数据编码完成: {len(enc_ev['input_ids'])} 样本")
        
        # 创建数据集
        ds_ev = UnifiedDataset(encodings=dict(enc_ev), labels=np.asarray(y_ev))
        
        # 创建Trainer
        training_args = TrainingArguments(
            output_dir="./tmp_eval",
            per_device_eval_batch_size=bert_config.eval_batch_size,
            report_to=[],
            fp16=bert_config.fp16 and model_manager.device.type == 'cuda',
        )
        
        data_collator = DataCollatorWithPadding(model_manager.tokenizer)
        
        trainer = Trainer(
            model=model_manager.model,
            args=training_args,
            eval_dataset=ds_ev,
            tokenizer=model_manager.tokenizer,
            data_collator=data_collator,
            compute_metrics=(lambda p: _compute_metrics(p, len(model_bundle["labels"]))),
        )
        
        # 评估
        log_info("📊 开始评估...")
        eval_start = time.time()
        
        with torch.no_grad():
            predictions = trainer.evaluate()
        
        eval_time = time.time() - eval_start
        log_info(f"✓ 评估完成，耗时 {fmt_sec(eval_time)}")
        
        # 获取预测结果
        pred_output = trainer.predict(ds_ev)
        logits = pred_output.predictions
        y_pred = np.argmax(logits, axis=1)
        
        # 计算概率
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        proba = e / e.sum(axis=1, keepdims=True)
        
        # 转换回原始标签
        y_pred_labels = model_manager.label_encoder.inverse_transform(y_pred)
        y_true_labels = model_manager.label_encoder.inverse_transform(y_ev)
        
        # 提取评估指标
        eval_metrics = {}
        for k in ["accuracy", "f1_weighted", "f1_macro", "hit@1", "hit@3", "hit@5", "hit@10",
                  "mrr", "ndcg@3", "ndcg@5", "ndcg@10", "coverage@3", "coverage@5", "coverage@10"]:
            if k in predictions:
                eval_metrics[k] = float(predictions[k])
        
        if "eval_loss" in predictions:
            eval_metrics["eval_loss"] = float(predictions["eval_loss"])
        
        # 计算性能指标
        performance_metrics = calculate_performance_metrics(eval_start, time.time(), len(y_ev))
        eval_metrics.update(performance_metrics)

        # 生成与 eval.py 对齐的逐样本预测文件，供开放集评估和 rerank 使用
        _export_open_set_predictions(
            df_ev=df_ev,
            outdir=data_config.outdir,
            y_raw=y_ev_f,
            y_proba_all=proba,
            label_encoder=model_manager.label_encoder,
        )
        
    else:
        # 加载TF-IDF模型
        if not _HAS_SKLEARN:
            raise RuntimeError("TF-IDF模型评估需要安装scikit-learn库")
        
        model_bundle = model_manager.load_model_bundle(model_path)
        
        if model_bundle["model_type"] != "tfidf":
            raise ValueError(f"模型类型不匹配，期望: tfidf, 实际: {model_bundle['model_type']}")
        
        # 提取模型组件
        classifier = model_bundle["model"]
        vectorizer = model_bundle["vectorizer"]
        label_encoder = model_bundle["label_encoder"]
        
        # 设置标签编码器
        model_manager.setup_label_encoder(label_encoder.classes_.tolist())
        
        # 过滤不在训练标签集的样本
        ev_mask = [lbl in set(label_encoder.classes_) for lbl in y_ev_raw]
        if not all(ev_mask):
            dropped = int(np.sum(~np.array(ev_mask)))
            log_info(f"[警告] eval 中有 {dropped} 条样本的标签未在训练集中出现（将被过滤）")
        
        X_ev_f = [t for t, m in zip(X_ev, ev_mask) if m]
        y_ev_f = [l for l, m in zip(y_ev_raw, ev_mask) if m]
        y_ev = label_encoder.transform(y_ev_f) if len(y_ev_f) > 0 else np.array([])
        
        # 评估
        log_info("📊 开始评估TF-IDF模型...")
        eval_start = time.time()
        
        # 特征提取
        X_ev_vec = vectorizer.transform(X_ev_f)
        
        # 预测
        y_pred = classifier.predict(X_ev_vec)
        y_proba = classifier.decision_function(X_ev_vec)
        
        # 处理概率
        if y_proba.ndim == 1:
            e = np.exp(y_proba - np.max(y_proba))
            y_proba = e / e.sum(axis=1, keepdims=True)
        else:
            y_proba = np.exp(y_proba - np.max(y_proba, axis=1, keepdims=True))
            y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        
        eval_time = time.time() - eval_start
        log_info(f"✓ 评估完成，耗时 {fmt_sec(eval_time)}")
        
        # 转换回原始标签
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_true_labels = label_encoder.inverse_transform(y_ev)
        
        # 计算指标
        acc = accuracy_score(y_ev, y_pred)
        f1w = f1_score(y_ev, y_pred, average="weighted")
        f1m = f1_score(y_ev, y_pred, average="macro")
        
        # 计算基础指标
        eval_metrics = {
            "accuracy": float(acc),
            "f1_weighted": float(f1w),
            "f1_macro": float(f1m),
            "hit@1": hit_at_k(y_ev, y_proba, 1),
            "hit@3": hit_at_k(y_ev, y_proba, 3),
            "hit@5": hit_at_k(y_ev, y_proba, 5),
            "hit@10": hit_at_k(y_ev, y_proba, 10),
        }
        
        # 计算高级指标
        eval_metrics["mrr"] = mean_reciprocal_rank(y_ev, y_proba)
        eval_metrics["ndcg@3"] = ndcg_at_k(y_ev, y_proba, 3)
        eval_metrics["ndcg@5"] = ndcg_at_k(y_ev, y_proba, 5)
        eval_metrics["ndcg@10"] = ndcg_at_k(y_ev, y_proba, 10)
        eval_metrics["coverage@3"] = coverage_at_k(y_proba, 3)
        eval_metrics["coverage@5"] = coverage_at_k(y_proba, 5)
        eval_metrics["coverage@10"] = coverage_at_k(y_proba, 10)
        
        # 计算性能指标
        performance_metrics = calculate_performance_metrics(eval_start, time.time(), len(y_ev))
        eval_metrics.update(performance_metrics)

        # 生成与 eval.py 对齐的逐样本预测文件，供开放集评估和 rerank 使用
        _export_open_set_predictions(
            df_ev=df_ev,
            outdir=data_config.outdir,
            y_raw=y_ev_f,
            y_proba_all=y_proba,
            label_encoder=label_encoder,
        )
    
    # 使用新的日志记录功能记录评估指标
    log_metrics(eval_metrics, "quality")
    
    # 记录性能指标（如果存在）
    performance_metrics = {k: v for k, v in eval_metrics.items()
                         if k in ['avg_latency', 'tp99_latency', 'tps', 'total_time', 'num_samples']}
    if performance_metrics:
        log_metrics(performance_metrics, "performance")
    
    # 预先准备 label_classes，供详细报告和实验摘要共用
    if model_type == 'bert':
        label_classes = model_manager.label_encoder.classes_
    else:
        label_classes = label_encoder.classes_

    # 生成详细报告（可选）
    if args.detailed_report:
        log_info("\n📋 生成详细报告...")
        
        # 分类报告
        report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # 混淆矩阵
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        cm_df = pd.DataFrame(cm, index=label_classes, columns=label_classes)
        
        # 错误类型分析
        log_info("🔍 分析错误类型...")
        error_analysis = calculate_error_analysis(
            np.array(y_true_labels),
            np.array(y_pred_labels),
            X_ev_f
        )
        
        # Top-N分布分析
        log_info("📊 分析Top-N分布...")
        top_n_distribution = calculate_top_n_distribution(
            proba if model_type == 'bert' else y_proba,
            np.array(y_ev),
            label_classes.tolist()
        )
        
        # 保存详细报告
        output_dir = data_config.experiment_outdir or data_config.outdir
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存指标
        metrics_path = os.path.join(output_dir, "eval_metrics.csv")
        pd.DataFrame([eval_metrics]).to_csv(metrics_path, index=False)
        log_info(f"📊 评估指标已保存到: {metrics_path}")
        
        # 保存分类报告
        report_path = os.path.join(output_dir, "classification_report.csv")
        report_df.to_csv(report_path)
        log_info(f"📋 分类报告已保存到: {report_path}")
        
        # 保存混淆矩阵
        cm_path = os.path.join(output_dir, "confusion_matrix.csv")
        cm_df.to_csv(cm_path)
        log_info(f"🔢 混淆矩阵已保存到: {cm_path}")
        
        # 保存错误分析
        error_analysis_df = pd.DataFrame([
            {
                'error_type': error_type,
                'count': info['count'],
                'percentage': info['percentage']
            }
            for error_type, info in error_analysis.items()
        ])
        error_analysis_path = os.path.join(output_dir, "error_analysis.csv")
        error_analysis_df.to_csv(error_analysis_path, index=False)
        log_info(f"❌ 错误分析已保存到: {error_analysis_path}")
        
        # 保存Top-N分布
        top_n_dfs = {}
        for top_k, distribution in top_n_distribution.items():
            top_n_df = pd.DataFrame([
                {
                    'item_name': item_name,
                    'count': info['count'],
                    'percentage': info['percentage']
                }
                for item_name, info in distribution.items()
            ]).sort_values('count', ascending=False)
            top_n_dfs[top_k] = top_n_df
            
            top_n_path = os.path.join(output_dir, f"top_{top_k}_distribution.csv")
            top_n_df.to_csv(top_n_path, index=False)
            log_info(f"📈 {top_k}分布已保存到: {top_n_path}")
        
        # 保存预测结果
        results_df = pd.DataFrame({
            "true_label": y_true_labels,
            "predicted_label": y_pred_labels,
            "correct": y_true_labels == y_pred_labels,
        })
        
        # 添加概率信息
        if model_type == 'bert':
            for i, label in enumerate(model_manager.label_encoder.classes_):
                results_df[f"prob_{label}"] = proba[:, i]
        else:
            for i, label in enumerate(label_encoder.classes_):
                results_df[f"prob_{label}"] = y_proba[:, i]
        
        results_path = os.path.join(output_dir, "eval_results.csv")
        results_df.to_csv(results_path, index=False)
        log_info(f"🎯 预测结果已保存到: {results_path}")
        
        # 显示错误样本
        if args.show_errors:
            error_df = results_df[results_df["correct"] == False]
            if len(error_df) > 0:
                log_info(f"\n❌ 错误样本 (前10个):")
                for idx, row in error_df.head(10).iterrows():
                    log_info(f"  真实: {row['true_label']}, 预测: {row['predicted_label']}")
        
        # 使用新的日志记录功能记录错误分析
        log_metrics(error_analysis, "error_analysis")
        
        # 记录Top-5热门项目
        if 'top_5' in top_n_distribution:
            top_5_items = {
                item_name: info for item_name, info in
                sorted(top_n_distribution['top_5'].items(),
                      key=lambda x: x[1]['count'], reverse=True)[:5]
            }
            log_metrics(top_5_items, "distribution")
    
    # 记录实验摘要
    experiment_info = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': model_type,
        'model_path': model_path,
        'data_info': {
            'eval_samples': len(y_ev),
            'num_classes': len(label_classes)
        },
        'key_metrics': {
            'accuracy': eval_metrics.get('accuracy', 0),
            'hit@1': eval_metrics.get('hit@1', 0),
            'mrr': eval_metrics.get('mrr', 0),
            'ndcg@5': eval_metrics.get('ndcg@5', 0)
        },
        'output_files': {
            'metrics': metrics_path if args.detailed_report else "未生成",
            'classification_report': report_path if args.detailed_report else "未生成",
            'confusion_matrix': cm_path if args.detailed_report else "未生成",
            'error_analysis': error_analysis_path if args.detailed_report else "未生成"
        }
    }
    
    log_experiment_summary(experiment_info)
    
    total_time = time.time() - global_start
    log_info(f"\n🎉 {model_type.upper()}评估完成！")
    log_info(f"⏱️  总耗时：{fmt_sec(total_time)}")
    log_info(f"📊 评估样本数：{len(y_ev)}")

    # 使用 Hit@1 作为“准确率”展示，避免缺少 accuracy 键导致报错
    hit1 = eval_metrics.get('hit@1') or eval_metrics.get('hit_1')
    if hit1 is not None:
        log_info(f"🎯 Hit@1（准确率）：{hit1:.4f}")
    else:
        log_info("🎯 未能获取 Hit@1 指标")
    
    # 清理内存
    model_manager.clear_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 模型类型选择 - 默认为BERT以保持向后兼容性
    parser.add_argument("--model-type", type=str, default="bert", choices=["bert", "tfidf"], help="模型类型")
    
    # 数据参数
    parser.add_argument("--eval-file", type=str, default="eval.csv", help="验证集文件名")
    parser.add_argument("--outdir", type=str, default="./output/2025_up_to_month_2", help="数据目录")
    parser.add_argument("--experiment-outdir", type=str, default=None, help="实验输出目录")
    parser.add_argument("--modelsdir", type=str, default="./models", help="模型目录")
    parser.add_argument("--model", type=str, required=True, help="模型文件名")
    
    # BERT 参数
    parser.add_argument("--bert-model", type=str, default="./models/google-bert/bert-base-chinese", help="BERT模型名称或路径")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", help="启用混合精度评估")
    parser.add_argument("--allow-online", type=_str2bool, default=False, help="允许在线下载HF模型")
    
    # 报告参数
    parser.add_argument("--detailed-report", action="store_true", help="生成详细报告")
    parser.add_argument("--show-errors", action="store_true", help="显示错误样本")
    
    # 系统参数
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    main(args)