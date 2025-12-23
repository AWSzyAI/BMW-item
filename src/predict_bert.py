#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一的预测脚本，支持BERT和TF-IDF模型
使用统一的模型管理器和配置管理系统
"""

import os
import json
import argparse
import warnings
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import nullcontext

# 导入我们的管理系统
from model_manager import ModelManager
from config_manager import get_config_manager
from error_handler import get_error_handler, log_info, log_warning, log_error, handle_exception, handle_oom, retry, log_metrics, log_experiment_summary
from utils import (
    ensure_single_label, build_text, hit_at_k, fmt_sec, _flex_read_csv,
    calculate_performance_metrics, coverage_at_k, mean_reciprocal_rank, ndcg_at_k
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


def _str2bool(v) -> bool:
    return str(v).lower() in {"1", "true", "t", "y", "yes"}


def _get_bundle_label_classes(bundle: dict) -> list[str]:
    if bundle.get("labels"):
        return list(bundle["labels"])
    le = bundle.get("label_encoder")
    if le is not None:
        return le.classes_.tolist()
    raise KeyError("模型 bundle 缺少 labels/label_encoder")


@handle_exception
@retry(max_retries=3, delay=2.0)
def main(args):
    """主预测函数"""
    global_start = time.time()
    
    # 初始化配置管理器和错误处理器
    config_manager = get_config_manager()
    config_manager.update_from_args(vars(args))
    
    error_handler = get_error_handler(
        log_file=f"./logs/predict_{time.strftime('%Y%m%d_%H%M%S')}.log",
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
    
    log_info(f"=== {model_type.upper()}模型预测开始 ===")
    
    # 初始化模型管理器
    model_manager = ModelManager()
    
    # 读取数据
    log_info("📖 正在读取测试数据...")
    log_info(f"   数据目录: {data_config.outdir}")
    # 统一使用 eval_file 作为预测输入（与 eval 配置保持一致）
    test_file = getattr(data_config, "test_file", None) or data_config.eval_file
    log_info(f"   测试文件: {test_file}")
    df_te = _flex_read_csv(data_config.outdir, test_file)
    log_info(f"✓ 测试数据读取完成: {df_te.shape}")
    
    # 检查必要列
    for col in ["case_title", "performed_work"]:
        if col not in df_te.columns:
            raise KeyError(f"测试数据缺少列：{col}")
    
    # 构建预测文本
    X_te = build_text(df_te).tolist()
    
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

        label_classes = _get_bundle_label_classes(model_bundle)
        model_manager.setup_label_encoder(label_classes)
        
        # 设置模型和分词器
        model_manager.setup_tokenizer(model_bundle["model_dir"], local_files_only=True)
        model_manager.setup_model(model_bundle["model_dir"], len(label_classes), local_files_only=True)
        
        # 编码数据
        def _tokenize(batch_texts: list[str]):
            return model_manager.tokenizer(
                batch_texts,
                padding=False,
                truncation=True,
                max_length=bert_config.max_length,
            )
        
        log_info("🔤 正在编码测试数据...")
        log_info(f"   最大序列长度: {bert_config.max_length}")
        enc_te = _tokenize(X_te)
        log_info(f"✓ 测试数据编码完成: {len(enc_te['input_ids'])} 样本")
        
        # 创建数据集
        ds_te = UnifiedDataset(encodings=dict(enc_te))
        
        # 创建Trainer
        training_args = TrainingArguments(
            output_dir="./tmp_predict",
            per_device_eval_batch_size=bert_config.eval_batch_size,
            report_to=[],
            fp16=bert_config.fp16 and model_manager.device.type == 'cuda',
        )
        
        data_collator = DataCollatorWithPadding(model_manager.tokenizer)
        
        trainer = Trainer(
            model=model_manager.model,
            args=training_args,
            eval_dataset=ds_te,
            tokenizer=model_manager.tokenizer,
            data_collator=data_collator,
        )
        
        # 预测
        log_info("🔮 开始预测...")
        predict_start = time.time()
        
        with torch.no_grad():
            predictions = trainer.predict(ds_te)
        
        predict_time = time.time() - predict_start
        log_info(f"✓ 预测完成，耗时 {fmt_sec(predict_time)}")
        
        # 处理预测结果
        logits = predictions.predictions
        y_pred = np.argmax(logits, axis=1)
        
        # 计算概率
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        proba = e / e.sum(axis=1, keepdims=True)
        
        # 转换回原始标签
        y_pred_labels = model_manager.label_encoder.inverse_transform(y_pred)
        
        # 创建结果DataFrame
        result_df = pd.DataFrame({
            "case_id": df_te["case_id"] if "case_id" in df_te.columns else range(len(y_pred_labels)),
            "predicted_label": y_pred_labels,
        })
        
        # 添加概率列
        for i, label in enumerate(model_manager.label_encoder.classes_):
            result_df[f"prob_{label}"] = proba[:, i]
        
        # 添加top-k预测
        for k in [1, 3, 5, 10]:
            if k <= len(model_manager.label_encoder.classes_):
                top_k_indices = np.argsort(proba, axis=1)[:, -k:][:, ::-1]
                top_k_labels = model_manager.label_encoder.inverse_transform(top_k_indices.flatten()).reshape(top_k_indices.shape)
                result_df[f"top_{k}_predictions"] = ["|".join(labels) for labels in top_k_labels]
        
    else:
        # 加载TF-IDF模型
        if not _HAS_SKLEARN:
            raise RuntimeError("TF-IDF模型预测需要安装scikit-learn库")
        
        model_bundle = model_manager.load_model_bundle(model_path)
        
        if model_bundle["model_type"] != "tfidf":
            raise ValueError(f"模型类型不匹配，期望: tfidf, 实际: {model_bundle['model_type']}")
        
        # 提取模型组件
        classifier = model_bundle["model"]
        vectorizer = model_bundle["vectorizer"]
        label_encoder = model_bundle["label_encoder"]
        
        # 设置标签编码器
        model_manager.setup_label_encoder(label_encoder.classes_.tolist())
        
        # 预测
        log_info("🔮 开始预测...")
        predict_start = time.time()
        
        # 特征提取
        X_te_vec = vectorizer.transform(X_te)
        
        # 预测
        y_pred = classifier.predict(X_te_vec)
        y_proba = classifier.decision_function(X_te_vec)
        
        # 处理概率
        if y_proba.ndim == 1:
            e = np.exp(y_proba - np.max(y_proba))
            y_proba = e / e.sum(axis=1, keepdims=True)
        else:
            y_proba = np.exp(y_proba - np.max(y_proba, axis=1, keepdims=True))
            y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        
        predict_time = time.time() - predict_start
        log_info(f"✓ 预测完成，耗时 {fmt_sec(predict_time)}")
        
        # 转换回原始标签
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        
        # 创建结果DataFrame
        result_df = pd.DataFrame({
            "case_id": df_te["case_id"] if "case_id" in df_te.columns else range(len(y_pred_labels)),
            "predicted_label": y_pred_labels,
        })
        
        # 添加概率列
        for i, label in enumerate(label_encoder.classes_):
            result_df[f"prob_{label}"] = y_proba[:, i]
        
        # 添加top-k预测
        for k in [1, 3, 5, 10]:
            if k <= len(label_encoder.classes_):
                top_k_indices = np.argsort(y_proba, axis=1)[:, -k:][:, ::-1]
                top_k_labels = label_encoder.inverse_transform(top_k_indices.flatten()).reshape(top_k_indices.shape)
                result_df[f"top_{k}_predictions"] = ["|".join(labels) for labels in top_k_labels]
    
    # 保存结果
    log_info("💾 正在保存预测结果...")
    # 输出文件名优先使用命令行参数，其次兼容 DataConfig 中可能存在的字段，最后回退到默认值
    output_file = getattr(args, "output_file", None) \
        or getattr(data_config, "output_file", None) \
        or "predictions.csv"
    output_path = os.path.join(data_config.outdir, output_file)
    result_df.to_csv(output_path, index=False)
    log_info(f"✓ 预测结果已保存到: {output_path}")
    
    # 创建JSON排序文件，包含预测标签、原始标签和item_title
    log_info("📄 正在生成JSON排序文件...")
    
    # 读取标签映射文件
    label_mapping_path = os.path.join(data_config.outdir, "label_mapping.csv")
    if os.path.exists(label_mapping_path):
        label_mapping_df = pd.read_csv(label_mapping_path)
        # 创建标签到item_title的映射
        label_to_title = dict(zip(label_mapping_df['linked_items'], label_mapping_df['item_title']))
    else:
        log_warning(f"未找到标签映射文件: {label_mapping_path}")
        label_to_title = {}
    
    # 创建JSON格式的预测结果
    predictions_json = []
    for idx, row in result_df.iterrows():
        pred_label = row['predicted_label']
        item_title = label_to_title.get(pred_label, "")
        
        # 获取top-k预测的item_title
        top_k_titles = []
        for k in [1, 3, 5, 10]:
            if f"top_{k}_predictions" in row:
                top_labels = row[f"top_{k}_predictions"].split("|")
                top_titles = [label_to_title.get(label, "") for label in top_labels]
                top_k_titles.append({
                    f"top_{k}_predictions": top_labels,
                    f"top_{k}_titles": top_titles
                })
        
        prediction_item = {
            "case_id": row['case_id'] if 'case_id' in row else idx,
            "predicted_label": pred_label,
            "predicted_item_title": item_title,
            "confidence": float(row[f"prob_{pred_label}"]) if f"prob_{pred_label}" in row else 0.0,
        }
        
        # 添加所有概率信息
        for col in result_df.columns:
            if col.startswith('prob_'):
                label = col.replace('prob_', '')
                prediction_item[f"prob_{label}"] = float(row[col])
        
        # 添加top-k信息
        for top_k_info in top_k_titles:
            prediction_item.update(top_k_info)
        
        predictions_json.append(prediction_item)
    
    # 按置信度排序
    predictions_json.sort(key=lambda x: x['confidence'], reverse=True)
    
    # 保存JSON文件
    json_output_path = os.path.join(data_config.outdir, "predictions_sorted.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions_json, f, ensure_ascii=False, indent=2)
    
    log_info(f"✓ JSON排序文件已保存到: {json_output_path}")
    
    # 计算性能指标
    performance_metrics = calculate_performance_metrics(predict_start, time.time(), len(result_df))
    
    # 计算预测质量指标（如果有概率信息）
    quality_metrics = {}
    if model_type == 'bert':
        proba_matrix = np.array([result_df[f"prob_{label}"].values for label in model_manager.label_encoder.classes_]).T
    else:
        proba_matrix = np.array([result_df[f"prob_{label}"].values for label in label_encoder.classes_]).T
    
    # 计算覆盖率指标
    quality_metrics["coverage@3"] = coverage_at_k(proba_matrix, 3)
    quality_metrics["coverage@5"] = coverage_at_k(proba_matrix, 5)
    quality_metrics["coverage@10"] = coverage_at_k(proba_matrix, 10)
    
    # 计算平均置信度
    quality_metrics["avg_confidence"] = np.mean(np.max(proba_matrix, axis=1))
    quality_metrics["min_confidence"] = np.min(np.max(proba_matrix, axis=1))
    quality_metrics["max_confidence"] = np.max(np.max(proba_matrix, axis=1))
    
    # 计算置信度分布
    confidence_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    confidence_distribution = {}
    for threshold in confidence_thresholds:
        count = np.sum(np.max(proba_matrix, axis=1) >= threshold)
        confidence_distribution[f"confidence_{threshold}+"] = {
            "count": int(count),
            "percentage": float(count / len(result_df) * 100)
        }
    
    # 使用新的日志记录功能记录性能指标
    log_metrics(performance_metrics, "performance")
    
    # 记录预测质量指标
    log_metrics(quality_metrics, "quality")
    
    # 记录置信度分布
    log_metrics(confidence_distribution, "confidence")
    
    # 显示一些统计信息
    log_info("\n📊 预测统计:")
    log_info(f"  总样本数: {len(result_df)}")
    log_info(f"  预测类别数: {len(result_df['predicted_label'].unique())}")
    
    # 显示预测分布
    label_counts = result_df['predicted_label'].value_counts()
    log_info("\n📈 预测分布 (前10个):")
    for label, count in label_counts.head(10).items():
        item_title = label_to_title.get(label, "")
        log_info(f"  {label} ({item_title}): {count}")
    
    # 保存性能指标到文件
    performance_df = pd.DataFrame([{
        **performance_metrics,
        **quality_metrics,
        "model_type": model_type,
        "model_path": model_path,
        "total_samples": len(result_df),
        "unique_predictions": len(result_df['predicted_label'].unique()),
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }])
    
    performance_path = os.path.join(data_config.outdir, "performance_metrics.csv")
    performance_df.to_csv(performance_path, index=False)
    log_info(f"📊 性能指标已保存到: {performance_path}")
    
    # 保存置信度分布到文件
    confidence_df = pd.DataFrame([
        {
            "threshold": threshold,
            "count": info["count"],
            "percentage": info["percentage"]
        }
        for threshold, info in confidence_distribution.items()
    ])
    
    confidence_path = os.path.join(data_config.outdir, "confidence_distribution.csv")
    confidence_df.to_csv(confidence_path, index=False)
    log_info(f"📊 置信度分布已保存到: {confidence_path}")
    
    # 记录实验摘要
    experiment_info = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': model_type,
        'model_path': model_path,
        'data_info': {
            'test_samples': len(result_df),
            'unique_predictions': len(result_df['predicted_label'].unique())
        },
        'key_metrics': {
            'avg_latency': performance_metrics['avg_latency'],
            'tps': performance_metrics['tps'],
            'avg_confidence': quality_metrics['avg_confidence'],
            'coverage@5': quality_metrics['coverage@5']
        },
        'output_files': {
            'csv_results': output_path,
            'json_results': json_output_path,
            'performance_metrics': performance_path,
            'confidence_distribution': confidence_path
        }
    }
    
    log_experiment_summary(experiment_info)
    
    total_time = time.time() - global_start
    log_info(f"\n🎉 {model_type.upper()}预测完成！")
    log_info(f"⏱️  总耗时：{fmt_sec(total_time)}")
    log_info(f"📁 CSV结果文件：{output_path}")
    log_info(f"📄 JSON排序文件：{json_output_path}")
    log_info(f"📊 性能指标文件：{performance_path}")
    log_info(f"📊 置信度分布文件：{confidence_path}")
    
    # 清理内存
    model_manager.clear_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 模型类型选择 - 默认为BERT以保持向后兼容性
    parser.add_argument("--model-type", type=str, default="bert", choices=["bert", "tfidf"], help="模型类型")
    
    # 数据参数
    parser.add_argument("--test-file", type=str, default="test.csv", help="测试集文件名")
    parser.add_argument("--outdir", type=str, default="./output/2025_up_to_month_2", help="数据目录")
    parser.add_argument("--modelsdir", type=str, default="./models", help="模型目录")
    parser.add_argument("--model", type=str, required=True, help="模型文件名")
    parser.add_argument("--output-file", type=str, default="predictions.csv", help="预测结果文件名")
    
    # BERT 参数
    parser.add_argument("--bert-model", type=str, default="./models/google-bert/bert-base-chinese", help="BERT模型名称或路径")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", help="启用混合精度预测")
    parser.add_argument("--allow-online", type=_str2bool, default=False, help="允许在线下载HF模型")
    
    # 系统参数
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    main(args)