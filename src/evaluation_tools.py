#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BMW Case-Item 项目评估工具
支持完整的评估指标体系，包括模型质量、结构性分析和工程性能
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import time
import warnings

warnings.filterwarnings("ignore")

# 导入我们的管理系统
from model_manager import ModelManager
from config_manager import get_config_manager
from error_handler import get_error_handler, log_info, log_warning, log_error
from utils import ensure_single_label, build_text, _flex_read_csv


def calculate_hit_at_k(predictions: List[List[str]], ground_truth: List[str], k: int) -> float:
    """计算 Hit@K"""
    if len(predictions) != len(ground_truth):
        raise ValueError("预测结果和真实标签数量不匹配")
    
    hits = 0
    for pred, true in zip(predictions, ground_truth):
        if true in pred[:k]:
            hits += 1
    return hits / len(predictions)


def calculate_mrr(predictions: List[List[str]], ground_truth: List[str]) -> float:
    """计算 MRR（平均倒数排名）"""
    if len(predictions) != len(ground_truth):
        raise ValueError("预测结果和真实标签数量不匹配")
    
    reciprocal_ranks = []
    for pred, true in zip(predictions, ground_truth):
        try:
            rank = pred.index(true) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_ndcg_at_k(predictions: List[List[str]], ground_truth: List[str], 
                    relevance_scores: Optional[Dict[str, float]], k: int) -> float:
    """计算 NDCG@K"""
    if len(predictions) != len(ground_truth):
        raise ValueError("预测结果和真实标签数量不匹配")
    
    # 如果没有提供相关性分数，假设所有正确答案相关性为1
    if relevance_scores is None:
        relevance_scores = {item: 1.0 for item in set(ground_truth)}
    
    dcg = 0.0
    idcg = 0.0
    
    # 计算 IDCG（理想DCG）
    sorted_relevance = sorted([relevance_scores.get(item, 0.0) for item in set(ground_truth)], reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted_relevance[:k]))
    
    if idcg == 0:
        return 0.0
    
    # 计算 DCG
    for pred, true in zip(predictions, ground_truth):
        for i, item in enumerate(pred[:k]):
            if item == true:
                dcg += relevance_scores.get(item, 0.0) / np.log2(i + 2)
                break
    
    return dcg / idcg


def calculate_coverage(all_items: List[str], predicted_items: List[List[str]], top_k: int) -> float:
    """计算覆盖率"""
    predicted_set = set()
    for pred in predicted_items:
        predicted_set.update(pred[:top_k])
    
    return len(predicted_set & set(all_items)) / len(set(all_items))


def analyze_corrected_cases(base_predictions: List[List[str]], rerank_predictions: List[List[str]], 
                           ground_truth: List[str], k: int = 3) -> Dict:
    """分析纠错案例"""
    corrected = 0
    misranked = 0
    corrected_cases = []
    misranked_cases = []
    
    for i, (base_pred, rerank_pred, true) in enumerate(zip(base_predictions, rerank_predictions, ground_truth)):
        base_hit = true in base_pred[:k]
        rerank_hit = true in rerank_pred[:k]
        
        if not base_hit and rerank_hit:
            corrected += 1
            corrected_cases.append({
                'index': i,
                'true_label': true,
                'base_prediction': base_pred[:k],
                'rerank_prediction': rerank_pred[:k],
                'base_rank': base_pred.index(true) + 1 if true in base_pred else None,
                'rerank_rank': rerank_pred.index(true) + 1 if true in rerank_pred else None
            })
        elif base_hit and not rerank_hit:
            misranked += 1
            misranked_cases.append({
                'index': i,
                'true_label': true,
                'base_prediction': base_pred[:k],
                'rerank_prediction': rerank_pred[:k],
                'base_rank': base_pred.index(true) + 1,
                'rerank_rank': rerank_pred.index(true) + 1
            })
    
    return {
        'corrected': corrected,
        'misranked': misranked,
        'corrected_rate': corrected / len(ground_truth),
        'misranked_rate': misranked / len(ground_truth),
        'corrected_cases': corrected_cases,
        'misranked_cases': misranked_cases
    }


def analyze_error_types(predictions: List[str], ground_truth: List[str], 
                   texts: List[str], k: int = 3) -> Dict:
    """分析错误类型"""
    error_types = {
        'text_noise': 0,      # 文本噪声（拼写错误、异常字段）
        'vague_description': 0,  # item 描述模糊
        'too_short': 0,        # case 过短
        'too_long': 0,         # case 过长
        'template_mismatch': 0, # 模板化 case → 某类 item 被固定错误预测
        'ambiguous': 0,         # 多义 case → 混淆多类
        'other': 0              # 其他错误
    }
    
    error_details = []
    
    for i, (pred, true, text) in enumerate(zip(predictions, ground_truth, texts)):
        if true not in pred[:k]:
            # 简单的错误类型分析（可以进一步优化）
            text_len = len(text)
            
            if text_len < 10:
                error_types['too_short'] += 1
                error_type = 'too_short'
            elif text_len > 500:
                error_types['too_long'] += 1
                error_type = 'too_long'
            elif any(char.isdigit() for char in text) and any(char.isalpha() for char in text):
                error_types['text_noise'] += 1
                error_type = 'text_noise'
            else:
                error_types['other'] += 1
                error_type = 'other'
            
            error_details.append({
                'index': i,
                'true_label': true,
                'predicted_label': pred[0],
                'text': text[:100] + '...' if len(text) > 100 else text,
                'error_type': error_type
            })
    
    total_errors = sum(error_types.values())
    error_percentages = {k: v/total_errors for k, v in error_types.items() if total_errors > 0}
    
    return {
        'error_counts': error_types,
        'error_percentages': error_percentages,
        'total_errors': total_errors,
        'error_details': error_details
    }


def plot_confusion_matrix(cm, class_names, output_path: str):
    """绘制混淆矩阵"""
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_comparison(metrics_df, output_path: str):
    """绘制指标对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Hit@K 对比
    metrics_df[['Hit@1', 'Hit@3', 'Hit@10']].plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Hit@K Comparison')
    axes[0, 0].set_ylabel('Hit Rate')
    
    # MRR 和 NDCG 对比
    metrics_df[['MRR', 'NDCG@3', 'NDCG@10']].plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Ranking Quality Comparison')
    axes[0, 1].set_ylabel('Score')
    
    # Coverage 和其他指标
    other_cols = [col for col in metrics_df.columns if col not in ['Hit@1', 'Hit@3', 'Hit@10', 'MRR', 'NDCG@3', 'NDCG@10']]
    if other_cols:
        metrics_df[other_cols].plot(kind='bar', ax=axes[0, 2])
        axes[0, 2].set_title('Other Metrics')
        axes[0, 2].set_ylabel('Value')
    
    # 工程性能指标（如果有）
    if 'Latency' in metrics_df.columns:
        metrics_df[['Latency']].plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Latency Comparison')
        axes[1, 0].set_ylabel('Latency (ms)')
    
    if 'TP99' in metrics_df.columns:
        metrics_df[['TP99']].plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('TP99 Latency')
        axes[1, 1].set_ylabel('TP99 (ms)')
    
    # TPS（如果有）
    if 'TPS' in metrics_df.columns:
        metrics_df[['TPS']].plot(kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('Throughput')
        axes[1, 2].set_ylabel('TPS')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model(predictions_file: str, ground_truth_file: str, 
                model_name: str, output_dir: str) -> Dict:
    """评估单个模型"""
    log_info(f"🔍 开始评估模型: {model_name}")
    
    # 读取预测结果
    if predictions_file.endswith('.json'):
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions_data = json.load(f)
        
        predictions = [item['top_10_predictions'] if 'top_10_predictions' in item else [item['predicted_label']] 
                     for item in predictions_data]
        confidences = [item['confidence'] for item in predictions_data]
    else:
        pred_df = pd.read_csv(predictions_file)
        predictions = [pred_df['top_10_predictions'].iloc[i] if 'top_10_predictions' in pred_df.columns 
                     else [pred_df['predicted_label'].iloc[i]] for i in range(len(pred_df))]
        confidences = pred_df['confidence'].tolist() if 'confidence' in pred_df.columns else [1.0] * len(pred_df)
    
    # 读取真实标签
    truth_df = pd.read_csv(ground_truth_file)
    ground_truth = truth_df['linked_items'].astype(str).tolist()
    
    # 读取文本（用于错误分析）
    texts = []
    if 'case_title' in truth_df.columns and 'performed_work' in truth_df.columns:
        texts = (truth_df['case_title'] + ' ' + truth_df['performed_work']).tolist()
    
    # 确保数据量一致
    min_len = min(len(predictions), len(ground_truth))
    predictions = predictions[:min_len]
    ground_truth = ground_truth[:min_len]
    texts = texts[:min_len]
    
    # 计算核心指标
    hit_1 = calculate_hit_at_k(predictions, ground_truth, 1)
    hit_3 = calculate_hit_at_k(predictions, ground_truth, 3)
    hit_10 = calculate_hit_at_k(predictions, ground_truth, 10)
    mrr = calculate_mrr(predictions, ground_truth)
    ndcg_3 = calculate_ndcg_at_k(predictions, ground_truth, None, 3)
    ndcg_10 = calculate_ndcg_at_k(predictions, ground_truth, None, 10)
    
    # 计算覆盖率
    all_items = list(set(ground_truth))
    coverage_3 = calculate_coverage(all_items, predictions, 3)
    coverage_10 = calculate_coverage(all_items, predictions, 10)
    
    # 分析错误类型
    error_analysis = analyze_error_types(predictions, ground_truth, texts, 3)
    
    # 生成混淆矩阵
    le = LabelEncoder()
    all_labels = list(set(ground_truth + [pred[0] for pred in predictions]))
    le.fit(all_labels)
    
    y_true_encoded = le.transform(ground_truth)
    y_pred_encoded = le.transform([pred[0] for pred in predictions])
    
    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    
    results = {
        'model_name': model_name,
        'total_samples': len(predictions),
        'hit_1': hit_1,
        'hit_3': hit_3,
        'hit_10': hit_10,
        'mrr': mrr,
        'ndcg_3': ndcg_3,
        'ndcg_10': ndcg_10,
        'coverage_3': coverage_3,
        'coverage_10': coverage_10,
        'error_analysis': error_analysis,
        'confusion_matrix': cm.tolist(),
        'class_names': le.classes_.tolist()
    }
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数值结果
    results_df = pd.DataFrame([{
        'Model': model_name,
        'Hit@1': hit_1,
        'Hit@3': hit_3,
        'Hit@10': hit_10,
        'MRR': mrr,
        'NDCG@3': ndcg_3,
        'NDCG@10': ndcg_10,
        'Coverage@3': coverage_3,
        'Coverage@10': coverage_10,
        'Total Errors': error_analysis['total_errors'],
        'Error Rate': error_analysis['total_errors'] / len(predictions)
    }])
    
    results_path = os.path.join(output_dir, f"{model_name}_metrics.csv")
    results_df.to_csv(results_path, index=False)
    log_info(f"✓ 评估指标已保存到: {results_path}")
    
    # 保存错误详情
    if error_analysis['error_details']:
        error_details_df = pd.DataFrame(error_analysis['error_details'])
        error_details_path = os.path.join(output_dir, f"{model_name}_error_details.csv")
        error_details_df.to_csv(error_details_path, index=False)
        log_info(f"✓ 错误详情已保存到: {error_details_path}")
    
    # 保存混淆矩阵
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(cm, le.classes_, cm_path)
    log_info(f"✓ 混淆矩阵已保存到: {cm_path}")
    
    return results


def compare_models(base_predictions: str, rerank_predictions: str, 
                 ground_truth: str, output_dir: str) -> Dict:
    """比较基础模型和rerank模型"""
    log_info("🔄 开始比较基础模型和rerank模型")
    
    # 读取预测结果
    with open(base_predictions, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    
    with open(rerank_predictions, 'r', encoding='utf-8') as f:
        rerank_data = json.load(f)
    
    # 读取真实标签
    truth_df = pd.read_csv(ground_truth)
    ground_truth = truth_df['linked_items'].astype(str).tolist()
    
    # 提取预测列表
    base_preds = [item['top_10_predictions'] if 'top_10_predictions' in item else [item['predicted_label']] 
                  for item in base_data]
    rerank_preds = [item['top_10_predictions'] if 'top_10_predictions' in item else [item['predicted_label']] 
                    for item in rerank_data]
    
    # 确保数据量一致
    min_len = min(len(base_preds), len(rerank_preds), len(ground_truth))
    base_preds = base_preds[:min_len]
    rerank_preds = rerank_preds[:min_len]
    ground_truth = ground_truth[:min_len]
    
    # 分析纠错案例
    corrected_analysis = analyze_corrected_cases(base_preds, rerank_preds, ground_truth, 3)
    
    # 计算各自指标
    base_hit_3 = calculate_hit_at_k(base_preds, ground_truth, 3)
    rerank_hit_3 = calculate_hit_at_k(rerank_preds, ground_truth, 3)
    
    base_mrr = calculate_mrr(base_preds, ground_truth)
    rerank_mrr = calculate_mrr(rerank_preds, ground_truth)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_results = {
        'base_hit_3': base_hit_3,
        'rerank_hit_3': rerank_hit_3,
        'delta_hit_3': rerank_hit_3 - base_hit_3,
        'base_mrr': base_mrr,
        'rerank_mrr': rerank_mrr,
        'delta_mrr': rerank_mrr - base_mrr,
        'corrected_cases': corrected_analysis['corrected'],
        'misranked_cases': corrected_analysis['misranked'],
        'corrected_rate': corrected_analysis['corrected_rate'],
        'misranked_rate': corrected_analysis['misranked_rate']
    }
    
    # 保存比较结果
    comparison_df = pd.DataFrame([comparison_results])
    comparison_path = os.path.join(output_dir, "rerank_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    log_info(f"✓ Rerank比较结果已保存到: {comparison_path}")
    
    # 保存纠错案例
    if corrected_analysis['corrected_cases']:
        corrected_df = pd.DataFrame(corrected_analysis['corrected_cases'])
        corrected_path = os.path.join(output_dir, "corrected_cases.csv")
        corrected_df.to_csv(corrected_path, index=False)
        log_info(f"✓ 纠错案例已保存到: {corrected_path}")
    
    if corrected_analysis['misranked_cases']:
        misranked_df = pd.DataFrame(corrected_analysis['misranked_cases'])
        misranked_path = os.path.join(output_dir, "misranked_cases.csv")
        misranked_df.to_csv(misranked_path, index=False)
        log_info(f"✓ 错排案例已保存到: {misranked_path}")
    
    return comparison_results


def generate_evaluation_report(tfidf_results: Optional[str], bert_results: Optional[str], 
                          rerank_results: Optional[str], output_dir: str):
    """生成完整的评估报告"""
    log_info("📊 生成综合评估报告")
    
    all_results = []
    
    if tfidf_results and os.path.exists(tfidf_results):
        tfidf_df = pd.read_csv(tfidf_results)
        all_results.append(tfidf_df)
    
    if bert_results and os.path.exists(bert_results):
        bert_df = pd.read_csv(bert_results)
        all_results.append(bert_df)
    
    if rerank_results and os.path.exists(rerank_results):
        rerank_df = pd.read_csv(rerank_results)
        all_results.append(rerank_df)
    
    if not all_results:
        log_warning("没有找到评估结果文件")
        return
    
    # 合并结果
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 生成对比图
    plot_path = os.path.join(output_dir, "metrics_comparison.png")
    plot_metric_comparison(combined_df, plot_path)
    log_info(f"✓ 指标对比图已保存到: {plot_path}")
    
    # 生成HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BMW Case-Item 评估报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .metric-table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            .metric-table th {{ background-color: #f2f2f2; }}
            .improvement {{ color: green; }}
            .degradation {{ color: red; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📘 BMW Case-Item 项目评估报告</h1>
            <p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📊 核心指标对比</h2>
            <table class="metric-table">
                <tr>
                    <th>模型</th>
                    <th>Hit@1</th>
                    <th>Hit@3</th>
                    <th>Hit@10</th>
                    <th>MRR</th>
                    <th>NDCG@3</th>
                    <th>NDCG@10</th>
                    <th>Coverage@3</th>
                    <th>Coverage@10</th>
                </tr>
    """
    
    for _, row in combined_df.iterrows():
        html_content += f"""
                <tr>
                    <td>{row['Model']}</td>
                    <td>{row['Hit@1']:.4f}</td>
                    <td>{row['Hit@3']:.4f}</td>
                    <td>{row['Hit@10']:.4f}</td>
                    <td>{row['MRR']:.4f}</td>
                    <td>{row['NDCG@3']:.4f}</td>
                    <td>{row['NDCG@10']:.4f}</td>
                    <td>{row['Coverage@3']:.4f}</td>
                    <td>{row['Coverage@10']:.4f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>📈 指标对比图</h2>
            <img src="metrics_comparison.png" alt="指标对比图" style="max-width: 100%;">
        </div>
        
        <div class="section">
            <h2>💡 改进建议</h2>
            <ul>
                <li>重点关注 Hit@3 指标，这是业务核心指标</li>
                <li>分析高频错误模式，针对性改进数据预处理</li>
                <li>考虑 Rerank 策略优化，减少 Mis-rank Cases</li>
                <li>监控 Coverage 指标，确保冷门 Item 有足够曝光</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    report_path = os.path.join(output_dir, "evaluation_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    log_info(f"✓ 评估报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="BMW Case-Item 项目评估工具")
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 评估单个模型
    eval_parser = subparsers.add_parser('evaluate', help='评估单个模型')
    eval_parser.add_argument('--predictions', required=True, help='预测结果文件（CSV或JSON）')
    eval_parser.add_argument('--ground-truth', required=True, help='真实标签文件（CSV）')
    eval_parser.add_argument('--model-name', required=True, help='模型名称')
    eval_parser.add_argument('--output-dir', required=True, help='输出目录')
    
    # 比较模型
    compare_parser = subparsers.add_parser('compare', help='比较基础模型和rerank模型')
    compare_parser.add_argument('--base-predictions', required=True, help='基础模型预测结果（JSON）')
    compare_parser.add_argument('--rerank-predictions', required=True, help='Rerank模型预测结果（JSON）')
    compare_parser.add_argument('--ground-truth', required=True, help='真实标签文件（CSV）')
    compare_parser.add_argument('--output-dir', required=True, help='输出目录')
    
    # 生成报告
    report_parser = subparsers.add_parser('report', help='生成综合评估报告')
    report_parser.add_argument('--tfidf-results', help='TF-IDF评估结果（CSV）')
    report_parser.add_argument('--bert-results', help='BERT评估结果（CSV）')
    report_parser.add_argument('--rerank-results', help='Rerank评估结果（CSV）')
    report_parser.add_argument('--output-dir', required=True, help='输出目录')
    
    args = parser.parse_args()
    
    # 初始化日志
    error_handler = get_error_handler(
        log_file=f"./logs/evaluation_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    if args.command == 'evaluate':
        results = evaluate_model(
            args.predictions, args.ground_truth, args.model_name, args.output_dir
        )
        log_info(f"✓ 模型 {args.model_name} 评估完成")
        
    elif args.command == 'compare':
        results = compare_models(
            args.base_predictions, args.rerank_predictions, args.ground_truth, args.output_dir
        )
        log_info("✓ 模型比较完成")
        
    elif args.command == 'report':
        generate_evaluation_report(
            args.tfidf_results, args.bert_results, args.rerank_results, args.output_dir
        )
        log_info("✓ 评估报告生成完成")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()