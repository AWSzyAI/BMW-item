#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BERT学习率自动化搜索脚本
自动测试不同学习率和调度器组合，找到最佳配置
"""

import os
import json
import argparse
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def run_single_experiment(lr, scheduler_type, warmup_ratio, patience, data_dir, other_args):
    """运行单个实验"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"lr_{lr}_sched_{scheduler_type}_warmup_{warmup_ratio}_{timestamp}"
    
    # 构建输出目录 - 所有模型文件都保存在checkpoint目录下
    exp_outdir = f"./output/lr_search/{exp_name}"
    exp_checkpoint_dir = f"./checkpoints/lr_search/{exp_name}"
    
    # 确保目录存在
    os.makedirs(exp_outdir, exist_ok=True)
    os.makedirs(exp_checkpoint_dir, exist_ok=True)
    
    # 确保父目录也存在
    os.makedirs(os.path.dirname(exp_outdir), exist_ok=True)
    os.makedirs(os.path.dirname(exp_checkpoint_dir), exist_ok=True)
    
    # 构建命令
    cmd = [
        "python", "src/train_bert.py",
        "--learning-rate", str(lr),
        "--lr-scheduler-type", scheduler_type,
        "--warmup-ratio", str(warmup_ratio),
        "--early-stopping-patience", str(patience),
        "--outdir", data_dir,  # 使用原始数据目录而不是实验输出目录
        "--experiment-outdir", exp_outdir,  # 实验输出目录用于保存结果
        "--checkpoint-dir", exp_checkpoint_dir,  # checkpoint目录用于保存模型
        "--outmodel", f"{exp_name}.joblib",
        "--num-train-epochs", "10",  # 限制epoch数以加快搜索
    ] + other_args
    
    print(f"\n🚀 开始实验: {exp_name}")
    print(f"📊 学习率: {lr}, 调度器: {scheduler_type}, 预热比例: {warmup_ratio}")
    print(f"🔧 命令: {' '.join(cmd)}")
    
    # 运行实验
    start_time = time.time()
    print(f"⏱️  开始时间: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    print("=" * 60)
    
    try:
        # 使用实时输出而不是 capture_output，以便看到训练进度
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, universal_newlines=True)
        
        # 实时输出日志
        last_log_time = time.time()
        training_started = False
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # 直接输出所有内容，让训练脚本的详细日志显示
                print(output.strip())
                
                # 检测训练是否开始
                if "🚀 开始训练..." in output:
                    training_started = True
                
                # 如果训练已经开始，每10秒输出一次实验进度信息
                if training_started:
                    current_time = time.time()
                    if current_time - last_log_time > 10:
                        elapsed = current_time - start_time
                        print(f"\n⏳ 实验进行中... 已耗时: {elapsed/60:.1f}分钟")
                        last_log_time = current_time
        
        # 等待进程完成
        return_code = process.poll()
        end_time = time.time()
        elapsed = end_time - start_time
        
        if return_code == 0:
            print("=" * 60)
            print(f"✅ 实验完成，总耗时: {elapsed/60:.1f}分钟")
            
            # 解析结果
            metrics_file = os.path.join(exp_outdir, "metrics_eval.csv")
            if os.path.exists(metrics_file):
                metrics_df = pd.read_csv(metrics_file)
                best_hit3 = metrics_df['hit@3'].iloc[0] if 'hit@3' in metrics_df.columns else float('nan')
                best_accuracy = metrics_df['accuracy'].iloc[0] if 'accuracy' in metrics_df.columns else float('nan')
                best_f1_macro = metrics_df['f1_macro'].iloc[0] if 'f1_macro' in metrics_df.columns else float('nan')
            else:
                # 如果metrics_eval.csv不存在，尝试从训练日志中解析评估指标
                print(f"⚠️  未找到评估指标文件 {metrics_file}，尝试从训练日志解析...")
                
                # 尝试从checkpoint目录中查找最佳模型的评估结果
                checkpoint_bert_dir = os.path.join(exp_checkpoint_dir, "bert")
                if os.path.exists(checkpoint_bert_dir):
                    # 查找trainer_state.json文件，其中包含训练历史
                    trainer_state_file = os.path.join(checkpoint_bert_dir, "trainer_state.json")
                    if os.path.exists(trainer_state_file):
                        try:
                            with open(trainer_state_file, 'r') as f:
                                trainer_state = json.load(f)
                            
                            # 从log_history中找到最佳评估结果
                            best_hit3 = best_accuracy = best_f1_macro = float('nan')
                            if 'log_history' in trainer_state:
                                for log_entry in trainer_state['log_history']:
                                    if 'eval_hit@3' in log_entry:
                                        best_hit3 = max(best_hit3, log_entry['eval_hit@3'])
                                    if 'eval_accuracy' in log_entry:
                                        best_accuracy = max(best_accuracy, log_entry['eval_accuracy'])
                                    if 'eval_f1_macro' in log_entry:
                                        best_f1_macro = max(best_f1_macro, log_entry['eval_f1_macro'])
                            
                            print(f"✓ 从训练日志解析得到: hit@3={best_hit3:.4f}, accuracy={best_accuracy:.4f}, f1_macro={best_f1_macro:.4f}")
                        except Exception as e:
                            print(f"⚠️  解析训练日志失败: {e}")
                            best_hit3 = best_accuracy = best_f1_macro = float('nan')
                    else:
                        print(f"⚠️  未找到训练状态文件 {trainer_state_file}")
                        best_hit3 = best_accuracy = best_f1_macro = float('nan')
                else:
                    print(f"⚠️  未找到模型检查点目录 {checkpoint_bert_dir}")
                    best_hit3 = best_accuracy = best_f1_macro = float('nan')
            
            return {
                'exp_name': exp_name,
                'learning_rate': lr,
                'scheduler_type': scheduler_type,
                'warmup_ratio': warmup_ratio,
                'patience': patience,
                'best_hit3': best_hit3,
                'best_accuracy': best_accuracy,
                'best_f1_macro': best_f1_macro,
                'training_time': elapsed,
                'status': 'success',
                'outdir': exp_outdir,
                'checkpoint_dir': exp_checkpoint_dir
            }
        else:
            print("=" * 60)
            print(f"❌ 实验失败，耗时: {elapsed/60:.1f}分钟")
            return {
                'exp_name': exp_name,
                'learning_rate': lr,
                'scheduler_type': scheduler_type,
                'warmup_ratio': warmup_ratio,
                'patience': patience,
                'best_hit3': float('nan'),
                'best_accuracy': float('nan'),
                'best_f1_macro': float('nan'),
                'training_time': elapsed,
                'status': 'failed',
                'error': f"进程返回码: {return_code}",
                'outdir': exp_outdir,
                'checkpoint_dir': exp_checkpoint_dir
            }
    
    except subprocess.TimeoutExpired:
        print("=" * 60)
        print(f"⏰ 实验超时（1小时）")
        return {
            'exp_name': exp_name,
            'learning_rate': lr,
            'scheduler_type': scheduler_type,
            'warmup_ratio': warmup_ratio,
            'patience': patience,
            'best_hit3': float('nan'),
            'best_accuracy': float('nan'),
            'best_f1_macro': float('nan'),
            'training_time': 3600,
            'status': 'timeout',
            'outdir': exp_outdir,
            'checkpoint_dir': exp_checkpoint_dir
        }

def visualize_results(results_df, output_dir):
    """可视化搜索结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 学习率 vs hit@3 热力图
    plt.figure(figsize=(12, 8))
    
    # 创建透视表
    pivot_hit3 = results_df.pivot_table(
        values='best_hit3', 
        index='learning_rate', 
        columns='scheduler_type', 
        aggfunc='max'
    )
    
    sns.heatmap(pivot_hit3, annot=True, fmt='.4f', cmap='YlOrRd', 
                xticklabels=True, yticklabels=True)
    plt.title('Learning Rate vs Scheduler Type (Best Hit@3)')
    plt.xlabel('Scheduler Type')
    plt.ylabel('Learning Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lr_scheduler_heatmap.png'), dpi=300)
    plt.close()
    
    # 2. 学习率 vs hit@3 折线图
    plt.figure(figsize=(12, 8))
    
    for scheduler in results_df['scheduler_type'].unique():
        scheduler_data = results_df[results_df['scheduler_type'] == scheduler]
        plt.plot(scheduler_data['learning_rate'], scheduler_data['best_hit3'], 
                marker='o', label=scheduler, linewidth=2, markersize=6)
    
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Best Hit@3')
    plt.title('Learning Rate vs Best Hit@3 by Scheduler Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lr_hit3_curves.png'), dpi=300)
    plt.close()
    
    # 3. 预热比例影响分析
    if 'warmup_ratio' in results_df.columns:
        plt.figure(figsize=(10, 6))
        warmup_pivot = results_df.groupby('warmup_ratio')['best_hit3'].mean().reset_index()
        plt.bar(warmup_pivot['warmup_ratio'], warmup_pivot['best_hit3'])
        plt.xlabel('Warmup Ratio')
        plt.ylabel('Average Best Hit@3')
        plt.title('Warmup Ratio Impact on Performance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'warmup_impact.png'), dpi=300)
        plt.close()
    
    # 4. 训练时间分析
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['best_hit3'], results_df['training_time'], alpha=0.6)
    plt.xlabel('Best Hit@3')
    plt.ylabel('Training Time (seconds)')
    plt.title('Performance vs Training Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_vs_time.png'), dpi=300)
    plt.close()

def main(args):
    print("🔍 BERT学习率自动化搜索开始")
    
    # 定义搜索空间
    learning_rates = args.learning_rates
    scheduler_types = args.scheduler_types
    warmup_ratios = args.warmup_ratios
    
    print(f"📊 搜索空间:")
    print(f"   学习率: {learning_rates}")
    print(f"   调度器: {scheduler_types}")
    print(f"   预热比例: {warmup_ratios}")
    
    # 生成实验组合
    experiments = []
    for lr in learning_rates:
        for scheduler in scheduler_types:
            for warmup in warmup_ratios:
                experiments.append((lr, scheduler, warmup))
    
    print(f"🧪 总共 {len(experiments)} 个实验组合")
    
    # 其他参数
    other_args = []
    if args.bert_model:
        other_args.extend(["--bert-model", args.bert_model])
    if args.init_hf_dir:
        other_args.extend(["--init-hf-dir", args.init_hf_dir])
    if args.allow_online:
        other_args.append("--allow-online")
    if args.train_batch_size:
        other_args.extend(["--train-batch-size", str(args.train_batch_size)])
    if args.eval_batch_size:
        other_args.extend(["--eval-batch-size", str(args.eval_batch_size)])
    if args.max_length:
        other_args.extend(["--max-length", str(args.max_length)])
    if args.fp16:
        other_args.append("--fp16")
    
    # 运行实验
    results = []
    total_start = time.time()
    experiment_times = []  # 记录每个实验的耗时，用于估算剩余时间
    
    for i, (lr, scheduler, warmup) in enumerate(experiments, 1):
        print(f"\n📍 进度: {i}/{len(experiments)} ({i/len(experiments)*100:.1f}%)")
        
        # 估算剩余时间
        if experiment_times:
            avg_time = sum(experiment_times) / len(experiment_times)
            remaining_experiments = len(experiments) - i
            eta_minutes = avg_time * remaining_experiments / 60
            eta_hours = eta_minutes / 60
            if eta_hours >= 1:
                print(f"⏱️  预计剩余时间: {eta_hours:.1f}小时")
            else:
                print(f"⏱️  预计剩余时间: {eta_minutes:.1f}分钟")
        
        result = run_single_experiment(lr, scheduler, warmup, args.patience, args.data_dir, other_args)
        results.append(result)
        
        # 记录实验耗时
        if result['training_time']:
            experiment_times.append(result['training_time'])
        
        # 保存中间结果
        results_df = pd.DataFrame(results)
        results_df.to_csv("./lr_search_results.csv", index=False)
        
        # 统计状态
        successful_count = sum(1 for r in results if r['status'] == 'success')
        failed_count = sum(1 for r in results if r['status'] == 'failed')
        timeout_count = sum(1 for r in results if r['status'] == 'timeout')
        
        print(f"📊 状态统计: 成功 {successful_count} | 失败 {failed_count} | 超时 {timeout_count}")
        
        # 如果是最佳结果，打印信息
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['best_hit3'])
            print(f"🏆 当前最佳: {best_result['exp_name']} (hit@3={best_result['best_hit3']:.4f})")
    
    total_time = time.time() - total_start
    total_hours = total_time / 3600
    if total_hours >= 1:
        print(f"\n🎉 搜索完成！总耗时: {total_hours:.1f}小时")
    else:
        print(f"\n🎉 搜索完成！总耗时: {total_time/60:.1f}分钟")
    
    # 分析结果
    results_df = pd.DataFrame(results)
    
    # 保存完整结果
    results_df.to_csv("./lr_search_results.csv", index=False)
    print(f"📊 结果已保存到: lr_search_results.csv")
    
    # 成功实验分析
    successful_results = results_df[results_df['status'] == 'success']
    if len(successful_results) > 0:
        print(f"\n✅ 成功实验: {len(successful_results)}/{len(results)}")
        
        # 最佳结果
        best_result = successful_results.loc[successful_results['best_hit3'].idxmax()]
        print(f"\n🏆 最佳配置:")
        print(f"   实验名称: {best_result['exp_name']}")
        print(f"   学习率: {best_result['learning_rate']}")
        print(f"   调度器: {best_result['scheduler_type']}")
        print(f"   预热比例: {best_result['warmup_ratio']}")
        print(f"   最佳hit@3: {best_result['best_hit3']:.4f}")
        print(f"   最佳准确率: {best_result['best_accuracy']:.4f}")
        print(f"   训练时间: {best_result['training_time']:.1f}秒")
        print(f"   模型路径: {best_result['checkpoint_dir']}")
        
        # 生成可视化
        print(f"\n📈 生成可视化...")
        visualize_results(successful_results, "./lr_search_visualizations")
        print(f"📊 可视化结果已保存到: lr_search_visualizations/")
        
        # 保存最佳配置
        best_config = {
            'learning_rate': best_result['learning_rate'],
            'lr_scheduler_type': best_result['scheduler_type'],
            'warmup_ratio': best_result['warmup_ratio'],
            'early_stopping_patience': args.patience,
            'best_hit3': best_result['best_hit3'],
            'best_accuracy': best_result['best_accuracy'],
            'model_path': best_result['checkpoint_dir']
        }
        
        with open("./best_lr_config.json", "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)
        print(f"⚙️ 最佳配置已保存到: best_lr_config.json")
    
    else:
        print(f"\n❌ 所有实验都失败了")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT学习率自动化搜索")
    
    # 搜索空间参数
    parser.add_argument("--learning-rates", type=float, nargs='+',
                       default=[1e-5, 3e-5, 5e-5, 1e-4, 3e-4],
                       help="学习率搜索列表")
    parser.add_argument("--scheduler-types", type=str, nargs='+',
                       choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                       default=["cosine", "linear", "polynomial"],
                       help="调度器类型搜索列表")
    parser.add_argument("--warmup-ratios", type=float, nargs='+',
                       default=[0.0, 0.05, 0.1, 0.2],
                       help="预热比例搜索列表")
    
    # 训练参数
    parser.add_argument("--data-dir", type=str, default="./output/2025_up_to_month_7", help="训练数据目录")
    parser.add_argument("--patience", type=int, default=3, help="早停耐心值")
    parser.add_argument("--bert-model", type=str, default="./models", help="BERT模型路径")
    parser.add_argument("--init-hf-dir", type=str, help="本地HF模型目录")
    parser.add_argument("--allow-online", action="store_true", help="允许在线下载")
    parser.add_argument("--train-batch-size", type=int, default=16, help="训练批次大小")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="评估批次大小")
    parser.add_argument("--max-length", type=int, default=256, help="最大序列长度")
    parser.add_argument("--fp16", action="store_true", help="启用混合精度")
    
    args = parser.parse_args()
    main(args)