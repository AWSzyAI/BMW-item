#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
学习率监控和可视化工具
实时监控BERT训练过程中的学习率变化和性能指标
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import re
from pathlib import Path

class LearningRateMonitor:
    """学习率监控器"""
    
    def __init__(self, log_dir="./log", output_dir="./lr_monitor_outputs"):
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def parse_training_logs(self, model_name):
        """解析训练日志文件"""
        log_file = self.log_dir / f"{model_name}_train.txt"
        
        if not log_file.exists():
            print(f"❌ 找不到日志文件: {log_file}")
            return None
            
        epochs = []
        losses = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # 解析格式: "epoch=1/10 loss=0.123456"
                    match = re.search(r'epoch=(\d+)/\d+\s+loss=([\d.]+)', line)
                    if match:
                        epoch = int(match.group(1))
                        loss = float(match.group(2))
                        epochs.append(epoch)
                        losses.append(loss)
            
            return pd.DataFrame({
                'epoch': epochs,
                'loss': losses
            })
            
        except Exception as e:
            print(f"❌ 解析日志文件失败: {e}")
            return None
    
    def parse_eval_metrics(self, model_name):
        """解析评估指标文件"""
        # 尝试从多个可能的目录读取评估指标
        possible_dirs = [
            Path("./output/2025_up_to_month_2"),
            Path("./output"),
            Path(f"./output/lr_search")
        ]
        
        for base_dir in possible_dirs:
            metrics_file = base_dir / "metrics_eval.csv"
            if metrics_file.exists():
                try:
                    metrics_df = pd.read_csv(metrics_file)
                    return metrics_df
                except Exception as e:
                    continue
        
        print(f"❌ 找不到评估指标文件")
        return None
    
    def extract_training_history(self, model_dir):
        """从BERT训练目录提取历史记录"""
        model_path = Path(model_dir)
        
        if not model_path.exists():
            print(f"❌ 模型目录不存在: {model_dir}")
            return None
            
        # 查找trainer_state.json
        state_file = None
        for root, dirs, files in os.walk(model_path):
            if "trainer_state.json" in files:
                state_file = Path(root) / "trainer_state.json"
                break
        
        if not state_file:
            print(f"❌ 找不到trainer_state.json")
            return None
            
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # 提取日志历史
            log_history = state_data.get("log_history", [])
            
            epochs = []
            learning_rates = []
            train_losses = []
            eval_losses = []
            eval_hit3 = []
            eval_accuracy = []
            
            for log_entry in log_history:
                if "epoch" in log_entry:
                    epoch = int(log_entry["epoch"]) + 1
                    
                    # 训练损失
                    if "train_loss" in log_entry:
                        train_losses.append(log_entry["train_loss"])
                        epochs.append(epoch)
                        
                        # 学习率
                        if "learning_rate" in log_entry:
                            learning_rates.append(log_entry["learning_rate"])
                        else:
                            learning_rates.append(np.nan)
                    
                    # 评估指标
                    if "eval_loss" in log_entry:
                        eval_losses.append(log_entry["eval_loss"])
                    if "eval_hit@3" in log_entry:
                        eval_hit3.append(log_entry["eval_hit@3"])
                    if "eval_accuracy" in log_entry:
                        eval_accuracy.append(log_entry["eval_accuracy"])
            
            return pd.DataFrame({
                'epoch': epochs,
                'learning_rate': learning_rates,
                'train_loss': train_losses,
                'eval_loss': eval_losses[:len(epochs)] if eval_losses else [np.nan] * len(epochs),
                'eval_hit3': eval_hit3[:len(epochs)] if eval_hit3 else [np.nan] * len(epochs),
                'eval_accuracy': eval_accuracy[:len(epochs)] if eval_accuracy else [np.nan] * len(epochs)
            })
            
        except Exception as e:
            print(f"❌ 解析训练历史失败: {e}")
            return None
    
    def create_comprehensive_report(self, model_name, model_dir=None):
        """创建综合监控报告"""
        print(f"📊 生成学习率监控报告: {model_name}")
        
        # 收集数据
        training_logs = self.parse_training_logs(model_name)
        eval_metrics = self.parse_eval_metrics(model_name)
        training_history = self.extract_training_history(model_dir) if model_dir else None
        
        if training_history is None and training_logs is None:
            print("❌ 无法获取训练数据")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Learning Rate Monitoring Report - {model_name}', fontsize=16)
        
        # 1. 学习率变化曲线
        if training_history is not None and 'learning_rate' in training_history.columns:
            ax1 = axes[0, 0]
            ax1.plot(training_history['epoch'], training_history['learning_rate'], 
                    marker='o', linewidth=2, markersize=4)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Learning Rate')
            ax1.set_title('Learning Rate Schedule')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
        
        # 2. 损失曲线
        ax2 = axes[0, 1]
        if training_history is not None:
            ax2.plot(training_history['epoch'], training_history['train_loss'], 
                    label='Train Loss', marker='o', linewidth=2)
            if 'eval_loss' in training_history.columns:
                eval_epochs = training_history['epoch'][~training_history['eval_loss'].isna()]
                eval_losses = training_history['eval_loss'][~training_history['eval_loss'].isna()]
                ax2.plot(eval_epochs, eval_losses, 
                        label='Eval Loss', marker='s', linewidth=2)
        elif training_logs is not None:
            ax2.plot(training_logs['epoch'], training_logs['loss'], 
                    label='Train Loss', marker='o', linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training and Evaluation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Hit@3 性能曲线
        ax3 = axes[1, 0]
        if training_history is not None and 'eval_hit3' in training_history.columns:
            hit3_epochs = training_history['epoch'][~training_history['eval_hit3'].isna()]
            hit3_values = training_history['eval_hit3'][~training_history['eval_hit3'].isna()]
            ax3.plot(hit3_epochs, hit3_values, 
                    marker='o', linewidth=2, color='green', markersize=6)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Hit@3')
            ax3.set_title('Hit@3 Performance')
            ax3.grid(True, alpha=0.3)
            
            # 标记最佳点
            if len(hit3_values) > 0:
                best_idx = hit3_values.idxmax()
                best_epoch = hit3_epochs.iloc[best_idx]
                best_hit3 = hit3_values.iloc[best_idx]
                ax3.scatter([best_epoch], [best_hit3], color='red', s=100, zorder=5)
                ax3.annotate(f'Best: {best_hit3:.4f}', 
                           xy=(best_epoch, best_hit3), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 4. 学习率 vs 性能散点图
        ax4 = axes[1, 1]
        if training_history is not None and 'learning_rate' in training_history.columns:
            # 计算每个epoch的平均学习率
            valid_lr = training_history['learning_rate'].dropna()
            valid_hit3 = training_history['eval_hit3'].dropna()
            
            if len(valid_lr) > 0 and len(valid_hit3) > 0:
                min_len = min(len(valid_lr), len(valid_hit3))
                ax4.scatter(valid_lr[:min_len], valid_hit3[:min_len], 
                          alpha=0.6, s=50)
                ax4.set_xlabel('Learning Rate')
                ax4.set_ylabel('Hit@3')
                ax4.set_title('Learning Rate vs Performance')
                ax4.set_xscale('log')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_file = self.output_dir / f"{model_name}_lr_monitor_report.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 监控报告已保存: {output_file}")
        
        # 生成详细数据报告
        self.generate_detailed_report(model_name, training_history, eval_metrics)
    
    def generate_detailed_report(self, model_name, training_history, eval_metrics):
        """生成详细的数据报告"""
        report_file = self.output_dir / f"{model_name}_detailed_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Learning Rate Monitoring Report - {model_name}\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            if training_history is not None:
                f.write("Training History Summary:\n")
                f.write("-" * 30 + "\n")
                
                # 基本统计
                f.write(f"Total Epochs: {len(training_history)}\n")
                
                if 'learning_rate' in training_history.columns:
                    lr_stats = training_history['learning_rate'].describe()
                    f.write(f"Learning Rate Range: {lr_stats['min']:.2e} - {lr_stats['max']:.2e}\n")
                    f.write(f"Mean Learning Rate: {lr_stats['mean']:.2e}\n")
                
                if 'train_loss' in training_history.columns:
                    loss_stats = training_history['train_loss'].describe()
                    f.write(f"Final Train Loss: {training_history['train_loss'].iloc[-1]:.6f}\n")
                    f.write(f"Loss Reduction: {training_history['train_loss'].iloc[0] - training_history['train_loss'].iloc[-1]:.6f}\n")
                
                if 'eval_hit3' in training_history.columns:
                    hit3_data = training_history['eval_hit3'].dropna()
                    if len(hit3_data) > 0:
                        best_hit3 = hit3_data.max()
                        best_epoch = hit3_data.idxmax() + 1
                        f.write(f"Best Hit@3: {best_hit3:.6f} (Epoch {best_epoch})\n")
                        f.write(f"Final Hit@3: {hit3_data.iloc[-1]:.6f}\n")
                        f.write(f"Hit@3 Improvement: {hit3_data.iloc[-1] - hit3_data.iloc[0]:.6f}\n")
                
                f.write("\n")
            
            if eval_metrics is not None:
                f.write("Final Evaluation Metrics:\n")
                f.write("-" * 30 + "\n")
                for col in eval_metrics.columns:
                    f.write(f"{col}: {eval_metrics[col].iloc[0]:.6f}\n")
        
        print(f"📄 详细报告已保存: {report_file}")
    
    def compare_lr_schedules(self, model_dirs, model_names):
        """比较不同学习率调度器的效果"""
        print("📊 比较不同学习率调度器...")
        
        plt.figure(figsize=(15, 10))
        
        # 1. 学习率调度曲线比较
        plt.subplot(2, 2, 1)
        for model_dir, model_name in zip(model_dirs, model_names):
            history = self.extract_training_history(model_dir)
            if history is not None and 'learning_rate' in history.columns:
                plt.plot(history['epoch'], history['learning_rate'], 
                        label=model_name, marker='o', linewidth=2, markersize=4)
        
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedules Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 2. Hit@3 性能比较
        plt.subplot(2, 2, 2)
        best_hit3_values = []
        model_labels = []
        
        for model_dir, model_name in zip(model_dirs, model_names):
            history = self.extract_training_history(model_dir)
            if history is not None and 'eval_hit3' in history.columns:
                hit3_data = history['eval_hit3'].dropna()
                if len(hit3_data) > 0:
                    best_hit3_values.append(hit3_data.max())
                    model_labels.append(model_name)
        
        if best_hit3_values:
            bars = plt.bar(model_labels, best_hit3_values)
            plt.ylabel('Best Hit@3')
            plt.title('Best Hit@3 Comparison')
            plt.xticks(rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars, best_hit3_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.4f}', ha='center', va='bottom')
        
        # 3. 收敛速度比较
        plt.subplot(2, 2, 3)
        convergence_epochs = []
        convergence_labels = []
        
        for model_dir, model_name in zip(model_dirs, model_names):
            history = self.extract_training_history(model_dir)
            if history is not None and 'eval_hit3' in history.columns:
                hit3_data = history['eval_hit3'].dropna()
                if len(hit3_data) > 0:
                    # 定义收敛为达到最佳性能的95%
                    best_hit3 = hit3_data.max()
                    threshold = best_hit3 * 0.95
                    converged_epoch = (hit3_data >= threshold).idxmax() + 1
                    convergence_epochs.append(converged_epoch)
                    convergence_labels.append(model_name)
        
        if convergence_epochs:
            bars = plt.bar(convergence_labels, convergence_epochs)
            plt.ylabel('Epochs to Convergence')
            plt.title('Convergence Speed Comparison')
            plt.xticks(rotation=45)
            
            for bar, value in zip(bars, convergence_epochs):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value}', ha='center', va='bottom')
        
        # 4. 训练稳定性比较
        plt.subplot(2, 2, 4)
        stability_scores = []
        stability_labels = []
        
        for model_dir, model_name in zip(model_dirs, model_names):
            history = self.extract_training_history(model_dir)
            if history is not None and 'eval_hit3' in history.columns:
                hit3_data = history['eval_hit3'].dropna()
                if len(hit3_data) > 1:
                    # 用标准差衡量稳定性（越小越稳定）
                    stability = hit3_data.std()
                    stability_scores.append(stability)
                    stability_labels.append(model_name)
        
        if stability_scores:
            bars = plt.bar(stability_labels, stability_scores)
            plt.ylabel('Hit@3 Standard Deviation')
            plt.title('Training Stability Comparison')
            plt.xticks(rotation=45)
            
            for bar, value in zip(bars, stability_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                        f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存比较图
        output_file = self.output_dir / "lr_schedules_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 调度器比较图已保存: {output_file}")

def main(args):
    monitor = LearningRateMonitor(args.log_dir, args.output_dir)
    
    if args.mode == "single":
        # 单个模型监控
        monitor.create_comprehensive_report(args.model_name, args.model_dir)
    
    elif args.mode == "compare":
        # 多个模型比较
        model_dirs = args.model_dirs.split(',')
        model_names = args.model_names.split(',')
        
        if len(model_dirs) != len(model_names):
            print("❌ 模型目录和名称数量不匹配")
            return
        
        monitor.compare_lr_schedules(model_dirs, model_names)
    
    elif args.mode == "auto":
        # 自动发现并监控所有模型
        print("🔍 自动发现模型...")
        
        # 查找所有训练日志
        log_files = list(Path(args.log_dir).glob("*_train.txt"))
        
        for log_file in log_files:
            model_name = log_file.stem.replace("_train", "")
            print(f"📊 处理模型: {model_name}")
            monitor.create_comprehensive_report(model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学习率监控和可视化工具")
    
    parser.add_argument("--mode", type=str, choices=["single", "compare", "auto"], 
                       default="auto", help="监控模式")
    
    # 单个模型监控参数
    parser.add_argument("--model-name", type=str, help="模型名称")
    parser.add_argument("--model-dir", type=str, help="模型目录路径")
    
    # 比较模式参数
    parser.add_argument("--model-dirs", type=str, help="多个模型目录，逗号分隔")
    parser.add_argument("--model-names", type=str, help="多个模型名称，逗号分隔")
    
    # 通用参数
    parser.add_argument("--log-dir", type=str, default="./log", help="日志目录")
    parser.add_argument("--output-dir", type=str, default="./lr_monitor_outputs", help="输出目录")
    
    args = parser.parse_args()
    main(args)