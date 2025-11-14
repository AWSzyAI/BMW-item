#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试BERT集成：训练、预测和评估的完整流程
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil
import subprocess
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data():
    """创建测试数据"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"创建测试数据在: {temp_dir}")
    
    # 创建训练数据
    train_data = {
        'case_title': ['发动机异响', '刹车失灵', '空调不制冷', '轮胎漏气', '发动机无法启动'] * 4,
        'performed_work': ['检查发动机', '更换刹车片', '添加制冷剂', '补胎', '检查电瓶'] * 4,
        'linked_items': ['发动机故障', '刹车系统故障', '空调系统故障', '轮胎故障', '电气系统故障'] * 4
    }
    train_df = pd.DataFrame(train_data)
    train_path = os.path.join(temp_dir, 'train.csv')
    train_df.to_csv(train_path, index=False)
    
    # 创建评估数据
    eval_data = {
        'case_title': ['发动机抖动', '刹车异响', '空调有异味', '转向沉重', '变速箱顿挫'],
        'performed_work': ['清洗节气门', '检查刹车盘', '更换空调滤芯', '检查转向系统', '更换变速箱油'],
        'linked_items': ['发动机故障', '刹车系统故障', '空调系统故障', '转向系统故障', '变速箱故障']
    }
    eval_df = pd.DataFrame(eval_data)
    eval_path = os.path.join(temp_dir, 'eval.csv')
    eval_df.to_csv(eval_path, index=False)
    
    return temp_dir, train_path, eval_path

def test_bert_training(temp_dir, train_path, eval_path):
    """测试BERT训练"""
    print("\n=== 测试BERT训练 ===")
    
    # 创建模型输出目录
    models_dir = os.path.join(temp_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 构建训练命令
    cmd = [
        'python', 'src/train_bert.py',
        '--train-file', train_path,
        '--eval-file', eval_path,
        '--outdir', temp_dir,
        '--modelsdir', models_dir,
        '--outmodel', 'test_bert_model.joblib',
        '--bert-model', './models',  # 使用本地下载的模型
        '--allow-online', 'False',
        '--num-train-epochs', '1.0',  # 只训练1个epoch用于测试
        '--train-batch-size', '4',
        '--eval-batch-size', '8',
        '--max-length', '128',
        '--skip-train-stats', 'True',
        '--post-train-stats-batch-size', '4',
        '--stats-on-cpu', 'True'
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            print("✓ BERT训练成功")
            print(result.stdout)
            return os.path.join(models_dir, 'test_bert_model.joblib')
        else:
            print("✗ BERT训练失败")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return None
    except Exception as e:
        print(f"✗ 训练过程中出现异常: {e}")
        return None

def test_bert_prediction(model_path, temp_dir):
    """测试BERT预测"""
    print("\n=== 测试BERT预测 ===")
    
    if not model_path or not os.path.exists(model_path):
        print("✗ 模型文件不存在，跳过预测测试")
        return False
    
    # 构建预测命令
    cmd = [
        'python', 'src/predict_bert.py',
        '--modelsdir', os.path.dirname(model_path),
        '--model', os.path.basename(model_path),
        '--outdir', temp_dir,
        '--infile', 'eval.csv'
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            print("✓ BERT预测成功")
            print(result.stdout)
            return True
        else:
            print("✗ BERT预测失败")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"✗ 预测过程中出现异常: {e}")
        return False

def test_bert_evaluation(model_path, temp_dir):
    """测试BERT评估"""
    print("\n=== 测试BERT评估 ===")
    
    if not model_path or not os.path.exists(model_path):
        print("✗ 模型文件不存在，跳过评估测试")
        return False
    
    # 构建评估命令
    cmd = [
        'python', 'src/eval_bert.py',
        '--modeldir', os.path.dirname(model_path),
        '--model', os.path.basename(model_path),
        '--outdir', temp_dir,
        '--path', 'eval.csv',
        '--mode', 'new'
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            print("✓ BERT评估成功")
            print(result.stdout)
            return True
        else:
            print("✗ BERT评估失败")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"✗ 评估过程中出现异常: {e}")
        return False

def main():
    """主测试函数"""
    print("=== BERT集成测试开始 ===")
    
    # 检查本地模型是否存在
    if not os.path.exists('./models'):
        print("✗ 本地模型目录不存在，请先运行:")
        print("modelscope download --model 'google-bert/bert-base-chinese' --local_dir './models'")
        return False
    
    # 创建测试数据
    temp_dir, train_path, eval_path = create_test_data()
    
    try:
        # 测试训练
        model_path = test_bert_training(temp_dir, train_path, eval_path)
        
        # 测试预测
        predict_success = test_bert_prediction(model_path, temp_dir)
        
        # 测试评估
        eval_success = test_bert_evaluation(model_path, temp_dir)
        
        # 总结结果
        print("\n=== 测试结果总结 ===")
        print(f"训练: {'✓ 成功' if model_path else '✗ 失败'}")
        print(f"预测: {'✓ 成功' if predict_success else '✗ 失败'}")
        print(f"评估: {'✓ 成功' if eval_success else '✗ 失败'}")
        
        overall_success = bool(model_path) and predict_success and eval_success
        print(f"整体: {'✓ 所有测试通过' if overall_success else '✗ 部分测试失败'}")
        
        return overall_success
        
    finally:
        # 清理临时文件
        try:
            shutil.rmtree(temp_dir)
            print(f"\n已清理临时目录: {temp_dir}")
        except Exception as e:
            print(f"清理临时目录失败: {e}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)