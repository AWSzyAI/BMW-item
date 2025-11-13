#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用现有的BERT.py代码调用本地下载的BERT模型
"""

import os
import sys
import argparse

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """
    使用本地BERT模型进行训练的示例
    """
    print("=== 使用本地BERT模型训练示例 ===\n")
    
    # 创建模拟的命令行参数
    args = argparse.Namespace()
    
    # 基本参数
    args.train_file = "train.csv"  # 训练数据文件
    args.eval_file = "eval.csv"    # 评估数据文件
    args.outdir = "./output"       # 输出目录
    args.modelsdir = "./models"    # 模型目录
    args.outmodel = "bert_model.joblib"  # 输出模型文件名
    
    # BERT特定参数 - 关键是这里指定本地模型路径
    args.bert_model = "./models"   # 指向你的本地模型目录
    args.init_hf_dir = "./models"  # 也可以使用这个参数指定本地模型目录
    args.allow_online = False      # 禁用在线下载，强制使用本地模型
    
    # 训练参数
    args.num_train_epochs = 3.0
    args.train_batch_size = 16
    args.eval_batch_size = 32
    args.learning_rate = 5e-5
    args.weight_decay = 0.01
    args.grad_accum_steps = 1
    args.max_length = 256
    args.fp16 = False
    
    # 其他参数
    args.early_stopping_patience = 3
    args.ooc_tau_percentile = 5.0
    args.ooc_temperature = 20.0
    args.resample_method = "none"
    args.save_hf_dir = None
    
    print("参数设置:")
    print(f"  本地模型路径: {args.bert_model}")
    print(f"  训练数据: {args.train_file}")
    print(f"  评估数据: {args.eval_file}")
    print(f"  输出模型: {args.outmodel}")
    print(f"  允许在线下载: {args.allow_online}")
    
    # 检查本地模型目录是否存在
    if not os.path.isdir(args.bert_model):
        print(f"\n错误: 本地模型目录不存在: {args.bert_model}")
        print("请确保你已经使用以下命令下载了模型:")
        print("modelscope download --model 'google-bert/bert-base-chinese' --local_dir './models'")
        return
    
    # 检查必要的模型文件
    required_files = ["config.json", "tokenizer.json", "vocab.txt"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(args.bert_model, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"\n错误: 本地模型目录缺少必要文件: {missing_files}")
        print("请确保模型下载完整。")
        return
    
    print("\n✓ 本地模型检查通过")
    
    # 导入并运行BERT训练
    try:
        from BERT import main as bert_main
        print("\n✓ 成功导入BERT.py")
        
        print("\n开始使用本地BERT模型进行训练...")
        print("注意: 如果没有训练数据，这将会报错。")
        print("请确保在outdir目录下有train.csv和eval.csv文件。")
        
        # 实际运行训练
        bert_main(args)
        
    except ImportError as e:
        print(f"\n错误: 无法导入BERT.py: {e}")
        print("请确保src/BERT.py文件存在且可访问。")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        print("这通常是因为缺少训练数据文件。")

def create_sample_data():
    """
    创建示例训练数据（可选）
    """
    import pandas as pd
    
    # 创建输出目录
    os.makedirs("./output", exist_ok=True)
    
    # 创建示例训练数据
    train_data = {
        'case_title': ['发动机异响', '刹车失灵', '空调不制冷', '轮胎漏气', '发动机无法启动'] * 4,
        'performed_work': ['检查发动机', '更换刹车片', '添加制冷剂', '补胎', '检查电瓶'] * 4,
        'linked_items': ['发动机故障', '刹车系统故障', '空调系统故障', '轮胎故障', '电气系统故障'] * 4
    }
    train_df = pd.DataFrame(train_data)
    train_df.to_csv("./output/train.csv", index=False)
    
    # 创建示例评估数据
    eval_data = {
        'case_title': ['发动机抖动', '刹车异响', '空调有异味'],
        'performed_work': ['清洗节气门', '检查刹车盘', '更换空调滤芯'],
        'linked_items': ['发动机故障', '刹车系统故障', '空调系统故障']
    }
    eval_df = pd.DataFrame(eval_data)
    eval_df.to_csv("./output/eval.csv", index=False)
    
    print("✓ 已创建示例训练数据在 ./output/ 目录")

if __name__ == "__main__":
    # 如果需要创建示例数据，取消下面的注释
    # create_sample_data()
    
    main()