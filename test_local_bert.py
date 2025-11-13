#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
演示如何调用本地下载的BERT模型进行文本分类
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def load_local_bert_model(model_path):
    """
    加载本地BERT模型和分词器
    
    Args:
        model_path: 本地模型目录路径
        
    Returns:
        tokenizer, model: 分词器和模型
    """
    print(f"正在加载本地模型: {model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print("✓ 分词器加载成功")
    
    # 加载模型 - 这里使用基础BERT模型，不是分类模型
    # 如果你有微调后的分类模型，请使用 AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=3,  # 假设3分类任务，根据你的实际任务调整
        local_files_only=True,
        ignore_mismatched_sizes=True  # 允许分类头尺寸不匹配
    )
    print("✓ 模型加载成功")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✓ 模型已加载到设备: {device}")
    
    return tokenizer, model, device

def create_sample_data():
    """
    创建示例数据用于演示
    """
    # 创建一些示例数据
    data = {
        'text': [
            '发动机异响，需要检查',
            '刹车系统失灵，紧急维修',
            '空调不制冷，需要添加制冷剂',
            '轮胎漏气，需要补胎',
            '发动机无法启动，检查电瓶',
            '变速箱换挡困难',
            '转向系统异响',
            '电瓶亏电，无法启动'
        ],
        'label': [
            '发动机故障',
            '刹车系统故障', 
            '空调系统故障',
            '轮胎故障',
            '电气系统故障',
            '变速箱故障',
            '转向系统故障',
            '电气系统故障'
        ]
    }
    
    df = pd.DataFrame(data)
    return df

def preprocess_data(texts, tokenizer, max_length=128):
    """
    预处理文本数据
    
    Args:
        texts: 文本列表
        tokenizer: 分词器
        max_length: 最大序列长度
        
    Returns:
        编码后的输入
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def predict_with_bert(texts, tokenizer, model, device, label_encoder=None):
    """
    使用BERT模型进行预测
    
    Args:
        texts: 待预测的文本列表
        tokenizer: 分词器
        model: BERT模型
        device: 计算设备
        label_encoder: 标签编码器（可选）
        
    Returns:
        预测结果
    """
    # 预处理文本
    inputs = preprocess_data(texts, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 模型预测
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
    
    # 转换为numpy数组
    predictions = predictions.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    # 如果有标签编码器，解码预测结果
    if label_encoder is not None:
        predicted_labels = label_encoder.inverse_transform(predictions)
    else:
        predicted_labels = predictions
    
    return {
        'predictions': predicted_labels,
        'probabilities': probabilities,
        'logits': logits.cpu().numpy()
    }

def main():
    """
    主函数：演示如何使用本地BERT模型
    """
    print("=== 本地BERT模型调用演示 ===\n")
    
    # 1. 加载本地模型
    model_path = "./models/google-bert/bert-base-chinese"  # 你的本地模型目录
    tokenizer, model, device = load_local_bert_model(model_path)
    
    # 2. 创建示例数据
    df = create_sample_data()
    print(f"\n✓ 创建了 {len(df)} 条示例数据")
    print("\n示例数据:")
    print(df.head())
    
    # 3. 准备标签编码器
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(df['label'])
    num_classes = len(label_encoder.classes_)
    print(f"\n✓ 标签编码完成，共 {num_classes} 个类别:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label}")
    
    # 4. 重新加载模型，确保分类头尺寸正确
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_classes,
        local_files_only=True,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    # 5. 预测示例
    test_texts = [
        "发动机有异响声音",
        "刹车不灵敏",
        "空调出风不冷"
    ]
    
    print(f"\n=== 预测测试 ===")
    print("测试文本:")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text}")
    
    results = predict_with_bert(test_texts, tokenizer, model, device, label_encoder)
    
    print("\n预测结果:")
    for i, (text, pred, probs) in enumerate(zip(test_texts, results['predictions'], results['probabilities'])):
        print(f"\n文本 {i+1}: {text}")
        print(f"预测类别: {pred}")
        print("各类别概率:")
        for j, (label, prob) in enumerate(zip(label_encoder.classes_, probs)):
            print(f"  {label}: {prob:.4f}")
    
    # 6. 评估模型性能（使用示例数据）
    print(f"\n=== 模型评估 ===")
    train_results = predict_with_bert(df['text'].tolist(), tokenizer, model, device, label_encoder)
    
    accuracy = accuracy_score(df['label'], train_results['predictions'])
    f1_macro = f1_score(df['label'], train_results['predictions'], average='macro')
    f1_weighted = f1_score(df['label'], train_results['predictions'], average='weighted')
    
    print(f"训练集准确率: {accuracy:.4f}")
    print(f"训练集F1-macro: {f1_macro:.4f}")
    print(f"训练集F1-weighted: {f1_weighted:.4f}")
    
    print("\n=== 演示完成 ===")
    print("你已经成功加载并使用了本地BERT模型！")

if __name__ == "__main__":
    main()