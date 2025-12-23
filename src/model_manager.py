#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一的模型管理器，处理模型加载、设备管理、内存清理等
"""

import os
import gc
import warnings
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import joblib

# 可选依赖处理
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    _HAS_IMBLEARN = True
except Exception:
    SMOTE = SMOTEENN = SMOTETomek = RandomOverSampler = None
    _HAS_IMBLEARN = False

# 可选依赖：matplotlib
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

from utils import ensure_single_label, build_text, hit_at_k, fmt_sec, _flex_read_csv


class ModelManager:
    """统一的模型管理器，处理模型加载、设备管理、内存清理等"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化模型管理器
        
        Args:
            config: 配置字典，包含模型路径、设备配置等
        """
        self.config = config or {}
        self.device = self._detect_device()
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.ooc_detector = None
        
    def _detect_device(self) -> torch.device:
        """自动检测可用设备"""
        try:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"✓ 检测到CUDA设备: {torch.cuda.get_device_name()}")
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                device = torch.device('mps')
                print("✓ 检测到MPS设备")
            else:
                device = torch.device('cpu')
                print("✓ 使用CPU设备")
            return device
        except Exception as e:
            print(f"⚠️ 设备检测失败，使用CPU: {e}")
            return torch.device('cpu')
    
    def setup_tokenizer(self, model_path: str, local_files_only: bool = True) -> None:
        """
        设置分词器
        
        Args:
            model_path: 模型路径
            local_files_only: 是否仅使用本地文件
        """
        try:
            print(f"📥 正在加载分词器: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                local_files_only=local_files_only
            )
            print(f"✓ 分词器加载成功，词汇表大小: {len(self.tokenizer)}")
        except Exception as e:
            raise RuntimeError(f"分词器加载失败: {e}")
    
    def setup_model(self, model_path: str, num_labels: int, local_files_only: bool = True) -> None:
        """
        设置模型
        
        Args:
            model_path: 模型路径
            num_labels: 分类数量
            local_files_only: 是否仅使用本地文件
        """
        try:
            print(f"🏗️ 正在加载模型: {model_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                local_files_only=local_files_only
            )
            self.model.to(self.device)
            print(f"✓ 模型加载成功，已移动到设备: {self.device}")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def setup_label_encoder(self, labels: List[str]) -> None:
        """
        设置标签编码器
        
        Args:
            labels: 标签列表
        """
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        print(f"✓ 标签编码器设置完成，共 {len(self.label_encoder.classes_)} 个类别")
    
    def clear_memory(self) -> None:
        """清理内存"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ GPU内存已清理")
        except Exception as e:
            print(f"⚠️ 内存清理失败: {e}")
    
    def switch_device(self, target_device: Union[str, torch.device]) -> bool:
        """
        切换设备
        
        Args:
            target_device: 目标设备
            
        Returns:
            bool: 切换是否成功
        """
        try:
            if isinstance(target_device, str):
                target_device = torch.device(target_device)
            
            if self.model is not None:
                self.model.to(target_device)
                self.device = target_device
                self.clear_memory()
                print(f"✓ 设备已切换到: {target_device}")
                return True
            return False
        except Exception as e:
            print(f"⚠️ 设备切换失败: {e}")
            return False
    
    def predict_proba_batched(self, texts: List[str], batch_size: int = 16, 
                           max_length: int = 256, use_amp: bool = False) -> np.ndarray:
        """
        分批预测概率，自动处理OOM错误
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            max_length: 最大序列长度
            use_amp: 是否使用混合精度
            
        Returns:
            np.ndarray: 预测概率
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型或分词器未初始化")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # 设备尝试序列
        devices = []
        try:
            if torch.cuda.is_available():
                devices.append(torch.device('cuda'))
            if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                devices.append(torch.device('mps'))
        except Exception:
            pass
        devices.append(torch.device('cpu'))
        
        last_err = None
        for device in devices:
            try:
                # 切换到目标设备
                if not self.switch_device(device):
                    continue
                
                # 逐步缩小批次大小
                for bs in [batch_size, max(1, batch_size // 2), max(1, batch_size // 4)]:
                    try:
                        use_amp_current = use_amp and device.type == 'cuda'
                        return self._predict_proba_in_batches(
                            texts, bs, max_length, use_amp_current
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"⚠️ OOM错误，批次大小 {bs} -> {max(1, bs // 2)}")
                            self.clear_memory()
                            continue
                        raise e
            except Exception as e:
                last_err = e
                continue
        
        # 所有尝试都失败
        if last_err is not None:
            raise last_err
        return np.zeros((0, 0), dtype=np.float32)
    
    def _predict_proba_in_batches(self, texts: List[str], batch_size: int, 
                               max_length: int, use_amp: bool = False) -> np.ndarray:
        """
        内部分批预测方法
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            max_length: 最大序列长度
            use_amp: 是否使用混合精度
            
        Returns:
            np.ndarray: 预测概率
        """
        self.model.eval()
        all_probs = []
        
        # 设置混合精度上下文
        amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if use_amp else torch.no_grad()
        
        with torch.inference_mode(), amp_ctx:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 分词
                enc = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # 移动到设备
                enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
                
                # 预测
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(**enc)
                    probs = torch.softmax(outputs.logits, dim=-1).to('cpu')
                
                all_probs.append(probs)
                
                # 清理中间变量
                del enc, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if not all_probs:
            return np.zeros((0, 0), dtype=np.float32)
        
        return torch.cat(all_probs, dim=0).numpy()
    
    def save_model(self, save_dir: str) -> None:
        """
        保存模型和分词器
        
        Args:
            save_dir: 保存目录
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型或分词器未初始化")
        
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            print(f"✓ 模型和分词器已保存到: {save_dir}")
        except Exception as e:
            raise RuntimeError(f"模型保存失败: {e}")
    
    def load_model_bundle(self, bundle_path: str) -> Dict[str, Any]:
        """
        加载模型bundle
        
        Args:
            bundle_path: bundle文件路径
            
        Returns:
            Dict: 包含模型、标签编码器等的bundle
        """
        try:
            bundle = joblib.load(bundle_path)
            print(f"✓ 模型bundle加载成功: {bundle_path}")
            return bundle
        except Exception as e:
            raise RuntimeError(f"模型bundle加载失败: {e}")
    
    def save_model_bundle(self, bundle_path: str, model_dir: str, 
                        model_type: str = "bert", **kwargs) -> None:
        """
        保存模型bundle
        
        Args:
            bundle_path: bundle保存路径
            model_dir: 模型目录
            model_type: 模型类型
            **kwargs: 其他参数
        """
        labels = kwargs.get("labels")
        if labels is None and self.label_encoder is not None:
            labels = self.label_encoder.classes_.tolist()

        bundle = {
            "model_type": model_type,
            "model_dir": model_dir,
            "label_encoder": self.label_encoder,
            "labels": labels,
            "ooc_detector": self.ooc_detector,
        }

        if model_type == "bert":
            bundle.update({
                "max_length": kwargs.get("max_length"),
                "fp16": kwargs.get("fp16", False),
            })
        else:
            bundle.update(kwargs)
        
        try:
            os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
            joblib.dump(bundle, bundle_path)
            print(f"✓ 模型bundle已保存到: {bundle_path}")
        except Exception as e:
            raise RuntimeError(f"模型bundle保存失败: {e}")
    
    def handle_imbalanced_data(self, X: List[str], y: np.ndarray, 
                           method: str = "none") -> tuple[List[str], np.ndarray]:
        """
        处理不平衡数据
        
        Args:
            X: 文本列表
            y: 标签数组
            method: 采样方法
            
        Returns:
            tuple: 处理后的文本和标签
        """
        if method == "none" or not X:
            return X, y
        
        print(f"🔧 使用不平衡处理方法(ROS): {method}")

        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        max_n = counts.max()
        rng = np.random.default_rng(42)
        indices = []
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                continue
            if len(idx) < max_n:
                extra = rng.choice(idx, size=max_n - len(idx), replace=True)
                idx = np.concatenate([idx, extra], axis=0)
            indices.append(idx)

        sel = np.concatenate(indices, axis=0)
        print(f"✓ 采样完成，样本数: {len(X)} -> {len(sel)}")
        return [X[i] for i in sel], y[sel]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息
        """
        info = {
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "label_encoder_loaded": self.label_encoder is not None,
        }
        
        if self.label_encoder is not None:
            info["num_classes"] = len(self.label_encoder.classes_)
            info["classes"] = list(self.label_encoder.classes_)
        
        if self.model is not None:
            info["model_type"] = type(self.model).__name__
            
        return info