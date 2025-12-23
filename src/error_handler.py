#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一的错误处理和日志记录系统
"""

import os
import sys
import traceback
import logging
import functools
import numpy as np
from typing import Any, Callable, Optional, Union, Type
from datetime import datetime


class ErrorHandler:
    """统一的错误处理器"""
    
    def __init__(self, log_file: Optional[str] = None, log_level: str = "INFO"):
        """
        初始化错误处理器
        
        Args:
            log_file: 日志文件路径
            log_level: 日志级别
        """
        self.log_file = log_file or f"./logs/error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # 创建日志目录
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # 配置日志
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """设置日志配置"""
        # 创建logger
        self.logger = logging.getLogger('BMW_BERT')
        self.logger.setLevel(self.log_level)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(self.log_level)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_error(self, message: str, exception: Optional[Exception] = None) -> None:
        """
        记录错误
        
        Args:
            message: 错误消息
            exception: 异常对象
        """
        if exception:
            self.logger.error(f"{message}: {str(exception)}")
            self.logger.debug(f"异常堆栈:\n{traceback.format_exc()}")
        else:
            self.logger.error(message)
    
    def log_warning(self, message: str) -> None:
        """
        记录警告
        
        Args:
            message: 警告消息
        """
        self.logger.warning(message)
    
    def log_info(self, message: str) -> None:
        """
        记录信息
        
        Args:
            message: 信息消息
        """
        self.logger.info(message)
    
    def log_debug(self, message: str) -> None:
        """
        记录调试信息
        
        Args:
            message: 调试消息
        """
        self.logger.debug(message)
    
    def handle_exception(self, func: Callable) -> Callable:
        """
        异常处理装饰器
        
        Args:
            func: 要装饰的函数
            
        Returns:
            装饰后的函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.log_error(f"函数 {func.__name__} 执行失败", e)
                raise
        return wrapper
    
    def handle_oom(self, func: Callable) -> Callable:
        """
        OOM错误处理装饰器
        
        Args:
            func: 要装饰的函数
            
        Returns:
            装饰后的函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.log_warning(f"检测到OOM错误，尝试清理内存: {e}")
                    self._clear_memory()
                    # 尝试降低批次大小重试
                    if 'batch_size' in kwargs:
                        original_batch_size = kwargs['batch_size']
                        new_batch_size = max(1, original_batch_size // 2)
                        self.log_info(f"降低批次大小重试: {original_batch_size} -> {new_batch_size}")
                        kwargs['batch_size'] = new_batch_size
                        try:
                            return func(*args, **kwargs)
                        except Exception as retry_e:
                            self.log_error(f"降低批次大小重试失败", retry_e)
                            raise
                    else:
                        raise
                else:
                    raise
        return wrapper
    
    def _clear_memory(self) -> None:
        """清理内存"""
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.log_info("GPU内存已清理")
        except Exception as e:
            self.log_warning(f"内存清理失败: {e}")
    
    def retry(self, max_retries: int = 3, delay: float = 1.0, 
              exceptions: tuple = (Exception,)) -> Callable:
        """
        重试装饰器
        
        Args:
            max_retries: 最大重试次数
            delay: 重试延迟（秒）
            exceptions: 需要重试的异常类型
            
        Returns:
            装饰器
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_retries:
                            self.log_warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                            self.log_info(f"等待 {delay} 秒后重试...")
                            import time
                            time.sleep(delay)
                        else:
                            self.log_error(f"函数 {func.__name__} 在 {max_retries + 1} 次尝试后仍然失败", e)
                            raise
                raise last_exception
            return wrapper
        return decorator
    
    def validate_file_exists(self, file_path: str, description: str = "文件") -> bool:
        """
        验证文件是否存在
        
        Args:
            file_path: 文件路径
            description: 文件描述
            
        Returns:
            bool: 文件是否存在
        """
        if not os.path.exists(file_path):
            self.log_error(f"{description}不存在: {file_path}")
            return False
        return True
    
    def validate_dir_exists(self, dir_path: str, description: str = "目录") -> bool:
        """
        验证目录是否存在
        
        Args:
            dir_path: 目录路径
            description: 目录描述
            
        Returns:
            bool: 目录是否存在
        """
        if not os.path.exists(dir_path):
            self.log_error(f"{description}不存在: {dir_path}")
            return False
        if not os.path.isdir(dir_path):
            self.log_error(f"路径不是目录: {dir_path}")
            return False
        return True
    
    def validate_model_files(self, model_dir: str) -> bool:
        """
        验证模型文件是否完整
        
        Args:
            model_dir: 模型目录
            
        Returns:
            bool: 模型文件是否完整
        """
        required_files = ["config.json"]
        optional_files = ["tokenizer.json", "vocab.txt", "pytorch_model.bin"]
        
        # 检查必需文件
        for file_name in required_files:
            file_path = os.path.join(model_dir, file_name)
            if not self.validate_file_exists(file_path, f"必需模型文件 {file_name}"):
                return False
        
        # 检查可选文件（至少需要一个）
        tokenizer_found = False
        for file_name in optional_files:
            file_path = os.path.join(model_dir, file_name)
            if os.path.exists(file_path):
                tokenizer_found = True
                break
        
        if not tokenizer_found:
            self.log_error(f"模型目录缺少分词器文件: {model_dir}")
            return False
        
        self.log_info(f"模型文件验证通过: {model_dir}")
        return True
    
    def get_log_file(self) -> str:
        """获取日志文件路径"""
        return self.log_file
    
    def log_metrics(self, metrics: dict, category: str = "general") -> None:
        """
        记录评估指标
        
        Args:
            metrics: 指标字典
            category: 指标类别（如 'performance', 'quality', 'error_analysis'）
        """
        self.log_info(f"=== {category.upper()} 指标 ===")
        
        # 按类别组织指标
        if category == "performance":
            self.log_info(f"  平均延迟: {metrics.get('avg_latency', 0):.4f}秒")
            self.log_info(f"  TP99延迟: {metrics.get('tp99_latency', 0):.4f}秒")
            self.log_info(f"  TPS: {metrics.get('tps', 0):.2f}")
            self.log_info(f"  总时间: {metrics.get('total_time', 0):.4f}秒")
            self.log_info(f"  样本数: {metrics.get('num_samples', 0)}")
        
        elif category == "quality":
            # 基础指标
            self.log_info("  🔍 基础指标:")
            for k in ["accuracy", "f1_weighted", "f1_macro"]:
                if k in metrics:
                    self.log_info(f"    {k}: {metrics[k]:.4f}")
            
            # Hit@K指标
            self.log_info("  🎯 Hit@K指标:")
            for k in ["hit@1", "hit@3", "hit@5", "hit@10"]:
                if k in metrics and not np.isnan(metrics[k]):
                    self.log_info(f"    {k}: {metrics[k]:.4f}")
            
            # 排序质量指标
            self.log_info("  📊 排序质量指标:")
            for k in ["mrr", "ndcg@3", "ndcg@5", "ndcg@10"]:
                if k in metrics and not np.isnan(metrics[k]):
                    self.log_info(f"    {k}: {metrics[k]:.4f}")
            
            # 覆盖率指标
            self.log_info("  🌐 覆盖率指标:")
            for k in ["coverage@3", "coverage@5", "coverage@10"]:
                if k in metrics and not np.isnan(metrics[k]):
                    self.log_info(f"    {k}: {metrics[k]:.4f}")
        
        elif category == "confidence":
            self.log_info(f"  平均置信度: {metrics.get('avg_confidence', 0):.4f}")
            self.log_info(f"  最低置信度: {metrics.get('min_confidence', 0):.4f}")
            self.log_info(f"  最高置信度: {metrics.get('max_confidence', 0):.4f}")
            
            # 置信度分布
            self.log_info("  置信度分布:")
            for k, v in metrics.items():
                if k.startswith("confidence_") and isinstance(v, dict):
                    threshold = k.replace("confidence_", "")
                    self.log_info(f"    {threshold}+: {v['count']} ({v['percentage']:.1f}%)")
        
        elif category == "error_analysis":
            self.log_info("  错误类型分析:")
            for error_type, info in metrics.items():
                if isinstance(info, dict):
                    self.log_info(f"    {error_type}: {info['count']} ({info['percentage']:.1f}%)")
        
        elif category == "distribution":
            self.log_info("  预测分布:")
            for item_name, info in metrics.items():
                if isinstance(info, dict):
                    self.log_info(f"    {item_name}: {info['count']} ({info['percentage']:.1f}%)")
        
        else:
            # 通用指标记录
            for k, v in metrics.items():
                if isinstance(v, float):
                    self.log_info(f"  {k}: {v:.4f}")
                elif isinstance(v, dict):
                    self.log_info(f"  {k}: {v}")
                else:
                    self.log_info(f"  {k}: {v}")
    
    def log_experiment_summary(self, experiment_info: dict) -> None:
        """
        记录实验摘要
        
        Args:
            experiment_info: 实验信息字典
        """
        self.log_info("=" * 50)
        self.log_info("实验摘要")
        self.log_info("=" * 50)
        
        # 基本信息
        self.log_info(f"实验时间: {experiment_info.get('timestamp', 'Unknown')}")
        self.log_info(f"模型类型: {experiment_info.get('model_type', 'Unknown')}")
        self.log_info(f"模型路径: {experiment_info.get('model_path', 'Unknown')}")
        
        # 数据信息
        if 'data_info' in experiment_info:
            data_info = experiment_info['data_info']
            self.log_info(f"训练样本数: {data_info.get('train_samples', 0)}")
            self.log_info(f"评估样本数: {data_info.get('eval_samples', 0)}")
            self.log_info(f"测试样本数: {data_info.get('test_samples', 0)}")
            self.log_info(f"类别数: {data_info.get('num_classes', 0)}")
        
        # 关键指标
        if 'key_metrics' in experiment_info:
            key_metrics = experiment_info['key_metrics']
            self.log_info("关键指标:")
            for metric_name, value in key_metrics.items():
                if isinstance(value, float):
                    self.log_info(f"  {metric_name}: {value:.4f}")
                else:
                    self.log_info(f"  {metric_name}: {value}")
        
        # 输出文件
        if 'output_files' in experiment_info:
            self.log_info("输出文件:")
            for file_type, file_path in experiment_info['output_files'].items():
                self.log_info(f"  {file_type}: {file_path}")
        
        self.log_info("=" * 50)


# 全局错误处理器实例
_error_handler = None


def get_error_handler(log_file: Optional[str] = None, 
                   log_level: str = "INFO") -> ErrorHandler:
    """获取全局错误处理器实例"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler(log_file, log_level)
    return _error_handler


def reset_error_handler() -> None:
    """重置全局错误处理器"""
    global _error_handler
    _error_handler = None


def log_error(message: str, exception: Optional[Exception] = None) -> None:
    """记录错误的便捷函数"""
    get_error_handler().log_error(message, exception)


def log_warning(message: str) -> None:
    """记录警告的便捷函数"""
    get_error_handler().log_warning(message)


def log_info(message: str) -> None:
    """记录信息的便捷函数"""
    get_error_handler().log_info(message)


def log_debug(message: str) -> None:
    """记录调试信息的便捷函数"""
    get_error_handler().log_debug(message)


def handle_exception(func: Callable) -> Callable:
    """异常处理装饰器的便捷函数"""
    return get_error_handler().handle_exception(func)


def handle_oom(func: Callable) -> Callable:
    """OOM错误处理装饰器的便捷函数"""
    return get_error_handler().handle_oom(func)


def retry(max_retries: int = 3, delay: float = 1.0,
          exceptions: tuple = (Exception,)) -> Callable:
    """重试装饰器的便捷函数"""
    return get_error_handler().retry(max_retries, delay, exceptions)


def log_metrics(metrics: dict, category: str = "general") -> None:
    """记录评估指标的便捷函数"""
    get_error_handler().log_metrics(metrics, category)


def log_experiment_summary(experiment_info: dict) -> None:
    """记录实验摘要的便捷函数"""
    get_error_handler().log_experiment_summary(experiment_info)