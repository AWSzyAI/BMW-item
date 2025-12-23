#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一的配置管理系统，处理所有配置参数和默认值
"""

import os
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class BERTConfig:
    """BERT模型配置"""
    model_path: str = "./models/google-bert/bert-base-chinese"
    init_hf_dir: Optional[str] = None
    allow_online: bool = False
    num_train_epochs: float = 3.0
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    grad_accum_steps: int = 1
    max_length: int = 256
    fp16: bool = False
    save_hf_dir: Optional[str] = None
    early_stopping_patience: int = 3
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    ooc_tau_percentile: float = 5.0
    ooc_temperature: float = 20.0
    skip_train_stats: bool = False
    post_train_stats_batch_size: int = 16
    stats_on_cpu: bool = False
    resample_method: str = "none"


@dataclass
class DataConfig:
    """数据配置"""
    outdir: str = "./output/2025_up_to_month_2"
    experiment_outdir: Optional[str] = None
    modelsdir: str = "./models"
    checkpoint_dir: Optional[str] = None
    outmodel: str = "bert_model.joblib"
    train_file: str = "train.csv"
    eval_file: str = "eval.csv"
    # 逻辑标签列统一命名为 y，原始列通过 label_col_raw 控制（linked_items/extern_id 等）
    label_col_raw: str = "linked_items"


@dataclass
class EvalConfig:
    """评估配置"""
    mode: str = "new"
    map_unknown_to_other: bool = False
    other_label: str = "__OTHER__"
    unknown_policy: str = "tag-not-in-train"
    reject_threshold: Optional[float] = None
    not_in_train_label: str = "__NOT_IN_TRAIN__"
    sweep_thresholds: Optional[str] = None


@dataclass
class PredictConfig:
    """预测配置"""
    infile: str = "eval.csv"
    index: int = -1
    ooc_decision_threshold: float = 0.5


@dataclass
class SystemConfig:
    """系统配置"""
    log_level: str = "INFO"
    random_seed: int = 42
    num_threads: int = 1
    use_uv: bool = True
    uv_link_mode: str = "copy"


class ConfigManager:
    """统一的配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file or "./config.json"
        self.bert_config = BERTConfig()
        self.data_config = DataConfig()
        self.eval_config = EvalConfig()
        self.predict_config = PredictConfig()
        self.system_config = SystemConfig()
        
        # 如果配置文件存在，加载配置
        if os.path.exists(self.config_file):
            self.load_config()
    
    def load_config(self) -> None:
        """从文件加载配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 更新各部分配置
            if "bert" in config_dict:
                self._update_dataclass(self.bert_config, config_dict["bert"])
            
            if "data" in config_dict:
                self._update_dataclass(self.data_config, config_dict["data"])
            
            if "eval" in config_dict:
                self._update_dataclass(self.eval_config, config_dict["eval"])
            
            if "predict" in config_dict:
                self._update_dataclass(self.predict_config, config_dict["predict"])
            
            if "system" in config_dict:
                self._update_dataclass(self.system_config, config_dict["system"])
            
            print(f"✓ 配置已从文件加载: {self.config_file}")
        except Exception as e:
            print(f"⚠️ 配置文件加载失败，使用默认配置: {e}")
    
    def save_config(self) -> None:
        """保存配置到文件"""
        try:
            config_dict = {
                "bert": asdict(self.bert_config),
                "data": asdict(self.data_config),
                "eval": asdict(self.eval_config),
                "predict": asdict(self.predict_config),
                "system": asdict(self.system_config)
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            print(f"✓ 配置已保存到文件: {self.config_file}")
        except Exception as e:
            print(f"⚠️ 配置文件保存失败: {e}")
    
    def _update_dataclass(self, dataclass_instance: Any, update_dict: Dict[str, Any]) -> None:
        """更新数据类实例的属性"""
        for key, value in update_dict.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
            else:
                print(f"⚠️ 未知配置项: {key} = {value}")
    
    def update_from_args(self, args: Dict[str, Any]) -> None:
        """从命令行参数更新配置"""
        # BERT配置
        bert_mapping = {
            "bert_model": "model_path",
            "init_hf_dir": "init_hf_dir",
            "allow_online": "allow_online",
            "num_train_epochs": "num_train_epochs",
            "train_batch_size": "train_batch_size",
            "eval_batch_size": "eval_batch_size",
            "learning_rate": "learning_rate",
            "weight_decay": "weight_decay",
            "grad_accum_steps": "grad_accum_steps",
            "max_length": "max_length",
            "fp16": "fp16",
            "save_hf_dir": "save_hf_dir",
            "early_stopping_patience": "early_stopping_patience",
            "lr_scheduler_type": "lr_scheduler_type",
            "warmup_ratio": "warmup_ratio",
            "warmup_steps": "warmup_steps",
            "ooc_tau_percentile": "ooc_tau_percentile",
            "ooc_temperature": "ooc_temperature",
            "skip_train_stats": "skip_train_stats",
            "post_train_stats_batch_size": "post_train_stats_batch_size",
            "stats_on_cpu": "stats_on_cpu",
            "resample_method": "resample_method"
        }
        
        # 数据配置
        data_mapping = {
            "outdir": "outdir",
            "experiment_outdir": "experiment_outdir",
            "modelsdir": "modelsdir",
            "checkpoint_dir": "checkpoint_dir",
            "outmodel": "outmodel",
            "train_file": "train_file",
            "eval_file": "eval_file",
            "label_col": "label_col"
        }
        
        # 评估配置
        eval_mapping = {
            "mode": "mode",
            "map_unknown_to_other": "map_unknown_to_other",
            "other_label": "other_label",
            "unknown_policy": "unknown_policy",
            "reject_threshold": "reject_threshold",
            "not_in_train_label": "not_in_train_label",
            "sweep_thresholds": "sweep_thresholds"
        }
        
        # 预测配置
        predict_mapping = {
            "infile": "infile",
            "index": "index",
            "ooc_decision_threshold": "ooc_decision_threshold"
        }
        
        # 系统配置
        system_mapping = {
            "log_level": "log_level",
            "random_seed": "random_seed",
            "num_threads": "num_threads"
        }
        
        # 更新配置
        self._update_from_mapping(args, bert_mapping, self.bert_config)
        self._update_from_mapping(args, data_mapping, self.data_config)
        self._update_from_mapping(args, eval_mapping, self.eval_config)
        self._update_from_mapping(args, predict_mapping, self.predict_config)
        self._update_from_mapping(args, system_mapping, self.system_config)
    
    def _update_from_mapping(self, args: Dict[str, Any], mapping: Dict[str, str], 
                           config_instance: Any) -> None:
        """根据映射更新配置"""
        for arg_key, config_key in mapping.items():
            if arg_key in args and args[arg_key] is not None:
                setattr(config_instance, config_key, args[arg_key])
    
    def get_bert_config(self) -> BERTConfig:
        """获取BERT配置"""
        return self.bert_config
    
    def get_data_config(self) -> DataConfig:
        """获取数据配置"""
        return self.data_config
    
    def get_eval_config(self) -> EvalConfig:
        """获取评估配置"""
        return self.eval_config
    
    def get_predict_config(self) -> PredictConfig:
        """获取预测配置"""
        return self.predict_config
    
    def get_system_config(self) -> SystemConfig:
        """获取系统配置"""
        return self.system_config
    
    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置"""
        return {
            "bert": asdict(self.bert_config),
            "data": asdict(self.data_config),
            "eval": asdict(self.eval_config),
            "predict": asdict(self.predict_config),
            "system": asdict(self.system_config)
        }
    
    def validate_configs(self) -> bool:
        """验证配置的有效性"""
        valid = True
        
        # 验证BERT配置
        if self.bert_config.num_train_epochs <= 0:
            print("⚠️ 训练轮数必须大于0")
            valid = False
        
        if self.bert_config.train_batch_size <= 0:
            print("⚠️ 训练批次大小必须大于0")
            valid = False
        
        if self.bert_config.eval_batch_size <= 0:
            print("⚠️ 评估批次大小必须大于0")
            valid = False
        
        if self.bert_config.learning_rate <= 0:
            print("⚠️ 学习率必须大于0")
            valid = False
        
        if self.bert_config.max_length <= 0:
            print("⚠️ 最大序列长度必须大于0")
            valid = False
        
        # 验证数据配置
        if not os.path.exists(self.data_config.outdir):
            print(f"⚠️ 数据目录不存在: {self.data_config.outdir}")
            valid = False
        
        # 验证评估配置
        if self.eval_config.mode not in ["new", "clean", "dirty"]:
            print(f"⚠️ 无效的评估模式: {self.eval_config.mode}")
            valid = False
        
        return valid
    
    def print_configs(self) -> None:
        """打印当前配置"""
        print("=== 当前配置 ===")
        print("\n[BERT配置]")
        for key, value in asdict(self.bert_config).items():
            print(f"  {key}: {value}")
        
        print("\n[数据配置]")
        for key, value in asdict(self.data_config).items():
            print(f"  {key}: {value}")
        
        print("\n[评估配置]")
        for key, value in asdict(self.eval_config).items():
            print(f"  {key}: {value}")
        
        print("\n[预测配置]")
        for key, value in asdict(self.predict_config).items():
            print(f"  {key}: {value}")
        
        print("\n[系统配置]")
        for key, value in asdict(self.system_config).items():
            print(f"  {key}: {value}")
        
        print("=" * 50)


# 全局配置管理器实例
_config_manager = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager


def reset_config_manager() -> None:
    """重置全局配置管理器"""
    global _config_manager
    _config_manager = None