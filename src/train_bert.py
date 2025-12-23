#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一的训练脚本，支持BERT和TF-IDF模型
使用统一的模型管理器和配置管理系统
"""

import os
import json
import argparse
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, f1_score
import joblib
from tqdm import tqdm
from contextlib import nullcontext

# 导入我们的管理系统
from model_manager import ModelManager
from config_manager import get_config_manager, ConfigManager
from error_handler import get_error_handler, log_info, log_warning, log_error, handle_exception, handle_oom, retry
from utils import ensure_single_label, build_text, hit_at_k, fmt_sec, _flex_read_csv

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
        TrainerCallback,
        EarlyStoppingCallback,
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


class LossRecorder(TrainerCallback):
    """记录训练损失"""
    def __init__(self):
        self.losses: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if state.is_world_process_zero and ("loss" in logs):
            try:
                self.losses.append(float(logs["loss"]))
            except Exception:
                pass


def _compute_metrics(eval_pred, num_labels: int) -> dict:
    """计算评估指标"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1w = f1_score(labels, preds, average="weighted")
    f1m = f1_score(labels, preds, average="macro")
    # 计算 hit@k
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    proba = e / e.sum(axis=1, keepdims=True)
    out = {
        "accuracy": float(acc),
        "f1_weighted": float(f1w),
        "f1_macro": float(f1m),
        "hit@1": hit_at_k(labels, proba, 1),
        "hit@3": hit_at_k(labels, proba, 3),
        "hit@5": hit_at_k(labels, proba, 5) if num_labels >= 5 else float("nan"),
        "hit@10": hit_at_k(labels, proba, 10) if num_labels >= 10 else float("nan"),
    }
    return out


def _str2bool(v) -> bool:
    return str(v).lower() in {"1", "true", "t", "y", "yes"}


def _read_split_or_combined(base_dir: str, base_filename: str) -> pd.DataFrame:
    """优先读取 X/Y 分离文件；若不存在则回退到单表 CSV。"""
    base_dir = os.path.abspath(base_dir)
    name = os.path.basename(base_filename)
    stem, ext = os.path.splitext(name)
    # 兼容传入 *_X.csv 或 *_y.csv 的情况，统一回到公共 stem
    if stem.endswith("_X"):
        stem = stem[:-2]
    if stem.endswith("_y"):
        stem = stem[:-2]

    x_name = f"{stem}_X.csv"
    y_name = f"{stem}_y.csv"

    def _exists_in_dir(fname: str) -> str | None:
        p = os.path.join(base_dir, fname)
        return p if os.path.exists(p) else None

    # 1) 优先尝试分表
    x_path = _exists_in_dir(x_name)
    y_path = _exists_in_dir(y_name)
    if x_path and y_path:
        X = _flex_read_csv(base_dir, os.path.basename(x_path))
        y = _flex_read_csv(base_dir, os.path.basename(y_name))

        # 兼容 y 列名
        if "linked_items" not in y.columns:
            if "label" in y.columns:
                y = y.rename(columns={"label": "linked_items"})
            elif "y" in y.columns:
                y = y.rename(columns={"y": "linked_items"})
            else:
                # 若多列，取第一列作为标签
                first_label_col = y.columns[0]
                warnings.warn(f"未找到 'linked_items'，使用 '{first_label_col}' 作为标签列")
                y = y.rename(columns={first_label_col: "linked_items"})

        # 只保留标签列
        y = y[["linked_items"]]
        if len(X) != len(y):
            raise ValueError(f"X/Y 行数不一致：X={len(X)} Y={len(y)}（stem={stem}）")
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        return df

    # 2) 回退：读取单表（例如 train.csv / eval.csv）
    warnings.warn(
        f"未找到分表 {x_name}+{y_name}，回退到单表 {name}（目录：{base_dir}）"
    )
    return _flex_read_csv(base_dir, name)


def _choose_label_column(df: pd.DataFrame) -> str:
    """选择标签列"""
    # 优先级：extend_id > linked_items > item_title
    for col in ["extend_id", "linked_items", "item_title"]:
        if col in df.columns:
            return col
    raise KeyError("未找到可用标签列（extend_id/linked_items/item_title）")


@handle_exception
@retry(max_retries=3, delay=2.0)
def main(args):
    """主训练函数"""
    global_start = time.time()
    
    # 初始化配置管理器和错误处理器
    config_manager = get_config_manager()
    config_manager.update_from_args(vars(args))
    
    error_handler = get_error_handler(
        log_file=f"./logs/train_{time.strftime('%Y%m%d_%H%M%S')}.log",
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
    
    log_info(f"=== {model_type.upper()}模型训练开始 ===")
    
    # 创建目录
    os.makedirs(data_config.outdir, exist_ok=True)
    os.makedirs(data_config.modelsdir, exist_ok=True)
    
    # 初始化模型管理器
    model_manager = ModelManager()
    
    # 读取数据
    log_info("📖 正在读取训练数据...")
    log_info(f"   数据目录: {data_config.outdir}")
    log_info(f"   训练文件: {data_config.train_file}")
    df_tr = _read_split_or_combined(data_config.outdir, data_config.train_file)
    log_info(f"✓ 训练数据读取完成: {df_tr.shape}")
    
    log_info("📖 正在读取评估数据...")
    log_info(f"   评估文件: {data_config.eval_file}")
    df_ev = _read_split_or_combined(data_config.outdir, data_config.eval_file)
    log_info(f"✓ 评估数据读取完成: {df_ev.shape}")
    
    label_col = _choose_label_column(df_tr)
    log_info(f"✓ 选择标签列: {label_col}")
    
    # 检查必要列
    for df_name, df in [("train", df_tr), ("eval", df_ev)]:
        for col in ["case_title", "performed_work", label_col]:
            if col not in df.columns:
                raise KeyError(f"{df_name}.csv 缺少列：{col}")
    
    # 清洗标签
    df_tr[label_col] = df_tr[label_col].apply(ensure_single_label).astype(str)
    df_ev[label_col] = df_ev[label_col].apply(ensure_single_label).astype(str)
    
    X_tr = build_text(df_tr).tolist()
    y_tr_raw = df_tr[label_col].astype(str).tolist()
    X_ev = build_text(df_ev).tolist()
    y_ev_raw = df_ev[label_col].astype(str).tolist()
    
    # 处理稀有标签
    log_info("🔍 检查稀有标签...")
    vc = df_tr[label_col].value_counts()
    rare_labels = vc[vc == 1].index.tolist()
    if rare_labels:
        rare_samples = df_tr[df_tr[label_col].isin(rare_labels)]
        df_tr = pd.concat([df_tr, rare_samples], ignore_index=True)
        log_info(f"⚠️  已复制 {len(rare_samples)} 个单样本类别，以平衡训练集。")
        X_tr = build_text(df_tr).tolist()
        y_tr_raw = df_tr[label_col].astype(str).tolist()
    else:
        log_info("✓ 无需复制稀有标签样本")
    
    # 编码标签
    log_info("🏷️  正在编码标签...")
    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr_raw)
    log_info(f"✓ 标签编码完成，共 {len(le.classes_)} 个类别")
    
    # 设置模型管理器的标签编码器
    model_manager.setup_label_encoder(le.classes_.tolist())
    
    # 过滤 eval 中不在训练标签集的样本
    ev_mask = [lbl in set(le.classes_) for lbl in y_ev_raw]
    if not all(ev_mask):
        dropped = int(np.sum(~np.array(ev_mask)))
        log_info(f"[警告] eval 中有 {dropped} 条样本的标签未在训练集中出现（记为 not_in_train）")
    X_ev_f = [t for t, m in zip(X_ev, ev_mask) if m]
    y_ev_f = [l for l, m in zip(y_ev_raw, ev_mask) if m]
    y_ev = le.transform(y_ev_f) if len(y_ev_f) > 0 else np.array([])
    
    if model_type == 'bert':
        # BERT模型训练
        if not _HAS_TRANSFORMERS:
            raise RuntimeError("BERT模型训练需要安装transformers库")
        
        # 设置模型和分词器
        init_path = bert_config.init_hf_dir or bert_config.model_path
        is_local = os.path.isdir(init_path)
        
        log_info(f"📂 模型路径: {init_path}")
        log_info(f"🌐 使用本地模型: {is_local}")
        log_info(f"🔗 允许在线下载: {bert_config.allow_online}")
        
        # 验证本地模型
        if is_local:
            needed_files = ["config.json"]
            has_tokenizer = any(
                os.path.exists(os.path.join(init_path, name))
                for name in ["tokenizer.json", "vocab.txt"]
            )
            if not has_tokenizer:
                raise RuntimeError(f"本地模型目录不完整：{init_path}")
        
        # 设置分词器和模型
        model_manager.setup_tokenizer(init_path, local_files_only=is_local)
        model_manager.setup_model(init_path, len(le.classes_), local_files_only=is_local)

        # 处理不平衡数据（需在 tokenizer 初始化后）
        if bert_config.resample_method != "none":
            X_tr, y_tr = model_manager.handle_imbalanced_data(
                X_tr, y_tr, bert_config.resample_method
            )
        
        # 编码数据
        def _tokenize(batch_texts: list[str]):
            return model_manager.tokenizer(
                batch_texts,
                padding=False,
                truncation=True,
                max_length=bert_config.max_length,
            )
        
        log_info("🔤 正在编码训练数据...")
        log_info(f"   最大序列长度: {bert_config.max_length}")
        enc_tr = _tokenize(X_tr)
        log_info(f"✓ 训练数据编码完成: {len(enc_tr['input_ids'])} 样本")
        
        if len(X_ev_f) > 0:
            log_info("🔤 正在编码评估数据...")
            enc_ev = _tokenize(X_ev_f)
            log_info(f"✓ 评估数据编码完成: {len(enc_ev['input_ids'])} 样本")
        else:
            enc_ev = _tokenize(["dummy"])
            log_warning("⚠️  评估数据为空，使用虚拟数据")
        
        # 创建数据集，确保标签为整数类型
        ds_tr = UnifiedDataset(encodings=dict(enc_tr), labels=np.asarray(y_tr, dtype=np.int64))
        ds_ev = UnifiedDataset(encodings=dict(enc_ev), labels=np.asarray(y_ev, dtype=np.int64) if len(X_ev_f) > 0 and len(y_ev) > 0 else None)
        
        # 设置运行目录
        if data_config.checkpoint_dir:
            run_dir = os.path.join(data_config.checkpoint_dir, "runs")
            os.makedirs(run_dir, exist_ok=True)
            log_info(f"📁 运行目录: {run_dir}")
        else:
            run_dir = os.path.join(data_config.modelsdir, os.path.splitext(os.path.basename(data_config.outmodel))[0] + "_runs")
            os.makedirs(run_dir, exist_ok=True)
            log_info(f"📁 运行目录: {run_dir}")
        
        # 训练参数
        use_fp16 = bert_config.fp16 and model_manager.device.type == 'cuda'
        
        training_args = TrainingArguments(
            output_dir=run_dir,
            per_device_train_batch_size=bert_config.train_batch_size,
            per_device_eval_batch_size=bert_config.eval_batch_size,
            learning_rate=bert_config.learning_rate,
            num_train_epochs=bert_config.num_train_epochs,
            weight_decay=bert_config.weight_decay,
            eval_strategy="epoch" if ds_ev is not None else "no",
            logging_strategy="epoch",
            save_strategy="epoch" if ds_ev is not None else "no",
            save_total_limit=1,
            report_to=[],
            load_best_model_at_end=bool(ds_ev is not None),
            metric_for_best_model="eval_loss" if ds_ev is not None else None,
            greater_is_better=False,
            remove_unused_columns=False,
            gradient_accumulation_steps=bert_config.grad_accum_steps,
            fp16=use_fp16,
            lr_scheduler_type=bert_config.lr_scheduler_type,
            warmup_ratio=bert_config.warmup_ratio,
            warmup_steps=bert_config.warmup_steps,
        )
        
        data_collator = DataCollatorWithPadding(model_manager.tokenizer)
        loss_recorder = LossRecorder()
        
        # 早停回调
        callbacks = [loss_recorder]
        if ds_ev is not None and bert_config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=bert_config.early_stopping_patience)
            )
        
        # 创建Trainer
        trainer = Trainer(
            model=model_manager.model,
            args=training_args,
            train_dataset=ds_tr,
            eval_dataset=ds_ev,
            tokenizer=model_manager.tokenizer,
            data_collator=data_collator,
            compute_metrics=(lambda p: _compute_metrics(p, len(le.classes_))) if ds_ev is not None else None,
            callbacks=callbacks,
        )
        
        # 训练
        log_info("🚀 开始训练...")
        train_start = time.time()
        trainer.train()
        train_time = time.time() - train_start
        log_info(f"\n🎉 训练完成！")
        log_info(f"⏱️  总训练时间: {fmt_sec(train_time)}")
        
        # 评估
        eval_metrics = {}
        if ds_ev is not None:
            log_info("\n📊 开始评估...")
            eval_start = time.time()
            e_out = trainer.evaluate()
            eval_time = time.time() - eval_start
            log_info(f"✓ 评估完成，耗时 {fmt_sec(eval_time)}")
            
            # 从 e_out 读取 compute_metrics 的指标
            log_info("\n📈 评估指标:")
            for k in ["accuracy", "f1_weighted", "f1_macro", "hit@1", "hit@3", "hit@5", "hit@10"]:
                if k in e_out:
                    eval_metrics[k] = float(e_out[k])
                    log_info(f"  {k}: {eval_metrics[k]:.4f}")
            if "eval_loss" in e_out:
                eval_metrics["eval_loss"] = float(e_out["eval_loss"])
                log_info(f"  eval_loss: {eval_metrics['eval_loss']:.6f}")
        
        # 保存模型
        log_info("\n💾 正在保存模型...")
        
        if data_config.checkpoint_dir:
            model_dir = os.path.join(data_config.checkpoint_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            log_info(f"📁 保存模型到: {model_dir}")
            model_manager.save_model(model_dir)
        else:
            default_dir = os.path.join(
                data_config.modelsdir,
                os.path.splitext(os.path.basename(data_config.outmodel))[0] + "_model",
            )
            model_dir = bert_config.save_hf_dir or default_dir
            os.makedirs(model_dir, exist_ok=True)
            log_info(f"📁 保存模型到: {model_dir}")
            model_manager.save_model(model_dir)
        
        # 保存模型bundle
        model_manager.save_model_bundle(
            os.path.join(data_config.checkpoint_dir or data_config.modelsdir, data_config.outmodel),
            model_dir,
            model_type="bert",
            label_col=label_col,
            max_length=bert_config.max_length,
            fp16=bert_config.fp16,
        )
        
        # 保存评估指标
        if eval_metrics:
            output_dir = data_config.experiment_outdir or data_config.outdir
            metrics_path = os.path.join(output_dir, "metrics_eval.csv")
            pd.DataFrame([eval_metrics]).to_csv(metrics_path, index=False)
            log_info(f"📊 评估指标已保存到: {metrics_path}")
        
        total_time = time.time() - global_start
        log_info(f"\n🎉 {model_type.upper()} 训练完成！")
        log_info(f"⏱️  总耗时：{fmt_sec(total_time)}")
        log_info(f"📁 模型目录：{model_dir}")
        log_info(f"📦 模型文件：{os.path.join(data_config.checkpoint_dir or data_config.modelsdir, data_config.outmodel)}")
        
    else:
        # TF-IDF模型训练
        if not _HAS_SKLEARN:
            raise RuntimeError("TF-IDF模型训练需要安装scikit-learn库")
        
        # 创建TF-IDF特征提取器
        log_info("🔤 创建TF-IDF特征提取器...")
        vectorizer = TfidfVectorizer(
            analyzer=getattr(args, 'tfidf_analyzer', 'char_wb'),
            ngram_range=(getattr(args, 'tfidf_ngram_min', 2), getattr(args, 'tfidf_ngram_max', 4)),
            max_features=getattr(args, 'tfidf_max_features', 100000),
            min_df=1,
            sublinear_tf=True
        )
        
        # 训练和测试数据
        X_train_vec = vectorizer.fit_transform(X_tr)
        X_test_vec = vectorizer.transform(X_ev_f) if len(X_ev_f) > 0 else None
        y_test = y_ev if len(y_ev_f) > 0 else None
        
        # 创建分类器
        log_info("🤖 创建分类器...")
        classifier = SGDClassifier(
            loss=getattr(args, 'loss', 'hinge'),
            penalty=getattr(args, 'penalty', 'l2'),
            alpha=getattr(args, 'alpha', 0.0001),
            max_iter=getattr(args, 'max_iter', 100),
            random_state=42,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            tol=1e-3
        )
        
        # 训练
        log_info("🚀 开始训练TF-IDF模型...")
        train_start = time.time()
        classifier.fit(X_train_vec, y_tr)
        train_time = time.time() - train_start
        log_info(f"\n🎉 TF-IDF训练完成！")
        log_info(f"⏱️  训练时间: {fmt_sec(train_time)}")
        
        # 评估
        eval_metrics = {}
        if X_test_vec is not None and y_test is not None:
            log_info("\n📊 开始评估TF-IDF模型...")
            eval_start = time.time()
            
            # 预测
            y_pred = classifier.predict(X_test_vec)
            y_proba = classifier.decision_function(X_test_vec)
            
            # 计算概率
            if y_proba.ndim == 1:
                e = np.exp(y_proba - np.max(y_proba))
                y_proba = e / e.sum(axis=1, keepdims=True)
            else:
                y_proba = np.exp(y_proba - np.max(y_proba, axis=1, keepdims=True))
                y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
            
            eval_time = time.time() - eval_start
            log_info(f"✓ 评估完成，耗时 {fmt_sec(eval_time)}")
            
            # 计算指标
            acc = accuracy_score(y_test, y_pred)
            f1w = f1_score(y_test, y_pred, average="weighted")
            f1m = f1_score(y_test, y_pred, average="macro")
            
            eval_metrics = {
                "accuracy": float(acc),
                "f1_weighted": float(f1w),
                "f1_macro": float(f1m),
                "hit@1": hit_at_k(y_test, y_proba, 1),
                "hit@3": hit_at_k(y_test, y_proba, 3),
                "hit@5": hit_at_k(y_test, y_proba, 5),
                "hit@10": hit_at_k(y_test, y_proba, 10),
            }
            
            log_info("\n📈 评估指标:")
            for k, v in eval_metrics.items():
                log_info(f"  {k}: {v:.4f}")
        
        # 保存模型
        log_info("\n💾 正在保存TF-IDF模型...")
        
        # 保存模型bundle
        model_bundle = {
            "model": classifier,
            "vectorizer": vectorizer,
            "label_encoder": le,
            "model_type": "tfidf",
            "label_col": label_col,
        }
        
        bundle_path = os.path.join(data_config.modelsdir, data_config.outmodel)
        joblib.dump(model_bundle, bundle_path)
        log_info(f"✓ TF-IDF模型已保存到: {bundle_path}")
        
        total_time = time.time() - global_start
        log_info(f"\n🎉 {model_type.upper()} 训练完成！")
        log_info(f"⏱️  总耗时：{fmt_sec(total_time)}")
        log_info(f"📦 模型文件：{bundle_path}")
    
    # 清理内存
    model_manager.clear_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 模型类型选择 - 默认为BERT以保持向后兼容性
    parser.add_argument("--model-type", type=str, default="bert", choices=["bert", "tfidf"], help="模型类型")
    
    # 数据参数
    parser.add_argument("--train-file", type=str, default="train.csv", help="训练集文件名")
    parser.add_argument("--eval-file", type=str, default="eval.csv", help="验证集文件名")
    parser.add_argument("--outdir", type=str, default="./output/2025_up_to_month_2", help="数据目录")
    parser.add_argument("--experiment-outdir", type=str, default=None, help="实验输出目录")
    parser.add_argument("--modelsdir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="checkpoint目录")
    parser.add_argument("--outmodel", type=str, default="bert_model.joblib", help="模型保存文件名")
    
    # BERT 参数
    parser.add_argument("--bert-model", type=str, default="./models/google-bert/bert-base-chinese", help="BERT模型名称或路径")
    parser.add_argument("--init-hf-dir", type=str, default=None, help="从本地 HF 目录初始化")
    parser.add_argument("--num-train-epochs", dest="num_train_epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", help="启用混合精度训练")
    parser.add_argument("--save-hf-dir", type=str, default=None, help="保存 Hugging Face 模型与分词器的目录")
    parser.add_argument("--allow-online", type=_str2bool, default=False, help="允许在线下载HF模型")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="早停耐心值")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine",
                       choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--ooc-tau-percentile", type=float, default=5.0)
    parser.add_argument("--ooc-temperature", type=float, default=20.0)
    parser.add_argument("--skip-train-stats", type=_str2bool, default=False)
    parser.add_argument("--post-train-stats-batch-size", type=int, default=16)
    parser.add_argument("--stats-on-cpu", type=_str2bool, default=False)
    parser.add_argument("--resample-method", type=str, default="none",
                       choices=["none", "ros", "smote", "smoteenn", "smotetomek"])
    
    # TF-IDF 参数
    parser.add_argument("--tfidf-analyzer", type=str, default="char_wb", choices=["char", "char_wb", "word"])
    parser.add_argument("--tfidf-ngram-min", type=int, default=2)
    parser.add_argument("--tfidf-ngram-max", type=int, default=4)
    parser.add_argument("--tfidf-max-features", type=int, default=100000)
    parser.add_argument("--loss", type=str, default="hinge", choices=["hinge", "log", "modified_huber", "squared_hinge", "perceptron"])
    parser.add_argument("--penalty", type=str, default="l2", choices=["l1", "l2", "elasticnet"])
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--calibrate", type=str, default="none", choices=["none", "sigmoid", "isotonic"])
    
    # 系统参数
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    main(args)