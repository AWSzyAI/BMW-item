#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用BERT模型替代TF-IDF的训练脚本，兼容原有train.py的接口和输出格式
"""

import os, json, argparse, warnings, time
import gc
import numpy as np
import pandas as pd
# 可选依赖：matplotlib
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, f1_score
import joblib
from tqdm import tqdm
from contextlib import nullcontext

warnings.filterwarnings("ignore")
from utils import ensure_single_label, build_text, hit_at_k, fmt_sec, _flex_read_csv

# 导入BERT相关组件
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

# 可选依赖：imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    _HAS_IMBLEARN = True
except Exception:
    SMOTE = SMOTEENN = SMOTETomek = RandomOverSampler = None
    _HAS_IMBLEARN = False


def _str2bool(v) -> bool:
    return str(v).lower() in {"1", "true", "t", "y", "yes"}


def _is_valid_local_hf_dir(path: str) -> bool:
    """检查本地Hugging Face模型目录是否有效"""
    if not os.path.isdir(path):
        return False
    needed = [
        os.path.join(path, "config.json"),
    ]
    has_tokenizer = any(
        os.path.exists(os.path.join(path, name))
        for name in ["tokenizer.json", "vocab.txt"]
    )
    if not has_tokenizer:
        return False
    return all(os.path.exists(p) for p in needed)


def _read_split_or_combined(base_dir: str, base_filename: str) -> pd.DataFrame:
    """优先读取 X/Y 分离文件；若不存在则回退到单表 CSV。

    约定：
    - base_filename 可为 train.csv / eval.csv 或 train_X.csv / eval_X.csv；
    - 若为单表名，将尝试在 base_dir 下寻找 <stem>_X.csv 与 <stem>_y.csv；
    - X 文件应包含文本特征列（如 case_title、performed_work 等），
      y 文件至少包含 'linked_items'（若为 'label'/'y' 会自动重命名）。
    """
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
        y = _flex_read_csv(base_dir, os.path.basename(y_path))

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


class BERTDataset(torch.utils.data.Dataset):
    """BERT数据集类"""
    def __init__(self, encodings: dict, labels: np.ndarray | None = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


class LossRecorder(TrainerCallback):
    """记录训练损失"""
    def __init__(self):
        self.losses: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # logging_strategy='epoch' 时，logs 内含 'loss'
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


class BERTModelWrapper:
    """BERT模型包装器，兼容sklearn接口，支持懒加载与分批推理以避免显存溢出"""
    def __init__(self, model_path, tokenizer, label_encoder, device='cpu'):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        # 统一为 torch.device
        try:
            self.device = torch.device(device) if not isinstance(device, torch.device) else device
        except Exception:
            self.device = torch.device('cpu')
        self.model = None

    def _ensure_model(self):
        """确保底层HF模型已加载至 self.model，并移动到 self.device。"""
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()

    def fit(self, X=None, y=None):
        """加载已训练的模型（与 sklearn 接口对齐）。"""
        self._ensure_model()
        return self

    def predict_proba_batched(self, texts, batch_size: int = 16, max_length: int = 256):
        """分批预测概率，自动在 CUDA/MPS/CPU 之间选择，并在 OOM 时回退。"""
        if isinstance(texts, str):
            texts = [texts]
        # 确保模型加载
        self._ensure_model()
        # 设备与尝试序列
        devs = []
        try:
            if torch.cuda.is_available():
                devs.append(torch.device('cuda'))
            if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                devs.append(torch.device('mps'))
        except Exception:
            pass
        devs.append(torch.device('cpu'))

        last_err = None
        for dev in devs:
            try:
                # 移动模型到目标设备
                try:
                    self.model.to(dev)
                    self.device = dev
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                # 逐步缩小 batch size
                for bs in [batch_size, max(1, batch_size // 2), max(1, batch_size // 4)]:
                    use_amp = (isinstance(dev, torch.device) and dev.type == 'cuda')
                    probs = _predict_proba_in_batches(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        texts=texts,
                        device=dev,
                        max_length=max_length,
                        batch_size=int(bs),
                        use_amp=use_amp,
                    )
                    return probs
            except Exception as e:
                last_err = e
                # OOM 或设备错误则继续尝试下一种组合
                continue
        # 若所有尝试均失败，则抛出最后的异常
        if last_err is not None:
            raise last_err
        # 极端兜底（理论上不会到此）
        return np.zeros((0, 0), dtype=np.float32)

    def predict_proba(self, texts):
        """预测概率（默认走分批推理，安全且稳健）。"""
        return self.predict_proba_batched(texts, batch_size=16, max_length=256)

    def predict(self, texts):
        """预测类别"""
        proba = self.predict_proba(texts)
        return self.label_encoder.inverse_transform(np.argmax(proba, axis=1))


def _predict_proba_in_batches(model, tokenizer, texts, device, max_length, batch_size=16, use_amp=False):
    """分批计算文本的类别概率，避免一次性占满显存"""
    if isinstance(texts, str):
        texts = [texts]
    model.eval()
    all_probs: list[torch.Tensor] = []
    amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if (use_amp and isinstance(device, torch.device) and device.type == 'cuda') else nullcontext()
    with torch.inference_mode(), amp_ctx:
        for i in range(0, len(texts), int(batch_size)):
            batch_texts = texts[i:i + int(batch_size)]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors='pt'
            )
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1).to('cpu')
            all_probs.append(probs)
            del enc, out
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    if not all_probs:
        return np.zeros((0, 0), dtype=np.float32)
    return torch.cat(all_probs, dim=0).numpy()


def main(args):
    global_start = time.time()
    print("=== BERT模型训练开始（兼容TF-IDF接口）===")

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.modelsdir, exist_ok=True)

    # 读取数据（使用与train.py相同的逻辑）
    print("📖 正在读取训练数据...")
    df_tr = _read_split_or_combined(args.outdir, args.train_file)
    print(f"✓ 训练数据读取完成: {df_tr.shape}")
    
    print("📖 正在读取评估数据...")
    df_ev = _read_split_or_combined(args.outdir, args.eval_file)
    print(f"✓ 评估数据读取完成: {df_ev.shape}")
    
    label_col = _choose_label_column(df_tr)
    print(f"✓ 选择标签列: {label_col}")

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

    # 如果某一标签下只有一个样本，那就把这个样本复制一份（极端少样本的兜底）
    print("🔍 检查稀有标签...")
    vc = df_tr[label_col].value_counts()
    rare_labels = vc[vc == 1].index.tolist()
    if rare_labels:
        rare_samples = df_tr[df_tr[label_col].isin(rare_labels)]
        df_tr = pd.concat([df_tr, rare_samples], ignore_index=True)
        print(f"⚠️  已复制 {len(rare_samples)} 个单样本类别，以平衡训练集。")
        # 更新文本与标签（复制后）
        X_tr = build_text(df_tr).tolist()
        y_tr_raw = df_tr[label_col].astype(str).tolist()
    else:
        print("✓ 无需复制稀有标签样本")

    print("🏷️  正在编码标签...")
    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr_raw)
    print(f"✓ 标签编码完成，共 {len(le.classes_)} 个类别")

    # 过滤 eval 中不在训练标签集的样本
    ev_mask = [lbl in set(le.classes_) for lbl in y_ev_raw]
    if not all(ev_mask):
        dropped = int(np.sum(~np.array(ev_mask)))
        print(f"[警告] eval 中有 {dropped} 条样本的标签未在训练集中出现（记为 not_in_train）")
    X_ev_f = [t for t, m in zip(X_ev, ev_mask) if m]
    y_ev_f = [l for l, m in zip(y_ev_raw, ev_mask) if m]
    y_ev = le.transform(y_ev_f) if len(y_ev_f) > 0 else np.array([])

    # ========== 类别不平衡处理 ==========
    resample_method = getattr(args, "resample_method", "none")
    if resample_method != "none":
        print(f"[Info] 启用不平衡采样：{resample_method}")
        
        # 简易随机过采样（无依赖回退）
        def _simple_ros(X, y):
            y = np.asarray(y)
            classes, counts = np.unique(y, return_counts=True)
            max_n = counts.max()
            idx_all = []
            rng = np.random.default_rng(42)
            for c in classes:
                idx_c = np.where(y == c)[0]
                if len(idx_c) == 0:
                    continue
                if len(idx_c) < max_n:
                    extra = rng.choice(idx_c, size=max_n - len(idx_c), replace=True)
                    idx_c = np.concatenate([idx_c, extra], axis=0)
                idx_all.append(idx_c)
            sel = np.concatenate(idx_all, axis=0)
            return [X[i] for i in sel], y[sel]
        
        sampler = None
        
        if resample_method == "smote":
            if _HAS_IMBLEARN and SMOTE is not None:
                sampler = SMOTE(random_state=42)
            else:
                print("[Warning] 未安装 imbalanced-learn 或导入失败，SMOTE 回退为随机过采样（简易实现）。")
                resample_method = "ros"
        
        if resample_method == "smoteenn":
            if _HAS_IMBLEARN and SMOTEENN is not None:
                sampler = SMOTEENN(random_state=42)
            else:
                print("[Warning] 未安装 imbalanced-learn 或导入失败，SMOTEENN 回退为随机过采样（简易实现）。")
                resample_method = "ros"
        
        if resample_method == "smotetomek":
            if _HAS_IMBLEARN and SMOTETomek is not None:
                sampler = SMOTETomek(random_state=42)
            else:
                print("[Warning] 未安装 imbalanced-learn 或导入失败，SMOTETomek 回退为随机过采样（简易实现）。")
                resample_method = "ros"
        
        if resample_method == "ros":
            if _HAS_IMBLEARN and RandomOverSampler is not None:
                sampler = RandomOverSampler(random_state=42)
            else:
                sampler = None  # 使用简易 ROS
        
        # 执行采样
        if sampler is not None:
            # 对于文本数据，我们需要先编码再采样
            temp_enc = tokenizer(X_tr, padding=False, truncation=True, max_length=512)
            # 将编码转换为numpy数组用于采样
            X_temp = np.array([np.array(ids) for ids in temp_enc["input_ids"]])
            X_temp, y_tr = sampler.fit_resample(X_temp.reshape(len(X_temp), -1), y_tr)
            # 重建文本列表（这里简化处理，实际中可能需要更复杂的处理）
            X_tr = [X_tr[i % len(X_tr)] for i in range(len(y_tr))]
        else:
            X_tr, y_tr = _simple_ros(X_tr, y_tr)
        
        print(f"[Info] 采样后训练集样本数: {len(X_tr)}")
    # ========== 不平衡处理结束 ==========

    # Tokenizer & Model
    # 支持从本地 models/ 目录加载：优先使用 --init-hf-dir；否则使用 --bert-model（可为本地目录或模型名）
    print("🤖 正在初始化BERT模型...")
    init_path = getattr(args, "init_hf_dir", None) or args.bert_model
    model_name = init_path
    is_local = os.path.isdir(init_path)
    allow_online = bool(getattr(args, "allow_online", False))

    print(f"📂 模型路径: {init_path}")
    print(f"🌐 使用本地模型: {is_local}")

    if is_local:
        print("🔍 验证本地模型目录...")
        if not _is_valid_local_hf_dir(init_path):
            # 尝试在其子目录中自动发现一个合法的HF模型目录（常见布局: ./models/<publisher>/<model_name>）
            discovered = None
            try:
                for root, dirs, files in os.walk(init_path):
                    # 只深入两层，避免扫描过多
                    depth = root[len(init_path):].count(os.sep)
                    if depth > 3:
                        continue
                    if "config.json" in files and ("tokenizer.json" in files or "vocab.txt" in files):
                        discovered = root
                        break
                if discovered:
                    print(f"[提示] 传入目录 {init_path} 不含模型文件，自动发现子目录: {discovered}")
                    init_path = discovered  # 替换为真实模型路径
                else:
                    raise RuntimeError(
                        f"本地模型目录不完整：{init_path}\n"
                        f"未找到包含 config.json 与 tokenizer.json/vocab.txt 的子目录。\n"
                        f"请使用 --bert-model 指向具体模型目录，例如: ./models/google-bert/bert-base-chinese"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"本地模型目录不完整：{init_path}\n自动发现子目录时发生错误: {e}\n"
                    f"请确保包含至少 config.json 与 tokenizer.json 或 vocab.txt。"
                )
        print("📥 加载本地分词器...")
        tokenizer = AutoTokenizer.from_pretrained(init_path, local_files_only=True)
    else:
        if not allow_online:
            raise RuntimeError(
                "未提供本地模型目录且禁用了联网下载。请使用以下其一：\n"
                "1) 先将模型离线下载到本地，并通过 --init-hf-dir 指向该目录；\n"
                "2) 运行时添加 --allow-online，并可设置 HF_ENDPOINT=https://hf-mirror.com 与清理代理环境以加速与避免解析失败。"
            )
        print("📥 从在线加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(init_path, local_files_only=False)
    
    num_labels = len(le.classes_)
    # 设备与混精度：仅在 CUDA 可用时启用 fp16
    device = (
        "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    )
    use_fp16 = bool(args.fp16) and device == "cuda"
    
    print(f"💻 使用设备: {device}")
    print(f"🏷️  标签数量: {num_labels}")
    print(f"⚡ 混合精度: {use_fp16}")

    # 若 init_path 分类头维度与当前任务标签数不一致，使用 ignore_mismatched_sizes 自动重建分类头
    print("🏗️  正在加载BERT模型...")
    if is_local:
        model = AutoModelForSequenceClassification.from_pretrained(
            init_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            init_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            local_files_only=False,
        )
    
    print("📤 正在将模型移动到设备...")
    model.to(device)
    print("✓ 模型加载完成")

    # 编码
    def _tokenize(batch_texts: list[str]):
        return tokenizer(
            batch_texts,
            padding=False,
            truncation=True,
            max_length=int(args.max_length),
        )

    print("🔤 正在编码训练数据...")
    enc_tr = _tokenize(X_tr)
    print(f"✓ 训练数据编码完成: {len(enc_tr['input_ids'])} 样本")
    
    if len(X_ev_f) > 0:
        print("🔤 正在编码评估数据...")
        enc_ev = _tokenize(X_ev_f)
        print(f"✓ 评估数据编码完成: {len(enc_ev['input_ids'])} 样本")
    else:
        enc_ev = _tokenize(["dummy"])  # 保证不为空
        print("⚠️  评估数据为空，使用虚拟数据")
    
    # 保持为 list-of-ids，由 DataCollatorWithPadding 在 batch 维度做 padding
    ds_tr = BERTDataset(dict(enc_tr), labels=np.asarray(y_tr))
    ds_ev = (BERTDataset(dict(enc_ev), labels=np.asarray(y_ev)) if len(X_ev_f) > 0 and len(y_ev) > 0 else None)

    run_dir = os.path.join(args.modelsdir, os.path.splitext(os.path.basename(args.outmodel))[0] + "_bert_runs")
    os.makedirs(run_dir, exist_ok=True)
    print(f"📁 运行目录: {run_dir}")

    # 训练参数
    training_args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=int(args.train_batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        learning_rate=float(args.learning_rate),
        num_train_epochs=float(args.num_train_epochs),
        weight_decay=float(args.weight_decay),
        eval_strategy="epoch" if ds_ev is not None else "no",  # 使用eval_strategy而不是evaluation_strategy
        logging_strategy="epoch",
        save_strategy="epoch" if ds_ev is not None else "no",
        save_total_limit=1,
        report_to=[],
        load_best_model_at_end=bool(ds_ev is not None),
        metric_for_best_model="eval_loss" if ds_ev is not None else None,
        greater_is_better=False,
        remove_unused_columns=False,
        gradient_accumulation_steps=int(args.grad_accum_steps),
        fp16=use_fp16,
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    loss_recorder = LossRecorder()
    
    # 早停回调
    callbacks = [loss_recorder]
    if ds_ev is not None and getattr(args, "early_stopping_patience", 0) > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=int(args.early_stopping_patience))
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tr,
        eval_dataset=ds_ev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(lambda p: _compute_metrics(p, num_labels)) if ds_ev is not None else None,
        callbacks=callbacks,
    )

    # 训练
    print("🚀 开始训练...")
    train_start = time.time()
    
    # 添加训练进度回调
    class TrainingProgressCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.epoch_start_time = None
            self.epochs_times = []
        
        def on_epoch_begin(self, args, state, control, **kwargs):
            current_epoch = int(state.epoch) + 1
            total_epochs = int(args.num_train_epochs)
            self.epoch_start_time = time.time()
            
            # 计算剩余时间估算
            if len(self.epochs_times) > 0:
                avg_epoch_time = sum(self.epochs_times) / len(self.epochs_times)
                remaining_epochs = total_epochs - current_epoch + 1
                eta_seconds = avg_epoch_time * remaining_epochs
                eta_str = fmt_sec(eta_seconds)
            else:
                eta_str = "计算中..."
            
            # 创建进度条
            progress = int((current_epoch - 1) / total_epochs * 30)
            progress_bar = "█" * progress + "░" * (30 - progress)
            
            print(f"\n📊 Epoch {current_epoch}/{total_epochs} [{progress_bar}] ETA: {eta_str}")
            print(f"   开始时间: {time.strftime('%H:%M:%S', time.localtime(self.epoch_start_time))}")
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            
            # 显示训练进度
            if state.is_world_process_zero and "loss" in logs:
                current_epoch = int(state.epoch) + 1
                total_epochs = int(args.num_train_epochs)
                loss = float(logs["loss"])
                
                # 计算当前epoch内的进度
                if hasattr(state, 'global_step') and hasattr(args, 'max_steps'):
                    # 尝试获取总步数（如果可用）
                    try:
                        current_step = state.global_step
                        steps_per_epoch = args.max_steps / total_epochs
                        step_in_epoch = current_step % steps_per_epoch
                        epoch_progress = step_in_epoch / steps_per_epoch
                        
                        # 创建更细粒度的进度条
                        progress = int(epoch_progress * 20)
                        mini_bar = "█" * progress + "░" * (20 - progress)
                        
                        print(f"\r   训练进度: {mini_bar} {epoch_progress*100:.1f}% | 损失: {loss:.6f}", end="", flush=True)
                    except:
                        print(f"\r   训练损失: {loss:.6f}", end="", flush=True)
                else:
                    print(f"\r   训练损失: {loss:.6f}", end="", flush=True)
        
        def on_epoch_end(self, args, state, control, **kwargs):
            current_epoch = int(state.epoch)
            total_epochs = int(args.num_train_epochs)
            
            # 记录epoch时间
            if self.epoch_start_time is not None:
                epoch_time = time.time() - self.epoch_start_time
                self.epochs_times.append(epoch_time)
            
            # 计算剩余时间
            if len(self.epochs_times) > 0:
                avg_epoch_time = sum(self.epochs_times) / len(self.epochs_times)
                remaining_epochs = total_epochs - current_epoch
                eta_seconds = avg_epoch_time * remaining_epochs
                eta_str = fmt_sec(eta_seconds)
            else:
                eta_str = "计算中..."
            
            print()  # 换行
            
            # 显示epoch总结
            if hasattr(state, 'log_history') and state.log_history:
                # 找到当前epoch的日志
                epoch_logs = [log for log in state.log_history if log.get('epoch') == current_epoch - 1]
                if epoch_logs:
                    last_log = epoch_logs[-1]
                    if 'train_loss' in last_log:
                        train_loss = last_log['train_loss']
                        print(f"   训练损失: {train_loss:.6f}")
                    if 'eval_loss' in last_log:
                        eval_loss = last_log['eval_loss']
                        print(f"   验证损失: {eval_loss:.6f}")
                    if 'eval_accuracy' in last_log:
                        eval_acc = last_log['eval_accuracy']
                        print(f"   验证准确率: {eval_acc:.4f}")
                    if 'eval_f1_macro' in last_log:
                        eval_f1 = last_log['eval_f1_macro']
                        print(f"   验证F1-macro: {eval_f1:.4f}")
            
            # 显示时间信息
            if self.epoch_start_time is not None:
                epoch_time = time.time() - self.epoch_start_time
                print(f"   Epoch耗时: {fmt_sec(epoch_time)}")
            
            print(f"   剩余时间: {eta_str}")
            
            # 创建总体进度条
            overall_progress = int(current_epoch / total_epochs * 30)
            overall_bar = "█" * overall_progress + "░" * (30 - overall_progress)
            print(f"   总体进度: [{overall_bar}] {current_epoch}/{total_epochs} ({current_epoch/total_epochs*100:.1f}%)")
    
    # 添加进度回调到现有回调列表
    progress_callback = TrainingProgressCallback()
    trainer.add_callback(progress_callback)
    
    # 添加批次进度回调（可选）
    class BatchProgressCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.last_log_time = time.time()
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            
            # 限制日志频率，避免输出过多
            current_time = time.time()
            if current_time - self.last_log_time < 10:  # 每10秒最多输出一次
                return
            
            if state.is_world_process_zero and "loss" in logs:
                current_epoch = int(state.epoch) + 1
                total_epochs = int(args.num_train_epochs)
                loss = float(logs["loss"])
                
                # 显示简化的进度信息
                print(f"\r🔄 Epoch {current_epoch}/{total_epochs} | 损失: {loss:.6f} | 时间: {time.strftime('%H:%M:%S')}", end="", flush=True)
                self.last_log_time = current_time
    
    # 添加批次进度回调（可选，注释掉以避免过多输出）
    # batch_callback = BatchProgressCallback()
    # trainer.add_callback(batch_callback)
    
    trainer.train()
    train_time = time.time() - train_start
    print(f"\n🎉 训练完成！")
    print(f"⏱️  总训练时间: {fmt_sec(train_time)}")
    print(f"📊 平均每epoch时间: {fmt_sec(train_time / int(args.num_train_epochs))}")

    # 评估
    eval_metrics = {}
    if ds_ev is not None:
        print("\n📊 开始评估...")
        eval_start = time.time()
        e_out = trainer.evaluate()
        eval_time = time.time() - eval_start
        print(f"✓ 评估完成，耗时 {fmt_sec(eval_time)}")
        
        # 从 e_out 读取 compute_metrics 的指标
        print("\n📈 评估指标:")
        for k in ["accuracy", "f1_weighted", "f1_macro", "hit@1", "hit@3", "hit@5", "hit@10"]:
            if k in e_out:
                eval_metrics[k] = float(e_out[k])
                print(f"  {k}: {eval_metrics[k]:.4f}")
        # 兜底：若 compute_metrics 未注册，至少输出 loss
        if "eval_loss" in e_out:
            eval_metrics["eval_loss"] = float(e_out["eval_loss"])
            print(f"  eval_loss: {eval_metrics['eval_loss']:.6f}")
    else:
        print("⚠️  [提示] eval 集为空或无可评估样本，跳过评估。")

    # 训练集/验证集概率（分批），用于 OOD MSP 阈值
    id_pmax_for_stats = None
    try:
        if not getattr(args, 'skip_train_stats', False):
            print("\n📊 正在计算训练后统计（分批）...")
            stats_device = torch.device('cpu') if getattr(args, 'stats_on_cpu', False) else torch.device(device)
            moved_to_cpu = False
            if getattr(args, 'stats_on_cpu', False) and (isinstance(device, str) and device != 'cpu' or isinstance(device, torch.device) and device.type != 'cpu'):
                model.to('cpu')
                moved_to_cpu = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            use_amp_stats = bool(getattr(args, 'fp16', False)) and (hasattr(stats_device, 'type') and stats_device.type == 'cuda')
            probs_tr = _predict_proba_in_batches(
                model=model,
                tokenizer=tokenizer,
                texts=X_tr,
                device=stats_device,
                max_length=int(args.max_length),
                batch_size=int(getattr(args, 'post_train_stats_batch_size', 16)),
                use_amp=use_amp_stats,
            )
            id_pmax_for_stats = probs_tr.max(axis=1)
            if moved_to_cpu:
                model.to(device)
        else:
            # 退化方案：使用 eval 的分布估计阈值；若 eval 为空则使用固定阈值
            if len(X_ev_f) > 0:
                print("\n📊 跳过训练集统计，改用评估集分布估计阈值（分批）...")
                stats_device = torch.device('cpu') if getattr(args, 'stats_on_cpu', False) else torch.device(device)
                moved_to_cpu = False
                if getattr(args, 'stats_on_cpu', False) and (isinstance(device, str) and device != 'cpu' or isinstance(device, torch.device) and device.type != 'cpu'):
                    model.to('cpu')
                    moved_to_cpu = True
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                use_amp_stats = bool(getattr(args, 'fp16', False)) and (hasattr(stats_device, 'type') and stats_device.type == 'cuda')
                probs_ev = _predict_proba_in_batches(
                    model=model,
                    tokenizer=tokenizer,
                    texts=X_ev_f,
                    device=stats_device,
                    max_length=int(args.max_length),
                    batch_size=int(getattr(args, 'post_train_stats_batch_size', 16)),
                    use_amp=use_amp_stats,
                )
                id_pmax_for_stats = probs_ev.max(axis=1)
                if moved_to_cpu:
                    model.to(device)
            else:
                print("\n⚠️  跳过统计且评估集为空，使用默认阈值 0.1。")
                id_pmax_for_stats = np.array([0.1])
    finally:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    tau = float(np.percentile(id_pmax_for_stats, getattr(args, "ooc_tau_percentile", 5.0)))
    temperature = float(getattr(args, "ooc_temperature", 20.0))
    ooc_detector = {"kind": "threshold", "tau": tau, "temperature": temperature}

    # 保存 loss 曲线与数据
    loss_curve = loss_recorder.losses
    if _HAS_MATPLOTLIB:
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(loss_curve, label="train")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("BERT Training Loss Curve")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(args.outdir, "loss_curve.png"))
            plt.close()
        except Exception:
            pass
    else:
        print("[提示] 未安装matplotlib，跳过损失曲线图保存")
    with open(os.path.join(args.outdir, "loss_data.json"), "w", encoding="utf-8") as f:
        json.dump({"losses": list(map(float, loss_curve))}, f, ensure_ascii=False, indent=2)
    
    # 写入训练日志
    log_dir = os.path.join(os.getcwd(), "log")
    os.makedirs(log_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.outmodel))[0]
    train_log_path = os.path.join(log_dir, f"{stem}_train.txt")
    try:
        with open(train_log_path, "w", encoding="utf-8") as f:
            for epoch, loss in enumerate(loss_curve, 1):
                msg = f"epoch={epoch}/{len(loss_curve)} loss={loss:.6f}"
                f.write(msg + "\n")
    except Exception:
        pass

    # 保存模型与标签编码器（可自定义目录）
    print("\n💾 正在保存模型...")
    default_dir = os.path.join(
        args.modelsdir,
        os.path.splitext(os.path.basename(args.outmodel))[0] + "_bert",
    )
    model_dir = args.save_hf_dir if getattr(args, "save_hf_dir", None) else default_dir
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"📁 保存模型到: {model_dir}")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("✓ 模型和分词器保存完成")

    # 创建兼容sklearn接口的模型包装器
    print("🔧 正在创建模型包装器...")
    bert_wrapper = BERTModelWrapper(model_dir, tokenizer, le, device)
    
    # 保存模型bundle（兼容原有格式）
    bundle = {
        "model": bert_wrapper,  # 使用包装器而不是Pipeline
        "model_type": "bert",
        "model_dir": model_dir,
        "tokenizer": model_name,
        "label_encoder": le,
        "label_col": label_col,
        "ooc_detector": ooc_detector,
    }
    
    bundle_path = os.path.join(args.modelsdir, args.outmodel)
    print(f"💾 正在保存模型bundle到: {bundle_path}")
    # 使用 joblib 保存 bundle（与 TF-IDF 的保存路径对齐）
    joblib.dump(bundle, bundle_path)
    print("✓ 模型bundle保存完成")

    # 保存评估指标
    if eval_metrics:
        metrics_path = os.path.join(args.outdir, "metrics_eval.csv")
        pd.DataFrame([eval_metrics]).to_csv(metrics_path, index=False)
        print(f"📊 评估指标已保存到: {metrics_path}")

    total_time = time.time() - global_start
    print(f"\n🎉 BERT 训练完成！")
    print(f"⏱️  总耗时：{fmt_sec(total_time)}")
    print(f"📁 模型目录：{model_dir}")
    print(f"📦 模型文件：{bundle_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, default="train.csv", help="训练集文件名（默认从 outdir 读取）")
    parser.add_argument("--eval-file", type=str, default="eval.csv", help="验证集文件名（默认从 outdir 读取）")
    parser.add_argument("--outdir", type=str, default="./output/2025_up_to_month_2", help="输出目录（读取数据与保存训练曲线/指标）")
    parser.add_argument("--modelsdir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--outmodel", type=str, default="9.joblib", help="模型保存文件名")
    
    # BERT 参数
    parser.add_argument("--bert-model", type=str, default="./models", help="BERT模型名称或路径")
    parser.add_argument("--init-hf-dir", type=str, default=None, help="从本地 HF 目录初始化（覆盖 --bert-model），支持继续微调")
    parser.add_argument("--num-train-epochs", dest="num_train_epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", help="启用混合精度训练（仅CUDA）")
    parser.add_argument("--save-hf-dir", type=str, default=None, help="保存 Hugging Face 模型与分词器的目录（默认 models/<stem>_bert）")
    # 在线/离线
    parser.add_argument("--allow-online", type=_str2bool, default=False, help="允许在线下载HF模型（True/False）")
    
    # 早停参数
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="早停耐心值（若连续 N 个 epoch 未提升则停止）")
    
    # OOD/MSP
    parser.add_argument("--ooc-tau-percentile", type=float, default=5.0, help="无 OOD 正样本时，p_max 的分位数阈值（百分位）")
    parser.add_argument("--ooc-temperature", type=float, default=20.0, help="将 (tau - p_max) 经 sigmoid 映射为概率的温度系数")
    # 训练后统计
    parser.add_argument("--skip-train-stats", type=_str2bool, default=False, help="训练后跳过对训练集整表概率/统计的计算（True/False）")
    parser.add_argument("--post-train-stats-batch-size", type=int, default=16, help="训练后计算概率的batch size")
    parser.add_argument("--stats-on-cpu", type=_str2bool, default=False, help="训练后统计阶段在CPU上执行（True/False）")
    
    # 不平衡处理参数
    parser.add_argument(
        "--resample-method",
        type=str,
        default="none",
        choices=["none", "ros", "smote", "smoteenn", "smotetomek"],
        help="不平衡处理方法：none/ros/smote/smoteenn/smotetomek",
    )
    
    args = parser.parse_args()
    main(args)