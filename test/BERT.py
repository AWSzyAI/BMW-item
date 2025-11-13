from __future__ import annotations

import os, json, time, argparse, warnings
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    EarlyStoppingCallback,
)

from utils import ensure_single_label, build_text, hit_at_k, fmt_sec, _flex_read_csv

warnings.filterwarnings("ignore")

# 可选依赖：imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    _HAS_IMBLEARN = True
except Exception:
    SMOTE = SMOTEENN = SMOTETomek = RandomOverSampler = None
    _HAS_IMBLEARN = False


# # Load model directly
# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("/home/szy/bmw/2025-10-27/BMW-item/models/bert-base-chinese")
# model = AutoModelForMaskedLM.from_pretrained("/home/szy/bmw/2025-10-27/BMW-item/models/bert-base-chinese")

def _is_valid_local_hf_dir(path: str) -> bool:
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
    # 优先级：extend_id > linked_items > item_title
    for col in ["extend_id", "linked_items", "item_title"]:
        if col in df.columns:
            return col
    raise KeyError("未找到可用标签列（extend_id/linked_items/item_title）")


@dataclass
class HFInputExample:
    text: str
    label: int


class NPDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict[str, Any], labels: np.ndarray | None = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self) -> int:
        # encodings 中的各字段为 list（可变长序列），此处使用样本数计数
        return len(self.encodings["input_ids"])


class LossRecorder(TrainerCallback):
    def __init__(self):
        self.losses: List[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # logging_strategy='epoch' 时，logs 内含 'loss'
        if state.is_world_process_zero and ("loss" in logs):
            try:
                self.losses.append(float(logs["loss"]))
            except Exception:
                pass


def _compute_metrics(eval_pred, num_labels: int) -> Dict[str, float]:
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


def main(args: argparse.Namespace) -> None:
    t0_global = time.time()
    print("=== BERT模型训练开始（使用 train.csv / eval.csv）===")
    
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.modelsdir, exist_ok=True)

    # 读取数据（使用与train.py相同的逻辑）
    df_tr = _read_split_or_combined(args.outdir, args.train_file)
    df_ev = _read_split_or_combined(args.outdir, args.eval_file)
    label_col = _choose_label_column(df_tr)

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
    vc = df_tr[label_col].value_counts()
    rare_labels = vc[vc == 1].index.tolist()
    if rare_labels:
        rare_samples = df_tr[df_tr[label_col].isin(rare_labels)]
        df_tr = pd.concat([df_tr, rare_samples], ignore_index=True)
        print(f"已复制 {len(rare_samples)} 个单样本类别，以平衡训练集。")
        # 更新文本与标签（复制后）
        X_tr = build_text(df_tr).tolist()
        y_tr_raw = df_tr[label_col].astype(str).tolist()

    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr_raw)

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
    init_path = getattr(args, "init_hf_dir", None) or args.bert_model
    model_name = init_path  # 修复未定义的model_name变量
    is_local = os.path.isdir(init_path)
    allow_online = bool(getattr(args, "allow_online", False))

    if is_local:
        if not _is_valid_local_hf_dir(init_path):
            raise RuntimeError(
                f"本地模型目录不完整：{init_path}\n"
                f"请确保包含至少 config.json 与 tokenizer.json 或 vocab.txt。\n"
                f"可以先用 huggingface_hub.snapshot_download 下载到本地再指定 --init-hf-dir。"
            )
        tokenizer = AutoTokenizer.from_pretrained(init_path, local_files_only=True)
    else:
        if not allow_online:
            raise RuntimeError(
                "未提供本地模型目录且禁用了联网下载。请使用以下其一：\n"
                "1) 先将模型离线下载到本地，并通过 --init-hf-dir 指向该目录；\n"
                "2) 运行时添加 --allow-online，并可设置 HF_ENDPOINT=https://hf-mirror.com 与清理代理环境以加速与避免解析失败。"
            )
        tokenizer = AutoTokenizer.from_pretrained(init_path, local_files_only=False)
    num_labels = len(le.classes_)
    # 设备与混精度：仅在 CUDA 可用时启用 fp16
    device = (
        "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    )
    use_fp16 = bool(args.fp16) and device == "cuda"

    # 若 init_path 分类头维度与当前任务标签数不一致，使用 ignore_mismatched_sizes 自动重建分类头
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
    model.to(device)

    # 编码
    def _tokenize(batch_texts: List[str]):
        return tokenizer(
            batch_texts,
            padding=False,
            truncation=True,
            max_length=int(args.max_length),
        )

    enc_tr = _tokenize(X_tr)
    enc_ev = _tokenize(X_ev_f) if len(X_ev_f) > 0 else _tokenize(["dummy"])  # 保证不为空
    # 保持为 list-of-ids，由 DataCollatorWithPadding 在 batch 维度做 padding
    ds_tr = NPDataset(dict(enc_tr), labels=np.asarray(y_tr))
    ds_ev = (NPDataset(dict(enc_ev), labels=np.asarray(y_ev)) if len(X_ev_f) > 0 and len(y_ev) > 0 else None)

    run_dir = os.path.join(args.modelsdir, os.path.splitext(os.path.basename(args.outmodel))[0] + "_bert_runs")
    os.makedirs(run_dir, exist_ok=True)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=int(args.train_batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        learning_rate=float(args.learning_rate),
        num_train_epochs=float(args.num_train_epochs),
        weight_decay=float(args.weight_decay),
        evaluation_strategy="epoch" if ds_ev is not None else "no",
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
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    print(f"训练完成，耗时 {fmt_sec(train_time)}")

    # 评估
    eval_metrics = {}
    if ds_ev is not None:
        eval_start = time.time()
        e_out = trainer.evaluate()
        eval_time = time.time() - eval_start
        print(f"评估完成，耗时 {fmt_sec(eval_time)}")
        
        # 从 e_out 读取 compute_metrics 的指标
        for k in ["accuracy", "f1_weighted", "f1_macro", "hit@1", "hit@3", "hit@5", "hit@10"]:
            if k in e_out:
                eval_metrics[k] = float(e_out[k])
        # 兜底：若 compute_metrics 未注册，至少输出 loss
        if "eval_loss" in e_out:
            eval_metrics["eval_loss"] = float(e_out["eval_loss"])
        
        print(f"Eval metrics: {eval_metrics}")
    else:
        print("[提示] eval 集为空或无可评估样本，跳过评估。")

    # 训练集/验证集概率，用于 OOD MSP 阈值
    def _proba_for_texts(texts: List[str]) -> np.ndarray:
        enc = tokenizer(texts, padding=True, truncation=True, max_length=int(args.max_length), return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        model.eval()
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits.detach().cpu().numpy()
            e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

    # OOD 检测：与 TF-IDF 逻辑一致，阈值基于 train 分布（或 eval 中 ID 样本）
    id_pmax_for_stats = _proba_for_texts(X_tr).max(axis=1)
    tau = float(np.percentile(id_pmax_for_stats, getattr(args, "ooc_tau_percentile", 5.0)))
    temperature = float(getattr(args, "ooc_temperature", 20.0))
    ooc_detector = {"kind": "threshold", "tau": tau, "temperature": temperature}

    # 保存 loss 曲线与数据
    loss_curve = loss_recorder.losses
    try:
        import matplotlib.pyplot as plt

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
    default_dir = os.path.join(
        args.modelsdir,
        os.path.splitext(os.path.basename(args.outmodel))[0] + "_bert",
    )
    model_dir = args.save_hf_dir if getattr(args, "save_hf_dir", None) else default_dir
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    bundle = {
        "model_type": "bert",
        "model_dir": model_dir,
        "tokenizer": model_name,
        "label_encoder": le,
        "label_col": label_col,
        "ooc_detector": ooc_detector,
    }
    # 使用 joblib 保存 bundle（与 TF-IDF 的保存路径对齐）
    import joblib

    joblib.dump(bundle, os.path.join(args.modelsdir, args.outmodel))

    # 保存评估指标
    if eval_metrics:
        pd.DataFrame([eval_metrics]).to_csv(os.path.join(args.outdir, "metrics_eval.csv"), index=False)

    print(f"BERT 训练完成，总耗时：{fmt_sec(time.time() - t0_global)} | 模型目录：{model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, default="train.csv", help="训练集文件名（默认从 outdir 读取）")
    parser.add_argument("--eval-file", type=str, default="eval.csv", help="验证集文件名（默认从 outdir 读取）")
    parser.add_argument("--outdir", type=str, default="./output/2025_up_to_month_2", help="输出目录（读取数据与保存训练曲线/指标）")
    parser.add_argument("--modelsdir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--outmodel", type=str, default="9.joblib", help="模型保存文件名")
    
    # BERT 参数
    parser.add_argument("--bert-model", type=str, default="bert-base-chinese", help="BERT模型名称或路径")
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
    parser.add_argument("--allow-online", action="store_true", help="允许在线下载模型")
    
    # 早停参数
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="早停耐心值（若连续 N 个 epoch 未提升则停止）")
    
    # OOD/MSP
    parser.add_argument("--ooc-tau-percentile", type=float, default=5.0, help="无 OOD 正样本时，p_max 的分位数阈值（百分位）")
    parser.add_argument("--ooc-temperature", type=float, default=20.0, help="将 (tau - p_max) 经 sigmoid 映射为概率的温度系数")
    
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
