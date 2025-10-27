from __future__ import annotations

import os, json, time, argparse
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
)

from utils import ensure_single_label, build_text, hit_at_k, fmt_sec, _flex_read_csv


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
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.modelsdir, exist_ok=True)

    # 读取数据
    df_tr = _flex_read_csv(args.outdir, args.train_file)
    df_ev = _flex_read_csv(args.outdir, args.eval_file)
    label_col = _choose_label_column(df_tr)

    # 清洗标签
    df_tr[label_col] = df_tr[label_col].apply(ensure_single_label).astype(str)
    df_ev[label_col] = df_ev[label_col].apply(ensure_single_label).astype(str)

    X_tr = build_text(df_tr).tolist()
    y_tr_raw = df_tr[label_col].astype(str).tolist()
    X_ev = build_text(df_ev).tolist()
    y_ev_raw = df_ev[label_col].astype(str).tolist()

    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr_raw)

    # 过滤 eval 中不在训练标签集的样本
    ev_mask = [lbl in set(le.classes_) for lbl in y_ev_raw]
    X_ev_f = [t for t, m in zip(X_ev, ev_mask) if m]
    y_ev_f = [l for l, m in zip(y_ev_raw, ev_mask) if m]
    y_ev = le.transform(y_ev_f) if len(y_ev_f) > 0 else np.array([])

    # Tokenizer & Model
    # 支持从本地 models/ 目录加载：优先使用 --init-hf-dir；否则使用 --bert-model（可为本地目录或模型名）
    init_path = getattr(args, "init_hf_dir", None) or args.bert_model
    # 纯离线友好：local_files_only=True（若本地不存在会报错，避免意外联网）
    tokenizer = AutoTokenizer.from_pretrained(init_path, local_files_only=True)
    num_labels = len(le.classes_)
    # 设备与混精度：仅在 CUDA 可用时启用 fp16
    device = (
        "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    )
    use_fp16 = bool(args.fp16) and device == "cuda"

    # 若 init_path 分类头维度与当前任务标签数不一致，使用 ignore_mismatched_sizes 自动重建分类头
    model = AutoModelForSequenceClassification.from_pretrained(
        init_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        local_files_only=True,
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
        save_strategy="no",
        report_to=[],
        load_best_model_at_end=False,
        remove_unused_columns=False,
        gradient_accumulation_steps=int(args.grad_accum_steps),
        fp16=use_fp16,
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    loss_recorder = LossRecorder()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tr,
        eval_dataset=ds_ev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(lambda p: _compute_metrics(p, num_labels)) if ds_ev is not None else None,
        callbacks=[loss_recorder],
    )

    # 训练
    trainer.train()

    # 评估
    eval_metrics = {}
    if ds_ev is not None:
        e_out = trainer.evaluate()
        # 从 e_out 读取 compute_metrics 的指标
        for k in ["accuracy", "f1_weighted", "f1_macro", "hit@1", "hit@3", "hit@5", "hit@10"]:
            if k in e_out:
                eval_metrics[k] = float(e_out[k])
        # 兜底：若 compute_metrics 未注册，至少输出 loss
        if "eval_loss" in e_out:
            eval_metrics["eval_loss"] = float(e_out["eval_loss"])

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
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--eval-file", type=str, default="eval.csv")
    parser.add_argument("--outdir", type=str, default="./output/2025_up_to_month_9")
    parser.add_argument("--modelsdir", type=str, default="./models")
    parser.add_argument("--outmodel", type=str, default="9.joblib")
    # BERT 参数
    parser.add_argument("--bert-model", type=str, default="/home/szy/bmw/2025-10-27/BMW-item/models/bert-base-chinese")
    parser.add_argument("--init-hf-dir", type=str, default=None, help="从本地 HF 目录初始化（覆盖 --bert-model），支持继续微调")
    parser.add_argument("--num-train-epochs", dest="num_train_epochs", type=float, default=2.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save-hf-dir", type=str, default=None, help="保存 Hugging Face 模型与分词器的目录（默认 models/<stem>_bert）")
    # OOD/MSP
    parser.add_argument("--ooc-tau-percentile", type=float, default=5.0)
    parser.add_argument("--ooc-temperature", type=float, default=20.0)
    args = parser.parse_args()
    main(args)
