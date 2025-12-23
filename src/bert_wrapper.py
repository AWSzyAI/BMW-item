"""Utility wrapper around a fine-tuned HuggingFace classifier used in train/predict flows."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BERTModelWrapper:
    """Thin helper exposing predict/predict_proba for joblib bundles."""

    def __init__(
        self,
        model_dir: str,
        label_encoder,
        *,
        device: str | None = None,
        max_length: int = 256,
        fp16: bool = False,
    ) -> None:
        self.model_dir = model_dir
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.fp16 = fp16

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()

    def _batchify(self, texts: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    def predict_proba(self, texts: Sequence[str] | str, batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, len(self.label_encoder.classes_)), dtype=np.float32)

        use_fp16 = self.fp16 and self.device.type == "cuda"
        probs: List[np.ndarray] = []

        with torch.inference_mode():
            for batch in self._batchify(texts, batch_size):
                encoded = self.tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                amp_ctx = torch.cuda.amp.autocast() if use_fp16 else nullcontext()
                with amp_ctx:
                    logits = self.model(**encoded).logits
                logits = logits.float()
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

        return np.vstack(probs)

    def predict(self, texts: Sequence[str] | str, batch_size: int = 32) -> List[str]:
        proba = self.predict_proba(texts, batch_size=batch_size)
        if proba.size == 0:
            return []
        pred_idx = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(pred_idx)
