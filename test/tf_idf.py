"""TF-IDF 训练封装。

复用现有的 src/train.py 实现，暴露 train_tfidf(args) 供 orchestrator 调用。
"""
from __future__ import annotations

import types
from typing import Any


def train_tfidf(args: Any) -> None:
    # 延迟导入，避免作为模块导入时触发 CLI 解析
    import src.train as train_impl  # type: ignore

    # 直接复用现有实现
    train_impl.main(args)
