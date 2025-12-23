#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多次调用 predict.py 单样本预测，统计 rerank 前后 hit@k 的 ↑/↓/– 情况。

目标：在“行为尽量等同于反复执行 make predict”的前提下，自动化跑多次样本，
汇总 Qwen rerank 对 hit@1/3/5/10 的影响分布。

实现思路：
- 直接复用 `predict.py` 中的模型加载与 `_predict_single_example` 逻辑；
- 在同一进程内多次随机抽样不同样本进行预测（index<0 时的行为），保持与交互模式一致；
- 对每次预测中打印的 base/new hit@k 进行重新计算并汇总统计；
- 可通过 `--runs` 控制重复次数，通过 `--seed` 控制样本抽样的可复现性。
"""

import argparse
import random
from pathlib import Path

import numpy as np

import predict as _predict_mod
from utils import build_text, ensure_single_label


def _load_model_bundle(modelsdir: str, model_name: str):
    """复用 predict.main 中的模型 / label_encoder / ooc_detector 加载逻辑。

    这里不直接调用 main，而是仿照其实现手动完成加载，避免重复 CLI 解析。
    """

    import os, joblib
    from bert_wrapper import BERTModelWrapper

    bundle_path = os.path.join(modelsdir, model_name)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Model not found: {bundle_path}")

    bundle = joblib.load(bundle_path)
    model_type = bundle.get("model_type", None)
    le = _predict_mod._extract_label_encoder(bundle)
    if le is None:
        raise ValueError("模型 bundle 缺少标签编码信息")
    ooc_detector = bundle.get("ooc_detector")

    if model_type == "bert":
        model_dir = bundle.get("model_dir")
        if not model_dir:
            raise ValueError("BERT bundle 缺少 model_dir")
        model = BERTModelWrapper(
            model_dir,
            le,
            max_length=bundle.get("max_length", 256),
            fp16=bundle.get("fp16", False),
        )
    else:
        model = bundle.get("model")
        if model is None:
            raise ValueError("模型 bundle 缺少可用的 model 对象")

    return model, le, ooc_detector, bundle_path, bundle


def _prepare_label_info(outdir: str, le, bundle) -> tuple[dict, dict]:
    """复用 predict.py 中的标签映射构建逻辑。"""

    label_mapping, raw_to_label_id = _predict_mod._load_label_mapping(outdir)
    test_data = _predict_mod._load_test_data(outdir)
    data_csv_mapping = _predict_mod._load_data_csv_mapping(outdir)

    # 合并 data.csv
    for linked_items, info in data_csv_mapping.items():
        if linked_items in label_mapping:
            existing = label_mapping[linked_items]
            for key in ("extern_id", "item_title", "itemcreationdate"):
                new_val = info.get(key, "")
                if new_val:
                    existing[key] = new_val
        else:
            label_mapping[linked_items] = info

    # 合并 test_raw/test
    for linked_items, info in test_data.items():
        if linked_items in label_mapping:
            existing = label_mapping[linked_items]
            for key in ("extern_id", "item_title", "itemcreationdate"):
                new_val = info.get(key, "")
                if new_val and existing.get(key, "") != new_val:
                    existing[key] = new_val
        else:
            label_mapping[linked_items] = info

    label_to_info = _predict_mod._create_label_to_info_mapping(le, label_mapping)
    return label_to_info, raw_to_label_id


def _load_eval_dataframe(outdir: str, infile: str):
    """按照 predict.main 的规则读取单表/切分表，并构造文本。"""

    import os
    import pandas as pd

    infile_path = infile if os.path.isabs(infile) else os.path.join(outdir, infile)
    df = _predict_mod._read_split_or_combined(outdir, infile_path)
    label_column = _predict_mod._detect_label_column(df, le=None)  # 这里只为保持列名处理一致
    if label_column and label_column in df.columns:
        df[label_column] = df[label_column].apply(ensure_single_label).astype(str)
    texts = build_text(df).tolist()
    if not texts:
        raise ValueError("输入文件为空或无法构造文本。")
    return df.reset_index(drop=True), texts


def _compute_hit_at_k_from_candidates(cands: list[dict], true_lbl: str | None, k: int) -> int:
    if true_lbl is None:
        return 0
    labels = [str(c.get("label", "")) for c in cands[:k]]
    return 1 if str(true_lbl) in labels else 0


def run_multi_eval(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1) 加载模型等，与 predict.main 一致
    model, le, ooc_detector, bundle_path, bundle = _load_model_bundle(
        args.modelsdir, args.model
    )
    label_to_info, raw_to_label_id = _prepare_label_info(args.outdir, le, bundle)

    # 2) 加载 eval 数据和文本
    df, texts = _load_eval_dataframe(args.outdir, args.infile)
    n_samples = len(texts)
    print(f"[MULTI] 可用样本数: {n_samples} (infile={args.infile})")

    # 预先算好所有样本的 probs，避免重复前向
    print("[MULTI] 预计算所有样本的概率分布……")
    probs_matrix = model.predict_proba(texts)

    # 准备 args for _predict_single_example
    class DummyArgs:
        pass

    dummy = DummyArgs()
    dummy.reject_threshold = args.reject_threshold
    dummy.ooc_decision_threshold = args.ooc_decision_threshold
    dummy.rerank_strategy = args.rerank_strategy
    dummy.rerank_seed = args.rerank_seed
    dummy.batch_disable_rerank = False

    # 统计结构
    stats = {1: {"up": 0, "down": 0, "same": 0},
             3: {"up": 0, "down": 0, "same": 0},
             5: {"up": 0, "down": 0, "same": 0},
             10: {"up": 0, "down": 0, "same": 0}}

    runs = args.runs
    indices = [random.randint(0, n_samples - 1) for _ in range(runs)]
    print(f"[MULTI] 即将执行 {runs} 次单样本预测 (随机索引)。")

    for t, idx in enumerate(indices, 1):
        text = texts[idx]
        month_value = ""  # 对 eval.csv 来说通常为空或来自列，这里不影响评估

        # 调用原有单样本预测逻辑（含 Qwen rerank + 打印详细信息）
        result = _predict_mod._predict_single_example(
            df,
            idx,
            text,
            month_value,
            model,
            le,
            ooc_detector,
            label_to_info,
            bundle_path,
            args.outdir,
            dummy,
            label_column=None,
            precomputed_probs=probs_matrix,
            rerank_enabled=True,
            verbose=True,
        )

        true_label = result["true_label"]
        base_cands = result["record"]["topk"]
        reranked_cands = result["reranked_candidates"]

        for k in (1, 3, 5, 10):
            base_hit = _compute_hit_at_k_from_candidates(base_cands, true_label, k)
            new_hit = _compute_hit_at_k_from_candidates(reranked_cands, true_label, k)
            if base_hit == new_hit:
                stats[k]["same"] += 1
            elif new_hit > base_hit:
                stats[k]["up"] += 1
            else:
                stats[k]["down"] += 1

        print(f"[MULTI] 完成第 {t}/{runs} 次预测 (idx={idx})\n")

    # 汇总结果
    print("\n[MULTI] 汇总统计 (基于单样本 hit@k 变化)：")
    for k in (1, 3, 5, 10):
        up = stats[k]["up"]
        down = stats[k]["down"]
        same = stats[k]["same"]
        total = up + down + same
        if total == 0:
            continue
        print(f"  hit@{k}:  ↑={up}  ↓={down}  –={same}  (↑={up/total:.3f}, ↓={down/total:.3f}, –={same/total:.3f})")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多次运行单样本 predict 并统计 rerank hit@k 变化")
    p.add_argument("--modelsdir", type=str, default="./models")
    p.add_argument("--model", type=str, default="7.joblib")
    p.add_argument("--outdir", type=str, default="./output/2025_up_to_month_0")
    p.add_argument("--infile", type=str, default="eval.csv", help="输入文件名（相对 outdir 或绝对路径）")
    p.add_argument("--runs", type=int, default=100, help="重复单样本预测次数")
    p.add_argument("--seed", type=int, default=42, help="随机种子，用于样本索引抽样")
    p.add_argument("--reject-threshold", type=float, default=None, help="MSP：最大类别概率 p_max 的拒判阈值")
    p.add_argument("--ooc-decision-threshold", type=float, default=0.5, help="LogReg 检测器下的概率阈值")
    p.add_argument("--rerank-strategy", type=str, default="qwen2", choices=["random", "beacon", "qwen2"], help="Rerank 策略，默认使用 qwen2")
    p.add_argument("--rerank-seed", type=int, default=42, help="Rerank 随机种子（若策略为 random）")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_multi_eval(args)
