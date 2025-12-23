# predict.py
import os, argparse, random, sys, time, json
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from utils import _flex_read_csv, build_text, ensure_single_label
from bert_wrapper import BERTModelWrapper
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List
from rerank import (
    append_record,
    load_month_records,
    random_rerank,
    qwen_rerank,
    qwen_rerank_single,
    save_records,
    get_raw_path,
    get_reranked_path,
)
from datetime import datetime


# Compatibility: some saved joblib bundles reference LossCallback under __main__
# (e.g. when model was saved from train.py run as __main__). To allow
# unpickling here in predict.py, inject train.LossCallback into this module's
# namespace under the common names pickle may look for.
try:
    import train as _train_module
    _mod = sys.modules.get("__main__")
    if _mod is not None and not hasattr(_mod, "LossCallback"):
        setattr(_mod, "LossCallback", _train_module.LossCallback)
except Exception:
    pass

# 兼容从 train_bert.py 序列化的 BERT 包装器（BERTModelWrapper、LossRecorder 等）
try:
    import train_bert as _train_bert_module  # noqa: F401
    _mod = sys.modules.get("__main__")
    if _mod is not None:
        # 某些情况下，joblib 在保存时记录的模块为 __main__，需要在当前 __main__ 注入同名类
        for _name in ("BERTModelWrapper", "LossRecorder"):
            if hasattr(_train_bert_module, _name) and not hasattr(_mod, _name):
                setattr(_mod, _name, getattr(_train_bert_module, _name))
except Exception:
    # 若未找到也不致命；仅在反序列化 BERT 模型 bundle 时需要
    pass

def _ooc_prob_from_detector(ooc_detector, p_max_scalar: float, override_tau: float | None = None) -> float:
    """从保存的 ooc_detector 计算 not-in-train 概率。
    支持两种形式：
    - {kind:'logreg', estimator: sklearn model}
    - {kind:'threshold', tau: float, temperature: float}
    也兼容直接保存的 sklearn 模型。
    """
    if ooc_detector is None:
        return 0.0
    # dict 格式
    if isinstance(ooc_detector, dict):
        kind = ooc_detector.get("kind")
        if kind == "logreg" and "estimator" in ooc_detector:
            est = ooc_detector["estimator"]
            try:
                return float(est.predict_proba(np.array([[p_max_scalar]]))[:, 1][0])
            except Exception:
                return 0.0
        if kind == "threshold":
            tau = float(override_tau if override_tau is not None else ooc_detector.get("tau", 0.5))
            T = float(ooc_detector.get("temperature", 20.0))
            # 低于阈值越像 OOC：σ((tau - p_max) * T)
            return float(1.0 / (1.0 + np.exp((p_max_scalar - tau) * T)))
    # 直接当作 sklearn 模型
    try:
        return float(ooc_detector.predict_proba(np.array([[p_max_scalar]]))[:, 1][0])
    except Exception:
        return 0.0

def _normalize_linked_items(value):
    """标准化linked_items的格式，处理浮点数、整数、字符串等不同格式"""
    if pd.isna(value):
        return ""
    
    # 如果是浮点数，转换为整数再转字符串
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        else:
            return str(value)
    
    # 如果是整数，直接转字符串
    if isinstance(value, int):
        return str(value)
    
    # 如果是字符串
    if isinstance(value, str):
        # 处理字符串形式的linked_items（如'[35645]')
        if value.startswith('[') and value.endswith(']'):
            try:
                import ast
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return str(parsed[0])
            except Exception:
                pass
        return value.strip()
    
    return str(value)


def _clean_value(value):
    """将值转换为字符串，去除NaN和无意义的小数."""
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        v = float(s)
        if v.is_integer():
            return str(int(v))
    except Exception:
        pass
    return s


def _normalize_label_key(value) -> str:
    """统一标签键格式，确保 '123.0' 与 '123' 等价。"""
    return _clean_value(value)


def _extract_label_encoder(bundle: dict) -> LabelEncoder | None:
    le = bundle.get("label_encoder")
    if le is not None:
        return le
    labels = bundle.get("labels")
    if labels:
        le = LabelEncoder()
        le.fit(labels)
        return le
    return None


def _load_data_csv_mapping(outdir):
    """从 data.csv 加载元数据映射

    返回: linked_items/extern_id -> 详情字典
    仅用于补充展示信息，不直接参与标签空间映射。
    """
    data_path = os.path.join(outdir, "data.csv")
    if not os.path.exists(data_path):
        print(f"[DEBUG] 未找到 data.csv: {data_path}")
        return {}

    print(f"[DEBUG] 正在加载 data.csv: {data_path}")
    try:
        # 只读取需要的列以节省内存
        usecols = [
            'linked_items', 'extern_id', 'item_title', 'itemcreationdate',
            'case_id', 'case_title', 'performed_work', 'case_submitted_date'
        ]
        # 检查文件头，避免读取不存在的列
        header = pd.read_csv(data_path, nrows=0).columns.tolist()
        actual_cols = [c for c in usecols if c in header]
        
        if 'linked_items' not in actual_cols:
            print(f"[DEBUG] data.csv 缺少 linked_items 列")
            return {}

        df = pd.read_csv(data_path, usecols=actual_cols)
        print(f"[DEBUG] data.csv 行数: {len(df)}")

        mapping = {}
        for _, row in df.iterrows():
            linked_items = _normalize_linked_items(row['linked_items']) if 'linked_items' in df.columns else ""
            eid = _clean_value(row.get('extern_id', '')) or _clean_value(row.get('linked_items', ''))
            title = (
                _clean_value(row.get('item_title', ''))
                or _clean_value(row.get('case_title', ''))
                or _clean_value(row.get('performed_work', ''))
            )
            date = _clean_value(row.get('itemcreationdate', '')) or _clean_value(row.get('case_submitted_date', ''))
            case_id = _clean_value(row.get('case_id', ''))

            keys = []
            if linked_items:
                keys.append(linked_items)
            if eid and eid not in keys:
                keys.append(eid)

            if not keys:
                continue

            for key in keys:
                if key not in mapping:
                    mapping[key] = {
                        'extern_id': eid,
                        'item_title': title,
                        'itemcreationdate': date,
                        'case_id': case_id,
                        'case_title': _clean_value(row.get('case_title', '')),
                        'performed_work': _clean_value(row.get('performed_work', '')),
                        'case_submitted_date': _clean_value(row.get('case_submitted_date', ''))
                    }
                else:
                    existing = mapping[key]
                    if eid:
                        existing['extern_id'] = eid
                    if title:
                        existing['item_title'] = title
                    if date:
                        existing['itemcreationdate'] = date
                    if case_id:
                        existing['case_id'] = case_id
                    if row.get('case_title'):
                        existing['case_title'] = _clean_value(row.get('case_title', ''))
                    if row.get('performed_work'):
                        existing['performed_work'] = _clean_value(row.get('performed_work', ''))
                    if row.get('case_submitted_date'):
                        existing['case_submitted_date'] = _clean_value(row.get('case_submitted_date', ''))
        
        print(f"[DEBUG] 从 data.csv 加载了 {len(mapping)} 条映射")
        return mapping
    except Exception as e:
        print(f"Warning: 无法加载 data.csv: {e}")
        return {}

def _load_label_mapping(outdir):
    """加载标签映射文件。

    返回:
      - mapping: 任意 key (linked_items / extern_id / label_id) -> 详细信息
      - raw_to_label_id: 业务 id (linked_items / extern_id) -> 训练时的 label_id
    这样后续可以把测试集里的业务 id 回映射到与 le.classes_ 一致的标签空间。
    """
    mapping_path = os.path.join(outdir, "label_mapping.csv")
    if not os.path.exists(mapping_path):
        return {}, {}

    try:
        df = pd.read_csv(mapping_path)
        mapping = {}
        raw_to_label_id = {}

        for _, row in df.iterrows():
            linked_items_raw = row.get('linked_items', '')
            linked_items = _normalize_linked_items(linked_items_raw)
            label_id = _clean_value(row.get('label', ''))

            info = {
                'extern_id': _clean_value(row.get('extern_id', '')),
                'item_title': _clean_value(row.get('item_title', '')),
                'label': label_id,
                'itemcreationdate': _clean_value(row.get('itemcreationdate', '')),
            }

            if linked_items:
                mapping[linked_items] = info
                if label_id:
                    raw_to_label_id[linked_items] = label_id

            if label_id:
                # 允许通过 label_id 直接索引 info
                if label_id not in mapping:
                    mapping[label_id] = info

        return mapping, raw_to_label_id
    except Exception as e:
        print(f"Warning: 无法加载标签映射文件 {mapping_path}: {e}")
        return {}, {}


def _load_test_data(outdir):
    """加载测试数据文件，获取完整的原始数据包括itemcreationdate

    返回: linked_items -> 详情字典（extern_id、item_title、itemcreationdate 等）
    """
    # 优先尝试 test_raw.csv (包含完整列)，其次 test.csv
    candidates = ["test_raw.csv", "test.csv"]
    test_path = None
    for c in candidates:
        p = os.path.join(outdir, c)
        if os.path.exists(p):
            test_path = p
            break
            
    if not test_path:
        print(f"[DEBUG] 未找到测试数据文件，尝试过的路径: {[os.path.join(outdir, c) for c in candidates]}")
        return {}
    
    print(f"[DEBUG] 正在加载测试数据: {test_path}")
    try:
        df = pd.read_csv(test_path)
        print(f"[DEBUG] 测试数据列名: {df.columns.tolist()}")
        print(f"[DEBUG] 测试数据行数: {len(df)}")
        
        if 'linked_items' not in df.columns:
             print(f"[DEBUG] 测试数据缺少 linked_items 列")
             return {}

        mapping = {}
        for _, row in df.iterrows():
            linked_items = _normalize_linked_items(row['linked_items'])
            
            eid = _clean_value(row.get('extern_id', ''))
            title = _clean_value(row.get('item_title', ''))
            date = _clean_value(row.get('itemcreationdate', ''))

            if linked_items not in mapping:
                mapping[linked_items] = {
                    'extern_id': eid,
                    'item_title': title,
                    'itemcreationdate': date
                }
            else:
                # Update only if new info is not empty
                if eid: mapping[linked_items]['extern_id'] = eid
                if title: mapping[linked_items]['item_title'] = title
                if date: mapping[linked_items]['itemcreationdate'] = date

        print(f"[DEBUG] 从测试数据加载了 {len(mapping)} 条映射")
        return mapping
    except Exception as e:
        print(f"Warning: 无法加载测试数据文件 {test_path}: {e}")
        return {}

def _create_label_to_info_mapping(label_encoder, label_mapping):
    """创建从标签编码器类名到详细信息的映射"""
    mapping = {}
    
    # 首先为训练集中的标签创建映射
    for class_name in label_encoder.classes_:
        raw_key = str(class_name)
        norm_key = _normalize_label_key(raw_key)
        info = label_mapping.get(raw_key)
        if info is None and norm_key != raw_key:
            info = label_mapping.get(norm_key)
        if info is None:
            info = {
                'extern_id': '',
                'item_title': '',
                'label': '',
                'itemcreationdate': ''
            }
        mapping[raw_key] = info
        if norm_key and norm_key != raw_key:
            mapping[norm_key] = info
    
    # 然后为所有标签映射中的标签创建映射（包括测试集中的标签）
    for linked_items, info in label_mapping.items():
        mapping[linked_items] = info
        norm_key = _normalize_label_key(linked_items)
        if norm_key and norm_key != linked_items:
            mapping[norm_key] = info
    
    return mapping


def _get_label_info(mapping, label):
    """根据原始或归一化标签值获取信息。"""
    key = str(label)
    info = mapping.get(key)
    if info:
        return info
    norm_key = _normalize_label_key(key)
    if norm_key and norm_key != key:
        return mapping.get(norm_key, {})
    return {}


def _prepare_topk_payload(preds, scores, label_to_info):
    payload = []
    for lbl, sc in zip(preds, scores):
        info = _get_label_info(label_to_info, lbl)
        payload.append({
            "label": str(lbl),
            "score": float(sc),
            "extern_id": info.get('extern_id', ''),
            "item_title": info.get('item_title', ''),
        })
    return payload


def _print_candidate_block(
    title: str,
    candidates: List[dict],
    true_label: str | None,
    display_label: str | None = None,
) -> None:
    print(f"\n{title}:")
    if not candidates:
        print("  [empty]")
        return
    for i, cand in enumerate(candidates, 1):
        lbl = str(cand.get("label", ""))
        score = cand.get("score", 0.0)
        mark_train = "✅train" if true_label is not None and lbl == str(true_label) else ""
        mark_display = "✅disp" if display_label is not None and lbl == str(display_label) else ""
        mark = " ".join(token for token in (mark_train, mark_display) if token)
        mark_suffix = f" {mark}" if mark else ""
        print(f"{i:2d}. {lbl:<8}\t{float(score):.4f}{mark_suffix}")
        print(f"     extern_id: {cand.get('extern_id', '')}")
        print(f"     item_title: {cand.get('item_title', '')}")
        if i < len(candidates):
            print()


def _detect_label_column(df: pd.DataFrame, le) -> str | None:
    classes = set(str(c) for c in getattr(le, "classes_", []))
    preferred = [
        "linked_items",
        "extern_id",
        "label",
        "labels",
        "target",
        "y",
    ]
    candidates = [col for col in preferred if col in df.columns]
    if not candidates:
        return None

    best_col = candidates[0]
    best_hits = -1
    for col in candidates:
        try:
            series = df[col].apply(ensure_single_label).astype(str)
        except Exception:
            continue
        hits = int(series.isin(classes).sum()) if classes else 0
        if hits > best_hits:
            best_col = col
            best_hits = hits
        if classes and hits == len(series) and len(series) > 0:
            break
    if best_hits > 0:
        return best_col
    if "extern_id" in df.columns:
        return "extern_id"
    if "linked_items" in df.columns:
        return "linked_items"
    return best_col


def _read_split_or_combined(base_dir: str, base: str) -> pd.DataFrame:
    if base and os.path.isabs(base) and os.path.exists(base):
        stem = os.path.splitext(os.path.basename(base))[0]
        if stem.endswith("_X"):
            y_abs = base[:-6] + "_y.csv"
            if os.path.exists(y_abs):
                X_df = pd.read_csv(base)
                y_df = pd.read_csv(y_abs)
                if "linked_items" in y_df.columns:
                    return pd.concat([X_df.reset_index(drop=True), y_df[["linked_items"]].reset_index(drop=True)], axis=1)
                return X_df.reset_index(drop=True)
        return pd.read_csv(base)

    name = os.path.basename(base)
    stem = os.path.splitext(name)[0]
    if stem.endswith("_X"):
        stem = stem[:-2]
    if stem.endswith("_y"):
        stem = stem[:-2]
    x_path = os.path.join(base_dir, f"{stem}_X.csv")
    y_path = os.path.join(base_dir, f"{stem}_y.csv")

    if not os.path.exists(x_path):
        x_path_prefix = os.path.join(base_dir, f"X_{stem}.csv")
        y_path_prefix = os.path.join(base_dir, f"Y_{stem}.csv")
        if os.path.exists(x_path_prefix):
            x_path = x_path_prefix
            y_path = y_path_prefix

    if os.path.exists(x_path):
        X_df = pd.read_csv(x_path)
        if os.path.exists(y_path):
            y_df = pd.read_csv(y_path)
            if "linked_items" in y_df.columns:
                return pd.concat([X_df.reset_index(drop=True), y_df[["linked_items"]].reset_index(drop=True)], axis=1)
        return X_df.reset_index(drop=True)

    return _flex_read_csv(base_dir, base)


def _predict_single_example(
    df: pd.DataFrame,
    idx: int,
    text: str,
    month_value: str,
    model,
    le,
    ooc_detector,
    label_to_info,
    bundle_path: str,
    outdir: str,
    args,
    label_column: str | None,
    raw_to_label_id=None,
    *,
    precomputed_probs=None,
    rerank_enabled: bool = True,
    verbose: bool = True,
) -> dict:
    if precomputed_probs is None:
        t3 = time.time()
        probs = model.predict_proba([text])[0]
        predict_elapsed = time.time() - t3
    else:
        probs = precomputed_probs[idx]
        predict_elapsed = None

    sorted_idx = np.argsort(-probs)
    top_k = min(10, probs.shape[0])
    preds = le.inverse_transform(sorted_idx[:top_k])
    scores = probs[sorted_idx[:top_k]]
    topk_payload = _prepare_topk_payload(preds, scores, label_to_info)

    p_max = float(probs.max())
    decision_mode = "threshold"
    decision_threshold = None
    if isinstance(ooc_detector, dict) and ooc_detector.get("kind") == "logreg":
        ooc_proba = _ooc_prob_from_detector(ooc_detector, p_max)
        prob_thr = float(getattr(args, "ooc_decision_threshold", 0.5))
        is_ooc = (ooc_proba >= prob_thr)
        decision_info = f"(prob_thr={prob_thr:.3f})"
        decision_mode = "logreg"
        decision_threshold = prob_thr
    else:
        if getattr(args, "reject_threshold", None) is not None:
            thr = float(args.reject_threshold)
        elif isinstance(ooc_detector, dict) and ("tau" in ooc_detector):
            thr = float(ooc_detector["tau"])
        else:
            thr = 0.5
        is_ooc = (p_max < thr)
        ooc_proba = _ooc_prob_from_detector(ooc_detector, p_max, override_tau=thr)
        decision_info = f"(p_max={p_max:.4f}, thr={thr:.4f})"
        decision_threshold = thr

    true_label = None
    true_label_raw = None
    true_info = {}
    in_train = False
    label_source = None
    candidate_cols: list[str] = []
    if "extern_id" in df.columns:
        candidate_cols.append("extern_id")
    if label_column and label_column in df.columns and label_column not in candidate_cols:
        candidate_cols.append(label_column)
    for fallback_col in ("linked_items", "label", "labels", "target", "y"):
        if fallback_col in df.columns and fallback_col not in candidate_cols:
            candidate_cols.append(fallback_col)

    for col in candidate_cols:
        raw_value = df.iloc[idx].get(col)
        if raw_value is None:
            continue
        try:
            normalized_value = ensure_single_label(raw_value)
        except Exception:
            normalized_value = raw_value
        cleaned_value = _clean_value(normalized_value)
        if not cleaned_value:
            continue
        lowered = cleaned_value.lower()
        if lowered in {"nan", "none"}:
            continue
        true_label = cleaned_value
        true_label_raw = cleaned_value
        label_source = col
        break

    # 若从原始数据中抽到的是业务ID（如 linked_items / extern_id），尝试用 raw_to_label_id
    # 映射回训练标签ID，使其与 le.classes_ 空间一致，避免标签不对齐。
    if true_label and raw_to_label_id:
        mapped = raw_to_label_id.get(_normalize_label_key(true_label))
        if mapped:
            true_label = mapped

    if true_label:
        if isinstance(true_label, str) and true_label.endswith('.0'):
            true_label = true_label[:-2]
        true_info = _get_label_info(label_to_info, true_label)
        in_train = str(true_label) in set(str(c) for c in le.classes_)
    else:
        true_label = None

    display_label = true_label
    display_col = label_source or ("linked_items" if "linked_items" in df.columns else "")
    ext_val = None
    if "extern_id" in df.columns:
        ext_val = _clean_value(df.iloc[idx].get("extern_id"))
        if ext_val not in (None, "", "nan", "None"):
            display_label = str(ext_val)
            display_col = "extern_id"

    if verbose:
        if predict_elapsed is not None:
            print(f"[PERF] 模型预测耗时: {predict_elapsed:.3f}s")
        # 打印时仅在 ext_val 有值时附加，避免未初始化错误
        if ext_val not in (None, "", "nan", "None"):
            print(f"\n[预测结果] 样本 #{idx} #{ext_val}\n")
        else:
            print(f"\n[预测结果] 样本 #{idx}\n")
        print(f"Text: {text[:200]} ...")
        if true_label is not None:
            suffix = f" (col={display_col})" if display_col else ""
            # 展示时优先用原始业务ID（若存在），计算/对齐时使用 true_label
            disp_val = display_label or true_label_raw or true_label
            print(f"True label: {disp_val}{suffix}  (in_train={in_train})")
            print(f"  - extern_id: {true_info.get('extern_id', '')}")
            print(f"  - item_title: {true_info.get('item_title', '')}")
        else:
            if candidate_cols:
                missing_hint = candidate_cols[0]
            elif label_column:
                missing_hint = label_column
            else:
                missing_hint = "label"
            print(f"True label: [列 {missing_hint} 缺失]")
        _print_candidate_block(
            "Top-10 predictions (pre-rerank)",
            topk_payload,
            true_label,
            display_label,
        )
        mark = " ✅" if is_ooc else ""
        print(f"\nNot-in-train probability: {ooc_proba:.4f} {decision_info} {mark}".rstrip())

    record = {
        "record_id": f"{month_value or 'unknown'}_{idx}_{int(time.time()*1000)}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "month": month_value,
        "query_index": int(idx),
        "text": text,
        "true": {
            "label": true_label,
            "label_display": display_label,
            "label_column": label_source or display_col,
            "extern_id": true_info.get('extern_id', ''),
            "item_title": true_info.get('item_title', ''),
        },
        "topk": topk_payload,
        "model_bundle": bundle_path,
        "is_ooc": bool(is_ooc),
        "ooc_probability": float(ooc_proba),
    }
    raw_path = append_record(outdir, month_value, record)

    reranked_candidates = topk_payload
    if rerank_enabled:
        try:
            if args.rerank_strategy == "random":
                # 随机重排仍然基于整月记录，方便线下分析
                raw_records = load_month_records(outdir, month_value, reranked=False)
                if verbose:
                    print(f"[RERANK] 读取原始记录 {len(raw_records)} 条")
                reranked_records = random_rerank(raw_records, seed=args.rerank_seed)
                rerank_path = save_records(reranked_records, get_reranked_path(outdir, month_value))
                if verbose:
                    print(f"[RERANK] rerank 结果已保存 -> {rerank_path}")
                rerank_entry = next((rec for rec in reranked_records if rec.get("record_id") == record["record_id"]), None)
                if rerank_entry and rerank_entry.get("reranked_topk"):
                    reranked_candidates = rerank_entry["reranked_topk"]
            elif args.rerank_strategy in {"beacon", "qwen2"}:
                # Qwen2 策略：仅对当前样本单次调用，避免对整月历史记录多次重排
                reranked_record = qwen_rerank_single(record)
                reranked_candidates = reranked_record.get("reranked_topk", topk_payload)
                # 依然将单条结果追加保存，便于后续离线分析
                month_rerank_path = get_reranked_path(outdir, month_value)
                save_records([reranked_record], month_rerank_path)
                if verbose:
                    print(f"[RERANK] 当前样本 Qwen2 rerank 结果已追加保存 -> {month_rerank_path}")
            else:
                raise ValueError(f"未知 rerank 策略: {args.rerank_strategy}")
        except Exception as rerank_exc:
            if verbose:
                print(f"[RERANK][WARN] rerank 流程失败，沿用原始排序: {rerank_exc}")
    else:
        if verbose:
            print(f"[RERANK] 批量模式跳过 rerank；记录已保存 -> {raw_path}")

    # 计算 pre-rerank 的 hit@k 指标（仅对当前样本）
    def _compute_hit_at_k(cands: List[dict], true_lbl, k: int) -> int:
        if true_lbl is None:
            return 0
        top_labels = [str(c.get("label", "")) for c in cands[:k]]
        return 1 if str(true_lbl) in top_labels else 0

    base_hit1 = _compute_hit_at_k(topk_payload, true_label, 1)
    base_hit3 = _compute_hit_at_k(topk_payload, true_label, 3)
    base_hit5 = _compute_hit_at_k(topk_payload, true_label, 5)
    base_hit10 = _compute_hit_at_k(topk_payload, true_label, 10)

    final_label = None
    if not is_ooc:
        final_label = reranked_candidates[0]["label"] if reranked_candidates else (str(preds[0]) if len(preds) > 0 else "")

    if verbose:
        _print_candidate_block(
            "Top-10 predictions (reranked)",
            reranked_candidates,
            true_label,
            display_label,
        )

        # 从 rerank 元数据中读取 LLM 信号，用于 TUI 高亮提示
        if record.get("rerank_metadata"):
            rm = record["rerank_metadata"]
            all_irrelevant = bool(rm.get("all_irrelevant", False))
            suggested_new_title = rm.get("suggested_new_title") or None
        else:
            all_irrelevant = False
            suggested_new_title = None

        # Final 行：同时考虑 not-in-train 与 LLM 的 all_irrelevant 判断
        if is_ooc:
            print("Final: Not-in-train")
        elif all_irrelevant:
            # LLM 明确认为10个候选都与工单无关，给出强提示
            print("Final: ⚠️ All 10 candidates deemed IRRELEVANT by LLM (keep top-1 for metric only)")
        else:
            print(f"Final: {final_label}（Known, reranked）")

        # 如果 LLM 给出了建议标题，在 TUI 中显著打印出来
        if all_irrelevant and suggested_new_title:
            print("\n[LLM 标记结果]")
            print("  -> 本样本的10个候选都被Qwen标记为【不相关】。")
            print("  -> 建议新标题 (建议录入知识库/后续标注)：")
            print(f"     {suggested_new_title}")

        # 基于 rerank 后的候选计算 hit@k，并打印 ↑↓/– 情况
        new_hit1 = _compute_hit_at_k(reranked_candidates, true_label, 1)
        new_hit3 = _compute_hit_at_k(reranked_candidates, true_label, 3)
        new_hit5 = _compute_hit_at_k(reranked_candidates, true_label, 5)
        new_hit10 = _compute_hit_at_k(reranked_candidates, true_label, 10)

        def _delta_symbol(base: int, new: int) -> str:
            if base == new:
                return "–"
            return "↑" if new > base else "↓"

        print("\n[HIT@K 变化] (当前样本)")
        print(
            f"  hit@1:  {base_hit1} -> {new_hit1} {_delta_symbol(base_hit1, new_hit1)}\n"
            f"  hit@3:  {base_hit3} -> {new_hit3} {_delta_symbol(base_hit3, new_hit3)}\n"
            f"  hit@5:  {base_hit5} -> {new_hit5} {_delta_symbol(base_hit5, new_hit5)}\n"
            f"  hit@10: {base_hit10} -> {new_hit10} {_delta_symbol(base_hit10, new_hit10)}"
        )

    return {
        "record": record,
        "raw_path": raw_path,
        "topk_payload": topk_payload,
        "reranked_candidates": reranked_candidates,
        "is_ooc": bool(is_ooc),
        "ooc_probability": float(ooc_proba),
        "decision_info": decision_info,
        "decision_mode": decision_mode,
        "decision_threshold": decision_threshold,
        "true_label": true_label,
        "final_label": final_label,
    }


def _run_batch_mode(batch_df: pd.DataFrame, model, le, ooc_detector, label_to_info, bundle_path, outdir, args):
    if batch_df is None or batch_df.empty:
        print("[BATCH] 输入数据为空，结束批量预测。")
        return
    texts = build_text(batch_df).tolist()
    if len(texts) == 0:
        print("[BATCH] 无法构造文本，结束批量预测。")
        return
    print(f"[BATCH] 即将处理 {len(texts)} 条样本 (来源: {args.batch_path})")
    t_batch = time.time()
    probs_matrix = model.predict_proba(texts)
    ooc_rows = []
    label_column = _detect_label_column(batch_df, le)
    if "extern_id" in batch_df.columns:
        label_column = "extern_id"
    if label_column:
        batch_df[label_column] = batch_df[label_column].apply(ensure_single_label).astype(str)
    for idx, text in enumerate(texts):
        month_value = _clean_value(batch_df.iloc[idx].get("month", "")) if "month" in batch_df.columns else ""
        result = _predict_single_example(
            batch_df,
            idx,
            text,
            month_value,
            model,
            le,
            ooc_detector,
            label_to_info,
            bundle_path,
            outdir,
            args,
            label_column,
            precomputed_probs=probs_matrix,
            rerank_enabled=not getattr(args, "batch_disable_rerank", False),
            verbose=True,
        )
        if result["is_ooc"]:
            row = {col: batch_df.iloc[idx].get(col) for col in batch_df.columns}
            row = {k: _clean_value(v) for k, v in row.items()}
            row.update({
                "query_index": idx,
                "text_excerpt": text[:200],
                "ooc_probability": result["ooc_probability"],
                "decision_threshold": result["decision_threshold"],
                "decision_mode": result["decision_mode"],
                "top1_label": result["final_label"],
                "top10_labels": "|".join([cand["label"] for cand in result["reranked_candidates"]]),
                "top10_scores": "|".join([f"{cand['score']:.4f}" for cand in result["reranked_candidates"]]),
                "top10_json": json.dumps(result["reranked_candidates"], ensure_ascii=False),
            })
            ooc_rows.append(row)

    if ooc_rows:
        output_path = args.batch_ooc_output or os.path.join(outdir, "batch_not_in_train.csv")
        pd.DataFrame(ooc_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[BATCH] 已保存 {len(ooc_rows)} 条 not-in-train 记录 -> {output_path}")
    else:
        print("[BATCH] 没有超过阈值的 not-in-train 样本。")
    print(f"[BATCH] 总耗时: {time.time() - t_batch:.3f}s")



def main(args):
    start_time = time.time()
    
    # 解析模型路径
    modelsdir = args.modelsdir
    model_name = args.model
    bundle_path = os.path.join(modelsdir, model_name)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Model not found: {bundle_path}")
    bundle = joblib.load(bundle_path)
    model_type = bundle.get("model_type", None)
    le = _extract_label_encoder(bundle)
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
    
    # 加载标签映射和测试数据
    outdir = args.outdir
    label_mapping, raw_to_label_id = _load_label_mapping(outdir)
    test_data = _load_test_data(outdir)
    data_csv_mapping = _load_data_csv_mapping(outdir)
    
    # 优先合并 data.csv 的信息（因为它包含完整的原始数据）
    print(f"[DEBUG] 合并 data.csv 前 label_mapping 大小: {len(label_mapping)}")
    for linked_items, info in data_csv_mapping.items():
        if linked_items in label_mapping:
            existing = label_mapping[linked_items]
            for key in ('extern_id', 'item_title', 'itemcreationdate'):
                new_val = info.get(key, '')
                # 如果现有值为空，或者新值不为空且不同，则更新（这里倾向于相信 data.csv 的原始数据）
                if new_val: 
                    existing[key] = new_val
        else:
            label_mapping[linked_items] = info
    print(f"[DEBUG] 合并 data.csv 后 label_mapping 大小: {len(label_mapping)}")

    # 合并测试数据到标签映射中，补充itemcreationdate信息
    print(f"[DEBUG] 合并前 label_mapping 大小: {len(label_mapping)}")
    print(f"[DEBUG] test_data 大小: {len(test_data)}")
    
    for linked_items, info in test_data.items():
        if linked_items in label_mapping:
            existing = label_mapping[linked_items]
            updated = False
            for key in ('extern_id', 'item_title', 'itemcreationdate'):
                new_val = info.get(key, '')
                if new_val and existing.get(key, '') != new_val:
                    existing[key] = new_val
                    updated = True
        else:
            # 如果标签映射中没有，添加进去
            label_mapping[linked_items] = info
            print(f"[DEBUG] 添加新标签 {linked_items}: {info}")
    
    # 调试信息
    print(f"[DEBUG] 合并后 label_mapping 大小: {len(label_mapping)}")
    print(f"[DEBUG] test_data 大小: {len(test_data)}")
    
    # 创建从标签编码器类名到详细信息的映射
    t1 = time.time()
    label_to_info = _create_label_to_info_mapping(le, label_mapping)
    print(f"[PERF] 标签映射构建耗时: {time.time() - t1:.3f}s")
    
    # 调试信息 - 查看训练标签的信息
    print(f"[DEBUG] 训练标签信息:")
    for label in le.classes_[:5]:
        info = _get_label_info(label_to_info, label)
        print(f"  标签 {label}: extern_id={info.get('extern_id', 'N/A')}, item_title={info.get('item_title', 'N/A')[:30]}...")
    

    if args.batch_path:
        batch_df = _read_split_or_combined(outdir, args.batch_path)
        _run_batch_mode(batch_df, model, le, ooc_detector, label_to_info, bundle_path, outdir, args)
        total_time = time.time() - start_time
        print(f"[PERF] 总耗时: {total_time:.3f}s")
        return

    # 读取输入样本（支持 *_X.csv + *_y.csv 分离，或单表）
    infile = args.infile if args.infile else args.path  # 兼容旧参数 --path
    if not infile:
        raise ValueError("必须提供 --infile 或 --batch-path。")

    t2 = time.time()
    df = _read_split_or_combined(outdir, infile)
    label_column = _detect_label_column(df, le)
    if label_column:
        df[label_column] = df[label_column].apply(ensure_single_label).astype(str)
    texts = build_text(df).tolist()
    print(f"[PERF] 数据读取和文本构建耗时: {time.time() - t2:.3f}s")
    
    if len(texts) == 0:
        raise ValueError("输入文件为空或无法构造文本。")

    # 选择索引
    idx = args.index
    if idx is None or idx < 0 or idx >= len(texts):
        idx = random.randint(0, len(texts) - 1)
    text = texts[idx]
    month_value = _clean_value(df.iloc[idx].get("month", "")) if "month" in df.columns else ""

    _predict_single_example(
        df,
        idx,
        text,
        month_value,
        model,
        le,
        ooc_detector,
        label_to_info,
        bundle_path,
        outdir,
        args,
        label_column,
        raw_to_label_id=raw_to_label_id,
    )

    total_time = time.time() - start_time
    print(f"[PERF] 总耗时: {total_time:.3f}s")


def predict(texts, model_path=None, top_k=10, *, reject_threshold=None, not_in_train_label="__NOT_IN_TRAIN__"):
    """按 eval.py 逻辑进行预测，返回包含 hit@k 一致含义的结果。

    - texts: str | list[str] | pandas.DataFrame
      * 若为 DataFrame，则使用 `build_text` 与 eval.py 完全一致地构造输入文本；
      * 若为 str 或 list[str]，则直接视为已构造好的文本。
    - model_path: joblib bundle 路径；若为 None，则默认使用 `./models/8.joblib`，与 eval.py 默认一致。
    - top_k: 返回前 K 个候选标签（不含 not_in_train 标签）。
    - reject_threshold: 若不为 None，则启用与 eval.py `--reject-threshold` 相同的 MSP 拒判逻辑：
        * 设每个样本的最大类别概率为 p_max；
        * 当 p_max < reject_threshold 时，将该样本预测为 `not_in_train_label`，并不返回真实类别候选；
        * 当 p_max >= 阈值时，返回 top_k 个类别预测；
      这保证 predict 与 eval.py 的开放集评估策略统一。

    返回：list[dict]，每个元素包含：
      - "topk_labels": List[str]
      - "topk_scores": List[float]
      - "final_label": str（可能等于 not_in_train_label）
      - "p_max": float
      - "rejected": bool
    """

    # 1. 加载模型 bundle，与 eval.py 一致
    if model_path is None:
        # 与 eval.py __main__ 默认保持一致：models 目录 + 默认模型名
        model_path = os.path.join("./models", "8.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    bundle = joblib.load(model_path)
    model_type = bundle.get("model_type", None)
    le = _extract_label_encoder(bundle)
    if le is None:
        raise ValueError("模型 bundle 缺少标签编码信息。")

    if model_type == "bert":
        model_dir = bundle.get("model_dir")
        if not model_dir:
            raise ValueError("BERT bundle 缺少 model_dir。")
        model = BERTModelWrapper(
            model_dir,
            le,
            max_length=bundle.get("max_length", 256),
            fp16=bundle.get("fp16", False),
        )
    else:
        model = bundle.get("model")
        if model is None:
            raise ValueError("Loaded bundle does not contain a serialized model.")

    # 2. 构造文本，与 eval.py / utils.build_text 完全一致
    if isinstance(texts, pd.DataFrame):
        X = build_text(texts).tolist()
    elif isinstance(texts, str):
        X = [texts]
    else:
        X = list(texts)

    if len(X) == 0:
        return []

    # 3. 概率预测，与 eval.py 的开放集评估保持一致（使用 predict_proba）
    probs = model.predict_proba(X)
    n_classes = probs.shape[1]
    top_k = min(top_k, n_classes)

    results = []
    classes = le.classes_

    for i in range(len(X)):
        p = probs[i]
        p_max = float(p.max())
        # MSP 拒判逻辑：当设置阈值时才启用，与 eval.py 的 _eval_open_set 一致
        rejected = False
        final_label = None

        if reject_threshold is not None:
            if p_max < float(reject_threshold):
                rejected = True
                final_label = not_in_train_label

        idxs = np.argsort(-p)[:top_k]
        preds = le.inverse_transform(idxs)
        scores = p[idxs]

        # 若未被拒判，则 final_label 为 top1 预测；否则保持 not_in_train_label
        if not rejected:
            final_label = str(preds[0]) if len(preds) > 0 else ""

        results.append({
            "topk_labels": list(map(str, preds)),
            "topk_scores": list(map(float, scores)),
            "final_label": final_label,
            "p_max": p_max,
            "rejected": rejected,
        })

    return results

class TorchLinearModel(BaseEstimator, ClassifierMixin):
    """A PyTorch-based linear classifier compatible with sklearn's partial_fit interface."""
    def __init__(self, input_dim=None, num_classes=None, device="cuda", lr=1e-3, weight_decay=1e-4, class_weight=None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device_name = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weight = class_weight
        
        # Lazy initialization to allow sklearn cloning
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.classes_ = None
        
    def _init_model(self):
        if self.model is not None:
            return
            
        self.device = torch.device(self.device_name if torch.cuda.is_available() else "cpu")
        print(f"[Info] TorchLinearModel initialized on {self.device}")
        self.model = nn.Linear(self.input_dim, self.num_classes).to(self.device)
        
        # Initialize weights similar to sklearn
        nn.init.xavier_uniform_(self.model.weight)
        nn.init.zeros_(self.model.bias)
        
        weight_tensor = None
        if self.class_weight is not None:
            weight_tensor = torch.tensor(self.class_weight, dtype=torch.float32).to(self.device)
            
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.classes_ = np.arange(self.num_classes)

    def __sklearn_is_fitted__(self):
        return self.model is not None
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Backwards compatibility for old pickles
        if "lr" not in self.__dict__: self.lr = 1e-3
        if "weight_decay" not in self.__dict__: self.weight_decay = 1e-4
        if "class_weight" not in self.__dict__: self.class_weight = None
        if "device_name" not in self.__dict__: 
            if "device" in self.__dict__:
                self.device_name = str(self.device)
            else:
                self.device_name = "cuda"

    def fit(self, X, y):
        # Dummy fit for sklearn compatibility checks
        return self
        
    def partial_fit(self, X, y, classes=None):
        if self.model is None:
            # If input_dim/num_classes not set, infer from data (not ideal for partial_fit but helpful)
            if self.input_dim is None:
                self.input_dim = X.shape[1]
            if classes is not None and self.num_classes is None:
                self.num_classes = len(classes)
            self._init_model()
            
        self.model.train()
        
        # Handle sparse input
        if issparse(X):
            X = X.toarray()
            
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(X_t)
        loss = self.criterion(outputs, y_t)
        loss.backward()
        self.optimizer.step()
        return self
        
    def predict_proba(self, X):
        if self.model is None:
             # Should not happen if fitted
             return np.zeros((X.shape[0], self.num_classes))
             
        self.model.eval()
        n_samples = X.shape[0]
        batch_size = 2048
        probs_list = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                X_batch = X[i:end]
                if issparse(X_batch):
                    X_batch = X_batch.toarray()
                
                X_t = torch.tensor(X_batch, dtype=torch.float32).to(self.device)
                outputs = self.model(X_t)
                probs = torch.softmax(outputs, dim=1)
                probs_list.append(probs.cpu().numpy())
                
        if len(probs_list) > 0:
            return np.vstack(probs_list)
        return np.array([])

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
        
    def decision_function(self, X):
        if self.model is None:
             return np.zeros((X.shape[0], self.num_classes))

        self.model.eval()
        n_samples = X.shape[0]
        batch_size = 2048
        logits_list = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                X_batch = X[i:end]
                if issparse(X_batch):
                    X_batch = X_batch.toarray()
                    
                X_t = torch.tensor(X_batch, dtype=torch.float32).to(self.device)
                outputs = self.model(X_t)
                logits_list.append(outputs.cpu().numpy())
                
        if len(logits_list) > 0:
            return np.vstack(logits_list)
        return np.array([])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelsdir", type=str, default="./models")
    parser.add_argument("--model", type=str, default="7.joblib")
    parser.add_argument("--outdir", type=str, default="./output/2025_up_to_month_0")
    parser.add_argument("--infile", type=str, default="eval.csv", help="输入文件名（在 outdir 下，或提供绝对路径）")
    parser.add_argument("--index", type=int, default=-1, help="样本下标；<0 则随机")
    parser.add_argument("--batch-path", type=str, default=None, help="批量预测 CSV 路径（相对 outdir 或绝对路径）")
    parser.add_argument("--batch-ooc-output", type=str, default=None, help="批量预测下保存 not-in-train 结果的 CSV 路径")
    parser.add_argument("--batch-disable-rerank", action="store_true", help="批量模式下跳过 rerank 流程以提升速度")
    # MSP: p_max 阈值（与 eval 对齐）；LogReg: 概率阈值
    parser.add_argument("--reject-threshold", type=float, default=None, help="MSP：最大类别概率 p_max 的拒判阈值")
    parser.add_argument("--ooc-decision-threshold", type=float, default=0.5, help="LogReg 检测器下的概率阈值")
    parser.add_argument("--rerank-strategy", type=str, default="random", choices=["random","beacon","qwen2"], help="Rerank 策略（默认随机打乱，可选 qwen2/beacon）")
    parser.add_argument("--rerank-seed", type=int, default=42, help="Rerank 随机种子")
    # 兼容旧参数
    parser.add_argument("--path", type=str, default=None)
    args = parser.parse_args()
    main(args)
