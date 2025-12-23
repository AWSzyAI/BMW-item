"""Shared rerank interface for predict.py / eval.py and standalone CLI."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from beacon import ask_Qwen2

RAW_SUBDIR = "rerank_records"
RERANK_SUFFIX = "reranked"

# ==========================================
# Debug Switch
# ==========================================
DEBUG_MODE = True  # 设置为 True 会打印 LLM 的原始回复，方便排查

__all__ = [
    "append_record",
    "load_records",
    "load_month_records",
    "save_records",
    "random_rerank",
    "qwen_rerank",
    "qwen_rerank_single",
    "evaluate_records",
    "get_raw_path",
    "get_reranked_path",
]


def _normalize_month(month: Optional[str]) -> str:
    value = month or "unknown"
    return value.replace("/", "-").replace(" ", "_")


def get_raw_path(outdir: str, month: Optional[str]) -> Path:
    return Path(outdir) / RAW_SUBDIR / f"{_normalize_month(month)}.jsonl"


def get_reranked_path(outdir: str, month: Optional[str], suffix: str = RERANK_SUFFIX) -> Path:
    return Path(outdir) / RAW_SUBDIR / f"{_normalize_month(month)}.{suffix}.jsonl"


def append_record(outdir: str, month: Optional[str], record: Dict[str, Any]) -> Path:
    target = get_raw_path(outdir, month)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_month_records(outdir: str, month: Optional[str], reranked: bool = False, suffix: str = RERANK_SUFFIX) -> List[Dict[str, Any]]:
    path = get_reranked_path(outdir, month, suffix) if reranked else get_raw_path(outdir, month)
    if not path.exists():
        return []
    return load_records(path)


def save_records(records: Iterable[Dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return target


def random_rerank(records: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    output: List[Dict[str, Any]] = []
    for rec in records:
        candidates = rec.get("topk", [])
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        rec_copy = rec.copy()
        rec_copy["reranked_topk"] = shuffled
        rec_copy["final_prediction"] = shuffled[0]["label"] if shuffled else None
        rec_copy["rerank_metadata"] = {
            "strategy": "random",
            "seed": seed,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        output.append(rec_copy)
    return output


def _parse_llm_json_response(response: str) -> Dict[str, Any]:
    """Helper to robustly parse JSON from LLM response."""
    # 1. 尝试直接清理 markdown 标记
    clean_resp = response.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean_resp)
    except json.JSONDecodeError:
        pass

    # 2. 正则提取 JSON 对象
    match = re.search(r'(\{.*\})', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # 3. 如果连 JSON 都不是，尝试硬提取 list 结构
    match_list = re.search(r'(\[.*\])', response, re.DOTALL)
    if match_list:
        try:
            return {"reranked_ids": json.loads(match_list.group(1)), "thought": "extracted_list_only"}
        except:
            pass

    raise ValueError("Could not parse JSON from response")


def qwen_rerank(
    records: List[Dict[str, Any]],
    *,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    
    print(f"Starting Qwen rerank on {len(records)} records...", file=sys.stderr)

    for rec_idx, rec in enumerate(records):
        candidates = rec.get("topk", [])
        
        if not candidates:
            rec_copy = rec.copy()
            rec_copy["reranked_topk"] = []
            rec_copy["final_prediction"] = None
            rec_copy["rerank_metadata"] = {"strategy": "empty_input"}
            output.append(rec_copy)
            continue

        query = rec.get("text", "")

        # 1. 计算特征
        scores = [c.get('score', 0.0) for c in candidates]
        max_score = max(scores) if scores else 0.0

        # 这里不再对 Top1 做任何特殊保护，全部候选交给 LLM 重排
        llm_candidates = candidates
        id_offset = 0

        # 2. 构造描述（对全部 llm_candidates 编号）
        items_desc_list: List[str] = []
        for local_idx, cand in enumerate(llm_candidates, 1):
            s = cand.get('score', 0.0)
            global_rank = local_idx + id_offset
            rank_hint = " (Current Top 1)" if global_rank == 1 else ""
            items_desc_list.append(
                f"[ID:{local_idx}] Score:{s:.4f}{rank_hint} | Label:{cand.get('label','')} | Title:{cand.get('item_title','')}"
            )
        items_desc_str = "\n".join(items_desc_list)

        all_ids = list(range(1, len(llm_candidates) + 1))
        all_ids_str = ",".join(map(str, all_ids))

        # ==========================================
        # 升级版 Prompt：增加“全员无关”检测和“新标题建议”
        # ==========================================
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个工单故障匹配专家。你的任务是优化故障候选列表的排序。\n\n"
                    "### 核心决策逻辑：\n"
                    "1. **尊重显著的 Score 差异**：如果 [ID:1] Score 远高于其他项，除非语义完全风马牛不相及，否则应尽量保留在首位。\n"
                    "2. **处理模糊地带**：当 Score 差异不大时，完全依赖语义相似度重排。\n"
                    "3. **全员无关检测（重要）**：\n"
                    "   - 如果你发现候选列表中**没有任何一个**选项与工单语义匹配（哪怕只有一点点相关也算相关），请标记 `all_irrelevant: true`。\n"
                    "   - 此时，请根据工单内容，拟定一个**简短的、标准化的英文故障标题**，放入 `suggested_new_title` 字段。\n\n"
                    "### 输出规范（JSON）：\n"
                    "{\n"
                    '  "thought": "分析过程...",\n'
                    '  "reranked_ids": [1, 10, 2...], // 必须包含所有ID\n'
                    '  "all_irrelevant": false, // 是否所有候选项都无关\n'
                    '  "suggested_new_title": null // 如果 all_irrelevant 为 true，请填写建议标题，否则为 null\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"工单内容：{query}\n\n"
                    f"Score 显著性提示：Top1 优势信息仅供参考\n\n"
                    "候选列表：\n"
                    f"{items_desc_str}\n\n"
                    f"请给出重排后的 JSON，确保包含所有 ID ({all_ids_str})。"
                ),
            },
        ]

        try:
            resp = ask_Qwen2(messages) or ""
            
            if DEBUG_MODE and rec_idx < 3:
                print(f"\n--- [DEBUG Record {rec_idx}] ---")
                print(f"Query: {query}")
                print(f"LLM Raw Response: {resp}")

            # 解析 JSON
            thought = ""
            all_irrelevant = False
            suggested_new_title = None
            order_indices = []

            try:
                data = _parse_llm_json_response(resp)
                raw_ids = data.get('reranked_ids', [])
                thought = data.get('thought', "")
                all_irrelevant = data.get('all_irrelevant', False)
                suggested_new_title = data.get('suggested_new_title', None)
                
                for x in raw_ids:
                    if isinstance(x, int):
                        order_indices.append(x - 1)
                    elif isinstance(x, str) and x.isdigit():
                        order_indices.append(int(x) - 1)
            except Exception as e:
                thought = f"JSON Parse Failed: {str(e)}"
                nums = re.findall(r'\d+', resp)
                order_indices = [int(x) - 1 for x in nums]

            # 把 LLM 的局部下标映射回原始 candidates 的全局下标
            valid_indices = [idx + id_offset for idx in order_indices if 0 <= (idx + id_offset) < len(candidates)]

            if not valid_indices:
                raise ValueError("No valid indices found")

            seen: set[int] = set()
            ordered: List[Dict[str, Any]] = []
            for idx in valid_indices:
                if idx not in seen:
                    seen.add(idx)
                    ordered.append(candidates[idx])
            for i, cand in enumerate(candidates):
                if i not in seen:
                    ordered.append(cand)

            # ==========================================
            # 不做任何 Top1 保护或分数兜底，完全采用 LLM 排序
            # ==========================================
            final_ordered = ordered
            is_safeguarded = False

            rec_copy = rec.copy()
            rec_copy["reranked_topk"] = final_ordered[:top_k]
            rec_copy["final_prediction"] = (
                rec_copy["reranked_topk"][0]["label"]
                if rec_copy["reranked_topk"]
                else None
            )
            
            # 保存丰富的元数据
            rec_copy["rerank_metadata"] = {
                "strategy": "qwen2_with_discovery",
                "thought": thought,
                "all_irrelevant": all_irrelevant,  # 核心新字段
                "suggested_new_title": suggested_new_title, # 核心新字段
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "is_fallback": False,
                "safeguard_triggered": is_safeguarded
            }
            output.append(rec_copy)

        except Exception as e:
            if DEBUG_MODE:
                print(f"!!! Error processing record {rec_idx}: {e}")
            rec_copy = rec.copy()
            rec_copy["reranked_topk"] = candidates
            rec_copy["final_prediction"] = candidates[0]["label"] if candidates else None
            rec_copy["rerank_metadata"] = {
                "strategy": "qwen2_fallback",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            output.append(rec_copy)

    return output

def qwen_rerank_single(record: Dict[str, Any], *, top_k: int = 10) -> Dict[str, Any]:
    wrapper_list = [record]
    results = qwen_rerank(wrapper_list, top_k=top_k)
    return results[0]


def evaluate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = 0
    mismatches: List[Dict[str, Any]] = []
    
    # 统计新发现
    discovery_count = 0
    
    for rec in records:
        true_label = str(rec.get("true", {}).get("label"))
        pred = rec.get("final_prediction")
        pred = str(pred) if pred is not None else "None"
        
        meta = rec.get("rerank_metadata", {})
        if meta.get("all_irrelevant"):
            discovery_count += 1
            
        if pred == true_label:
            correct += 1
        else:
            mismatches.append({
                "record_id": rec.get("record_id"),
                "text": rec.get("text", "")[:60] + "...",
                "true": true_label,
                "pred": pred,
                "thought": meta.get("thought", "N/A"),
                "new_title_suggestion": meta.get("suggested_new_title", "N/A"), # 评估报告里展示建议标题
                "error": meta.get("error", "")
            })
            
    acc = correct / total if total else 0.0
    return {
        "samples": total,
        "correct": correct,
        "accuracy_top1": acc,
        "discovery_count": discovery_count,
        "mismatches": mismatches,
    }


# ---------------- CLI helpers ---------------- #

def cmd_random(args: argparse.Namespace) -> None:
    records = load_records(args.input)
    reranked = random_rerank(records, args.seed)
    save_records(reranked, args.output)
    print(f"Saved {len(reranked)} reranked queries -> {args.output}")


def cmd_eval(args: argparse.Namespace) -> None:
    records = load_records(args.input)
    metrics = evaluate_records(records)
    print(f"Samples: {metrics['samples']}")
    print(f"Top-1 accuracy: {metrics['accuracy_top1']:.4f} ({metrics['correct']} / {metrics['samples']})")
    print(f"Queries flagged as 'All Irrelevant': {metrics['discovery_count']}") # 打印统计
    
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Report saved -> {args.report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerank prediction logs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_random = sub.add_parser("random", help="Generate reranked JSON via random shuffle")
    p_random.add_argument("--input", required=True, help="Path to original query jsonl")
    p_random.add_argument("--output", required=True, help="Path to write reranked jsonl")
    p_random.add_argument("--seed", type=int, default=42, help="Random seed")
    p_random.set_defaults(func=cmd_random)

    p_eval = sub.add_parser("eval", help="Evaluate reranked results")
    p_eval.add_argument("--input", required=True, help="Path to reranked jsonl")
    p_eval.add_argument("--report", help="Optional path to save metrics json")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()