#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析 predict.py 多次运行产生的 rerank 日志，统计 hit@1/3/5/10 的 ↑/↓/– 次数。

用法示例：

  python src/parse_rerank_log.py --log-file rerank.txt

其中 rerank.txt 由类似下述命令生成：

  for i in $(seq 1 100); do \
    python src/predict.py \
      --modelsdir ./models \
      --model 2_tfidf.joblib \
      --outdir ./output/2025_up_to_month_2 \
      --infile eval.csv \
      --index -1 \
      --rerank-strategy qwen2; \
  done > rerank.txt

脚本会扫描形如：

  [HIT@K 变化] (当前样本)
    hit@1:  0 -> 1 ↑
    hit@3:  1 -> 1 –
    ...

的块，并统计每个 k 的 ↑/↓/– 次数及比例。
"""

import argparse


def parse_log(path: str) -> None:
    stats = {
        1: {"up": 0, "down": 0, "same": 0},
        3: {"up": 0, "down": 0, "same": 0},
        5: {"up": 0, "down": 0, "same": 0},
        10: {"up": 0, "down": 0, "same": 0},
    }

    current_block = False

    def _update(k: int, line: str) -> None:
        # line 形如: "  hit@1:  0 -> 1 ↑"
        if "↑" in line:
            stats[k]["up"] += 1
        elif "↓" in line:
            stats[k]["down"] += 1
        elif "–" in line or "-" in line:
            stats[k]["same"] += 1

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if "[HIT@K 变化]" in line:
                current_block = True
                continue
            if current_block:
                stripped = line.strip()
                if stripped.startswith("hit@1"):
                    _update(1, stripped)
                elif stripped.startswith("hit@3"):
                    _update(3, stripped)
                elif stripped.startswith("hit@5"):
                    _update(5, stripped)
                elif stripped.startswith("hit@10"):
                    _update(10, stripped)
                    # 当前块到此结束
                    current_block = False

    print(f"[PARSE] 解析完成: {path}")
    print("[PARSE] 汇总统计 (基于日志中的 [HIT@K 变化] 块)：")
    for k in (1, 3, 5, 10):
        up = stats[k]["up"]
        down = stats[k]["down"]
        same = stats[k]["same"]
        total = up + down + same
        if total == 0:
            print(f"  hit@{k}: 未在日志中找到记录")
            continue
        up_r = up / total
        down_r = down / total
        same_r = same / total
        print(
            f"  hit@{k}:  ↑={up}  ↓={down}  –={same}  "
            f"(↑={up_r:.3f}, ↓={down_r:.3f}, –={same_r:.3f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 rerank 日志中的 hit@k ↑/↓/– 情况")
    parser.add_argument("--log-file", type=str, required=True, help="由多次运行 predict.py 生成的日志文件路径")
    args = parser.parse_args()
    parse_log(args.log_file)


if __name__ == "__main__":
    main()
