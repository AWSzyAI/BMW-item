# 模型开关：tfidf 或 bert
# 用法：`make ALGO=bert all` 或在文件顶部改为 ALGO=bert
ALGO?=tfidf
# ALGO?=bert

# 设置UV链接模式为复制，避免跨文件系统硬链接警告
export UV_LINK_MODE=copy

month=7
outdir=./output/2025_up_to_month_${month}
model=${month}_${ALGO}.joblib
trainlog=${month}_train.txt
evallog=${month}_eval.txt
# evallog_rerank=${month}_eval_rerank.txt
predictlog=${month}_predict.txt
labelcol=extern_id
# labelcol=linked_items

BATCH_PATH?=${outdir}/test.csv
BATCH_OOC_OUTPUT?=${outdir}/batch_not_in_train.csv

# # Rerank 策略（供 predict.py 使用）：random / beacon / qwen2
# RERANK_STRATEGY?=qwen2
# RERANK_SEED?=42

# RERANK?=
# RERANK_INPUT?=
# RERANK_RECORDS_DIR:=${outdir}/rerank_records
# RERANK_RAW_CANDIDATES:=$(filter-out %.reranked.jsonl,$(wildcard $(RERANK_RECORDS_DIR)/*.jsonl))
# RERANK_RAW_DEFAULT:=$(firstword $(RERANK_RAW_CANDIDATES))
# RERANK_RERANKED_DEFAULT:=$(firstword $(wildcard $(RERANK_RECORDS_DIR)/*.reranked.jsonl))
# RERANK_BOOL_TRUE:=$(filter 1 true TRUE yes YES,$(strip $(RERANK)))
# RERANK_FLAG:=$(if $(RERANK_BOOL_TRUE),--rerank-use-reranked-topk,)
# RERANK_SOURCE?=$(if $(strip $(RERANK_INPUT)),$(RERANK_INPUT),$(if $(RERANK_BOOL_TRUE),$(if $(strip $(RERANK_RERANKED_DEFAULT)),$(RERANK_RERANKED_DEFAULT),$(RERANK_RAW_DEFAULT)),$(RERANK_RAW_DEFAULT)))

# 统一的模型目录变量（两条路线都会用到）
modelsdir=./models

# BERT 相关参数（当 ALGO=bert 时使用）
# - 默认使用在线下载 bert-base-chinese；如需离线本地目录，请设置 INIT_HF_DIR 指向本地HF模型目录
#   例如：`make ALGO=bert INIT_HF_DIR=./models/bert-base-chinese`（目录需包含config.json与tokenizer）
BERT_MODEL?=chinese-macbert-base
ALLOW_ONLINE?=0
INIT_HF_DIR?=./models/hfl/chinese-macbert-base

# 学习率调度器参数
LR_SCHEDULER_TYPE?=cosine
WARMUP_RATIO?=0.1
WARMUP_STEPS?=0


all: data train eval

data:
	@echo "设置环境变量：UV_LINK_MODE=copy"
	@export UV_LINK_MODE=copy && uv run src/data_split.py --outdir ${outdir} --month ${month} --label-col ${labelcol}

train:
ifeq ($(ALGO),bert)
	@echo "设置环境变量：UV_LINK_MODE=copy"
	@export UV_LINK_MODE=copy && uv run src/train_bert.py \
		--outdir ${outdir} \
		--checkpoint-dir ./checkpoints/${month}_bert \
		--outmodel ${model} \
		--bert-model ${BERT_MODEL} \
		--allow-online ${ALLOW_ONLINE} \
		$(if $(strip ${INIT_HF_DIR}),--init-hf-dir ${INIT_HF_DIR},) \
		--train-file train.csv \
		--eval-file eval.csv \
		--num-train-epochs 16 \
		--train-batch-size 16 \
		--eval-batch-size 32 \
		--learning-rate 2e-5 \
		--lr-scheduler-type ${LR_SCHEDULER_TYPE} \
		--warmup-ratio ${WARMUP_RATIO} \
		--warmup-steps ${WARMUP_STEPS} \
		--early-stopping-patience 3
else
	@echo "设置环境变量：UV_LINK_MODE=copy"
	@export UV_LINK_MODE=copy && uv run src/train.py \
		--outdir ${outdir} \
		--modelsdir ${modelsdir} \
		--outmodel ${model} \
		--max-epochs 100 \
		--tfidf-analyzer char_wb \
		--ngram-min 2 \
		--ngram-max 4 \
		--tfidf-max-features 200000 \
		--sgd-alpha 0.0001 \
		--sgd-penalty l2 \
		--resample-method none \
		--class-weight-balanced \
		--backend torch \
		--device cuda \
		--train-file train.csv \
		--eval-file eval.csv \
		--calibrate none \
		--shuffle
endif

eval:
ifeq ($(ALGO),bert)
	@echo "设置环境变量：UV_LINK_MODE=copy"
	@mkdir -p ./log
	@export UV_LINK_MODE=copy && uv run src/eval_bert.py \
		--modelsdir ./checkpoints/${month}_bert \
		--model ${month}_${ALGO}.joblib \
		--outdir ${outdir} \
		--experiment-outdir ${outdir} \
		--detailed-report \
		> ./log/${evallog}
else
	@export UV_LINK_MODE=copy && uv run src/eval.py --model ${model} --outdir ${outdir} > ./log/${evallog}
endif



predict:
ifeq ($(ALGO),bert)
	@echo "设置环境变量：UV_LINK_MODE=copy"
	@mkdir -p ./log
	@export UV_LINK_MODE=copy && uv run src/predict.py \
		--modelsdir ./checkpoints/${month}_bert \
		--model ${month}_${ALGO}.joblib \
		--outdir ${outdir} \
		--rerank-strategy ${RERANK_STRATEGY} \
		--rerank-seed ${RERANK_SEED} \
		> ./log/${predictlog}
else
	@echo "设置环境变量：UV_LINK_MODE=copy"
	@mkdir -p ./log
ifeq ($(BATCH),1)
	@export UV_LINK_MODE=copy && uv run src/predict.py \
		--model ${model} \
		--outdir ${outdir} \
		--batch-path ${BATCH_PATH} \
		--batch-ooc-output ${BATCH_OOC_OUTPUT} \
		$(if $(BATCH_DISABLE_RERANK),--batch-disable-rerank,) \
		--rerank-strategy ${RERANK_STRATEGY} \
		--rerank-seed ${RERANK_SEED} \
		> ./log/${predictlog}
else
	@export UV_LINK_MODE=copy && uv run src/predict.py \
		--model ${model} \
		--outdir ${outdir} \
		--rerank-strategy ${RERANK_STRATEGY} \
		--rerank-seed ${RERANK_SEED} \
		> ./log/${predictlog}
endif
endif

# 删除.gitignore对应的所有内容
clean:
	rm -rf output log .vscode *__pycache__

# 学习率搜索
lr-search:
	@echo "设置环境变量：UV_LINK_MODE=copy"
	@export UV_LINK_MODE=copy && uv run src/bert_lr_search.py \
		--bert-model ${BERT_MODEL} \
		$(if $(strip ${INIT_HF_DIR}),--init-hf-dir ${INIT_HF_DIR},) \
		--data-dir ${outdir} \
		--learning-rates 1e-5 3e-5 5e-5 1e-4 \
		--scheduler-types cosine linear polynomial \
		--warmup-ratios 0.0 0.1 0.2 \
		--patience 3

# 学习率监控
lr-monitor:
	@echo "设置环境变量：UV_LINK_MODE=copy"
	@export UV_LINK_MODE=copy && uv run src/learning_rate_monitor.py --mode auto

# 安装依赖
install-deps:
	@echo "=== BMW-item 依赖安装 ==="
	@echo "尝试使用 UV 安装依赖..."
	@export UV_LINK_MODE=copy && uv add -r requirements.txt || \
	 (echo "UV 安装失败，尝试使用 pip..." && pip install -r requirements.txt) || \
	 (echo "pip 安装失败，尝试使用国内镜像..." && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple) || \
	 (echo "所有方法都失败了，请查看 install_deps.sh 脚本获取更多选项")

# 快速安装（仅核心依赖）
install-core:
	@echo "安装核心依赖..."
	@export UV_LINK_MODE=copy && uv add torch transformers scikit-learn pandas numpy matplotlib seaborn joblib || \
	 pip install torch transformers scikit-learn pandas numpy matplotlib seaborn joblib

# 帮助信息
help:
	@echo "BMW-item BERT训练系统 - 使用说明"
	@echo ""
	@echo "基本用法："
	@echo "  make ALGO=bert all                    # 完整流程（数据准备+训练+评估）"
	@echo "  make ALGO=bert train                  # 仅训练BERT模型"
	@echo "  make ALGO=bert eval                   # 评估模型"
	@echo "  make ALGO=bert predict                # 预测"
	@echo "  make predict BATCH=1 BATCH_PATH=...   # 批量预测 CSV，输出 not-in-train 报表"
	@echo ""
	@echo "学习率优化功能："
	@echo "  make lr-search                        # 自动学习率搜索"
	@echo "  make lr-monitor                       # 学习率监控可视化"
	@echo ""
	@echo "依赖安装："
	@echo "  make install-deps                     # 安装所有依赖（多种方法尝试）"
	@echo "  make install-core                     # 仅安装核心依赖"
	@echo "  ./install_deps.sh                     # 运行安装脚本（更多选项）"
	@echo ""
	@echo "参数配置："
	@echo "  ALGO=berty|tfidf                    # 算法选择（默认：bert）"
	@echo "  month=7                             # 数据月份（默认：7）"
	@echo "  BERT_MODEL=chinese-macbert-base      # BERT模型名称"
	@echo "  ALLOW_ONLINE=0                       # 是否允许在线下载（默认：0）"
	@echo "  INIT_HF_DIR=./models/hfl/...         # 本地HF模型目录"
	@echo ""
	@echo "学习率参数："
	@echo "  LR_SCHEDULER_TYPE=cosine             # 调度器类型：cosine/linear/polynomial等"
	@echo "  WARMUP_RATIO=0.1                    # 预热比例（默认：0.1）"
	@echo "  WARMUP_STEPS=0                      # 预热步数（默认：0）"
	@echo ""
	@echo "示例："
	@echo "  make ALGO=bert LR_SCHEDULER_TYPE=cosine WARMUP_RATIO=0.1 train"
	@echo "  make lr-search"
	@echo "  make lr-monitor"
	@echo "  make install-deps"
	@echo ""
	@echo "更多信息请参考："
	@echo "  - README_LEARNING_RATE.md（学习率优化指南）"
	@echo "  - README_LOCAL_BERT.md（本地BERT使用指南）"
	@echo "  - README_UV_WARNING.md（UV安装问题解决方案）"

.PHONY: all train eval predict data lr-search lr-monitor install-deps install-core help

