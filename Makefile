# 模型开关：tfidf 或 bert
# 用法：`make ALGO=bert all` 或在文件顶部改为 ALGO=bert
# ALGO?=tfidf
ALGO?=bert

month=7
outdir=./output/2025_up_to_month_${month}
model=${month}_${ALGO}.joblib
trainlog=${month}_train.txt
evallog=${month}_eval.txt
predictlog=${month}_predict.txt
labelcol=extern_id
# labelcol=linked_items

# 统一的模型目录变量（两条路线都会用到）
modelsdir=./models

# BERT 相关参数（当 ALGO=bert 时使用）
# - 默认使用在线下载 bert-base-chinese；如需离线本地目录，请设置 INIT_HF_DIR 指向本地HF模型目录
#   例如：`make ALGO=bert INIT_HF_DIR=./models/bert-base-chinese`（目录需包含config.json与tokenizer）
BERT_MODEL?=chinese-macbert-base
ALLOW_ONLINE?=0
INIT_HF_DIR?=./models/hfl/chinese-macbert-base


all: data train eval

data:
	uv run src/2025.py --outdir ${outdir} --month ${month} --label-col ${labelcol}
	uv run src/5-fold.py --outdir ${outdir} --label-col ${labelcol}

train:
ifeq ($(ALGO),bert)
	uv run src/train_bert.py \
		--outdir ${outdir} \
		--modelsdir ${modelsdir} \
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
		--early-stopping-patience 3
else
	uv run src/train.py \
		--outdir ${outdir} \
		--modelsdir ${modelsdir} \
		--outmodel ${model} \
		--max-epochs 60 \
		--tfidf-analyzer char_wb \
		--ngram-min 2 \
		--ngram-max 4 \
		--tfidf-max-features 100000 \
		--sgd-alpha 0.0001 \
		--sgd-penalty l2 \
		--resample-method ros \
		--train-file train.csv \
		--eval-file eval.csv \
		--calibrate none \
		--shuffle
endif

eval:
ifeq ($(ALGO),bert)
	uv run src/eval_bert.py --modeldir ${modelsdir} --model ${model} --outdir ${outdir} > ./log/${evallog}
else
	uv run src/eval.py --model ${model} --outdir ${outdir} > ./log/${evallog}
endif

predict:
ifeq ($(ALGO),bert)
	uv run src/predict_bert.py --modelsdir ${modelsdir} --model ${model} --outdir ${outdir} > ./log/${predictlog}
else
	uv run src/predict.py --model ${model} --outdir ${outdir} > ./log/${predictlog}
endif

# 删除.gitignore对应的所有内容
clean:
	rm -r output log


.PHONY: all train eval predict data

