month=2
outdir=./output/2025_up_to_month_${month}
model=${month}.joblib
trainlog=${month}_train.txt
evallog=${month}_eval.txt
predictlog=${month}_predict.txt

all: data train eval

data:
	uv run src/5-fold.py --outdir ${outdir}

train:
	uv run src/train.py \
		--outdir ./output/2025_up_to_month_7 \
		--modelsdir ./models \
		--outmodel achar_wb_n2-4_f100000_alpha0.0001_penl2_ros.joblib \
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

eval:
	uv run src/eval.py  --model ${model} --outdir ${outdir} > ./log/${evallog}

predict:
	uv run src/predict.py --model ${model} --outdir ${outdir} > ./log/${predictlog}

# 删除.gitignore对应的所有内容
clean:


.PHONY: all train eval predict data

