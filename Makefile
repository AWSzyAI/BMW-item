month=8
outdir=./output/2025_up_to_month_${month}
model=${month}.joblib
trainlog=${month}_train.txt
evallog=${month}_eval.txt
predictlog=${month}_predict.txt

all: train eval

train:
	uv run src/train.py --outdir ${outdir} --outmodel ${model} > ./log/${trainlog}

eval:
	uv run src/eval.py  --model ${model} --outdir ${outdir} > ./log/${evallog}

predict:
	uv run src/predict.py --model ${model} --outdir ${outdir} > ./log/${predictlog}

.PHONY: all train eval predict

