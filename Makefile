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
	uv run src/train.py --outdir ${outdir} --outmodel ${model} > ./log/${trainlog}

eval:
	uv run src/eval.py  --model ${model} --outdir ${outdir} > ./log/${evallog}

predict:
	uv run src/predict.py --model ${model} --outdir ${outdir} > ./log/${predictlog}

# 删除.gitignore对应的所有内容
clean:


.PHONY: all train eval predict data

