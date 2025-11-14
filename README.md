# BMW Intern - CASE 1: Case -> Item

把Makefile中的month换成对应的月份

```bash
python 2025.py #从data生成每个月的数据集
make
uv run src/train_bert.py --bert-model ./models/google-bert/bert-base-chinese --skip-train-stats True --stats-on-cpu True --post-train-stats-batch-size 4 --num-train-epochs 1 --train-batch-size 4 --eval-batch-size 8 --max-length 128

uv run src/train_bert.py \
  --train-file train.csv \
  --eval-file eval.csv \
  --outdir ./output/2025_up_to_month_2 \
  --modelsdir ./models \
  --outmodel bert_model.joblib \
  --bert-model ./models/google-bert/bert-base-chinese \
  --allow-online False \
  --num-train-epochs 3.0 \
  --train-batch-size 16 \
  --eval-batch-size 32 \
  --max-length 256

uv run src/predict.py --modelsdir ./models --model bert_model.joblib --outdir ./output/2025_up_to_month_2 --infile eval.csv


uv run src/eval_bert.py --modeldir ./models --model bert_model.joblib --outdir ./output/2025_up_to_month_2 --path eval.csv --mode new

uv run src/eval_bert.py --modeldir ./models --model bert_model.joblib --outdir ./output/2025_up_to_month_2 --path eval.csv --mode new --reject-threshold 0.7

uv run src/eval_bert.py --modeldir ./models --model bert_model.joblib --outdir ./output/2025_up_to_month_2 --path eval.csv --mode new --sweep-thresholds "0.5:0.9:0.05"

在 Makefile 顶部修改标签列为 extern_id：

```
labelcol=extern_id
```

然后执行：

```
make data
make train
make eval
```
```


### Result
```bash
            acc   | f1_macro | hit@1 | hit@3 | hit@5 | hit@10
1-6>>>7     0.714 | 0.226    | 0.714 | 0.784 | 0.814 | 0.849
1-7>>>8     0.733 | 0.234    | 0.733 | 0.807 | 0.832 | 0.864
1-8>>>9     0.705 | 0.237    | 0.705 | 0.780 | 0.808 | 0.839
1-9>>>10    0.415 | 0.099    | 0.415 | 0.598 | 0.653 | 0.708

fix item_title test
            acc   | f1_macro | hit@1 | hit@3 | hit@5 | hit@10
1-6>>>7     0.428 | 0.059    | 0.428 | 0.533 | 0.567 | 0.617
1-7>>>8     0.427 | 0.057    | 0.427 | 0.538 | 0.575 | 0.636
1-8>>>9     0.398 | 0.057    | 0.398 | 0.505 | 0.546 | 0.607
1-9>>>10    0.396 | 0.071    | 0.396 | 0.517 | 0.567 | 0.630

fix item_title eval 
            acc    | f1_weighted | f1_macro | hit@1    | hit@3    | hit@5    | hit@10   | AUC
1-6>>>7   0.48669  |  0.397759   | 0.060242 | 0.48669  | 0.60481  | 0.642643 | 0.699135 | 0.7461
1-7>>>8   0.477505 | 0.383731    | 0.053635 | 0.477505 | 0.594016 | 0.629632 | 0.687599 | 0.7303
1-8>>>9   0.469482 | 0.375217    | 0.050958 | 0.469482 | 0.585994 | 0.626913 | 0.68545  | 0.7271
1-9>>>10  0.457876 | 0.366022    | 0.048724 | 0.457876 | 0.576433 | 0.616728 | 0.675036 | 0.7256

```


uv pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://pypi.org/simple

```
# 示例：以月9的数据目录为例（替换为你要评测的 outdir）
export DATA_OUTDIR=./output/2025_up_to_month_2

# 小空间并行搜索（并行 32 进程，可根据 CPU/内存上调/下调）
uv run src/tfidf_tune.py \
  --data-outdir "$DATA_OUTDIR" \
  --exp-root ./output/tfidf_tune \
  --modelsdir ./models \
  --max-epochs 60 \
  --resample-method ros \
  --calibrate none \
  --n-jobs 32 \
  --analyzers char char_wb \
  --ngram-mins 2 3 \
  --ngram-maxs 4 5 \
  --max-features 20000 50000 100000 \
  --sgd-alphas 0.0001 0.00005 0.00001 \
  --sgd-penalties l2 elasticnet \
  --shuffle

export DATA_OUTDIR=./output/2025_up_to_month_2

# 小空间并行搜索（并行 32 进程，可根据 CPU/内存上调/下调）
python src/tfidf_tune.py \
  --data-outdir "$DATA_OUTDIR" \
  --exp-root ./output/tfidf_tune \
  --modelsdir ./models \
  --max-epochs 60 \
  --resample-method ros \
  --calibrate none \
  --n-jobs 32 \
  --analyzers char char_wb \
  --ngram-mins 2 3 \
  --ngram-maxs 4 5 \
  --max-features 20000 50000 100000 \
  --sgd-alphas 0.0001 0.00005 0.00001 \
  --sgd-penalties l2 elasticnet \
  --shuffle


export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python src/tfidf_tune.py \
  --data-outdir "$DATA_OUTDIR" \
  --exp-root ./output/tfidf_tune \
  --modelsdir ./models \
  --max-epochs 60 \
  --resample-method ros \
  --calibrate none \
  --n-jobs 8 \
  --analyzers char char_wb \
  --ngram-mins 2 3 \
  --ngram-maxs 4 5 \
  --max-features 20000 50000 100000 \
  --sgd-alphas 0.0001 0.00005 0.00001 \
  --sgd-penalties l2 elasticnet \
  --shuffle



set -x DATA_OUTDIR (pwd)/output/2025_up_to_month_2
ls $DATA_OUTDIR/train_X.csv $DATA_OUTDIR/train_y.csv $DATA_OUTDIR/eval_X.csv $DATA_OUTDIR/eval_y.csv

set -x OMP_NUM_THREADS 1
set -x MKL_NUM_THREADS 1
set -x OPENBLAS_NUM_THREADS 1

python src/tfidf_tune.py \
  --data-outdir $DATA_OUTDIR \
  --exp-root ./output/tfidf_tune \
  --modelsdir ./models \
  --max-epochs 60 \
  --resample-method ros \
  --calibrate none \
  --n-jobs 8 \
  --analyzers char char_wb \
  --ngram-mins 2 3 \
  --ngram-maxs 4 5 \
  --max-features 20000 50000 100000 \
  --sgd-alphas 0.0001 0.00005 0.00001 \
  --sgd-penalties l2 elasticnet \
  --shuffle


```

modelscope download --model 'google-bert/bert-base-chinese' --local_dir './models'

BERT
```
python src/train_bert.py \
    --bert-model ./models/bert-base-chinese \
    --num-train-epochs 3.0 \
    --train-batch-size 16 \
    --eval-batch-size 32

python src/predict_bert.py \
    --modelsdir ./models \
    --model bert_model.joblib \
    --infile eval.csv

python src/eval_bert.py \
    --modeldir ./models \
    --model bert_model.joblib \
    --path eval.csv \
    --mode new

```
