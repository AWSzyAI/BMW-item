from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    # 通用参数（与现有 train.py/BERT.py 对齐）
    parser.add_argument("--strategy", type=str, default="tfidf", choices=["tfidf", "bert"], help="训练策略：tfidf 或 bert")
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--eval-file", type=str, default="eval.csv")
    parser.add_argument("--outdir", type=str, default="./output/2025_up_to_month_9")
    parser.add_argument("--modelsdir", type=str, default="./models")
    parser.add_argument("--outmodel", type=str, default="9.joblib")

    # TF-IDF 相关（透传给 train.py）
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--loss-sample-size", type=int, default=20000)
    parser.add_argument("--tfidf-analyzer", type=str, default="char", choices=["char", "char_wb"])
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--tfidf-max-features", type=int, default=100_000)
    parser.add_argument("--sgd-penalty", type=str, default="l2", choices=["l2", "l1", "elasticnet"])
    parser.add_argument("--sgd-alpha", type=float, default=0.0001)
    parser.add_argument("--sgd-average", action="store_true")
    parser.add_argument("--class-weight-balanced", action="store_true")
    parser.add_argument("--calibrate", type=str, default="none", choices=["none", "sigmoid", "isotonic"])
    parser.add_argument("--ooc-tau-percentile", type=float, default=5.0)
    parser.add_argument("--ooc-temperature", type=float, default=20.0)

    # BERT 相关（透传给 BERT.py）
    parser.add_argument("--bert-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--init-hf-dir", type=str, default=None)
    parser.add_argument("--num-train-epochs", dest="num_train_epochs", type=float, default=2.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save-hf-dir", type=str, default=None)

    args = parser.parse_args()

    if args.strategy == "tfidf":
        from src.tf_idf import train_tfidf

        train_tfidf(args)
    else:
        from BERT import main as bert_main

        bert_main(args)


if __name__ == "__main__":
    main()
