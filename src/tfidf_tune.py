#!/usr/bin/env python3
import argparse, os, sys, itertools, subprocess, json, time, glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

def run_cmd(cmd, env=None):
    start = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    out_lines = []
    for line in p.stdout:  # noqa: E701
        out_lines.append(line)
        print(line, end="")
    p.wait()
    return p.returncode, "".join(out_lines), time.time() - start

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-outdir", type=str, required=True,
                        help="数据目录（包含 train/eval/test_X,y 或 train.csv/eval.csv）")
    parser.add_argument("--exp-root", type=str, default="./output/tfidf_tune",
                        help="实验结果根目录（每组配置建子目录）")
    parser.add_argument("--modelsdir", type=str, default="./models", help="模型目录")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--resample-method", type=str, default="none",
                        choices=["none", "ros", "smote", "smoteenn", "smotetomek"])
    parser.add_argument("--calibrate", type=str, default="none", choices=["none", "sigmoid", "isotonic"])
    parser.add_argument("--class-weight-balanced", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=32, help="并行进程数（建议 <= CPU 物理核）")

    # 搜索空间（可按需修改/扩展）
    parser.add_argument("--analyzers", type=str, nargs="+", default=["char", "char_wb"])
    parser.add_argument("--ngram-mins", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--ngram-maxs", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--max-features", type=int, nargs="+", default=[20000, 50000, 100000])
    parser.add_argument("--sgd-alphas", type=float, nargs="+", default=[1e-4, 5e-5, 1e-5])
    parser.add_argument("--sgd-penalties", type=str, nargs="+", default=["l2", "elasticnet"])
    parser.add_argument("--shuffle", action="store_true")

    args = parser.parse_args()
    Path(args.exp_root).mkdir(parents=True, exist_ok=True)
    Path(args.modelsdir).mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(
        args.analyzers, args.ngram_mins, args.ngram_maxs, args.max_features,
        args.sgd_alphas, args.sgd_penalties
    ))
    # 过滤非法 ngram（min<=max）
    grid = [g for g in grid if g[1] <= g[2]]

    def build_exp(analyzer, nmin, nmax, maxf, alpha, penalty):
        exp = f"a{analyzer}_n{nmin}-{nmax}_f{maxf}_alpha{alpha}_pen{penalty}_{args.resample_method}"
        outdir = os.path.join(args.exp_root, exp)
        return exp, outdir

    tasks = []
    for (analyzer, nmin, nmax, maxf, alpha, penalty) in grid:
        exp, outdir = build_exp(analyzer, nmin, nmax, maxf, alpha, penalty)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "src/train.py",
            "--outdir", outdir,
            "--modelsdir", args.modelsdir,
            "--outmodel", f"{exp}.joblib",
            "--max-epochs", str(args.max_epochs),
            "--tfidf-analyzer", analyzer,
            "--ngram-min", str(nmin),
            "--ngram-max", str(nmax),
            "--tfidf-max-features", str(maxf),
            "--sgd-alpha", str(alpha),
            "--sgd-penalty", penalty,
            "--resample-method", args.resample_method,
            "--calibrate", args.calibrate,
        ]
        if args.class_weight_balanced:
            cmd.append("--class-weight-balanced")
        if args.shuffle:
            cmd.append("--shuffle")

        # 提示：train/eval 的读取由 train.py 内部在 --outdir 下完成（支持 X/y 或单表）
        tasks.append((exp, outdir, cmd))

    print(f"共 {len(tasks)} 组实验，将并行执行：n_jobs={args.n_jobs}")
    results = []
    env = os.environ.copy()

    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        fut2meta = {ex.submit(run_cmd, t[2], env): (t[0], t[1]) for t in tasks}
        for fut in as_completed(fut2meta):
            exp, outdir = fut2meta[fut]
            rc, logs, dur = fut.result()
            metrics_path = os.path.join(outdir, "metrics_eval.csv")
            rec = {
                "experiment": exp, "outdir": outdir, "returncode": rc,
                "duration_sec": round(dur, 2),
                "metrics_path": (metrics_path if os.path.exists(metrics_path) else ""),
            }
            # 读取 metrics（如果有）
            if os.path.exists(metrics_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(metrics_path)
                    if len(df) > 0:
                        for k, v in df.iloc[0].to_dict().items():
                            rec[k] = v
                except Exception:
                    pass
            results.append(rec)

    # 汇总保存
    df_sum = pd.DataFrame(results)
    sum_path = os.path.join(args.exp_root, "summary_metrics.csv")
    df_sum.sort_values(["returncode", "experiment"]).to_csv(sum_path, index=False)
    print("汇总结果已写入：", sum_path)

if __name__ == "__main__":
    main()