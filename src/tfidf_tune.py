#!/usr/bin/env python3
import argparse, os, sys, itertools, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from tqdm import tqdm  # 进度条

# =============== Helpers ===============

def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except Exception:
        import shutil
        shutil.copy(src, dst)

def ensure_data_in_exp_dir(
    exp_dir: Path,
    data_dir: Path,
    train_base: str,
    eval_base: str
) -> None:
    """
    确保 exp_dir 下具备 train/eval 所需数据文件。
    支持：
      - 单表：train.csv / eval.csv
      - 分表：train_X.csv + train_y.csv / eval_X.csv + eval_y.csv
    """
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 单表路径
    train_csv = data_dir / train_base
    eval_csv  = data_dir / eval_base

    # 分表路径（与 train_base 无关，按约定命名）
    train_X = data_dir / "train_X.csv"
    train_y = data_dir / "train_y.csv"
    eval_X  = data_dir / "eval_X.csv"
    eval_y  = data_dir / "eval_y.csv"

    have_single = train_csv.exists() and eval_csv.exists()
    have_split  = train_X.exists()   and train_y.exists() and eval_X.exists() and eval_y.exists()

    if not have_single and not have_split:
        existing = sorted([p.name for p in data_dir.glob("*.csv")])
        raise FileNotFoundError(
            "未在数据目录找到可用的训练/评估数据。\n"
            f"- 期望其一：\n"
            f"  1) 单表：{train_base} 和 {eval_base}\n"
            f"  2) 分表：train_X.csv, train_y.csv, eval_X.csv, eval_y.csv\n"
            f"- 数据目录：{data_dir}\n"
            f"- 实际存在：{existing}"
        )

    if have_single:
        _link_or_copy(train_csv, exp_dir / train_base)
        _link_or_copy(eval_csv,  exp_dir / eval_base)

    if have_split:
        _link_or_copy(train_X, exp_dir / "train_X.csv")
        _link_or_copy(train_y, exp_dir / "train_y.csv")
        _link_or_copy(eval_X,  exp_dir / "eval_X.csv")
        _link_or_copy(eval_y,  exp_dir / "eval_y.csv")


def run_cmd(cmd, env=None, cwd=None, log_path: Path | None = None):
    """运行子进程，同步打印 stdout，并写入日志文件；返回 (rc, duration_sec)"""
    start = time.time()
    log_fh = open(log_path, "w", encoding="utf-8") if log_path else None
    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=cwd
        )
        for line in p.stdout:
            if log_fh:
                log_fh.write(line)
            print(line, end="")
        p.wait()
        rc = p.returncode
    finally:
        if log_fh:
            log_fh.flush()
            log_fh.close()
    return rc, round(time.time() - start, 2)

# =============== New: resume/summary helpers ===============

def metrics_csv_path(outdir: Path) -> Path:
    return outdir / "metrics_eval.csv"

def is_exp_done(outdir: Path) -> bool:
    """判断该实验是否已经完成（可用于断点续跑的跳过条件）"""
    mpath = metrics_csv_path(outdir)
    if not mpath.exists():
        return False
    try:
        df = pd.read_csv(mpath)
        return len(df) > 0
    except Exception:
        return False

def load_metrics_record(exp: str, outdir: Path, model_path: Path, log_path: Path,
                        returncode: int | None = None, duration: float | None = None) -> dict:
    """从 metrics_eval.csv 读取一行指标，组装统一的记录字典"""
    rec = {
        "experiment": exp,
        "outdir": str(outdir),
        "returncode": (returncode if returncode is not None else ""),
        "duration_sec": (duration if duration is not None else ""),
        "log_path": str(log_path),
        "metrics_path": "",
        "status": "",  # done|running|failed|skipped
    }
    mpath = metrics_csv_path(outdir)
    if mpath.exists():
        rec["metrics_path"] = str(mpath)
        try:
            df = pd.read_csv(mpath)
            if len(df) > 0:
                for k, v in df.iloc[0].to_dict().items():
                    rec[k] = v
        except Exception as e:
            rec["metrics_read_error"] = str(e)
    return rec

def write_summary(exp_root: Path, records: list[dict]) -> Path:
    """将所有记录写入 summary_metrics.csv（去重按 experiment 保留最后一条）"""
    sum_path = exp_root / "summary_metrics.csv"
    if len(records) == 0:
        # 若此前已有旧文件，不清空；仅在有新记录时才覆盖更新
        return sum_path
    df = pd.DataFrame(records)
    # 去重：最后出现的覆盖之前的（保持“最新状态”）
    df = df.drop_duplicates(subset=["experiment"], keep="last")
    df.sort_values(by=["status", "returncode", "experiment"], inplace=True, na_position="last")
    df.to_csv(sum_path, index=False, encoding="utf-8")
    return sum_path

# =============== Main ===============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-outdir", type=str, required=True,
                        help="数据目录（包含 train/eval 单表或 train_X,y / eval_X,y 分表）")
    parser.add_argument("--exp-root", type=str, default="./output/tfidf_tune",
                        help="实验结果根目录（每组配置建子目录）")
    parser.add_argument("--modelsdir", type=str, default="./models", help="模型目录")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--resample-method", type=str, default="none",
                        choices=["none", "ros", "smote", "smoteenn", "smotetomek"])
    parser.add_argument("--calibrate", type=str, default="none", choices=["none", "sigmoid", "isotonic"])
    parser.add_argument("--class-weight-balanced", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=32, help="并行进程数（建议 <= CPU 物理核）")

    # 兼容可自定义单表文件名（默认 train.csv/eval.csv）
    parser.add_argument("--train-basename", type=str, default="train.csv",
                        help="单表训练文件名（与数据目录中的文件名对应）")
    parser.add_argument("--eval-basename", type=str, default="eval.csv",
                        help="单表评估文件名（与数据目录中的文件名对应）")

    # 搜索空间
    parser.add_argument("--analyzers", type=str, nargs="+", default=["char", "char_wb"])
    parser.add_argument("--ngram-mins", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--ngram-maxs", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--max-features", type=int, nargs="+", default=[20000, 50000, 100000])
    parser.add_argument("--sgd-alphas", type=float, nargs="+", default=[1e-4, 5e-5, 1e-5])
    parser.add_argument("--sgd-penalties", type=str, nargs="+", default=["l2", "elasticnet"])
    parser.add_argument("--shuffle", action="store_true")

    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    models_dir = Path(args.modelsdir)
    data_dir = Path(args.data_outdir)

    exp_root.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 生成网格并过滤非法 ngram
    grid = list(itertools.product(
        args.analyzers, args.ngram_mins, args.ngram_maxs, args.max_features,
        args.sgd_alphas, args.sgd_penalties
    ))
    grid = [g for g in grid if g[1] <= g[2]]

    def build_exp(analyzer, nmin, nmax, maxf, alpha, penalty):
        exp = f"a{analyzer}_n{nmin}-{nmax}_f{maxf}_alpha{alpha}_pen{penalty}_{args.resample_method}"
        outdir = exp_root / exp
        return exp, outdir

    # 限制 BLAS/OMP 线程，避免过度并行
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    # 项目根目录（src 的上一级），确保以此为 cwd 调用 train.py
    project_root = Path(__file__).resolve().parent.parent

    # ====== 构建全部任务，并划分：已完成 / 待运行 ======
    all_tasks_meta = []  # [(exp, outdir, model_path, log_path, cmd)]
    for (analyzer, nmin, nmax, maxf, alpha, penalty) in grid:
        exp, outdir = build_exp(analyzer, nmin, nmax, maxf, alpha, penalty)
        # 先把数据软链/复制到 outdir（必要时）
        ensure_data_in_exp_dir(outdir, data_dir, args.train_basename, args.eval_basename)

        model_path = models_dir / f"{exp}.joblib"
        log_path = outdir / "train_eval.log"

        cmd = [
            sys.executable, "src/train.py",
            "--outdir", str(outdir),
            "--modelsdir", str(models_dir),
            "--outmodel", str(model_path.name),
            "--max-epochs", str(args.max_epochs),
            "--tfidf-analyzer", analyzer,
            "--ngram-min", str(nmin),
            "--ngram-max", str(nmax),
            "--tfidf-max-features", str(maxf),
            "--sgd-alpha", str(alpha),
            "--sgd-penalty", penalty,
            "--resample-method", args.resample_method,
            "--calibrate", args.calibrate,
            "--train-file", args.train_basename,
            "--eval-file", args.eval_basename,
        ]
        if args.class_weight_balanced:
            cmd.append("--class-weight-balanced")
        if args.shuffle:
            cmd.append("--shuffle")

        all_tasks_meta.append((exp, outdir, model_path, log_path, cmd))

    # 将已完成的实验收集入 results（status=done），并且不重复提交
    results = []
    pending = []
    for (exp, outdir, model_path, log_path, cmd) in all_tasks_meta:
        if is_exp_done(outdir):
            rec = load_metrics_record(exp, outdir, model_path, log_path,
                                      returncode=0, duration=None)
            rec["status"] = "done"
            results.append(rec)
        else:
            pending.append((exp, outdir, model_path, log_path, cmd))

    total = len(all_tasks_meta)
    done0 = total - len(pending)
    print(f"总实验数：{total} | 已完成：{done0} | 待运行：{len(pending)}")
    # 先把当前已完成的写一版 summary，便于你马上分析
    sum_path = write_summary(exp_root, results)
    if done0 > 0:
        print(f"已将已完成实验写入：{sum_path}")

    # ====== 提交剩余任务并带总体进度条 ======
    if len(pending) == 0:
        print("没有剩余任务需要运行。")
        return

    print(f"将并行执行剩余任务：n_jobs={args.n_jobs}")
    with tqdm(total=total, initial=done0, desc="Overall progress") as pbar:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            fut2meta = {
                ex.submit(run_cmd, cmd, env, project_root, log_path): (exp, outdir, model_path, log_path)
                for (exp, outdir, model_path, log_path, cmd) in pending
            }
            for fut in as_completed(fut2meta):
                exp, outdir, model_path, log_path = fut2meta[fut]
                try:
                    rc, dur = fut.result()
                except Exception as e:
                    rc, dur = 1, -1
                    print(f"[{exp}] 运行异常：{e}")

                rec = load_metrics_record(exp, outdir, model_path, log_path, returncode=rc, duration=dur)
                rec["status"] = "done" if (rc == 0 and is_exp_done(outdir)) else ("failed" if rc != 0 else "unknown")
                results.append(rec)

                # 实时刷新汇总文件 & 进度条
                sum_path = write_summary(exp_root, results)
                pbar.update(1)
                pbar.set_postfix_str(f"last={exp} rc={rc}")

    # ====== 最终收尾 ======
    sum_path = write_summary(exp_root, results)
    # 最后再输出一份统计
    ok = sum(1 for r in results if r.get("status") == "done")
    fail = sum(1 for r in results if r.get("status") == "failed")
    print(f"全部结束：完成 {ok}/{total}，失败 {fail}，汇总：{sum_path}")

if __name__ == "__main__":
    main()