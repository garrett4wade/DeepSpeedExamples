from pathlib import Path
from typing import *
from setting import MODEL_SIZE_TO_N_NODES_BAISC, get_common_flags, run_debug_cmd, run_interruptable_cmd_on_js_h100, N_NODES_TO_BATCH_SIZE, CTX, PROMPT_LEN
from collections import defaultdict
import numpy as np
import pandas as pd
import itertools
import argparse

from scipy.stats import t

pd.set_option("display.precision", 2)  # Set precision to 4 decimal places
np.set_printoptions(precision=2)  # Set precision to 4 decimal places


def t_score_ci(data):
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)

    # Define confidence level (e.g., 95%)
    confidence_level = 0.95

    # Degrees of freedom (n-1 for a sample)
    degrees_of_freedom = len(data) - 1

    # Calculate the critical value based on the confidence level and degrees of freedom
    t_score = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

    # Calculate the margin of error
    margin_of_error = t_score * (std_dev / np.sqrt(len(data)))

    # Calculate the confidence interval
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound

benchmark_db = defaultdict(list)

def _parselog(
    actor_size: int,
    critic_size: int,
    bs: int,
    ctx: int,
    prompt_len: int,
    nr: int,
    nt: int,
) -> Tuple[bool, bool]:
    logfile = Path("/mnt/bs_fs/dschat-logs")
    exp_name = "mlsys"
    trial_name = f"a{actor_size}c{critic_size}b{bs}ct{ctx}p{prompt_len}nr{nr}nt{nt}"
    logpath = str(logfile / exp_name / trial_name / "output.log")

    oom = False
    parse_success = False

    time_records = []
    thpt_records = []
    try:
        with open(logpath, "r", errors="ignore") as f:
            lines = f.readlines()
            for line in lines:
                if "CUDA out of memory" in line or "not enough memory" in line:
                    oom = True
                    break
                elif (
                    "torch.distributed.DistBackendError: NCCL error"
                    in line
                ):
                    oom = True
                    break
                if "End-to-End => " in line:
                    step_time = float(line.split("End-to-End => Latency: ")[1].split("s,")[0])
                    time_records.append(step_time)
                    thpt = float(line.split(", Samples/sec: ")[1].split(",")[0])
                    thpt_records.append(thpt)
                if "Benchmarking finishes" in line:
                    oom = False
                if "CUBLAS_STATUS_ALLOC_FAILED" in line:
                    oom = True
                if "Fatal Python error: Segmentation fault" in line:
                    oom = True
    except FileNotFoundError:
        return parse_success, oom

    # time_records = time_records[1:]
    if not oom:
        if len(time_records) == 0 or len(thpt_records) == 0:
            return False, oom
        avg_time = np.mean(time_records)
        var_time = np.var(time_records)
        min_time = np.min(time_records)
        max_time = np.max(time_records)
        n_time = len(time_records)
        thpt = np.mean(thpt_records)
    else:
        avg_time = float("inf")
        n_time = 0
        min_time = float("nan")
        max_time = float("nan")
        var_time = float("nan")
        thpt = -float("inf")
    
    w = MODEL_SIZE_TO_N_NODES_BAISC[actor_size] * 8
    d = dict(
        a=actor_size,
        c=critic_size,
        n_gpus=w,
        ctx=ctx,
        bs=bs,
        plen=prompt_len,
        avg_t=avg_time,
        log_path=logpath,
    )
    if not oom:
        for k, v in d.items():
            benchmark_db[k].append(v)
    return True, oom

def main():
    for size in [7, 13, 34, 70]:
        for scale_both in [True, False]:
            factors = [2**x for x in range(7)]
            for nr, nt in itertools.product(factors, factors):
                parse_success, oom = _parselog(
                    actor_size=size,
                    critic_size=7 if not scale_both else 13,
                    bs=N_NODES_TO_BATCH_SIZE[MODEL_SIZE_TO_N_NODES_BAISC[size]],
                    ctx=CTX,
                    prompt_len=PROMPT_LEN,
                    nr=nr,
                    nt=nt,
                )
                if parse_success and not oom:
                    break
    df = pd.DataFrame(benchmark_db)
    print(df.to_string(index=False))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_size", "-x", type=int, required=True)
    # parser.add_argument("--scale_both", "-s", action="store_true")
    # args = parser.parse_args()
    # main(args)
    main()