from pathlib import Path
from typing import *
from setting import MODEL_SIZE_TO_N_NODES_BAISC, get_common_flags, run_debug_cmd, run_interruptable_cmd_on_js_h100
from collections import defaultdict
import numpy as np
import pandas as pd
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
    except FileNotFoundError:
        return parse_success, oom

    time_records = time_records[2:]
    if not oom:
        if len(time_records) == 0 or len(thpt_records) == 0:
            return False, oom
        avg_time = np.mean(time_records[1:])
        var_time = np.var(time_records)
        min_time = np.min(time_records)
        max_time = np.max(time_records)
        cil, cih = t_score_ci(time_records)
        n_time = len(time_records)
        thpt = np.mean(thpt_records[1:])
    else:
        avg_time = float("inf")
        n_time = 0
        cil = cih = float("nan")
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
        plen=prompt_len,
        avg_t=avg_time,
        var_t=var_time,
        min_t=min_time,
        max_t=max_time,
        cil=cil,
        cih=cih,
        n=n_time,
        log_path=logpath,
    )
    for k, v in d.items():
        benchmark_db[k].append(v)
    return True, oom

def main(args):
    parse_success, oom = _parselog(
        actor_size=args.model_size,
        critic_size=7 if not args.scale_both else args.model_size,
        bs=256,
        ctx=2048,
        prompt_len=1024,
        nr=1,
        nt=1,
    )
    print(parse_success, oom)
    df = pd.DataFrame(benchmark_db)
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", "-x", type=int, required=True)
    parser.add_argument("--scale_both", "-s", action="store_true")
    args = parser.parse_args()
    main(args)