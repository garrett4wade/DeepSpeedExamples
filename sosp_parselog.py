import json
import os
import subprocess
import numpy as np
import itertools
import pandas as pd
from collections import defaultdict
import argparse
import pickle
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

def get_default_n_gpus(model_size: int):
    if model_size in [7]:
        return 8
    elif model_size == 13:
        return 16
    elif model_size in [34]:
        return 32
    elif model_size == 70:
        return 64

def _parselog(
    actor_size: int,
    critic_size: int,
    actor_zero_stage: int,
    critic_zero_stage: int,
    seqlen: int,
    gen_bs: int,
    offload: bool,
    gpu_scale_factor: int,
):
    exp_name = f"rerun-dschat-a{actor_size}-z{actor_zero_stage}-c{critic_size}r7-cz{critic_zero_stage}-seqlen{seqlen}-g{gen_bs}"
    if offload:
        exp_name += "-offload"
    if gpu_scale_factor > 1:
        exp_name += f"-x{gpu_scale_factor:.1f}"
    logpath = f"/lustre/aigc/llm/logs/fw/{exp_name}/benchmark/rlhf-0"
    oom = False
    time_records = []
    tflops_records = []
    thpt_records = []
    max_mem = 0.0
    gen_time_record = []
    inf_time_record = []
    train_time_record = []
    try:
        with open(logpath, "r", errors="ignore") as f:
            lines = f.readlines()
            for line in lines:
                if "CUDA out of memory" in line or "not enough memory" in line:
                    oom = True
                    break
                elif (
                    "torch.distributed.DistBackendError: NCCL error in: /tmp/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1331, internal error - please report this issue to the NCCL developers, NCCL version 2.18.5"
                    in line
                ):
                    oom = True
                    break
                if "End-to-End" in line:
                    step_time = float(line.split("End-to-End => Latency: ")[1].split("s,")[0])
                    time_records.append(step_time)
                    tflops = float(line.split("TFLOPs: ")[1].split(",")[0])
                    tflops_records.append(tflops)
                    thpt = float(line.split(", Samples/sec: ")[1].split(",")[0])
                    thpt_records.append(thpt)
                if "Compute Utilization - " in line:
                    mem = float(line.split("Used Memory - ")[1].split("MB,")[0])
                    max_mem = max(max_mem, mem)
                if "pure generate time" in line:
                    gen_time = float(line.split("pure generate time ")[1].split("s")[0])
                    gen_time_record.append(gen_time)
                    inf_time = float(line.split(", inf time ")[1].split("s")[0])
                    inf_time_record.append(inf_time)
                    train_time = float(line.split(", train time ")[1].split("s")[0])
                    train_time_record.append(train_time * 4)
    except FileNotFoundError:
        return False
    time_records = time_records[2:]
    if not oom:
        if len(time_records) == 0 or len(tflops_records) == 0 or len(thpt_records) == 0 or max_mem == 0.0:
            return False
        avg_time = np.mean(time_records[1:])
        var_time = np.var(time_records)
        min_time = np.min(time_records)
        max_time = np.max(time_records)
        cil, cih = t_score_ci(time_records)
        n_time = len(time_records)
        avg_train_time = np.mean(train_time_record)
        avg_gen_time = np.mean(gen_time_record)
        avg_inf_time = np.mean(inf_time_record)
        avg_tflops = np.mean(tflops_records[1:])
        thpt = np.mean(thpt_records[1:])
    else:
        avg_time = float("inf")
        avg_train_time = float("inf")
        avg_gen_time = avg_actor_train_time = avg_critic_train_time = avg_inf_time = float("inf")
        n_time = 0
        cil = cih = float("nan")
        min_time = float("nan")
        max_time = float("nan")
        var_time = float("nan")
        avg_tflops = -float("inf")
        thpt = -float("inf")
        max_mem = 0.0
    assert actor_zero_stage == critic_zero_stage
    d = dict(
        a=actor_size,
        c=critic_size,
        n_gpus=gpu_scale_factor * get_default_n_gpus(max(actor_size, critic_size)),
        z=actor_zero_stage,
        s=seqlen,
        # gen_bs=gen_bs,
        offload=offload,
        avg_t=avg_time,
        var_t=var_time,
        min_t=min_time,
        max_t=max_time,
        cil=cil,
        cih=cih,
        avg_gt=avg_gen_time,
        avg_tt=avg_train_time,
        avg_it=avg_time - avg_train_time - avg_gen_time,
        avg_att=avg_train_time / (actor_size + critic_size) * actor_size,
        avg_ctt=avg_train_time / (actor_size + critic_size) * critic_size,
        avg_rfit=avg_inf_time / (actor_size + critic_size * 2) * actor_size,
        avg_cit=avg_inf_time / (actor_size + critic_size * 2) * critic_size,
        avg_rit=avg_inf_time / (actor_size + critic_size * 2) * critic_size,
        n=n_time,
        # OOM=oom,
        # Throughput=thpt,
        # MaxGPUMemory=max_mem,
        # avg_tflops=avg_tflops,
        log_path=logpath,
    )
    for k, v in d.items():
        benchmark_db[k].append(v)
    return True


def parselog(actor_size: int, critic_size: int):
    # takeaways
    # 1. when actor zero=3, critic zero=2 or 3 makes no difference. So by default we set critic zero=3;
    # 2. when actor zero=2, inference_tp_size must be 1;
    # 3. max GPU memory used is usually determined by gen_bs;
    gpu_scale_factors = [1, 2, 4, 8]
    critic_zero_stages = [3, 2]
    actor_zero_stages = [3, 2]
    seqlens_global_bs = [(128, 512), (384, 256), (896, 128)]
    offloads = [True, False]
    for gpu_scale_factor in gpu_scale_factors:
        if max(actor_size, critic_size) == 7:
            n_gpus = 8 * gpu_scale_factor
        elif max(actor_size, critic_size) == 13:
            n_gpus = 16 * gpu_scale_factor
        elif max(actor_size, critic_size) == 34:
            n_gpus = 32 * gpu_scale_factor
        elif max(actor_size, critic_size) == 70:
            n_gpus = 64 * gpu_scale_factor
        for critic_zero_stage, actor_zero_stage in itertools.product(critic_zero_stages, actor_zero_stages):
            for (max_answer_len, global_bs), offload in itertools.product(seqlens_global_bs, offloads):
                gen_bs = global_bs // n_gpus
                _parselog(
                    actor_size,
                    critic_size,
                    actor_zero_stage,
                    critic_zero_stage,
                    max_answer_len,
                    gen_bs,
                    offload,
                    gpu_scale_factor,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor_size", "-a", type=int, default=[7, 13, 34, 70], choices=[7, 13, 34, 70], nargs="+"
    )
    parser.add_argument(
        "--critic_size", "-c", type=int, default=[7, 13, 34, 70], choices=[7, 13, 34, 70], nargs="+"
    )
    parser.add_argument("--max", action="store_true")
    parser.add_argument("--dump_to_file", type=str, default=None)
    parser.add_argument("--no_print", action="store_true")
    args = parser.parse_args()
    for actor_size, critic_size in itertools.product(args.actor_size, args.critic_size):
        parselog(actor_size, critic_size)
    df = pd.DataFrame(benchmark_db)
    if args.max:
        df = df.loc[df.groupby(["a", "s", "c", "n_gpus"])["avg_t"].idxmin()]
    if not args.no_print:
        print(df.to_string(index=False))
    if args.dump_to_file:
        with open(args.dump_to_file, "wb") as f:
            pickle.dump(df, f)
