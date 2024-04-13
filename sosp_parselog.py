import json
import os
import subprocess
import numpy as np
import itertools
import pandas as pd
from collections import defaultdict
import argparse
import pickle

benchmark_db = defaultdict(list)


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
    except FileNotFoundError:
        return False
    if not oom:
        if len(time_records) == 0 or len(tflops_records) == 0 or len(thpt_records) == 0 or max_mem == 0.0:
            return False
        avg_time = np.mean(time_records[1:])
        avg_tflops = np.mean(tflops_records[1:])
        thpt = np.mean(thpt_records[1:])
    else:
        avg_time = float("inf")
        avg_tflops = -float("inf")
        thpt = -float("inf")
        max_mem = 0.0
    d = dict(
        actor_size=actor_size,
        critic_size=critic_size,
        actor_zero_stage=actor_zero_stage,
        critic_zero_stage=critic_zero_stage,
        seqlen=seqlen,
        gen_bs=gen_bs,
        offload=offload,
        avg_time=avg_time,
        OOM=oom,
        Throughput=thpt,
        MaxGPUMemory=max_mem,
        avg_tflops=avg_tflops,
        gpu_scale_factor=gpu_scale_factor,
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
        df = df.loc[df.groupby(["actor_size", "seqlen", "critic_size", "gpu_scale_factor"])["Throughput"].idxmax()]
    if not args.no_print:
        print(df.to_string(index=False))
    if args.dump_to_file:
        with open(args.dump_to_file, "wb") as f:
            pickle.dump(df, f)
