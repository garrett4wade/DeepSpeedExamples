import json
import os
import subprocess
import numpy as np
import itertools


def _parselog(model_size: int, actor_zero_stage: int, critic_zero_stage: int):
    exp_name = f"sosp-baseline-dschat-a{model_size}-z{actor_zero_stage}-c7r7-cz{critic_zero_stage}"
    logpath = f"/lustre/aigc/llm/logs/fw/{exp_name}/benchmark/rlhf-0"
    oom = False
    time_records = []
    tflops_records = []
    with open(logpath, "r", errors="ignore") as f:
        lines = f.readlines()
        for line in lines:
            if "CUDA out of memory" in line or "not enough memory" in line:
                oom = True
                break
            if "End-to-End" in line:
                step_time = float(line.split("End-to-End => Latency: ")[1].split("s,")[0])
                time_records.append(step_time)
                tflops = float(line.split("TFLOPs: ")[1].split(",")[0])
                tflops_records.append(tflops)
    time_records = time_records[2:17]
    if oom:
        print(
            f"Model size: {model_size}, Actor zero stage: {actor_zero_stage}, Critic zero stage: {critic_zero_stage}, OOM"
        )
    else:
        print(
            f"Model size: {model_size}, Actor zero stage: {actor_zero_stage}, Critic zero stage: {critic_zero_stage}"
            f", Average step time: {np.mean(time_records):.2f}, average TFLOPS: {np.mean(tflops_records):.2f}"
        )


def parselog(model_size: int):
    print("-" * 40)
    for actor_zero_stage, critic_zero_stage in itertools.product([3, 2], [3, 2]):
        _parselog(model_size, actor_zero_stage, critic_zero_stage)
    print("-" * 40)


if __name__ == "__main__":
    for model_size in [7, 13, 34, 70]:
        parselog(model_size)
