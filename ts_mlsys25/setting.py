import itertools
import math
import os
import signal
import socket
import subprocess
from pathlib import Path
from typing import *

IS_FRL = socket.gethostname().startswith("frl")
if IS_FRL:
    MODEL_SIZE_TO_PATH = {
        7: "/lustre/public/pretrained_model_weights/llama2/Llama-2-7b-hf",
    }
else:
    MODEL_SIZE_TO_PATH = {
        7: "/mnt/bs_fs/models/CodeLlama-7b-hf/",
        13: "/mnt/bs_fs/models/CodeLlama-13b-hf/",
        34: "/mnt/bs_fs/models/CodeLlama-34b-hf/",
        70: "/mnt/bs_fs/models/CodeLlama-70b-hf/",
    }

MODEL_SIZE_TO_N_NODES_BAISC = {7: 2, 13: 4, 34: 8, 70: 16}
MODEL_SIZE_TO_HE_TP_SIZE = {7: 1, 13: 2, 34: 4, 70: 8}


def get_common_flags(actor_size, critic_size, offload: bool):
    actor_path = MODEL_SIZE_TO_PATH[actor_size]
    critic_path = MODEL_SIZE_TO_PATH[critic_size]
    flags = [
        "python3",
        "/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/ray_main.py",
        f"--actor_model_name_or_path {actor_path}",
        f"--critic_model_name_or_path {critic_path}",
        "--actor_gradient_checkpointing",
        "--critic_gradient_checkpointing",
        "--enable_hybrid_engine",
        "--offload_reference_model",
        "--actor_zero_stage 3",
        "--critic_zero_stage 3",
        f"--inference_tp_size {MODEL_SIZE_TO_HE_TP_SIZE[actor_size]}",
    ]
    if offload:
        flags += ["--offload"]
    return flags


def log_stream_cmd(cmd, logfile):
    return f"stdbuf -oL {cmd} 2>&1 | tee -a {logfile}"


def run_debug_cmd(cmd, logfile, verbose=True):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    os.system(f"touch {logfile}")
    if verbose:
        print(" Running command ".center(100, "=") + f"\n{cmd}\n" + "=" * 100 + "\n")
    try:
        pro = subprocess.Popen(
            log_stream_cmd(cmd, logfile),
            shell=True,
            preexec_fn=os.setsid,
        )
        pro.wait()
    except KeyboardInterrupt:
        for _ in range(3):
            pro.send_signal(signal.SIGINT)
        try:
            pro.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pro.terminate()
        try:
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass


def run_interruptable_cmd_on_js_h100(cmd, nodelist, logfile, verbose=True):
    assert nodelist is not None
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    os.system(f"touch {logfile}")
    cmd = f"/home/fw/dschat-bundle/rayrun {nodelist} {cmd}"
    if verbose:
        print(" Running command ".center(100, "=") + f"\n{cmd}\n" + "=" * 100 + "\n")
    try:
        pro = subprocess.Popen(
            log_stream_cmd(cmd, logfile),
            shell=True,
            preexec_fn=os.setsid,
        )
        pro.wait()
    except KeyboardInterrupt:
        for _ in range(3):
            pro.send_signal(signal.SIGINT)
        try:
            pro.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pro.terminate()
        try:
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
