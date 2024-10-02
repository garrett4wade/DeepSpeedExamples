from pathlib import Path
from typing import *
import argparse
import itertools
import re
import os
import time

from setting import N_NODES_TO_BATCH_SIZE, CTX, PROMPT_LEN, MODEL_SIZE_TO_N_NODES_BAISC, get_common_flags, run_debug_cmd, run_interruptable_cmd_on_js_h100
from parse_log import _parselog


def build_cmd(
    model_size: int,
    bs: int,
    ctx: int,
    prompt_len: int,
    scale_both: bool,
    rollout_n_mbs: int,
    train_n_mbs: int,
    offload: bool = False,
    n_ppo_mbs=8,
):
    """
    Denote the world size as W, batch size B, rollout_n_mbs as Nr, train_n_mbs as Nt, and number of PPO batches as Np.

    `generation_batches` is Nr.
    `per_device_generation_batch_size` is the per-device mini-batch size of generation, which is B // W // Nr.
    `per_device_training_batch_size` is the per-device miro-batch size of a single PPO minibatch, which is B // W // Np // Nt.
    `gradient_accumulation_steps` is Nt.
    """
    critic_size = model_size if scale_both else 7
    exp_name = "mlsys"
    trial_name = f"a{model_size}c{critic_size}b{bs}ct{ctx}p{prompt_len}nr{rollout_n_mbs}nt{train_n_mbs}"

    logfile = Path("/mnt/bs_fs/dschat-logs") / exp_name / trial_name / "output.log"

    w = MODEL_SIZE_TO_N_NODES_BAISC[model_size] * 8

    flags = get_common_flags(model_size, critic_size, offload)
    flags += [
        f"--per_device_generation_batch_size {bs // w // rollout_n_mbs}",
        f"--per_device_training_batch_size {bs // w // n_ppo_mbs // train_n_mbs}",
        f"--generation_batches {rollout_n_mbs}",
        f"--max_answer_seq_len {ctx - prompt_len}",
        f"--max_prompt_seq_len {prompt_len}",
        f"--gradient_accumulation_steps {train_n_mbs}",
    ]
    flags += [
        f"-w {w}",
        f"--experiment_name {exp_name}",
        f"--trial_name {trial_name}",
    ]
    if bs // w // rollout_n_mbs < 1 or bs // w // n_ppo_mbs // train_n_mbs < 1:
        return None
    return " ".join(flags), logfile

def extract_node_integers(input_string):
    # Regular expression to match the pattern
    pattern = r'(\w+)\[(\d+)-(\d+)\]'
    
    # Try to match the pattern in the input string
    match = re.match(pattern, input_string)
    
    if match:
        # Extract the base name and the range
        base_name = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))
        
        return start, end
    else:
        raise ValueError('Invalid input string')

def main(args):
    for size in args.model_size:
        n_gpus = MODEL_SIZE_TO_N_NODES_BAISC[size] * 8
        bs = N_NODES_TO_BATCH_SIZE[MODEL_SIZE_TO_N_NODES_BAISC[size]]
        for nt, nr in [(4,16),(4,32)]:
            cmd_logfile = build_cmd(
                model_size=size,
                bs=bs,
                ctx=CTX,
                prompt_len=PROMPT_LEN,
                rollout_n_mbs=nr,
                train_n_mbs=nt,
                scale_both=args.scale_both,
                offload=False,
            )
            if cmd_logfile is not None:
                parse_success, oom = _parselog(
                    actor_size=size,
                    critic_size=7 if not args.scale_both else size,
                    bs=bs,
                    ctx=CTX,
                    prompt_len=PROMPT_LEN,
                    nr=nr,
                    nt=nt,
                )
                if parse_success:
                    if not oom:
                        print(f">>>>> Find existing successful logfile: {cmd_logfile[1]}")
                        break
                    else:
                        print(f">>>>> Find existing OOM logfile, skip it, continue searching")
                        continue

                s, e = extract_node_integers(args.nodelist)
                assert e -s + 1 == MODEL_SIZE_TO_N_NODES_BAISC[size]
                os.system(f"python3 /mnt/bs_fs/rayc.py stop -s {s} -e {e}")
                time.sleep(10)
                run_interruptable_cmd_on_js_h100(cmd_logfile[0], args.nodelist, cmd_logfile[1])
                os.system(f"python3 /mnt/bs_fs/rayc.py stop -s {s} -e {e}")
                time.sleep(10)
                
                parse_success, oom = _parselog(
                    actor_size=size,
                    critic_size=7 if not args.scale_both else size,
                    bs=bs,
                    ctx=CTX,
                    prompt_len=PROMPT_LEN,
                    nr=nr,
                    nt=nt,
                )
                if parse_success and not oom:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", "-x", type=int, required=True, nargs='+')
    parser.add_argument("--scale_both", "-s", action="store_true")
    parser.add_argument("--nodelist", type=str, default=None)
    args = parser.parse_args()
    main(args)
