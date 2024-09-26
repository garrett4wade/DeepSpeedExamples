from pathlib import Path
from typing import *
import argparse

from setting import IS_FRL, MODEL_SIZE_TO_N_NODES_BAISC, get_common_flags, run_debug_cmd, run_interruptable_cmd_on_js_h100
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
    if IS_FRL:
        logfile = (
            Path("/lustre/fw/mlsys25-dschat-logs")
            / exp_name
            / trial_name
            / "output.log"
        )
    else:
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


def main(args):
    n_gpus = MODEL_SIZE_TO_N_NODES_BAISC[args.model_size] * 8
    max_rollout_n_mbs = 256 // n_gpus
    max_train_n_mbs = 256 // n_gpus // 8
    for nr, nt in itertools.product(range(1,max_rollout_n_mbs +1), range(1,max_train_n_mbs+1)):
        cmd_logfile = build_cmd(
            model_size=args.model_size,
            bs=256,
            ctx=2048,
            prompt_len=1024,
            rollout_n_mbs=nr,
            train_n_mbs=nt,
            scale_both=args.scale_both,
            offload=False,
        )
        if cmd_logfile is not None:
            if args.debug:
                run_debug_cmd(*cmd_logfile)
            else:
                run_interruptable_cmd_on_js_h100(cmd_logfile[0], args.nodelist, cmd_logfile[1])
            
            parse_success, oom = _parselog(
                actor_size=args.model_size,
                critic_size=7 if not args.scale_both else args.model_size,
                bs=256,
                ctx=2048,
                prompt_len=1024,
                nr=nr,
                nt=nt,
            )
            if parse_success and not oom:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", "-x", type=int, required=True)
    parser.add_argument("--scale_both", "-s", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nodelist", type=str, default=None)
    args = parser.parse_args()
    main(args)
