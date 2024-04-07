import os
import subprocess
import itertools
import argparse
from sosp_parselog import _parselog


# fmt: off
interested_settings = [
    # model size 7
    # dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=128, offload=False, gen_bs=128),
    # dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=384, offload=False, gen_bs=64),
    # dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=896, offload=False, gen_bs=32),
    # dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=128, offload=True, gen_bs=128),
    # dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=384, offload=True, gen_bs=64),
    # dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=896, offload=True, gen_bs=32),
    # # model size 13
    # dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=128, offload=True, gen_bs=64),
    # dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=384, offload=True, gen_bs=32),
    # dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=896, offload=True, gen_bs=16),
    # dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=128, offload=False, gen_bs=64),
    # dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=384, offload=False, gen_bs=32),
    # dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=896, offload=False, gen_bs=16),
    # # model size 34
    # dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=128, offload=False, gen_bs=32),
    # dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=384, offload=False, gen_bs=16),
    # dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=896, offload=False, gen_bs=8),
    # dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=128, offload=True, gen_bs=32),
    # dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=384, offload=True, gen_bs=16),
    # dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=896, offload=True, gen_bs=8),
    # # model size 70
    # dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=128, offload=True, gen_bs=16),
    # dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=384, offload=True, gen_bs=8),
    # dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=896, offload=True, gen_bs=4),
    # dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=128, offload=False, gen_bs=16),
    # dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=384, offload=False, gen_bs=8),
    # dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=896, offload=False, gen_bs=4),
]

# fmt: on
for model_size in [7, 13, 34, 70]:
    if model_size <= 13:
        zero_stages = [2, 3]
    else:
        zero_stages = [3]
    for zero_stage, offload, (global_bs, genlen) in itertools.product(
        zero_stages, [False, True], [(128, 896), (256, 384), (512, 128)]
    ):
        if model_size == 7:
            n_gpus = 8
        elif model_size == 13:
            n_gpus = 16
        elif model_size == 34:
            n_gpus = 32
        elif model_size == 70:
            n_gpus = 64
        assert global_bs % n_gpus == 0
        assert global_bs * (128 + genlen) == 2**17
        if global_bs // n_gpus >= 4:
            interested_settings.append(
                dict(
                    model_size=model_size,
                    actor_zero_stage=zero_stage,
                    critic_zero_stage=zero_stage,
                    max_answer_len=genlen,
                    gen_bs=global_bs // n_gpus,
                    offload=offload,
                )
            )


def build_default_sweep_settings(model_size: int):
    settings = []
    return settings


def sweep(model_size: int, verbose_only: bool, scale_actor: bool, scale_critic: bool):
    assert scale_actor or scale_critic
    actor_size = model_size if scale_actor else 7
    critic_size = model_size if scale_critic else 7
    if actor_size in [7, 13]:
        tp_gather_partition_size = 1
        if actor_size == 7:
            inference_tp_size = 1
        elif actor_size == 13:
            inference_tp_size = 2
    else:
        tp_gather_partition_size = 1
        if actor_size == 34:
            inference_tp_size = 4
        elif actor_size == 70:
            inference_tp_size = 8

    global interested_settings
    interested_settings = list(filter(lambda x: x["model_size"] == model_size, interested_settings))
    assert len(interested_settings) > 0
    if len(interested_settings) == 0:
        settings = build_default_sweep_settings(model_size)
        print(
            f">>>>>>>>>>>>>>>> No interested settings for actor {actor_size} critic {critic_size} found. Using default {len(settings)} settings. <<<<<<<<<<<<<<<<"
        )
    else:
        settings = interested_settings
        print(
            f">>>>>>>>>>>>>>>> Found interested settings for actor {actor_size} critic {critic_size}! Run interested {len(settings)} settings only. <<<<<<<<<<<<<<<<"
        )
    for setting in settings:
        actor_zero_stage = setting["actor_zero_stage"]
        critic_zero_stage = setting["critic_zero_stage"]
        max_answer_len = setting["max_answer_len"]
        gen_bs = setting["gen_bs"]
        offload = setting["offload"]
        # skip if there exists a log file
        # if "force", forcely re-run the experiment
        if not args.force and _parselog(
            actor_size, critic_size, actor_zero_stage, critic_zero_stage, max_answer_len, gen_bs, offload
        ):
            continue
        if actor_zero_stage == 2:
            inference_tp_size = 1
        exp_name = f"rerun-dschat-a{actor_size}-z{actor_zero_stage}-c{critic_size}r7-cz{critic_zero_stage}-seqlen{max_answer_len}-g{gen_bs}"
        if offload:
            exp_name += "-offload"
        trial_name = "benchmark"
        cmd = (
            f"python3 applications/DeepSpeed-Chat/slurm_launch.py"
            f" -e {exp_name} -f {trial_name} "
            f"--actor_size {actor_size} "
            f"--critic_size {critic_size} "
            f"--actor_zero_stage {actor_zero_stage} --critic_zero_stage {critic_zero_stage} "
            f"--max_answer_len {max_answer_len} --offload_ref --use_hybrid_engine "
            f"--inference_tp_size {inference_tp_size} --tp_gather_partition_size {tp_gather_partition_size} "
            f"--gen_bs {gen_bs} --train_bs {gen_bs} "
        )
        if offload:
            cmd += "--offload "
        if not verbose_only:
            os.system(cmd)
        else:
            print(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", "-x", type=int, choices=[7, 13, 34, 70], required=True, nargs="+")
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--scale_critic", "-c", action="store_true", default=False)
    parser.add_argument("--scale_actor", "-a", action="store_true", default=False)
    parser.add_argument("--verbose_only", "-v", action="store_true")
    args = parser.parse_args()
    for model_size in args.model_size:
        sweep(model_size, args.verbose_only, args.scale_actor, args.scale_critic)
