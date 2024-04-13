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


def get_default_n_gpus(model_size: int):
    if model_size in [7]:
        return 8
    elif model_size == 13:
        return 16
    elif model_size in [34]:
        return 32
    elif model_size == 70:
        return 64


def build_default_sweep_settings(model_size: int, gpu_scale_factor: int):
    settings = []
    for model_size in [7, 13, 34, 70]:
        if model_size <= 13:
            zero_stages = [2, 3]
        else:
            zero_stages = [3]
        default_n_gpus = get_default_n_gpus(model_size)
        n_gpus = default_n_gpus * gpu_scale_factor
        global_bs_seqlens = [(128, 896), (256, 384), (512, 128)] if default_n_gpus == n_gpus else [(256, 384)]
        for zero_stage, offload, (global_bs, genlen) in itertools.product(
            zero_stages, [False, True], global_bs_seqlens
        ):
            if global_bs < n_gpus:
                continue
            assert global_bs % n_gpus == 0, (global_bs, n_gpus)
            assert global_bs * (128 + genlen) == 2**17
            
            if zero_stage == 2 and offload:
                # NOTE: This is due to the limitation of DSChat
                # This config will report (step3_rlhf_finetuning/main.py, line 454)
                # ValueError: The combination of [actor_zero_stage==2, critic_zero_stage==2, enable_hybrid_engine=True, offload=True, lora=False] is currently unsupported due to training instability
                continue
            
            if global_bs // n_gpus >= 4:
                settings.append(
                    dict(
                        model_size=model_size,
                        actor_zero_stage=zero_stage,
                        critic_zero_stage=zero_stage,
                        max_answer_len=genlen,
                        gen_bs=global_bs // n_gpus,
                        offload=offload,
                        n_gpus=n_gpus,
                    )
                )
    return settings


def sweep(model_size: int, verbose_only: bool, scale_actor: bool, scale_critic: bool, gpu_scale_factor: int):
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
    if len(interested_settings) == 0:
        settings = build_default_sweep_settings(model_size, gpu_scale_factor=gpu_scale_factor)
        settings = list(filter(lambda x: x["model_size"] == model_size, settings))
        print(
            f">>>>>>>>>>>>>>>> No interested settings for actor {actor_size} critic {critic_size} found. Using default {len(settings)} settings. <<<<<<<<<<<<<<<<"
        )
    else:
        settings = interested_settings
        print(
            f">>>>>>>>>>>>>>>> Found interested settings for actor {actor_size} critic {critic_size}! Run interested {len(settings)} settings only. <<<<<<<<<<<<<<<<"
        )
    for setting in settings:
        default_n_gpus = get_default_n_gpus(max(actor_size, critic_size))
        n_gpus = setting["n_gpus"]

        actor_zero_stage = setting["actor_zero_stage"]
        critic_zero_stage = setting["critic_zero_stage"]
        max_answer_len = setting["max_answer_len"]
        gen_bs = setting["gen_bs"]
        offload = setting["offload"]
        # skip if there exists a log file
        # if "force", forcely re-run the experiment
        if not args.force and _parselog(
            actor_size, critic_size, actor_zero_stage, critic_zero_stage, max_answer_len, gen_bs, offload, gpu_scale_factor,
        ):
            continue
        if actor_zero_stage == 2:
            inference_tp_size = 1
        exp_name = f"rerun-dschat-a{actor_size}-z{actor_zero_stage}-c{critic_size}r7-cz{critic_zero_stage}-seqlen{max_answer_len}-g{gen_bs}"
        if offload:
            exp_name += "-offload"
        if n_gpus != default_n_gpus:
            exp_name += f"-x{n_gpus / default_n_gpus:.1f}"

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
        if n_gpus != default_n_gpus:
            cmd += f"--n_gpus {n_gpus} "
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
    parser.add_argument("--gpu_scale_factor", "-g", type=int, default=[1], nargs='+')
    parser.add_argument("--verbose_only", "-v", action="store_true")
    args = parser.parse_args()
    for model_size, gpu_scale_factor in itertools.product(args.model_size, args.gpu_scale_factor):
        sweep(model_size, args.verbose_only, args.scale_actor, args.scale_critic, gpu_scale_factor)
