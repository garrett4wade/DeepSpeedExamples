import os
import subprocess
import itertools
import argparse
from sosp_parselog import _parselog


# fmt: off
interested_settings = [
    # model size 7
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=False, gen_bs=26),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=False, gen_bs=28),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=False, gen_bs=30),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=False, gen_bs=19),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=False, gen_bs=20),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=False, gen_bs=21),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=False, gen_bs=9),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=False, gen_bs=10),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=False, gen_bs=11),

    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=True, gen_bs=47),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=True, gen_bs=48),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=True, gen_bs=49),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=True, gen_bs=28),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=True, gen_bs=29),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=True, gen_bs=30),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=True, gen_bs=15),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=True, gen_bs=16),
    dict(model_size=7, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=True, gen_bs=17),
    # model size 13
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=True, gen_bs=27),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=True, gen_bs=28),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=True, gen_bs=29),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=True, gen_bs=17),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=True, gen_bs=18),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=True, gen_bs=19),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=True, gen_bs=9),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=True, gen_bs=10),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=True, gen_bs=11),

    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=False, gen_bs=19),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=False, gen_bs=20),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=False, gen_bs=21),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=False, gen_bs=11),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=False, gen_bs=12),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=False, gen_bs=13),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=False, gen_bs=6),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=False, gen_bs=7),
    dict(model_size=13, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=False, gen_bs=8),
    # model size 34
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=False, gen_bs=11),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=False, gen_bs=12),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=False, gen_bs=13),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=False, gen_bs=7),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=False, gen_bs=8),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=False, gen_bs=9),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=False, gen_bs=3),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=False, gen_bs=4),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=False, gen_bs=5),
    
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=True, gen_bs=17),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=True, gen_bs=18),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=True, gen_bs=19),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=True, gen_bs=11),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=True, gen_bs=12),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=True, gen_bs=13),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=True, gen_bs=5),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=True, gen_bs=6),
    dict(model_size=34, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=True, gen_bs=7),
    # model size 70
    dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=True, gen_bs=10),
    dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=True, gen_bs=11),
    dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=256, offload=True, gen_bs=12),
    dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=True, gen_bs=6),
    dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=True, gen_bs=7),
    dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=512, offload=True, gen_bs=8),
    dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=True, gen_bs=3),
    dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=True, gen_bs=4),
    dict(model_size=70, actor_zero_stage=3, critic_zero_stage=3, max_answer_len=1024, offload=True, gen_bs=5),
]
# fmt: on


def build_default_sweep_settings(model_size: int):
    settings = []
    for offload in [False, True]:
        critic_zero_stages = [3]
        actor_zero_stages = [3, 2]
        if not offload:
            gen_batch_sizes = [4, 8, 10, 12]
        else:
            gen_batch_sizes = [8, 16, 24, 32, 48]
        seqlens = [256, 512, 1024]
        if model_size >= 34:
            actor_zero_stages = [3]
        for critic_zero_stage, actor_zero_stage in itertools.product(critic_zero_stages, actor_zero_stages):
            for max_answer_len, gen_bs in itertools.product(seqlens, gen_batch_sizes):
                train_bs = gen_bs
                if not offload:
                    if model_size <= 13 and gen_bs <= 4:
                        continue
                    if max_answer_len == 256 and gen_bs <= 2:
                        continue
                    if max_answer_len == 512 and gen_bs >= 8:
                        continue
                    if max_answer_len == 1024 and gen_bs >= 8:
                        continue
                else:
                    if model_size <= 13 and gen_bs <= 4:
                        continue
                    if max_answer_len == 256 and gen_bs <= 8:
                        continue
                    if max_answer_len == 512 and gen_bs >= 32:
                        continue
                    if max_answer_len == 1024 and gen_bs >= 32:
                        continue
                settings.append(
                    dict(
                        model_size=model_size,
                        actor_zero_stage=actor_zero_stage,
                        critic_zero_stage=critic_zero_stage,
                        max_answer_len=max_answer_len,
                        gen_bs=gen_bs,
                        train_bs=train_bs,
                        offload=offload,
                    )
                )
    return settings


def sweep(model_size: int, verbose_only: bool):
    if model_size in [7, 13]:
        tp_gather_partition_size = 1
        if model_size == 7:
            inference_tp_size = 1
        elif model_size == 13:
            inference_tp_size = 2
    else:
        tp_gather_partition_size = 1
        if model_size == 34:
            inference_tp_size = 4
        elif model_size == 70:
            inference_tp_size = 8
    global interested_settings
    interested_settings = list(filter(lambda x: x["model_size"] == model_size, interested_settings))
    if len(interested_settings) == 0:
        settings = build_default_sweep_settings(model_size)
        print(
            f">>>>>>>>>>>>>>>> No interested settings for model size {model_size} found. Using default {len(settings)} settings. <<<<<<<<<<<<<<<<"
        )
    else:
        settings = interested_settings
        print(
            f">>>>>>>>>>>>>>>> Found interested settings for model size {model_size}! Run interested {len(settings)} settings only. <<<<<<<<<<<<<<<<"
        )
    for setting in settings:
        model_size = setting["model_size"]
        actor_zero_stage = setting["actor_zero_stage"]
        critic_zero_stage = setting["critic_zero_stage"]
        max_answer_len = setting["max_answer_len"]
        gen_bs = setting["gen_bs"]
        offload = setting["offload"]
        # skip if there exists a log file
        # if "force", forcely re-run the experiment
        if not args.force and _parselog(
            model_size, actor_zero_stage, critic_zero_stage, max_answer_len, gen_bs, offload
        ):
            continue
        if actor_zero_stage == 2:
            inference_tp_size = 1
        exp_name = f"rerun-dschat-a{model_size}-z{actor_zero_stage}-c7r7-cz{critic_zero_stage}-seqlen{max_answer_len}-g{gen_bs}"
        if offload:
            exp_name += "-offload"
        trial_name = "benchmark"
        cmd = (
            f"python3 applications/DeepSpeed-Chat/slurm_launch.py"
            f" -e {exp_name} -f {trial_name} "
            f"--actor_size {model_size} "
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
    parser.add_argument("--verbose_only", "-v", action="store_true")
    args = parser.parse_args()
    for model_size in args.model_size:
        sweep(model_size, args.verbose_only)
