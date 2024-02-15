import os
import subprocess
import itertools


def sweep(model_size: int):
    if model_size in [7, 13]:
        tp_gather_partition_size = 1
        if model_size == 7:
            inference_tp_size = 2
        elif model_size == 13:
            inference_tp_size = 4
    else:
        tp_gather_partition_size = 1
        if model_size == 34:
            inference_tp_size = 4
        elif model_size == 70:
            inference_tp_size = 8
    critic_zero_stages = [3, 2]
    actor_zero_stages = [3, 2]
    gen_batch_sizes = [1, 2, 4]
    train_batch_sizes = [1, 2, 3]
    seqlens = [256, 512, 1024]
    if model_size >= 34:
        critic_zero_stages = [3]
        actor_zero_stages = [3]
        gen_batch_sizes = [1, 2]
        seqlens = [256, 512]
    for critic_zero_stage, actor_zero_stage in itertools.product(critic_zero_stages, actor_zero_stages):
        for max_answer_len, gen_bs, train_bs in itertools.product(
            seqlens, gen_batch_sizes, train_batch_sizes
        ):
            for offload in [True, False]:
                exp_name = f"sosp-baseline-dschat-a{model_size}-z{actor_zero_stage}-c7r7-cz{critic_zero_stage}-seqlen{max_answer_len}-g{gen_bs}t{train_bs}"
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
                    f"--gen_bs {gen_bs} --train_bs {train_bs} "
                )
                if offload:
                    cmd += "--offload "
                os.system(cmd)
                # print(cmd)


if __name__ == "__main__":
    # sweep(13)
    sweep(34)
    sweep(70)
