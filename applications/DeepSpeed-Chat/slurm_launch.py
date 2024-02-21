import logging
import argparse
import subprocess
import os
import getpass
import scheduler.client as sched_client
import name_resolve
from cluster import spec as cluster_spec

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", "-e", type=str, required=True)
parser.add_argument("--trial_name", "-f", type=str, required=True)
parser.add_argument("--task_type", "-t", type=str, choices=["sft", "rw", "rlhf"], default="rlhf")
parser.add_argument("--mode", type=str, default="slurm")
parser.add_argument("--actor_size", type=int, choices=[7, 13, 34, 70], default=7)
parser.add_argument("--critic_size", type=int, choices=[7, 13, 34, 70], default=7)
parser.add_argument("--actor_zero_stage", type=int, default=3)
parser.add_argument("--critic_zero_stage", type=int, default=3)
parser.add_argument("--offload", action="store_true")
parser.add_argument("--offload_ref", action="store_true")
parser.add_argument("--max_answer_len", type=int, default=256)
parser.add_argument("--use_hybrid_engine", action="store_true")
parser.add_argument("--inference_tp_size", type=int, default=1)
parser.add_argument("--tp_gather_partition_size", type=int, default=1)
parser.add_argument("--gen_bs", type=int, default=1)
parser.add_argument("--train_bs", type=int, default=1)
args = parser.parse_args()

USER_NAMESPACE = getpass.getuser()

task_mapping = {
    "sft": "step1_supervised_finetuning",
    "rw": "step2_reward_model_finetuning",
    "rlhf": "step3_rlhf_finetuning",
}

n_ppo_mbs = 4
max_prompt_len = 256
max_answer_len = args.max_answer_len


def get_path_from_model_size(model_size: int):
    if model_size == 7:
        model_path = "/lustre/public/pretrained_model_weights/Llama-2-7b-hf"
    elif model_size == 13:
        model_path = "/lustre/public/pretrained_model_weights/Llama-2-13b-hf"
    elif model_size == 34:
        model_path = "/lustre/public/pretrained_model_weights/CodeLlama-34b-hf"
    elif model_size == 70:
        model_path = "/lustre/public/pretrained_model_weights/Llama-2-70b-hf"
    else:
        raise NotImplementedError()
    return model_path


def get_ngpus_and_nodelist_from_model_size(model_size: int):
    if model_size in [7]:
        return 8, "QH-com16"
    elif model_size == 13:
        return 16, "QH-com[17-18]"
    elif model_size in [34]:
        return 32, "QH-com[44-47]"
    elif model_size == 70:
        return 64, "QH-com[25-28,30-33]"


def main(args):
    name_resolve.clear_subtree(name_root=f"{USER_NAMESPACE}/{args.experiment_name}/{args.trial_name}")

    logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))
    logger = logging.getLogger("DeepSpeed Slurm Launch")

    entry_file = os.path.join(os.path.dirname(__file__), "training", task_mapping[args.task_type], "main.py")
    cmd = (
        f"python3 {entry_file} -e {args.experiment_name} -f {args.trial_name}"
        f" --deepspeed --slurm_launch -i {{jobstep_id}} -g {{n_jobsteps}} -r {{worker_submission_index}} "
        f"-p {{wprocs_per_jobstep}} -j {{wprocs_in_job}} -o {{wproc_offset}}"
    )

    actor_path = get_path_from_model_size(args.actor_size)
    critic_path = get_path_from_model_size(args.critic_size)
    n_actor_gpus, nodelist = get_ngpus_and_nodelist_from_model_size(args.actor_size)
    per_device_gen_bs = args.gen_bs
    per_device_batch_size = n_ppo_mbs * args.train_bs

    if args.task_type == "sft":
        flags = [
            "--data_path local/jsonfile",
            "--data_split 2,4,4",
            "--model_name_or_path /hddlustre/llm/public/checkpoints/pretrained/opt-1.3b",
            "--per_device_train_batch_size 4",
            "--per_device_eval_batch_size 4",
            "--max_seq_len 512",
            "--learning_rate 1e-4",
            "--weight_decay 0.1",
            "--num_train_epochs 1",
            "--gradient_accumulation_steps 1",
            "--lr_scheduler_type cosine",
            "--num_warmup_steps 0",
            "--seed 1",
            "--gradient_checkpointing",
            "--zero_stage 2",
            "--lora_dim 128",
            "--lora_module_name decoder.layers.",
        ]
    elif args.task_type == "rw":
        flags = [
            "--data_path local/jsonfile",
            "--data_split 2,4,4",
            "--model_name_or_path /hddlustre/llm/public/checkpoints/pretrained/opt-1.3b",
            "--num_padding_at_beginning 1",
            "--per_device_train_batch_size 2",
            "--per_device_eval_batch_size 2",
            "--max_seq_len 512",
            "--learning_rate 5e-5",
            "--weight_decay 0.1",
            "--disable_dropout",
            "--num_train_epochs 1",
            "--gradient_accumulation_steps 1",
            "--lr_scheduler_type cosine",
            "--num_warmup_steps 0",
            "--seed 1",
            "--zero_stage 2",
        ]
    elif args.task_type == "rlhf":
        flags = [
            "--data_path local/jsonfile",
            "--data_split 0,0,10",
            f"--actor_model_name_or_path {actor_path}",
            f"--critic_model_name_or_path {critic_path}",
            "--num_padding_at_beginning 0",
            f"--per_device_generation_batch_size {per_device_gen_bs}",
            f"--per_device_training_batch_size {args.train_bs}",
            f"--generation_batches {per_device_batch_size // per_device_gen_bs}",
            "--ppo_epochs 1",
            f"--max_answer_seq_len {max_answer_len}",
            f"--max_prompt_seq_len {max_prompt_len}",
            "--actor_learning_rate 5e-6",
            "--critic_learning_rate 5e-6",
            "--num_train_epochs 1",
            "--lr_scheduler_type cosine",
            "--gradient_accumulation_steps 1",
            "--num_warmup_steps 100",
            "--seed 1234",
            f"--actor_zero_stage {args.actor_zero_stage}",
            f"--critic_zero_stage {args.critic_zero_stage}",
            "--actor_gradient_checkpointing",
            "--critic_gradient_checkpointing",
            "--enable_test_mode",
            "--test_stop_step 20",
        ]
        if args.use_hybrid_engine:
            flags.append("--enable_hybrid_engine")
        if args.offload and "--offload" not in flags:
            flags.append("--offload")
        if args.offload_ref and "--offload_reference_model" not in flags:
            flags.append("--offload_reference_model")
        flags.append(f"--inference_tp_size {args.inference_tp_size}")
        flags.append(f"--tp_gather_partition_size {args.tp_gather_partition_size}")
    cmd = " ".join([cmd] + flags)

    log_path = f"/lustre/aigc/llm/logs/fw/{args.experiment_name}/{args.trial_name}/{args.task_type}-0"
    if os.path.exists(log_path):
        os.system(f"rm -rf {log_path}")

    env_vars = {
        # "PYTHONPATH":
        # os.path.join(os.path.dirname(os.path.dirname(__file__)), "training",
        #              "step1_supervised_finetuning"),
    }
    # print(os.path.join(os.path.dirname(os.path.dirname(__file__)), "training",
    #                  "step1_supervised_finetuning", "main.py"))

    sched = sched_client.make(mode=args.mode, expr_name=args.experiment_name, trial_name=args.trial_name)
    sched.submit_array(
        args.task_type,
        cmd,
        count=n_actor_gpus,
        cpu=4,
        gpu_type="tesla",
        gpu=1,
        mem=120 * 1024,
        env_vars=env_vars,
        container_image="llm/llm-dschat",
        container_mounts=cluster_spec.default_mount,
        nodelist=nodelist,
        hostfile=True,
    )

    try:
        sched.wait()
    except (KeyboardInterrupt, sched_client.JobException, TimeoutError, Exception) as e:
        sched.stop_all()
        raise e


if __name__ == "__main__":
    main(args)
