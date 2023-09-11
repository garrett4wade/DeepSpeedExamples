import logging
import argparse
import subprocess
import os
import getpass
import scheduler.client as sched_client
import name_resolve

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", "-e", type=str, required=True)
parser.add_argument("--trial_name", "-f", type=str, required=True)
parser.add_argument("--ntasks", "-n", type=int, default=1)
parser.add_argument("--mode", type=str, default='slurm')
args = parser.parse_args()

USER_NAMESPACE = getpass.getuser()


def main(args):
    name_resolve.clear_subtree(
        name_root=f"{USER_NAMESPACE}/{args.experiment_name}/{args.trial_name}")

    logging.basicConfig(format=LOG_FORMAT,
                        datefmt=DATE_FORMAT,
                        level=os.environ.get("LOGLEVEL", "INFO"))
    logger = logging.getLogger("DeepSpeed Slurm Launch")
    if args.mode == "local" and not os.path.exists("/data"):
        logger.warning(
            "No /data directory found in local mode. Have you mounted it in srun?"
        )
    output_dir = f"/data/aigc/llm/benchmarking/DeepSpeed-Chat/{args.experiment_name}/{args.trial_name}"
    os.makedirs(output_dir, exist_ok=True)

    entry_file = os.path.join(os.path.dirname(__file__), "training",
                              "step1_supervised_finetuning", "main.py")
    cmd = (
        f"python3 {entry_file} -e {args.experiment_name} -f {args.trial_name}"
        f" --deepspeed --slurm_launch -i {{index}} -n {{count}}")
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
        "--lr_scheduler_type cosine",
        "--num_warmup_steps 0",
        "--seed 1",
        "--gradient_checkpointing",
        "--zero_stage 2",
        "--lora_dim 128",
        "--lora_module_name decoder.layers.",
        f"--output_dir {output_dir}",
    ]
    cmd = " ".join([cmd] + flags)

    env_vars = {
        # "PYTHONPATH":
        # os.path.join(os.path.dirname(os.path.dirname(__file__)), "training",
        #              "step1_supervised_finetuning"),
    }
    # print(os.path.join(os.path.dirname(os.path.dirname(__file__)), "training",
    #                  "step1_supervised_finetuning", "main.py"))

    sched = sched_client.make(
        mode=args.mode, job_name=f"{args.experiment_name}_{args.trial_name}")
    sched.submit_array(
        "dschat-sft",
        cmd,
        count=args.ntasks,
        cpu=4,
        gpu_type="tesla",
        gpu=1,
        mem=60 * 1024,
        env_vars=env_vars,
        container_image='llm/llm-gpu',
        container_mounts="/data:/data,/lustre:/lustre,/hddlustre:/hddlustre",
        hostfile=True,
    )

    try:
        sched.wait()
    except (KeyboardInterrupt, sched_client.TaskException):
        sched.stop_all()
        raise


if __name__ == "__main__":
    main(args)
