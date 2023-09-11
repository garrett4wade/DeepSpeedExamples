import logging
import argparse
import subprocess
import os

from utils.scheduler.client import make as make_scheduler_client
import utils.scheduler.client

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", "-e", type=str, required=True)
parser.add_argument("--trial_name", "-f", type=str, required=True)
parser.add_argument("--ntasks", type=int, default=1)
args = parser.parse_args()


def main(args):
    logging.basicConfig(format=LOG_FORMAT,
                        datefmt=DATE_FORMAT,
                        level=os.environ.get("LOGLEVEL", "INFO"))
    output_dir = f"/data/aigc/llm/benchmarking/DeepSpeed-Chat/{args.experimental_name}/{args.trial_name}"
    os.makedirs(output_dir, exist_ok=True)
    sched = make_scheduler_client("slurm")
    cmd = (
        f"python3 -m training.step1_supervised_finetuning.main -e {args.experiment_name} -f {args.trial_name}"
        f" --deepspeed --slurm_launch")
    flags = [
        "--data_path /data/aigc/datasets/rm-static/",
        "--data_spit 2,4,4",
        "--model_name_or_path facebook/opt-1.3b",
        "--per_device_train_batch_size 4",
        "--per_device_eval_batch_size 4",
        "--max_seq_len 512",
        "--learning_rate 1e-4",
        "--weight_decay 0.1",
        "--num_train_epochs 1",
        "--lr_scheduler_type cosine",
        "--warmup_steps 0",
        "--seed 1",
        "--gradient_checkpointing",
        "--zero_stage 2",
        "--lora_dim 128",
        "--lora_module_name decoder.layers.",
        f"--output_dir {output_dir}",
    ]
    cmd = " ".join([cmd] + flags)

    sched.submit_array("dschat-sft",
                       cmd,
                       count=argparse.ntasks,
                       cpu=4,
                       gpu_type="tesla",
                       gpu=1,
                       mem=60 * 1024,
                       env_vars={},
                       container_image='llm/llm-gpu',
                       container_mounts="/data:/data",
                       hostfile=True)

    try:
        sched.wait()
    except (KeyboardInterrupt, utils.scheduler.client.TaskException):
        sched.stop_all()
        raise


if __name__ == "__main__":
    main(args)
