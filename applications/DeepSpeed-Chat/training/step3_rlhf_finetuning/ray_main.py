#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""

engine = DeepSpeedRLHFEngine(actor_model_name_or_path=actor_model_name_or_path,
                             critic_model_name_or_path=critic_model_name_or_path,
                             tokenizer=tokenizer,
                             args=args)
trainer = DeepSpeedPPOTrainer(engine=engine, args=args)

for prompt_batch in prompt_train_dataloader:
    out = trainer.generate_experience(prompt_batch)
    actor_loss, critic_loss = trainer.train_rlhf(out)

"""
import argparse
import os
import random
import re
import socket
import sys
import time
from contextlib import closing
from pathlib import Path

import pynvml
import ray
import torch
import torch.distributed
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

sys.path.append(str(Path(__file__).parent.parent.parent))
# sys.path.append("/home/fw/workspace/DeepSpeedExamples/applications/DeepSpeed-Chat/")
# print(sys.path)


import logging

import deepspeed
from deepspeed.accelerator import get_accelerator
from dschat.rlhf.ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedPPOTrainerUnsupervised
from dschat.rlhf.rlhf_engine import DeepSpeedRLHFEngine
from dschat.utils.data.data_utils import (
    DataCollatorRLHF,
    MiniDataset,
    create_prompt_dataset,
    get_unsupervised_data,
)
from dschat.utils.module.lora import convert_lora_to_linear_layer
from dschat.utils.perf import print_throughput_step3
from dschat.utils.utils import (
    ExponentialMovingAverage,
    get_all_reduce_mean,
    load_hf_tokenizer,
    moving_average,
    print_rank_0,
    save_hf_format,
    save_zero_three_model,
    set_random_seed,
    to_device,
)
from transformers import SchedulerType, default_data_collator

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"

logging.basicConfig(
    format=LOG_FORMAT, datefmt=DATE_FORMAT, level=os.environ.get("LOGLEVEL", "INFO")
)
logger = logging.getLogger("DeepSpeed Ray Launch")


def parse_args():
    parser = argparse.ArgumentParser(description="(Step 3) RLHF training arguments")
    parser.add_argument("--world_size", "-w", type=int, required=True)
    parser.add_argument("--experiment_name", "-e", type=str, required=True)
    parser.add_argument("--trial_name", "-f", type=str, required=True)

    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["local/jsonfile"],
        help="Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="0,0,10",
        help="Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` "
        "will use 60%% of data for phase 1, 20%% for phase 2 and 20%% for phase 3.",
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        default="/tmp/data_files",
        help="Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)",
    )
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--unsup_coef",
        type=float,
        default=27.8,
        help="""gamma in Equation 2 from InstructGPT paper""",
    )
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--critic_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=0,
        help="OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_generation_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader and generation purpose.",
    )
    parser.add_argument(
        "--per_device_training_batch_size",
        type=int,
        default=16,
        help="Mini Batch size (per device) for the training dataloader and training purpose.",
    )
    parser.add_argument(
        "--generation_batches",
        type=int,
        default=1,
        help="Generate x batches to go to training mode.",
    )
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.",
    )
    parser.add_argument(
        "--max_prompt_seq_len",
        type=int,
        default=256,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_answer_seq_len",
        type=int,
        default=256,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--actor_weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--critic_weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )

    # DeepSpeed
    parser.add_argument(
        "--enable_hybrid_engine",
        action="store_true",
        help="Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed.",
    )
    parser.add_argument(
        "--unpin_actor_parameters",
        action="store_true",
        help="Unpin actor's parameters during generation. This makes generation slower but requires less memory.",
    )
    parser.add_argument(
        "--release_inference_cache",
        action="store_true",
        help="Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size.",
    )
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help="Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature.",
    )
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help="Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature.",
    )
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Training data type",
    )
    parser.add_argument(
        "--offload_reference_model",
        action="store_true",
        help="Enable ZeRO Offload techniques for reference model",
    )
    parser.add_argument(
        "--actor_zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )
    parser.add_argument(
        "--critic_zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Critic model (and reward).",
    )
    parser.add_argument(
        "--actor_gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for Actor model.",
    )
    parser.add_argument(
        "--critic_gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for Critic model.",
    )
    parser.add_argument(
        "--actor_dropout",
        type=float,
        default=None,
        help="If actor dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the actor model.",
    )
    parser.add_argument(
        "--critic_dropout",
        type=float,
        default=None,
        help="If critic dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the critic model.",
    )
    ## LoRA for efficient training setting
    parser.add_argument(
        "--actor_lora_dim",
        type=int,
        default=0,
        help="If > 0, use LoRA for efficient training.",
    )
    parser.add_argument(
        "--actor_lora_module_name",
        type=str,
        default=".layers.",
        help="The scope of LoRA.",
    )
    parser.add_argument(
        "--critic_lora_dim",
        type=int,
        default=0,
        help="If > 0, use LoRA for efficient training.",
    )
    parser.add_argument(
        "--critic_lora_module_name",
        type=str,
        default=".layers.",
        help="The scope of LoRA.",
    )
    parser.add_argument(
        "--only_optimize_lora",
        action="store_true",
        help="Only optimize the LoRA parameters.",
    )
    parser.add_argument(
        "--actor_lora_learning_rate",
        type=float,
        default=5e-4,
        help="Initial actor LoRA learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--critic_lora_learning_rate",
        type=float,
        default=5e-4,
        help="Initial critic LoRA learning rate (after the potential warmup period) to use.",
    )
    ## Make EMA as an optional feature
    parser.add_argument(
        "--enable_ema", action="store_true", help="Enable EMA checkpoint for the model."
    )
    ## Mixed Precision ZeRO++
    parser.add_argument(
        "--enable_mixed_precision_lora",
        action="store_true",
        help="Enable Mixed Precision ZeRO++ for training and generation.",
    )
    ## low precision
    parser.add_argument(
        "--compute_fp32_loss",
        action="store_true",
        help="Relevant for low precision dtypes (fp16, bf16, etc.). "
        "If specified, loss is calculated in fp32."
        "This applies for both actor and critic models.",
    )
    ## Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="step3_tensorboard")
    ## Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action="store_true",
        help="Add <|endoftext|> as additional special token to tokenizer",
    )
    ## Actor/critic model overflow alignment
    parser.add_argument(
        "--align_overflow",
        action="store_true",
        help="Align loss scale overflow between actor and critic",
    )
    ## Print actor model answers during training
    parser.add_argument(
        "--print_answers",
        action="store_true",
        help="Print prompt and answers during training",
    )
    parser.add_argument(
        "--print_answers_interval",
        type=int,
        default=1,
        help="If --print_answers enabled, controls the printing interval.",
    )
    ## Testing
    parser.add_argument(
        "--enable_test_mode",
        action="store_true",
        help="Enable a testing mode that terminates training based on args.test_stop_step",
    )
    parser.add_argument(
        "--test_stop_step",
        type=int,
        default=20,
        help="Training non-overflow step at which to terminate training during testing.",
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    if (
        args.actor_zero_stage == 2
        and args.critic_zero_stage == 2
        and args.enable_hybrid_engine
        and args.offload
        and args.actor_lora_dim == 0
    ):
        raise ValueError(
            "The combination of [actor_zero_stage==2, critic_zero_stage==2, enable_hybrid_engine=True, offload=True, lora=False] is currently unsupported due to training instability!"
        )

    return args


def create_datasets(args, tokenizer, train_phase=3):
    unsupervised_training_enabled = (
        args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    )
    prompt_train_dataset, _ = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_prompt_seq_len,
    )
    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    data_collator = DataCollatorRLHF(args.max_prompt_seq_len, args.inference_tp_size)
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(unsupervised_train_dataset)
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_generation_batch_size,
    )
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=args.per_device_generation_batch_size,
        )
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader
        )  # basically a dummy dataloader

    num_update_steps_per_epoch = (
        min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))
        * (args.per_device_generation_batch_size / args.per_device_training_batch_size)
        * args.ppo_epochs
        / args.gradient_accumulation_steps
    )
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    return prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters


class DSChatRayRemoteWorker:
    def __init__(self, rank, world_size, dist_addr):
        self.rank = rank
        self.world_size = world_size
        self.dist_addr = dist_addr

    def init_process_group(self):
        torch.distributed.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"tcp://{self.dist_addr}",
            group_name="dschat",
        )

    def run_dschat(self, args):
        device = torch.device("cuda")
        deepspeed.init_distributed(auto_mpi_discovery=False)
        os.environ['LOCAL_RANK'] = str(0)

        args.global_rank = torch.distributed.get_rank()

        unsupervised_training_enabled = (
            args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
        )
        if unsupervised_training_enabled:
            # if we enable unsupervised training, we need to double the batch size for actor model
            args.gradient_accumulation_steps_actor = (
                args.gradient_accumulation_steps * 2
            )
        else:
            args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

        # If passed along, set the training seed now.
        set_random_seed(args.seed)
        torch.distributed.barrier()

        # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
        args.end_of_conversation_token = "<|endoftext|>"
        additional_special_tokens = (
            args.end_of_conversation_token if args.add_eot_token else None
        )
        tokenizer = load_hf_tokenizer(
            args.actor_model_name_or_path,
            fast_tokenizer=True,
            add_special_tokens=additional_special_tokens,
        )

        vocab_size = tokenizer.vocab_size
        prompt_train_dataloader = [
            dict(prompt=torch.randint(0, vocab_size, (args.per_device_generation_batch_size, args.max_prompt_seq_len), dtype=torch.long),
            prompt_att_mask=torch.ones(args.per_device_generation_batch_size, args.max_prompt_seq_len, dtype=torch.bool))
        ]
        prompt_train_dataloader = prompt_train_dataloader * 500
        unsupervised_train_dataloader = [None for _ in range(500)]
        num_update_steps_per_epoch = (
            min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))
            * (args.per_device_generation_batch_size / args.per_device_training_batch_size)
            * args.ppo_epochs
            / args.gradient_accumulation_steps
        )
        num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

        # prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters = (
        #     create_datasets(args=args, tokenizer=tokenizer, train_phase=3)
        # )

        # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
        rlhf_engine = DeepSpeedRLHFEngine(
            actor_model_name_or_path=args.actor_model_name_or_path,
            critic_model_name_or_path=args.critic_model_name_or_path,
            tokenizer=tokenizer,
            num_total_iters=num_total_iters,
            args=args,
        )

        # Mixed Precision ZeRO++
        if args.enable_mixed_precision_lora:
            assert (
                args.actor_lora_dim > 0
            ), "Mixed Precision LoRA requires LoRA to be enabled"
            assert (
                args.actor_zero_stage == 3
            ), "Mixed Precision LoRA requires Zero stage 3"
            rlhf_engine.actor.optimizer.quantize_nontrainable_params()
            print_rank_0("Mixed Precision ZeRO++ enabled")

        ppo_trainer = (
            DeepSpeedPPOTrainerUnsupervised
            if unsupervised_training_enabled
            else DeepSpeedPPOTrainer
        )
        trainer = ppo_trainer(rlhf_engine, args)

        # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
        exp_mini_dataset = MiniDataset(
            args.generation_batches, args.per_device_training_batch_size
        )
        unsup_mini_dataset = MiniDataset(
            args.generation_batches, args.per_device_training_batch_size
        )

        # Train!
        print_rank_0(
            f"***** Running training (total_iters={num_total_iters}) *****",
            args.global_rank,
        )

        non_overflow_step_count = 0
        step_average_reward = 0.0
        ema_reward_score = ExponentialMovingAverage()

        valid_train_cnt = 0

        train_start_tik = time.perf_counter()
        train_step_gen_time = 0
        for epoch in range(args.num_train_epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}",
                args.global_rank,
            )
            for step, (batch_prompt, batch_unsupervised) in enumerate(
                zip(prompt_train_dataloader, unsupervised_train_dataloader)
            ):

                batch_prompt = to_device(batch_prompt, device)

                # prompts = batch_prompt['prompt']
                # length = prompts.size(-1)
                # if length > args.max_prompt_seq_len:
                #     prompts = prompts[:, length - args.max_prompt_seq_len:]
                #     raise ValueError("Prompt length is too long")

                gen_tik = time.perf_counter()
                out = trainer.generate_experience(
                    batch_prompt["prompt"], batch_prompt["prompt_att_mask"], step
                )
                train_step_gen_time += time.perf_counter() - gen_tik

                # training_start = time.time()
                if batch_unsupervised is not None:
                    batch_unsupervised = to_device(batch_unsupervised, device)
                    unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
                else:
                    unsup_dataset = unsup_mini_dataset.add(
                        [[None] * args.per_device_generation_batch_size]
                    )

                exp_dataset = exp_mini_dataset.add(out)

                if exp_dataset is not None:
                    train_step_tik = time.perf_counter()
                    inner_iter = 0
                    actor_loss_sum, critic_loss_sum, unsup_loss_sum = 0, 0, 0
                    average_reward = 0

                    if args.actor_gradient_checkpointing:
                        rlhf_engine.actor.gradient_checkpointing_enable()

                    assert args.ppo_epochs == 1
                    for ppo_ep in range(args.ppo_epochs):
                        for i, (exp_data, unsup_data) in enumerate(
                            zip(exp_dataset, unsup_dataset)
                        ):
                            actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                            actor_loss_sum += actor_loss.item()
                            critic_loss_sum += critic_loss.item()
                            average_reward += exp_data["rewards"].mean()

                            if unsupervised_training_enabled:
                                unsup_loss = trainer.train_unsupervised(
                                    unsup_data, args.unsup_coef
                                )
                                unsup_loss_sum += unsup_loss.item()

                            inner_iter += 1
                            if args.enable_ema:
                                moving_average(
                                    rlhf_engine.actor,
                                    rlhf_engine.actor_ema,
                                    zero_stage=args.actor_zero_stage,
                                )

                        random.shuffle(exp_dataset)
                        random.shuffle(unsup_dataset)
                    valid_train_cnt += 1
                    training_time = time.perf_counter() - train_step_tik

                    # end = time.time()
                    # training_time = end - training_start
                    e2e_time = time.perf_counter() - train_start_tik
                    train_start_tik = time.perf_counter()
                    generate_time = train_step_gen_time
                    train_step_gen_time = 0
                    # e2e_time = training_time + trainer.generate_time * args.generation_batches  # it is an approximation, we did not include, e.g., rw forward time etc

                    print_rank_0(
                        f"Epoch: {epoch} | Step: {step} | PPO Epoch: {ppo_ep+1} | Actor Loss: {actor_loss_sum/inner_iter} | Critic Loss: {critic_loss_sum/inner_iter} | Unsupervised Loss: {unsup_loss_sum/inner_iter}",
                        args.global_rank,
                    )
                    print(
                        f">>>>> pure generate time {trainer.generate_time:.2f}s, inf time {trainer.inference_time:.2f}s, train time {trainer.actor_critic_train_time:.2f}s"
                    )
                    print_throughput_step3(
                        rlhf_engine.actor.module,
                        rlhf_engine.critic,
                        args,
                        e2e_time,
                        generate_time,
                        training_time,
                        args.global_rank,
                    )

                    average_reward = get_all_reduce_mean(average_reward).item()
                    step_average_reward += (
                        average_reward / args.gradient_accumulation_steps_actor
                    )
                    if (step + 1) % args.gradient_accumulation_steps_actor == 0:
                        ema_reward_score.update(step_average_reward)
                        step_average_reward = 0.0

                    print_rank_0(
                        f"Average reward score: {average_reward/inner_iter} | EMA reward score: {ema_reward_score.get()}",
                        args.global_rank,
                    )
                    print_rank_0(
                        "-------------------------------------------------------------------------------------",
                        args.global_rank,
                    )

                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_disable()

                actor_overflow, critic_overflow = trainer.get_overflow()

                if not actor_overflow and not critic_overflow:
                    non_overflow_step_count += 1

                if valid_train_cnt >= 3:
                    if torch.distributed.get_rank() == 0:
                        print("=" * 100)
                        print(f" Benchmarking finishes after {valid_train_cnt} steps ".center(100, "="))
                        print("=" * 100)
                    return

                # if args.enable_test_mode and non_overflow_step_count == args.test_stop_step:
                #     break

            # if args.enable_test_mode:
            #     break


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def main(args):

    ray.init()

    available_resources = ray.available_resources()
    assert available_resources["GPU"] == args.world_size, (
        available_resources,
        args.world_size,
    )
    assert available_resources["CPU"] >= args.world_size, (
        available_resources,
        args.world_size,
    )

    world_size = count = args.world_size
    available_nodes = [
        k
        for k in available_resources
        if re.match(r"node:(\b(?:\d{1,3}\.){3}\d{1,3}\b)", k)
    ]
    total_gpus = available_resources["GPU"]
    if total_gpus % len(available_nodes) != 0:
        raise ValueError(
            "Cannot schedule Ray jobs to nodes with heterogeneous numbers of GPUs."
        )
    n_gpus_per_node = int(total_gpus // len(available_nodes))
    assert n_gpus_per_node == 8
    if total_gpus < count:
        raise RuntimeError(
            "Available GPUs is smaller than the number of scheduled GPU workers."
        )

    ddp_addr = f"{socket.gethostbyname(socket.gethostname())}:{find_free_port()}"
    self_node = f"node:{socket.gethostbyname(socket.gethostname())}"
    assert self_node in available_nodes, (self_node, available_nodes)
    available_nodes = [self_node] + [n for n in available_nodes if n != self_node]
    jobs = []
    for node_idx, i in enumerate(range(0, count, n_gpus_per_node)):
        for _idx in range(n_gpus_per_node):
            job = ray.remote(
                num_cpus=1,
                num_gpus=1,
                name=f"dschat/{_idx + i}",
                resources={available_nodes[node_idx]: 1 / n_gpus_per_node},
            )(DSChatRayRemoteWorker).remote(
                _idx + i,
                world_size,
                ddp_addr,
            )
            jobs.append(job)
            print(ray.available_resources(), flush=True)
    print("init process group...", flush=True)
    init_jobs = [job.init_process_group.remote() for job in jobs[:7]]
    time.sleep(2)
    init_jobs += [job.init_process_group.remote() for job in jobs[7:]]
    ray.get(init_jobs)
    try:
        ray.get([job.run_dschat.remote(args) for job in jobs])
    finally:
        for job in jobs:
            ray.kill(job)
        print("Shutting down...")
        ray.shutdown()


if __name__ == "__main__":
    envs = {"TRANSFORMERS_OFFLINE": "1",
    # "PYTORCH_KERNEL_CACHE_PATH": "/mnt/bs_fs/fw/.cache/pytorch-kernels/",
    # "TRITON_CACHE_DIR": "/mnt/bs_fs/fw/.cache/triton/",
    "TOKENIZERS_PARALLELISM": "true",
    # "TORCH_EXTENSIONS_DIR": "/mnt/bs_fs/fw/.cache/torch-ext/",
    }
    envs['NCCL_IB_GID_INDEX']=str(3)
    envs['NCCL_SOCKET_IFNAME']=str('bond0')
    envs['CUDA_DEVICE_MAX_CONNECTIONS']=str(1)
    envs['TORCH_NCCL_AVOID_RECORD_STREAMS']=str(1)
    envs['TOKENIZERS_PARALLELISM']=str('true')
    envs['OMP_NUM_THREADS']=str(32)
    envs['TRANSFORMERS_OFFLINE']=str(1)
    for k, v in envs.items():
        os.environ[k] = v
    args = parse_args()
    main(args)
