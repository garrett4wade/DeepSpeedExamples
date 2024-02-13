import itertools
import logging
import os
import platform
import socket
import time

import torch

import utils.name_resolve as name_resolve

from contextlib import closing
import socket
import getpass

logger = logging.getLogger("System-GPU")

USER_NAMESPACE = getpass.getuser()


def trainer_ddp_peer(experiment_name, trial_name, model_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/trainer_ddp_peer/{model_name}"


def trainer_ddp_local_peer(experiment_name, trial_name, host_name, model_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/trainer_ddp_local_peer/{host_name}/{model_name}"


def trainer_ddp_master(experiment_name, trial_name, model_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/trainer_ddp_master/{model_name}"


def find_free_port():
    """From, stackoverflow Issue 1365265
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def gethostname():
    return socket.gethostname()


def gethostip():
    return socket.gethostbyname(socket.gethostname())


def gpu_count():
    """Returns the number of gpus on a node. Ad-hoc to frl cluster.
    """
    if platform.system() == "Darwin":
        return 0
    elif platform.system() == "Windows":
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            return 0
    else:
        dev_directories = list(os.listdir("/dev/"))
        for cnt in itertools.count():
            if "nvidia" + str(cnt) in dev_directories:
                continue
            else:
                break
        return cnt


def reveal_ddp_identity(expr_name, trial_name, model_name, worker_index):
    global_peer_name = trainer_ddp_peer(expr_name, trial_name, model_name)
    name_resolve.add_subentry(global_peer_name, worker_index, keepalive_ttl=30)
    local_peer_name = trainer_ddp_local_peer(expr_name, trial_name,
                                             socket.gethostname(), model_name)
    name_resolve.add_subentry(local_peer_name, worker_index, keepalive_ttl=30)


def setup_ddp(expr_name, trial_name, model_name, worker_index):
    logger.info(f"Setup DDP {worker_index} for model {model_name}")

    global_peers = list(
        sorted(
            name_resolve.get_subtree(
                trainer_ddp_peer(expr_name, trial_name, model_name))))
    assert len(global_peers) == len(
        set(global_peers)), f"Duplicated trainer worker index. {global_peers}"
    world_size = len(global_peers)
    ddp_rank = global_peers.index(str(worker_index))

    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(
        ',')) == 1, os.environ['CUDA_VISIBLE_DEVICES']
    local_gpu_id = int(os.environ['CUDA_VISIBLE_DEVICES'])

    ddp_master_name = trainer_ddp_master(expr_name, trial_name, model_name)
    if ddp_rank == 0:
        host_ip = socket.gethostbyname(socket.gethostname())
        port = find_free_port()
        ddp_init_address = f"tcp://{host_ip}:{port}"
        name_resolve.add(ddp_master_name, ddp_init_address, keepalive_ttl=60)
    else:
        try:
            ddp_init_address = name_resolve.wait(ddp_master_name, timeout=60)
        except TimeoutError:
            raise TimeoutError(
                f"DDP trainer(index:{worker_index}), rank {ddp_rank} for model "
                f"{model_name} wait for ddp_init_method timeout.")

    torch_dist_kwargs = dict(world_size=world_size,
                             rank=ddp_rank,
                             init_method=ddp_init_address,
                             backend='nccl')
    torch.cuda.set_device(
        0)  # initialize CUDA here with only a single visible device
    # This environment variable is used by DeepSpeed.
    os.environ['LOCAL_RANK'] = "0"

    torch.distributed.init_process_group(**torch_dist_kwargs,
                                         group_name=model_name)

    return world_size, ddp_rank, local_gpu_id


def isolate_cuda_device(worker_type, rank, world_size, experiment_name,
                        trial_name):
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        return

    name_resolve_identifier = f"__type_{worker_type}"
    name_resolve.add_subentry(
        trainer_ddp_local_peer(experiment_name, trial_name,
                               socket.gethostname(), name_resolve_identifier),
        rank,
        keepalive_ttl=60,
    )
    name_resolve.add_subentry(
        trainer_ddp_peer(experiment_name, trial_name, name_resolve_identifier),
        rank,
        keepalive_ttl=30,
    )
    logger.info(f"Rank {rank} waiting for peers, world size {world_size}...")
    while len(
            name_resolve.get_subtree(
                trainer_ddp_peer(experiment_name, trial_name,
                                 name_resolve_identifier))) < world_size:
        time.sleep(0.1)
    logger.info(f"Rank {rank} discovers all peers, resolving local rank...")
    local_peer_name = trainer_ddp_local_peer(
        experiment_name,
        trial_name,
        socket.gethostname(),
        name_resolve_identifier,
    )
    local_peers = list([
        str(x) for x in sorted(
            [int(x) for x in name_resolve.get_subtree(local_peer_name)])
    ])
    logger.info(
        f"Rank {rank} discovers local peers with global ranks {local_peers}")

    local_peer_index = local_peers.index(str(rank))
    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == len(local_peers):
        local_gpu_id = list(
            map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(
                ',')))[local_peer_index]
    elif len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1:
        local_gpu_id = int(os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        if not os.environ.get('DLLM_MODE') == "LOCAL":
            raise RuntimeError(
                f"Unresolvable CUDA_VISIBLE_DEVICES {os.environ['CUDA_VISIBLE_DEVICES']}, "
                f"local peers (global ranks) {local_peers}, local peer index {local_peer_index}."
            )
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        local_gpu_id = int(devices[local_peer_index % len(devices)])

    logger.info(
        f"Worker type {worker_type} rank {rank} running on host {socket.gethostname()}, "
        f"local peer index: {local_peer_index}, local gpu id {local_gpu_id}.")

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_gpu_id)
    return local_gpu_id
