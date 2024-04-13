python3 applications/DeepSpeed-Chat/slurm_launch.py \
    -e dschat -f debug \
    --actor_size 34 --use_hybrid_engine --inference_tp_size 8 --tp_gather_partition_size 1 --offload_ref