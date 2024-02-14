python3 applications/DeepSpeed-Chat/slurm_launch.py \
    -e sosp-baseline-dschat-a13-z3-c7r7-cz3 -f debug --mode slurm -t rlhf --critic_size 7 \
    --actor_size 13 --actor_zero_stage 3 --critic_zero_stage 3