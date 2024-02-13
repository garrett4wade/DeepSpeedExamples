srun --container-image=llm/llm-dschat --exclude=QH-com48 \
    --container-mounts=/lustre:/lustre \
    --pty -c99 --mem=900G --gpus=8 --container-mount-home  bash