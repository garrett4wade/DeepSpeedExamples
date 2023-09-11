srun --nodelist=frl8a138 --container-image=llm/llm-gpu \
    --container-mounts=/data:/data,/hddlustre:/hddlustre,/lustre:/lustre \
    --pty -c40 --mem=200G --gpus=4 --container-mount-home  bash