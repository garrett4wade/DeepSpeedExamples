srun --container-image=llm/llm-dschat --exclude=QH-com29,QH-com35 \
    --container-mounts=/lustre:/lustre,/home/fw/workspace/DeepSpeedExamples/DeepSpeed:/DeepSpeed \
    --pty -c99 --mem=900G --gpus=8 --container-mount-home  bash