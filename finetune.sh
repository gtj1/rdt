# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG=INFO
# export NCCL_NVLS_ENABLE=0

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-finetune-1b"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="cutlass"

export WANDB_PROJECT="robotics_diffusion_transformer"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...

accelerate launch main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="rdt_model" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=8 \
    --sample_batch_size=32 \
    --max_train_steps=200 \
    --checkpointing_period=200 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=wandb
    --num_gpus=4

# deepspeed --num_gpus=4 main.py \
#     --deepspeed="./configs/zero2.json" \
#     --pretrained_model_name_or_path="rdt_model" \
#     --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
#     --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
#     --output_dir=$OUTPUT_DIR \
#     --train_batch_size=32 \
#     --sample_batch_size=64 \
#     --max_train_steps=1000 \
#     --checkpointing_period=1000 \
#     --sample_period=500 \
#     --checkpoints_total_limit=40 \
#     --lr_scheduler="constant" \
#     --learning_rate=1e-4 \
#     --mixed_precision="bf16" \
#     --dataloader_num_workers=8 \
#     --image_aug \
#     --dataset_type="finetune" \
#     --state_noise_snr=40 \
#     --load_from_hdf5 \
#     --report_to=wandb


# deepspeed --hostfile=hostfile.txt main.py \
#     --deepspeed="./configs/zero2.json" \
#     --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b" \
#     --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
#     --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
#     --output_dir=$OUTPUT_DIR \
#     --train_batch_size=32 \
#     --sample_batch_size=64 \
#     --max_train_steps=200000 \
#     --checkpointing_period=1000 \
#     --sample_period=500 \
#     --checkpoints_total_limit=40 \
#     --lr_scheduler="constant" \
#     --learning_rate=1e-4 \
#     --mixed_precision="bf16" \
#     --dataloader_num_workers=8 \
#     --image_aug \
#     --dataset_type="finetune" \
#     --state_noise_snr=40 \
#     --load_from_hdf5 \
#     --report_to=wandb

    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training
    # --precomp_lang_embed \
