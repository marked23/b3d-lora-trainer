#!/bin/bash

# Number of GPUs
NUM_GPUS=5

# Training configuration
MODEL_NAME="bigcode/starcoder2-7b"
BATCH_SIZE=2
LEARNING_RATE=5e-5
NUM_EPOCHS=3
MAX_LENGTH=1536

# LoRA configuration
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.1
GRAD_ACCUMULATION_STEPS=8


# Output directory
OUTPUT_DIR="./checkpoints"

# Create output directory
mkdir -p $OUTPUT_DIR

export NCCL_TIMEOUT=300
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export NCCL_BUFFSIZE=2097152
export NCCL_NTHREADS=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

### Try all this later
# # Enable P2P communication for GPUs on same PCIe complex
export NCCL_P2P_LEVEL=PHB  # Use P2P for GPUs on same PCIe root complex
export NCCL_P2P_DISABLE=0   # Enable P2P (0 means enabled)
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4  # Adjust based on your GPU IDs

# # NCCL optimizations for PCIe
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_NCHANNELS=4

# # PyTorch optimizations
# export OMP_NUM_THREADS=1  # Adjust based on CPU cores

# Run distributed training
OMP_NUM_THREADS=1 torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    train.py \
    --model-name $MODEL_NAME \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --num-epochs $NUM_EPOCHS \
    --max-length $MAX_LENGTH \
    --lora-rank $LORA_RANK \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT \
    --target-modules q_proj v_proj k_proj o_proj \
    --mixed-precision \
    --checkpoint-dir $OUTPUT_DIR \
    --save-every 1 \
    --gradient-accumulation-steps $GRAD_ACCUMULATION_STEPS