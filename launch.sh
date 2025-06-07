#!/bin/bash

# Number of GPUs
NUM_GPUS=5

# Training configuration
MODEL_NAME="bigcode/starcoder2-7b"
BATCH_SIZE=4
LEARNING_RATE=1e-5
NUM_EPOCHS=3
MAX_LENGTH=1536

# LoRA configuration
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.1
GRAD_ACCUMULATION_STEPS=4

# Output directory
OUTPUT_DIR="./checkpoints"

# Create output directory
mkdir -p $OUTPUT_DIR

export NCCL_TIMEOUT=300
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export NCCL_BUFFSIZE=8388608
export TOKENIZERS_PARALLELISM=false

# NCCL optimizations for PCIe
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=32
export NCCL_P2P_LEVEL=PHB  # Use P2P for GPUs on same PCIe root complex
export NCCL_P2P_DISABLE=0   # Enable P2P (0 means enabled)

export NCCL_NTHREADS=128
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# launch.sh - Add MSCCL++ disable
export NCCL_MSCCL_ENABLE=0
export RCCL_MSCCLPP_ENABLE=0
export MSCCL_ENABLE=0
export NCCL_ALGO=Ring,Tree,Fused    # Use standard algorithms instead

# ROCm-specific optimizations (optional)
export HSA_FORCE_FINE_GRAIN_PCIE=1  # Better PCIe performance
export ROCR_VISIBLE_DEVICES=0,1,2,3,4  # Explicit GPU visibility
export HIP_VISIBLE_DEVICES=0,1,2,3,4   # HIP-specific visibility

# HPC-specific ROCm settings
export HSA_ENABLE_INTERRUPT=1
export ROCR_ENABLE_PRE_VEGA_FINALIZATION=0
export HSA_XNACK=0  # Disable XNACK for better P2P performance

# Run distributed training
OMP_NUM_THREADS=1 torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    train.py \
    --model-name $MODEL_NAME \
    --batch-size 1 \
    --learning-rate $LEARNING_RATE \
    --num-epochs $NUM_EPOCHS \
    --max-length $MAX_LENGTH \
    --lora-rank $LORA_RANK \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT \
    --target-modules q_proj v_proj k_proj o_proj \
    --checkpoint-dir $OUTPUT_DIR \
    --save-every 1 \
    --gradient-accumulation-steps $GRAD_ACCUMULATION_STEPS \
    --mixed-precision \
    --no-resume
