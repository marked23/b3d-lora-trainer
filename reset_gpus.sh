#!/bin/bash

# Script to reset all AMD GPUs using rocm-smi

# Get the number of GPUs
NUM_GPUS=5

echo "Found $NUM_GPUS GPUs. Resetting each one..."

for ((i=0; i<NUM_GPUS; i++)); do
    echo "Resetting GPU $i..."
    sudo rocm-smi --gpureset -d $i
done

echo "All GPUs have been reset."

ps aux | grep -i python