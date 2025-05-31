# B3D-LoRA Trainer

A specialized LoRA (Low-Rank Adaptation) training project designed to enhance code completion capabilities for the [build123d](https://github.com/gumyr/build123d) Python library. This project fine-tunes a code language model to better understand and complete build123d-specific code patterns and syntax.

## Features
- Optimized for AMD GPUs using ROCm
- Built on PyTorch 2.6 with ROCm 6.4.1 support
- Distributed training support via FSDP2
- Mixed precision training capabilities
- Checkpoint management for training resumption

## Requirements
- AMD GPU with ROCm support
- PyTorch 2.6
- ROCm 6.4.1

## Project Structure

| File | Description |
|------|-------------|
| `train.py` | Core training script implementing distributed LoRA fine-tuning with FSDP2 |
| `model.py` | LoRA model implementation and configuration for StarCoder2 |
| `parameter_dataset.py` | Dataset implementation for build123d code samples and parameter combinations |
| `parameter_matrix.md` | Training data (abreviated) |
| `checkpoint.py` | FSDP2-compatible checkpoint management for model states and training progress |
| `utils.py` | Helper functions for memory monitoring, data processing, and training utilities |
| `launch.sh` | Distributed training launcher with environment configuration for multiple GPUs |
| `requirements.txt` | Project dependencies and package versions |
| `pyproject.toml` | Python project metadata and build configuration |


This is my first attempt to train a LoRA.
I'm trying to augment a coder model to improve completion accuracy for a specific python library called [build123d](https://github.com/gumyr/build123d)

I use AMD GPUs. The requirements.txt will install the ROCm specific PyTorch from AMD's wheels.</br>
[Install PyTorch for ROCm](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-pytorch.html#install-pytorch-via-pip)

- PyTorch 2.6
- ROCm 6.4.1

I used [https://github.com/pytorch/examples/tree/main/distributed/FSDP2](https://github.com/pytorch/examples/tree/main/distributed/FSDP2)
as my starting point, but it's probably unrecognizable now that Claude has had his way with it.

