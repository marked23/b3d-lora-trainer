# utils.py
import torch
import random
import numpy as np
import psutil
import GPUtil
from typing import Dict, Union


def setup_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_memory_stats(rank: int):
    """Print memory usage statistics."""
    if rank == 0:
        # CPU memory
        cpu_percent = psutil.virtual_memory().percent
        cpu_used = psutil.virtual_memory().used / 1024**3  # GB
        cpu_total = psutil.virtual_memory().total / 1024**3  # GB
        
        # GPU memory
        gpus = GPUtil.getGPUs()
        
        print(f"\n{'='*50}")
        print(f"Memory Usage (Rank {rank}):")
        print(f"CPU: {cpu_used:.2f}/{cpu_total:.2f} GB ({cpu_percent:.1f}%)")
        
        for gpu in gpus:
            gpu_used = gpu.memoryUsed / 1024  # GB
            gpu_total = gpu.memoryTotal / 1024  # GB
            gpu_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
            print(f"GPU {gpu.id}: {gpu_used:.2f}/{gpu_total:.2f} GB ({gpu_percent:.1f}%)")
        print(f"{'='*50}\n")


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_parameter_count(model) -> Dict[str, Union[int, float]]:
    """Get parameter counts for a model."""
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "percentage": (trainable_params / total_params) * 100
    }