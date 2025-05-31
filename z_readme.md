# LoRA Fine-tuning for Qwen2.5-Coder-7B with FSDP2

This project implements LoRA (Low-Rank Adaptation) fine-tuning for the Qwen2.5-Coder-7B model using PyTorch's Fully Sharded Data Parallel (FSDP2) for efficient distributed training.

## Features

- **LoRA Implementation**: Efficient parameter-efficient fine-tuning with configurable rank and alpha
- **FSDP2 Integration**: Full support for PyTorch's latest distributed training framework
- **Mixed Precision Training**: BF16 mixed precision for faster training and lower memory usage
- **Distributed Checkpointing**: Save and resume training with distributed checkpoint support
- **Code Instruction Dataset**: Toy dataset with Python coding examples for demonstration

## Project Structure

```
.
├── model.py          # LoRA model implementation
├── dataset.py        # Code instruction dataset
├── train.py          # Main training script
├── checkpoint.py     # Checkpoint management
├── utils.py          # Training utilities
├── inference.py      # Inference script for testing
├── launch.sh         # Launch script for distributed training
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPUs (recommended: 4+ GPUs with 24GB+ memory each)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

### Single GPU Training

```bash
python train.py \
    --model-name "Qwen/Qwen2.5-Coder-7B" \
    --batch-size 2 \
    --num-epochs 3 \
    --learning-rate 5e-5 \
    --lora-rank 16 \
    --mixed-precision
```

### Multi-GPU Training (Recommended)

Use the provided launch script for distributed training:

```bash
chmod +x launch.sh
./launch.sh
```

Or manually with torchrun:

```bash
torchrun --nproc_per_node=4 train.py \
    --model-name "Qwen/Qwen2.5-Coder-7B" \
    --batch-size 4 \
    --num-epochs 3 \
    --mixed-precision
```

## Training Arguments

### Model Configuration
- `--model-name`: Base model name or path (default: "Qwen/Qwen2.5-Coder-7B")
- `--lora-rank`: LoRA rank (default: 16)
- `--lora-alpha`: LoRA alpha scaling parameter (default: 16.0)
- `--lora-dropout`: LoRA dropout rate (default: 0.05)
- `--target-modules`: Modules to apply LoRA (default: ["q_proj", "v_proj", "k_proj", "o_proj"])

### Training Configuration
- `--batch-size`: Batch size per GPU (default: 4)
- `--num-epochs`: Number of training epochs (default: 3)
- `--learning-rate`: Learning rate (default: 5e-5)
- `--weight-decay`: Weight decay (default: 0.01)
- `--grad-clip`: Gradient clipping value (default: 1.0)
- `--max-length`: Maximum sequence length (default: 512)

### FSDP Configuration
- `--mixed-precision`: Enable BF16 mixed precision training

### Checkpoint Configuration
- `--checkpoint-dir`: Directory for saving checkpoints (default: "checkpoints")
- `--save-every`: Save checkpoint every N epochs (default: 1)

## Dataset

The project includes a toy code instruction dataset with Python programming examples:

- Factorial calculation
- Palindrome checking
- Binary search
- Stack implementation
- List merging
- Fibonacci numbers
- Linked list reversal
- Timing decorators

Each example is formatted as:
```
### Instruction:
[Task description]

### Response:
[Code implementation]
```

## Inference

Test the fine-tuned model with the inference script:

### Interactive Mode

```bash
python inference.py \
    --model-name "Qwen/Qwen2.5-Coder-7B" \
    --lora-checkpoint "checkpoints/lora_checkpoints/[timestamp]/lora_weights.pt" \
    --interactive
```

### Batch Mode

```bash
python inference.py \
    --model-name "Qwen/Qwen2.5-Coder-7B" \
    --lora-checkpoint "checkpoints/lora_checkpoints/[timestamp]/lora_weights.pt"
```

## Memory Requirements

Approximate GPU memory usage with BF16 mixed precision:

- **Training**: ~20-24GB per GPU (batch size 4, max length 512)
- **Inference**: ~15-18GB (single GPU)

Reduce batch size or sequence length if encountering OOM errors.

## LoRA Implementation Details

The implementation includes:

1. **LoRALayer**: Core LoRA adapter with A and B matrices
2. **LoRALinear**: Wrapper that combines base Linear layer with LoRA adapter
3. **Automatic application**: Applies LoRA to specified target modules
4. **Efficient checkpointing**: Saves only LoRA parameters (~30MB vs 14GB full model)

### Parameter Efficiency

With default settings (rank=16):
- Trainable parameters: ~10M (0.14% of total)
- Total parameters: ~7B
- Checkpoint size: ~30MB (LoRA weights only)

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size
- Reduce sequence length
- Enable gradient checkpointing (not implemented in this version)
- Use more GPUs

### Slow Training
- Ensure mixed precision is enabled
- Check GPU utilization with `nvidia-smi`
- Verify NCCL backend is being used

### Checkpoint Loading Issues
- Ensure checkpoint path is correct
- Check that LoRA configuration matches between training and inference

## Future Improvements

- Gradient checkpointing support
- Dynamic batching
- Evaluation metrics
- Integration with larger datasets
- Support for more model architectures
- Quantization support (4-bit, 8-bit)

## License

This project is provided as-is for educational and research purposes.