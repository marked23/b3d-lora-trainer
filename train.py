# train.py

import argparse
import os
import torch
import torch.distributed as dist
import torch.nn.utils

# Set tokenizers parallelism to false to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, FSDPModule
from torch.distributed.device_mesh import init_device_mesh


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.models.starcoder2.modeling_starcoder2 import Starcoder2DecoderLayer
from torch.utils.data import DataLoader, DistributedSampler
from functools import partial
from tqdm import tqdm
from model import LoRAModel
from parameter_dataset import CodeParametersDataset, CodeParametersDatasetFromDict, ParameterExample, SampleDict
from checkpoint import FSDP2Checkpointer

from utils import print_memory_stats, setup_seed

# Initialize tokenizer with parallelism enabled
tokenizer = None  # Will be initialized after fork

def get_tokenizer(model_name):
    """Get or create tokenizer instance."""
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def setup():
    """Initialize the distributed environment."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)
    
    return rank, world_size


def monitor_memory(rank, step):
    """Monitor memory usage"""
    if rank == 0 and step % 10 == 0:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Step {step}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


# Updated training configuration
def get_training_args():
    """Get memory-efficient training arguments"""
    return {
        "batch_size": 1,  # Start with 1
        "max_length": 512,  # Start with 512
        "gradient_accumulation_steps": 8,  # Use grad accumulation instead of large batches
        "mixed_precision": True,
        "activation_checkpointing": True,
    }


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


# Cast all model parameters to bfloat16 before FSDP wrapping
def cast_model_to_dtype(model, dtype):
    for param in model.parameters():
        param.data = param.data.to(dtype)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(dtype)

def estimate_memory_usage(model, batch_size, seq_length, vocab_size):
    """Estimate memory usage for the model"""
    
    # Model parameters (in bfloat16 = 2 bytes per param)
    total_params = sum(p.numel() for p in model.parameters())
    model_memory_gb = (total_params * 2) / (1024**3)
    
    # Optimizer states (AdamW stores 2 states per parameter)
    optimizer_memory_gb = model_memory_gb * 2
    
    # Gradients (same size as parameters)
    gradient_memory_gb = model_memory_gb
    
    # Activations (rough estimate)
    # For transformer: batch_size * seq_length * hidden_size * num_layers * 4 (rough multiplier)
    hidden_size = 4608  # From your config
    num_layers = 32
    activation_memory_gb = (batch_size * seq_length * hidden_size * num_layers * 4 * 2) / (1024**3)
    
    # Attention matrices: batch_size * num_heads * seq_length^2
    num_heads = 36
    attention_memory_gb = (batch_size * num_heads * seq_length**2 * 2) / (1024**3)
    
    total_memory_gb = (
        model_memory_gb + 
        optimizer_memory_gb + 
        gradient_memory_gb + 
        activation_memory_gb + 
        attention_memory_gb
    )
    
    print(f"Memory Estimation:")
    print(f"  Model parameters: {model_memory_gb:.2f} GB")
    print(f"  Optimizer states: {optimizer_memory_gb:.2f} GB") 
    print(f"  Gradients: {gradient_memory_gb:.2f} GB")
    print(f"  Activations: {activation_memory_gb:.2f} GB")
    print(f"  Attention matrices: {attention_memory_gb:.2f} GB")
    print(f"  Total estimated: {total_memory_gb:.2f} GB")
    print(f"  Per GPU (with {torch.cuda.device_count()} GPUs): {total_memory_gb/torch.cuda.device_count():.2f} GB")
    
    return total_memory_gb

def check_gpu_memory():
    """Check current GPU memory usage"""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")

def debug_fsdp_memory_usage(model):
    """Debug FSDP memory usage"""
    if hasattr(model, '_fsdp_wrapped_module'):
        print("FSDP Memory Stats:")
        
        # Check if parameters are sharded
        total_params = 0
        sharded_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            if hasattr(param, '_is_sharded') and param._is_sharded:
                sharded_params += param.numel()
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Sharded parameters: {sharded_params:,}")
        print(f"  Sharding efficiency: {(sharded_params/total_params)*100:.1f}%")

def validate_memory_before_training(model, args):
    """Validate memory requirements before starting training"""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if rank == 0:
        print("\n" + "="*50)
        print("MEMORY VALIDATION")
        print("="*50)
        
        # Check current memory
        check_gpu_memory()
        
        # Estimate total memory needed
        total_batch_size = args.batch_size * int(os.environ.get("WORLD_SIZE", 1))
        estimated_memory = estimate_memory_usage(model, total_batch_size, args.max_length, 49152)
        
        # Get available memory per GPU
        available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        memory_per_gpu = estimated_memory / torch.cuda.device_count()
        
        # Calculate memory headroom
        memory_headroom = available_memory - memory_per_gpu
        
        print(f"\nMemory Analysis:")
        print(f"  Available memory per GPU: {available_memory:.1f} GB")
        print(f"  Estimated memory per GPU: {memory_per_gpu:.1f} GB")
        print(f"  Memory headroom: {memory_headroom:.1f} GB")
        
        # Provide specific recommendations based on memory analysis
        print(f"\nRecommendations:")
        if memory_headroom < 2.0:
            print(f"  ⚠️  WARNING: Less than 2GB headroom available!")
            if args.batch_size > 1:
                suggested_batch = max(1, args.batch_size // 2)
                print(f"  • Consider reducing batch size from {args.batch_size} to {suggested_batch}")
            if args.max_length > 1024:
                suggested_length = min(1024, args.max_length // 2)
                print(f"  • Consider reducing sequence length from {args.max_length} to {suggested_length}")
            if args.gradient_accumulation_steps < 16:
                print(f"  • Consider increasing gradient accumulation steps from {args.gradient_accumulation_steps} to {args.gradient_accumulation_steps * 2}")
        else:
            print(f"  ✓ Memory headroom looks good ({memory_headroom:.1f} GB)")
            if memory_headroom > 8.0:
                print(f"  • You could potentially increase batch size or sequence length")
        
        print("="*50)
        print()


def apply_fsdp2_to_model(model, args, world_size):
    """Apply FSDP2 to model using new APIs"""
    
    # Create device mesh
    device_mesh = init_device_mesh("cuda", (world_size,))
    
    # Configure mixed precision if requested
    mp_policy = None
    if getattr(args, 'mixed_precision', False):
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        )
    
    # Apply FSDP2 to transformer layers first
    for name, module in model.named_modules():
        if isinstance(module, Starcoder2DecoderLayer):
            fully_shard(
                module,
                mesh=device_mesh,
                mp_policy=mp_policy,
            )
    
    # Apply FSDP2 to entire model
    fully_shard(
        model,
        mesh=device_mesh, 
        mp_policy=mp_policy,
    )
    
    return model


def train_epoch_with_accumulation(model, dataloader, optimizer, scheduler, epoch, rank, args):
    """Training with gradient accumulation"""
    model.train()
    total_loss = 0
    accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        # Only move tensor items to CUDA
        batch = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()
        }
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps  # Scale loss
        
        # Backward pass
        loss.backward()
        
        # Update every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # Monitor memory
        monitor_memory(rank, step)
    
    return total_loss / len(dataloader)

# def train_epoch(
#     model,
#     dataloader,
#     optimizer,
#     scheduler,
#     epoch,
#     rank,
#     args,
# ):
#     """Train for one epoch."""
#     model.train()
#     total_loss = 0
    
#     # Progress bar only on rank 0
#     if rank == 0:
#         pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
    
#     for step, batch in enumerate(dataloader):
#         # Move batch to device
#         batch = {k: v.cuda() for k, v in batch.items()}
        
#         # Forward pass
#         outputs = model(**batch)
#         loss = outputs.loss
        
#         # Backward pass
#         loss.backward()
        
#         # Gradient clipping
#         if args.grad_clip > 0:
#             model.clip_grad_norm_(args.grad_clip)
        
#         # Optimizer step
#         optimizer.step()
#         scheduler.step()
#         optimizer.zero_grad()
        
#         # Accumulate loss
#         total_loss += loss.item()
        
#         # Update progress bar
#         if rank == 0:
#             pbar.update(1)
#             pbar.set_postfix({
#                 "loss": f"{loss.item():.4f}",
#                 "lr": f"{scheduler.get_last_lr()[0]:.2e}"
#             })
        
#         # Print memory stats periodically
#         if step % 50 == 0 and rank == 0:
#             print_memory_stats(rank)
        
#         if step % 5 == 0:
#             torch.cuda.empty_cache()
    
#     if rank == 0:
#         pbar.close()
    
#     return total_loss / len(dataloader)


def print_model_shapes(model, rank):
    """Print shapes of model and its layers."""
    if rank != 0:
        return
        
    print("\nModel Architecture and Shapes:")
    print("=" * 50)
    
    # Print overall model shape
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {(trainable_params/total_params)*100:.4f}")
    
    # Print shapes of each module
    print("\nLayer Shapes:")
    print("-" * 50)
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            if hasattr(module.weight, 'shape'):
                print(f"{name}:")
                print(f"  Weight shape: {module.weight.shape}")
                if hasattr(module, 'bias') and module.bias is not None:
                    print(f"  Bias shape: {module.bias.shape}")
                if hasattr(module, 'in_features'):
                    print(f"  Input features: {module.in_features}")
                if hasattr(module, 'out_features'):
                    print(f"  Output features: {module.out_features}")
                print()
    
    # Print input/output shapes for key layers
    print("\nKey Layer Shapes:")
    print("-" * 50)
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            print(f"{name}:")
            if isinstance(module, torch.nn.Linear):
                print(f"  Input features: {module.in_features}")
                print(f"  Output features: {module.out_features}")
                print(f"  Weight shape: {module.weight.shape}")
                if module.bias is not None:
                    print(f"  Bias shape: {module.bias.shape}")
            elif isinstance(module, torch.nn.Embedding):
                print(f"  Vocab size: {module.num_embeddings}")
                print(f"  Embedding dim: {module.embedding_dim}")
                print(f"  Weight shape: {module.weight.shape}")
            print()


def get_data_collator(tokenizer):
    """Create a data collator for the dataset"""
    def collate_fn(batch):
        # Stack tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        # Get method names and contexts
        method_names = [item['method_name'] for item in batch]
        contexts = [item['context'] for item in batch]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'method_name': method_names,
            'context': contexts
        }
    return collate_fn

def main(args):
    # Setup distributed training
    rank, world_size = setup()
    setup_seed(42 + rank)
    
    # Initialize tokenizer after fork
    tokenizer = get_tokenizer(args.model_name)
    
    # Load base model first
    if rank == 0:
        print(f"Loading model: {args.model_name}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,  # Disable KV cache for training
    )
    
    # Cast model to bfloat16
    cast_model_to_dtype(base_model, torch.bfloat16)
    
    # Apply LoRA to the base model first
    lora_wrapper = LoRAModel(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        base_model=base_model,
    )
    
    # Get the PEFT model (not the wrapper)
    model = lora_wrapper.model
    
    # Ensure all parameters are in bfloat16
    for param in model.parameters():
        param.data = param.data.to(torch.bfloat16)
    
    # Move to GPU
    model = model.cuda()
    
    # Apply FSDP2
    model = apply_fsdp2_to_model(model, args, world_size)
    
    # Ensure all parameters are on the correct device and in the right dtype
    for param in model.parameters():
        if param.device != torch.cuda.current_device():
            param.data = param.data.to(torch.cuda.current_device())
        if param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)
    
    # Create dataset from parameter matrix
    dataset = CodeParametersDataset(
        parameter_matrix_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        include_context=True,
        augment_negatives=True
    )
    
    if rank == 0:
        print(f"Dataset size: {len(dataset)}")
        print(f"Data path: {args.data_path}")
        
        # Print a sample to verify
        sample = dataset[0]
        print("\nSample data:")
        print(f"Input shape: {sample['input_ids'].shape}")
        print(f"Method: {sample['method_name']}")
        print(f"Context: {sample['context']}")
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),  # Ensure batch size isn't larger than dataset
        sampler=sampler,
        collate_fn=get_data_collator(tokenizer),
        num_workers=4,
        pin_memory=True,
        drop_last=False,  # Don't drop incomplete batches for small datasets
    )
    
    if rank == 0:
        print(f"Dataloader size: {len(dataloader)}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
        print(f"Dataset size: {len(dataset)}")
        if len(dataloader) == 0:
            raise ValueError(
                f"No batches available. Dataset size ({len(dataset)}) is too small for "
                f"batch size {args.batch_size} with {world_size} GPUs. "
                "Please add more examples or reduce batch size."
            )
    
    # Print model shapes
    # print_model_shapes(model, rank)
    
    if rank == 0:
        print(f"\nModel configuration:")
        print(f"Input shape: {args.max_length}")
        print(f"Batch size: {args.batch_size}")
        print(f"World size: {world_size}")
        print(f"Hidden size: {base_model.config.hidden_size}")
        print(f"Vocab size: {base_model.config.vocab_size}")
        print(f"Number of layers: {base_model.config.num_hidden_layers}")
        print(f"Number of attention heads: {base_model.config.num_attention_heads}")
    
    # Initialize checkpointer
    checkpointer = FSDP2Checkpointer(args.checkpoint_dir)
    
    # Initialize optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Load checkpoint if exists and resume is not disabled
    if not checkpointer.is_empty() and not args.no_resume:
        if rank == 0:
            print(f"Loading checkpoint from: {checkpointer.last_training_time}")
        success = checkpointer.load(model, optimizer)
        if not success:
            if rank == 0:
                print("Failed to load checkpoint")
    elif rank == 0 and not args.no_resume:
        print("No checkpoint found, starting fresh training")
    elif rank == 0 and args.no_resume:
        print("Skipping checkpoint loading as --no-resume was specified")
    
    
    # Initialize scheduler
    total_steps = len(dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    
    # Training loop
    if rank == 0:
        print(f"Starting training for {args.num_epochs} epochs")
        print(f"Total batches per epoch: {len(dataloader)}")
        print(f"Batch size: {args.batch_size}")
        print(f"World size: {world_size}")
        
    # Validate memory before training
    validate_memory_before_training(model, args)
    
    for epoch in range(args.num_epochs):
        # Set epoch for sampler
        sampler.set_epoch(epoch)
        
        # Train for one epoch
        avg_loss = train_epoch_with_accumulation(
            model,
            dataloader,
            optimizer,
            scheduler,
            epoch + 1,
            rank,
            args,
        )
        
        if rank == 0:
            print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            if rank == 0:
                print(f"Saving checkpoint at epoch {epoch + 1}")
            
            metadata = {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "model_name": args.model_name,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "target_modules": args.target_modules,
            }
            
            checkpointer.save(model, optimizer, metadata)
            
            # Cleanup old checkpoints
            if rank == 0:
                checkpointer.cleanup_old_checkpoints(keep_last_n=3)
    
    # Final checkpoint
    if (epoch + 1) % args.save_every == 0:
        if rank == 0:
            print("Saving final checkpoint")
        
        metadata = {
            "epoch": args.num_epochs,
            "model_name": args.model_name,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules,
            "final": True,
        }
        
        checkpointer.save(model, optimizer, metadata)
        checkpointer.cleanup_old_checkpoints(keep_last_n=3)

    # Cleanup
    cleanup()
    
    if rank == 0:
        print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune StarCoder2 with LoRA using FSDP2")
    
    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="bigcode/starcoder2-7b",
        help="Name or path of the base model",
    )
    
    # LoRA arguments
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj", "k_proj", "o_proj"],
        help="Target modules for LoRA",
    )
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size per GPU")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps")

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="parameter_matrix.md",  # Match the example file name
        help="Path to parameter matrix markdown file",
    )
    
    # FSDP arguments
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision")
    
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--no-resume", action="store_false", help="Do not resume from previous checkpoint")
    
    args = parser.parse_args()
    main(args)