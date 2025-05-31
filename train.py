# train.py

# region imports
import argparse
import os
import torch
import torch.distributed as dist
import torch.nn.utils
from datetime import datetime, timedelta
import signal
import sys

# Set tokenizers parallelism to false to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set NCCL timeout to prevent hanging
os.environ["NCCL_TIMEOUT"] = "300"  # 5 minutes
os.environ["NCCL_BLOCKING_WAIT"] = "1"

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, FSDPModule, ShardingStrategy, BackwardPrefetch
from torch.distributed.device_mesh import init_device_mesh

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.models.starcoder2.modeling_starcoder2 import Starcoder2DecoderLayer
from torch.utils.data import DataLoader, DistributedSampler
from functools import partial
from tqdm import tqdm
from model import LoRAModel
from parameter_dataset import CodeParametersDataset, CodeParametersDatasetFromDict, ParameterExample, SampleDict
from checkpoint import FSDP2Checkpointer, safe_checkpoint_save

from utils import print_memory_stats, setup_seed
# endregion


tokenizer = None  # Will be initialized after fork
graceful_shutdown = False


def signal_handler(signum, frame):
    """Handle graceful shutdown on SIGTERM/SIGINT"""
    global graceful_shutdown
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        print(f"\nReceived signal {signum}, initiating graceful shutdown...")
    graceful_shutdown = True


def setup():
    """Initialize the distributed environment with timeout protection."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize process group with timeout
    try:
        dist.init_process_group(
            "nccl", 
            rank=rank, 
            world_size=world_size,
            timeout=timedelta(seconds=300)  # 5 minute timeout
        )
    except Exception as e:
        print(f"Failed to initialize process group: {e}")
        sys.exit(1)
    
    # Set device
    torch.cuda.set_device(rank)
    
    # Clear cache
    torch.cuda.empty_cache()
    
    return rank, world_size



def setup_memory_snapshot(enabled=True):
    """Setup memory snapshot recording with minimal overhead"""
    if not enabled:
        return
    
    os.makedirs("snapshots", exist_ok=True)

    # Only rank 0 needs to setup snapshots to avoid conflicts
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        print("🔍 Memory snapshot recording enabled")
        torch.cuda.memory._record_memory_history(
            max_entries=100000,  # Keep last 100k alloc/free events
            # context="all"      # Uncomment for stack traces (adds overhead)
        )


def dump_memory_snapshot(step, prefix="memory_snapshot"):
    """Dump memory snapshot to file"""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank != 0:
        return
    
    try:
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshots/{prefix}_step_{step}_{timestamp}.pkl"
        
        print(f"💾 Saving memory snapshot: {filename}")
        torch.cuda.memory._dump_snapshot(filename)
        print(f"✅ Snapshot saved. View at: https://pytorch.org/memory_viz")
        
    except Exception as e:
        print(f"❌ Failed to save memory snapshot: {e}")




def get_training_args():
    """Get memory-efficient training arguments"""
    return {
        "batch_size": 1,  # Start with 1
        "max_length": 512,  # Start with 512
        "gradient_accumulation_steps": 8,  # Use grad accumulation instead of large batches
        "mixed_precision": True,
        "activation_checkpointing": True,
    }



def get_fsdp_memory_breakdown(model):
    """Get detailed FSDP memory breakdown"""
    rank = dist.get_rank()
    
    if rank != 0:
        return
    
    print("\nFSDP Memory Breakdown:")
    print("-" * 40)
    
    # Count parameters by type
    total_params = 0
    lora_params = 0
    base_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if 'lora' in name.lower():
            lora_params += param_count
        else:
            base_params += param_count
    
    print(f"Total parameters: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,} ({lora_params/total_params*100:.2f}%)")
    print(f"Base parameters: {base_params:,} ({base_params/total_params*100:.2f}%)")
    
    # Memory estimates (bfloat16 = 2 bytes per param)
    total_param_memory = total_params * 2 / 1024**3
    lora_param_memory = lora_params * 2 / 1024**3
    
    print(f"\nParameter Memory (bfloat16):")
    print(f"Total: {total_param_memory:.2f} GB")
    print(f"LoRA: {lora_param_memory:.2f} GB")
    print(f"Per GPU (sharded): {total_param_memory / dist.get_world_size():.2f} GB")


def safe_batch_size_test(model, tokenizer, current_batch_size, max_length):
    """Test different batch sizes to find the maximum safe size"""
    rank = dist.get_rank()
    
    if rank != 0:
        return current_batch_size
    
    print("\n🧪 Testing safe batch sizes...")
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    test_sizes = [current_batch_size, current_batch_size * 2, current_batch_size * 4]
    safe_batch_size = current_batch_size
    
    for test_size in test_sizes:
        try:
            print(f"Testing batch size {test_size}...")
            
            # Create dummy batch
            dummy_input = torch.randint(0, tokenizer.vocab_size, 
                                      (test_size, max_length), 
                                      device=torch.cuda.current_device())
            
            dummy_attention = torch.ones_like(dummy_input)
            
            # Test forward pass
            with torch.no_grad():
                outputs = model(input_ids=dummy_input, attention_mask=dummy_attention)
            
            # Check memory usage
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"  Peak memory: {peak_memory:.2f}GB / {gpu_memory:.2f}GB ({peak_memory/gpu_memory*100:.1f}%)")
            
            if peak_memory < gpu_memory * 0.8:  # 80% threshold
                safe_batch_size = test_size
                print(f"  ✅ Safe at batch size {test_size}")
            else:
                print(f"  ❌ Too high memory usage at batch size {test_size}")
                break
                
            # Cleanup
            del dummy_input, dummy_attention, outputs
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  💥 OOM at batch size {test_size}")
                break
            else:
                raise e
    
    print(f"🎯 Recommended safe batch size: {safe_batch_size}")
    return safe_batch_size


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
            print(f"     WARNING: Less than 2GB headroom available!")
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
    """Apply FSDP2 to model using new APIs with better error handling"""
    
    try:
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
        
    except Exception as e:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        if rank == 0:
            print(f"Error applying FSDP2: {e}")
        raise e


def train_epoch_with_accumulation(model, dataloader, optimizer, scheduler, epoch, rank, args, checkpointer):
    """Training with gradient accumulation and improved checkpoint handling"""
    model.train()
    total_loss = 0
    accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    
    optimizer.zero_grad()

    # Reset peak memory stats at start of epoch
    if rank == 0:
        torch.cuda.reset_peak_memory_stats()
    
    checkpoint_saved = False
    
    for step, batch in enumerate(dataloader):
        # Check for graceful shutdown
        if graceful_shutdown:
            if rank == 0:
                print("Graceful shutdown requested, saving checkpoint...")
            
            if not checkpoint_saved:
                metadata = {
                    "epoch": epoch,
                    "step": step,
                    "partial_epoch": True,
                    "model_name": args.model_name,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "target_modules": args.target_modules,
                }
                safe_checkpoint_save(checkpointer, model, optimizer, metadata, max_retries=1)
                checkpoint_saved = True
            
            return total_loss / max(1, step)
        
        if step == 0 or step % 20 == 0:
            monitor_memory(step, log_every=1)

        # Only move tensor items to CUDA
        batch = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()
        }
        
        try:
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Clear cache periodically
                if (step + 1) % (accumulation_steps * 4) == 0:
                    torch.cuda.empty_cache()
            
            total_loss += loss.item() * accumulation_steps
            
            if step in [5, 10, 15, 20]:
                dump_memory_snapshot(step)

            # Emergency checkpoint save if memory is getting high
            if step > 0 and step % 50 == 0:
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                    
                if peak_memory > gpu_memory * 0.9:  # 90% usage
                    if rank == 0:
                        print(f"High memory usage detected ({peak_memory:.1f}GB), saving emergency checkpoint...")
                    
                    metadata = {
                        "epoch": epoch,
                        "step": step,
                        "emergency_save": True,
                        "model_name": args.model_name,
                        "lora_rank": args.lora_rank,
                        "lora_alpha": args.lora_alpha,
                        "target_modules": args.target_modules,
                    }
                    safe_checkpoint_save(checkpointer, model, optimizer, metadata, max_retries=1)
                    
                    # Clear cache after emergency save
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                if rank == 0:
                    print(f"OOM at step {step}, batch size {len(batch['input_ids'])}")
                monitor_memory(step, log_every=1)
                
                # Try to save emergency checkpoint before failing
                try:
                    metadata = {
                        "epoch": epoch,
                        "step": step,
                        "oom_error": True,
                        "model_name": args.model_name,
                        "lora_rank": args.lora_rank,
                        "lora_alpha": args.lora_alpha,
                        "target_modules": args.target_modules,
                    }
                    safe_checkpoint_save(checkpointer, model, optimizer, metadata, max_retries=1)
                except:
                    pass  # Don't fail if emergency save fails
                
                raise e
            else:
                raise e
    
    # Final memory report
    if rank == 0:
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nEpoch {epoch} Peak Memory: {peak_memory:.2f} GB")
    
    return total_loss / len(dataloader)


def monitor_memory(step, log_every=10):
    """Monitor memory usage with improved error handling"""
    if step % log_every != 0:
        return
        
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Get memory stats for current rank
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3
        
        if rank == 0:
            print(f"Step {step} - Rank 0 Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, peak: {max_allocated:.2f}GB")
            
            # Check for potential OOM risk
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if reserved > gpu_memory * 0.85:
                print("Warning: High memory usage detected!")
                torch.cuda.empty_cache()
                
    except Exception as e:
        # Don't fail training if memory monitoring fails
        pass


def cleanup():
    """Clean up the distributed environment with timeout protection."""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        if rank == 0:
            print(f"Warning: Failed to destroy process group cleanly: {e}")


def get_tokenizer(model_name):
    """Get or create tokenizer instance."""
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def cast_model_to_dtype(model, dtype):
    """Cast all model parameters to specified dtype"""
    for param in model.parameters():
        param.data = param.data.to(dtype)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(dtype)


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
    
    # Initialize checkpointer with shorter timeout
    checkpointer = FSDP2Checkpointer(
        args.checkpoint_dir, 
        save_peft_format=True,
        checkpoint_timeout=300  # 5 minutes timeout
    )
    
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
    
    setup_memory_snapshot(enabled=True)

    # Training loop
    if rank == 0:
        print(f"Starting training for {args.num_epochs} epochs")
        print(f"Total batches per epoch: {len(dataloader)}")
        print(f"Batch size: {args.batch_size}")
        print(f"World size: {world_size}")
        
    try:
        for epoch in range(args.num_epochs):
            # Check for graceful shutdown
            if graceful_shutdown:
                if rank == 0:
                    print("Graceful shutdown requested, breaking training loop")
                break
            
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
                checkpointer,
            )
            
            if rank == 0:
                print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint with improved error handling
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
                
                # Use safe checkpoint save with retries
                success = safe_checkpoint_save(checkpointer, model, optimizer, metadata)
                if not success and rank == 0:
                    print("Warning: Checkpoint save failed, continuing training...")
                
                # Cleanup old checkpoints
                if rank == 0:
                    try:
                        checkpointer.cleanup_old_checkpoints(keep_last_n=3)
                    except Exception as e:
                        print(f"Warning: Failed to cleanup old checkpoints: {e}")
        
        # Final checkpoint
        if not graceful_shutdown and (epoch + 1) % args.save_every != 0:
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
            
            safe_checkpoint_save(checkpointer, model, optimizer, metadata)
            
            if rank == 0:
                try:
                    checkpointer.cleanup_old_checkpoints(keep_last_n=3)
                except Exception as e:
                    print(f"Warning: Failed to cleanup old checkpoints: {e}")

    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user")
        # Save emergency checkpoint
        metadata = {
            "epoch": epoch + 1 if 'epoch' in locals() else 0,
            "interrupted": True,
            "model_name": args.model_name,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules,
        }
        safe_checkpoint_save(checkpointer, model, optimizer, metadata, max_retries=1)
        
    except Exception as e:
        if rank == 0:
            print(f"Training failed with error: {e}")
        # Save emergency checkpoint
        try:
            metadata = {
                "epoch": epoch + 1 if 'epoch' in locals() else 0,
                "error": str(e),
                "model_name": args.model_name,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "target_modules": args.target_modules,
            }
            safe_checkpoint_save(checkpointer, model, optimizer, metadata, max_retries=1)
        except:
            pass  # Don't fail if emergency save fails
        raise e
    
    finally:
        # Always try to cleanup
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
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Gradient accumulation steps")

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