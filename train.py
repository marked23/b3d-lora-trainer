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
import traceback
import time
import threading


# PREFER SETTING ENVIRONMENT VARIABLES IN launch.sh


from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.models.starcoder2.modeling_starcoder2 import Starcoder2DecoderLayer
from torch.utils.data import DataLoader, DistributedSampler
from functools import partial
from tqdm import tqdm
from model import LoRAModel
# from parameter_dataset import CodeParametersDataset, CodeParametersDatasetFromDict, ParameterExample, SampleDict
from build123d_dataset import CodeTextDataset
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
        print("Training will stop after completing current operations...")
    
    graceful_shutdown = True
    
    # Try to synchronize shutdown across all ranks with timeout
    if dist.is_initialized():
        try:
            # Give other ranks a moment to receive the signal
            time.sleep(1)
            
            # Use a timeout barrier to avoid hanging indefinitely
            if rank == 0:
                print("Coordinating graceful shutdown across all ranks...")
            
            # Use all_reduce to signal shutdown to all ranks
            shutdown_tensor = torch.tensor(1.0, device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.all_reduce(shutdown_tensor, op=dist.ReduceOp.MAX)
            
            if rank == 0:
                print("All ranks notified of graceful shutdown")
                
        except Exception as e:
            if rank == 0:
                print(f"Warning: Failed to coordinate shutdown: {e}")
    
    # Set up force exit on second signal
    signal.signal(signal.SIGTERM, force_exit_handler)
    signal.signal(signal.SIGINT, force_exit_handler)


def force_exit_handler(signum, frame):
    """Force exit on second signal"""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        print(f"\nReceived second signal {signum}, forcing immediate exit...")
    
    # Try to cleanup quickly without barriers
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass
    
    os._exit(1)  # Force exit without cleanup


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
    print("  Memory snapshot recording enabled")
    torch.cuda.memory._record_memory_history(
        max_entries=100000,  # Keep last 100k alloc/free events
    )


def dump_memory_snapshot(step, prefix="memory_snapshot"):
    """Dump memory snapshot to file"""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    try:
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshots/{prefix}_rank_{rank}_step_{step}_{timestamp}.pkl"
        
        # print(f"  Saving memory snapshot: {filename}")
        torch.cuda.memory._dump_snapshot(filename)
        
    except Exception as e:
        print(f"  Failed to save memory snapshot: {e}")


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
    
    # print("\nFSDP Memory Breakdown:")
    # print("-" * 40)
    
    # Count parameters by type
    total_params = 0
    lora_params = 0
    base_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        if isinstance(param, DTensor):
            # For DTensor, get the full size
            param_count = param.size().numel()
        
        total_params += param_count
        
        if 'lora' in name.lower():
            lora_params += param_count
        else:
            base_params += param_count
    
    # print(f"Total parameters: {total_params:,}")
    # print(f"LoRA parameters: {lora_params:,} ({lora_params/total_params*100:.2f}%)")
    # print(f"Base parameters: {base_params:,} ({base_params/total_params*100:.2f}%)")
    
    # Memory estimates (bfloat16 = 2 bytes per param)
    total_param_memory = total_params * 2 / 1024**3
    lora_param_memory = lora_params * 2 / 1024**3
    
    # print(f"\nParameter Memory (bfloat16):")
    # print(f"Total: {total_param_memory:.2f} GB")
    # print(f"LoRA: {lora_param_memory:.2f} GB")
    # print(f"Per GPU (sharded): {total_param_memory / dist.get_world_size():.2f} GB")


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
        # print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
        pass


def validate_memory_before_training(model, args):
    """Validate memory requirements before starting training"""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if rank == 0:
        # print("\n" + "="*50)
        # print("MEMORY VALIDATION")
        # print("="*50)
        
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
        
        # print(f"\nMemory Analysis:")
        # print(f"  Available memory per GPU: {available_memory:.1f} GB")
        # print(f"  Estimated memory per GPU: {memory_per_gpu:.1f} GB")
        # print(f"  Memory headroom: {memory_headroom:.1f} GB")
        
        # Provide specific recommendations based on memory analysis
        # print(f"\nRecommendations:")
        if memory_headroom < 2.0:
            # print(f"     WARNING: Less than 2GB headroom available!")
            if args.batch_size > 1:
                suggested_batch = max(1, args.batch_size // 2)
                # print(f"  • Consider reducing batch size from {args.batch_size} to {suggested_batch}")
            if args.max_length > 1024:
                suggested_length = min(1024, args.max_length // 2)
                # print(f"  • Consider reducing sequence length from {args.max_length} to {suggested_length}")
            if args.gradient_accumulation_steps < 16:
                # print(f"  • Consider increasing gradient accumulation steps from {args.gradient_accumulation_steps} to {args.gradient_accumulation_steps * 2}")
                pass
        else:
            # print(f"  ✓ Memory headroom looks good ({memory_headroom:.1f} GB)")
            if memory_headroom > 8.0:
                # print(f"  • You could potentially increase batch size or sequence length")
                pass
        
        # print("="*50)
        # print()
        pass


def apply_fsdp2_to_model(model, args, world_size):
    """Apply FSDP2 to model using new APIs"""
    
    try:
        # Create device mesh
        device_mesh = init_device_mesh("cuda", (world_size,))
        
        # Configure mixed precision
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16
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
    log_every = 10

    try:
        model.train()
        total_loss = 0
        accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        
        optimizer.zero_grad()

        # Reset peak memory stats at start of epoch
        if rank == 0:
            torch.cuda.reset_peak_memory_stats()
        
        checkpoint_saved = False
        
        for step, batch in enumerate(dataloader):
            # Check for graceful shutdown at the start of each step
            if graceful_shutdown:
                if rank == 0:
                    print("Graceful shutdown requested, saving checkpoint...")
                
                if not checkpoint_saved:
                    try:
                        metadata = {
                            "epoch": epoch,
                            "step": step,
                            "partial_epoch": True,
                            "graceful_shutdown": True,
                            "model_name": args.model_name,
                            "lora_rank": args.lora_rank,
                            "lora_alpha": args.lora_alpha,
                            "target_modules": args.target_modules,
                        }
                        # Use minimal retries for graceful shutdown to avoid hanging
                        success = safe_checkpoint_save(checkpointer, model, optimizer, metadata, max_retries=1)
                        checkpoint_saved = True
                        
                        if rank == 0:
                            if success:
                                print("✓ Graceful shutdown checkpoint saved successfully")
                            else:
                                print("✗ Failed to save graceful shutdown checkpoint")
                    except Exception as e:
                        if rank == 0:
                            print(f"Warning: Failed to save graceful shutdown checkpoint: {e}")
                        checkpoint_saved = True  # Don't retry indefinitely
                
                return total_loss / max(1, step)
            
            if rank == 0:
                # print(f"[Rank {rank}] About to fetch batch {step}")
                pass
            sys.stdout.flush()
            
            # Only move tensor items to CUDA
            batch = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()
            }
            
            try:
                if step == 0 and rank == 0:
                    # print("First batch:", batch)
                    pass
                
                # Forward pass
                outputs = model(**batch)
                
                # if step == 0 and rank == 0:
                #     print("Outputs:", outputs)
                #     print("Logits:", outputs.logits)
                #     print("Any NaN in logits?", torch.isnan(outputs.logits).any())

                loss = outputs.loss / accumulation_steps

                # print(f"[Rank {rank}] Step {step} loss: {loss}, logits: {outputs.logits}")
                # print(f"[Rank {rank}] Any NaN in logits? {torch.isnan(outputs.logits).any()}")
                # print(f"[Rank {rank}] Any NaN in loss? {torch.isnan(loss)}")

                # NaN loss handling
                if torch.isnan(loss):
                    print(f"[Rank {rank}] NaN loss detected at step {step}, aborting!")
                    if dist.is_initialized():
                        dist.destroy_process_group()
                    sys.exit(1)
                
                # Backward pass
                loss.backward()
                
                # Memory monitoring
                if step == 5 or step % 20 == 0:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    # print(f"[Rank {rank}] Step {step}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, peak={peak:.2f}GB")
                    dump_memory_snapshot(step)
                    monitor_memory(step, log_every=1)

                # Update every accumulation_steps
                if (step + 1) % accumulation_steps == 0:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps
                
                # Emergency checkpoint save if memory is getting high
                if step > 0 and step % 50 == 0:
                    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    if peak_memory > gpu_memory * 0.9:  # 90% usage
                        if rank == 0:
                            # print(f"High memory usage detected ({peak_memory:.1f}GB), saving emergency checkpoint...")
                            pass
                        
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

                if (step + 1) % log_every == 0 and rank == 0:
                    avg_loss_so_far = total_loss / (step + 1)
                    print(f"[Rank {rank}] Step {step + 1}: Average Loss so far: {avg_loss_so_far:.4f}")

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
            # print(f"\nEpoch {epoch} Peak Memory: {peak_memory:.2f} GB")
        
        return total_loss / len(dataloader)
    except Exception as e:
        print(f"Error in train_epoch_with_accumulation: {e}")
        print(f"Exception on rank {dist.get_rank()}:\n{traceback.format_exc()}")
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)


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
            # print(f"Step {step} - Rank 0 Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, peak: {max_allocated:.2f}GB")
            
            # Check for potential OOM risk
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if reserved > gpu_memory * 0.85:
                # print("Warning: High memory usage detected!")
                pass
                
    except Exception as e:
        # Don't fail training if memory monitoring fails
        pass


def cleanup():
    """Clean up the distributed environment with timeout protection."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    
    try:
        if dist.is_initialized():
            # Skip synchronization during graceful shutdown
            if graceful_shutdown:
                if rank == 0:
                    print("Skipping cleanup synchronization due to graceful shutdown")
            else:
                # Use timeout for cleanup barrier
                if rank == 0:
                    print("Synchronizing processes before cleanup...")
                
                sync_complete = threading.Event()
                
                def sync_thread():
                    try:
                        dist.barrier()
                        sync_complete.set()
                    except:
                        sync_complete.set()
                
                thread = threading.Thread(target=sync_thread)
                thread.daemon = True
                thread.start()
                
                if not sync_complete.wait(timeout=10):
                    if rank == 0:
                        print("Cleanup synchronization timeout, proceeding...")
            
            # Destroy process group
            if rank == 0:
                print("Destroying process group...")
            
            try:
                dist.destroy_process_group()
                if rank == 0:
                    print("Process group destroyed successfully")
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Failed to destroy process group: {e}")
                    
    except Exception as e:
        if rank == 0:
            print(f"Warning: Failed during cleanup: {e}")
    
    # Clear CUDA cache
    try:
        torch.cuda.empty_cache()
        if rank == 0:
            print("CUDA cache cleared")
    except Exception as e:
        if rank == 0:
            print(f"Warning: Failed to clear CUDA cache: {e}")


def get_tokenizer(model_name):
    """Get or create tokenizer instance."""
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def cast_model_to_dtype(model, dtype):
    """Cast all model parameters and buffers to specified dtype in-place."""
    for param in model.parameters():
        if param.dtype != dtype:
            param.data = param.data.to(dtype)
    for buffer in model.buffers():
        if buffer.dtype != dtype:
            buffer.data = buffer.data.to(dtype)


def get_data_collator(tokenizer=None):
    """Collate function for next-token code completion (input, label pairs)."""
    def collate_fn(batch):
        # Each batch item: (input_ids, labels)
        input_ids = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return {
            'input_ids': input_ids,
            'labels': labels
        }
    return collate_fn


def inspect_dtensor_parameters(model, rank):
    """Inspect DTensor parameters after FSDP2 wrapping"""
    if rank == 0:
        print("\n" + "="*50)
        print("FSDP2 DTensor Parameter Inspection")
        print("="*50)
        
        for name, param in model.named_parameters():
            if isinstance(param, DTensor):
                print(f"\nParameter: {name}")
                print(f"  Type: DTensor")
                print(f"  Global shape: {param.shape}")
                print(f"  Local shape: {param.to_local().shape}")
                print(f"  Placements: {param.placements}")
                print(f"  Device mesh: {param.device_mesh}")
            else:
                print(f"\nParameter: {name}")
                print(f"  Type: Regular Tensor")
                print(f"  Shape: {param.shape}")
        
        print("="*50)


def main(args):
    # Setup distributed training
    rank, world_size = setup()
    setup_seed(42 + rank)
    
    # Initialize tokenizer after fork   
    tokenizer = get_tokenizer(args.model_name)
    
    # Initialize model on meta device for FSDP2
    if rank == 0:
        print(f"Loading model: {args.model_name}")

    config = AutoConfig.from_pretrained(args.model_name)
    with torch.device("meta"):
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    # Materialize and load weights into base model
    base_model.to_empty(device="cuda")
    state_dict = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu"
    ).state_dict()
    missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        # print("Base model missing keys:", missing)
        # print("Base model unexpected keys:", unexpected)
        pass
    for name, param in base_model.named_parameters():
        if torch.isnan(param.data).any() or torch.isinf(param.data).any():
            print(f"Parameter {name} contains NaN or inf!")

    # Apply LoRA to the base model (now with weights loaded)
    lora_wrapper = LoRAModel(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        base_model=base_model,
    )
    model = lora_wrapper.model

    cast_model_to_dtype(model, torch.bfloat16)
  
    # Apply FSDP2
    model = apply_fsdp2_to_model(model, args, world_size)

    for name, param in model.named_parameters():
        # print(f"[Rank {rank}] param: {name}, shape: {param.shape}, device: {param.device}, dtype: {param.dtype}, is_dtensor: {hasattr(param, 'to_local')}")
        pass
    sys.stdout.flush()
    # print(f"[Rank {rank}] Model param count: {sum(p.numel() for p in model.parameters())}")


    # Inspect DTensor parameters
    # inspect_dtensor_parameters(model, rank)

    # Create dataset
    dataset = CodeTextDataset(args.data_path, tokenizer)

    if rank == 0:
        # print(f"Dataset size: {len(dataset)}")
        # print(f"Data path: {args.data_path}")
        pass

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        sampler=sampler,
        collate_fn=get_data_collator(tokenizer),
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    if rank == 0:
        # print(f"Dataloader size: {len(dataloader)}")
        # print(f"Batch size per GPU: {args.batch_size}")
        # print(f"Total batch size: {args.batch_size * world_size}")
        # print(f"Dataset size: {len(dataset)}")
        if len(dataloader) == 0:
            raise ValueError(
                f"No batches available. Dataset size ({len(dataset)}) is too small for "
                f"batch size {args.batch_size} with {world_size} GPUs. "
                "Please add more examples or reduce batch size."
            )
    
    # Initialize checkpointer
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
            # print(f"Loading checkpoint from: {checkpointer.last_training_time}")
            pass
        success = checkpointer.load(model, optimizer)
        if not success:
            if rank == 0:
                # print("Failed to load checkpoint")
                pass
    elif rank == 0 and not args.no_resume:
        # print("No checkpoint found, starting fresh training")
        pass
    elif rank == 0 and args.no_resume:
        # print("Skipping checkpoint loading as --no-resume was specified")
        pass
    
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
        
    print(f"[Rank {rank}] After barrier, before train step")
    sys.stdout.flush()

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
        # Handle cleanup differently for graceful vs abrupt shutdown
        if graceful_shutdown:
            if rank == 0:
                print("Performing graceful cleanup...")
            # Skip barriers during graceful shutdown to avoid hanging
            try:
                # Save one final checkpoint if possible
                if 'checkpointer' in locals() and 'model' in locals() and 'optimizer' in locals():
                    metadata = {
                        "epoch": epoch + 1 if 'epoch' in locals() else 0,
                        "graceful_shutdown": True,
                        "model_name": args.model_name,
                        "lora_rank": args.lora_rank,
                        "lora_alpha": args.lora_alpha,
                        "target_modules": args.target_modules,
                    }
                    # Use only 1 retry for graceful shutdown
                    safe_checkpoint_save(checkpointer, model, optimizer, metadata, max_retries=1)
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Failed to save final checkpoint during graceful shutdown: {e}")
        else:
            # Normal cleanup with barrier synchronization
            if dist.is_initialized():
                try:
                    if rank == 0:
                        print(f"Waiting for all ranks to reach cleanup on rank {rank}")
                    
                    # Use timeout on cleanup barrier
                    import threading
                    barrier_complete = threading.Event()
                    
                    def barrier_thread():
                        try:
                            dist.barrier()
                            barrier_complete.set()
                        except:
                            barrier_complete.set()
                    
                    thread = threading.Thread(target=barrier_thread)
                    thread.daemon = True
                    thread.start()
                    
                    if barrier_complete.wait(timeout=30):
                        print(f"Barrier passed on rank {rank}")
                        if rank == 0:
                            print(f"All ranks ready for cleanup on rank {rank}")
                    else:
                        if rank == 0:
                            print("Cleanup barrier timeout, proceeding with individual cleanup")
                            
                except Exception as e:
                    if rank == 0:
                        print(f"Cleanup barrier failed: {e}")
        
        # Always call cleanup, but make it robust
        try:
            cleanup()
        except Exception as e:
            if rank == 0:
                print(f"Warning: Cleanup failed: {e}")
    
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
        default="build123d_all_examples.txt",
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
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from previous checkpoint")
    
    args = parser.parse_args()
    main(args)