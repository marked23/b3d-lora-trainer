# checkpoint.py

import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.checkpoint.state_dict_saver import save
from torch.distributed.checkpoint.state_dict_loader import load
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.tensor import DTensor
from datetime import datetime
import json
import shutil
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path


class FSDP2Checkpointer:
    """FSDP2-compatible checkpointer using DTensor state dicts"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_peft_format: bool = True,
        checkpoint_timeout: int = 300,
        max_retries: int = 3,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_peft_format = save_peft_format
        self.checkpoint_timeout = checkpoint_timeout
        self.max_retries = max_retries
        self.last_training_time = None
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the latest checkpoint
        self._find_latest_checkpoint()
    
    def _find_latest_checkpoint(self):
        """Find the most recent checkpoint directory"""
        if not self.checkpoint_dir.exists():
            return
        
        checkpoint_dirs = [
            d for d in self.checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint_")
        ]
        
        if checkpoint_dirs:
            # Filter out incomplete checkpoints and sort by creation time
            valid_checkpoints = [
                d for d in checkpoint_dirs 
                if verify_checkpoint_integrity(d)
            ]
            
            if valid_checkpoints:
                latest_dir = max(valid_checkpoints, key=lambda x: x.stat().st_mtime)
                self.last_training_time = latest_dir.name
            else:
                self.last_training_time = None
    
    def is_empty(self) -> bool:
        """Check if there are any checkpoints"""
        return self.last_training_time is None
    
    def _get_checkpoint_path(self, timestamp: Optional[str] = None) -> Path:
        """Get checkpoint path for given timestamp"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.checkpoint_dir / f"checkpoint_{timestamp}"
    
    def _save_metadata(self, checkpoint_path: Path, metadata: Dict[str, Any]):
        """Save metadata to checkpoint directory"""
        metadata_path = checkpoint_path / "metadata.json"
        
        # Add timestamp and rank info
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "fsdp_version": "2.0",
        })
        
        # Only rank 0 saves metadata
        if not dist.is_initialized() or dist.get_rank() == 0:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
    
    def _load_metadata(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load metadata from checkpoint directory"""
        metadata_path = checkpoint_path / "metadata.json"
        
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save FSDP2 checkpoint using distributed checkpoint API"""
        
        if metadata is None:
            metadata = {}
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Create checkpoint directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self._get_checkpoint_path(timestamp)
        model_path = checkpoint_path.joinpath("model")
        optimizer_path = checkpoint_path.joinpath("optimizer")
        
        try:
            # if rank == 0:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            model_path.mkdir(exist_ok=True)
            optimizer_path.mkdir(exist_ok=True)

            if dist.is_initialized():
                # Use safe barrier with timeout instead of regular barrier
                barrier_success = safe_barrier_with_timeout(timeout_seconds=60)
                if not barrier_success:
                    if rank == 0:
                        print("Warning: Barrier failed/timed out, proceeding with checkpoint save")
                    # Continue with save attempt even if barrier fails

            # Get model and optimizer state dicts using FSDP2 API
            with torch.no_grad():
                model_state_dict, optimizer_state_dict = get_state_dict(
                    model,
                    optimizer,
                    options=None,  # Use default options for FSDP2
                )
            
            # Save model state dict
            save(
                state_dict={"model": model_state_dict},
                storage_writer=FileSystemWriter(str(model_path)),
                planner=None,  # Use default planner
            )
            
            # Save optimizer state dict
            save(
                state_dict={"optimizer": optimizer_state_dict},
                storage_writer=FileSystemWriter(str(optimizer_path)),
                planner=None,
            )
            
            # Save PEFT format if requested
            if self.save_peft_format:
                self._save_peft_format(model, checkpoint_path, rank)
            
            # Save metadata
            self._save_metadata(checkpoint_path, metadata)
            
            # Update last training time
            self.last_training_time = f"checkpoint_{timestamp}"
            
            if rank == 0:
                print(f"✓ Checkpoint saved successfully to {checkpoint_path}")
            
            return True
            
        except Exception as e:
            if rank == 0:
                print(f"✗ Failed to save checkpoint: {e}")
                # Cleanup failed checkpoint
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path, ignore_errors=True)
            return False
    
    def _save_peft_format(
        self,
        model: torch.nn.Module,
        checkpoint_path: Path,
        rank: int,
    ):
        """Save LoRA weights in PEFT format for easy loading"""
        try:
            # Only rank 0 saves PEFT format
            if rank != 0:
                return
            
            peft_path = checkpoint_path / "peft_model"
            peft_path.mkdir(exist_ok=True)
            
            # Extract LoRA weights
            lora_state_dict = {}
            
            for name, param in model.named_parameters():
                if "lora" in name.lower():
                    # Convert DTensor to regular tensor if needed
                    if isinstance(param, DTensor):
                        # Gather the full tensor on rank 0
                        full_param = param.full_tensor() if hasattr(param, 'full_tensor') else param.to_local()
                    else:
                        full_param = param
                    
                    lora_state_dict[name] = full_param.cpu()
            
            # Save LoRA weights
            torch.save(lora_state_dict, peft_path.joinpath("adapter_model.bin"))
            
            # Save adapter config (basic version)
            adapter_config = {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "r": 16,  # Default, should be updated from metadata
                "lora_alpha": 16,
                "lora_dropout": 0.05,
            }
            
            with open(peft_path / "adapter_config.json", "w") as f:
                json.dump(adapter_config, f, indent=2)
            
            print(f"✓ PEFT format saved to {peft_path}")
            
        except Exception as e:
            print(f"Warning: Failed to save PEFT format: {e}")
    
    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_name: Optional[str] = None,
    ) -> bool:
        """Load FSDP2 checkpoint using distributed checkpoint API"""
        
        if checkpoint_name is None:
            if self.last_training_time is None:
                return False
            checkpoint_name = self.last_training_time
        
        checkpoint_path = self.checkpoint_dir.joinpath(checkpoint_name)
        
        if not checkpoint_path.exists():
            return False
        
        # Verify checkpoint integrity before attempting to load
        if not verify_checkpoint_integrity(checkpoint_path):
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(f"Checkpoint {checkpoint_path} is incomplete or corrupted, skipping...")
            return False
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        try:
            if rank == 0:
                print(f"Loading checkpoint from: {checkpoint_path}")
            
            # Load model state dict
            model_path = checkpoint_path.joinpath("model")
            model_state_dict = {"model": {}}
            
            load(
                state_dict=model_state_dict,
                storage_reader=FileSystemReader(str(model_path)),
            )
            
            # Load optimizer state dict
            optimizer_path = checkpoint_path.joinpath("optimizer")
            optimizer_state_dict = {"optimizer": {}}
            
            load(
                state_dict=optimizer_state_dict,
                storage_reader=FileSystemReader(str(optimizer_path)),
            )
            
            # Apply state dicts to model and optimizer
            set_state_dict(
                model,
                optimizer,
                model_state_dict=model_state_dict["model"],
                optim_state_dict=optimizer_state_dict["optimizer"],
            )
            
            # Load and return metadata
            metadata = self._load_metadata(checkpoint_path)
            
            if rank == 0:
                print(f"✓ Checkpoint loaded successfully")
                if metadata:
                    epoch = metadata.get("epoch", "unknown")
                    print(f"  Resumed from epoch: {epoch}")
            
            return True
            
        except Exception as e:
            if rank == 0:
                print(f"✗ Failed to load checkpoint: {e}")
            return False
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """Clean up old checkpoints, keeping only the most recent N"""
        
        if not dist.is_initialized() or dist.get_rank() != 0:
            return
        
        try:
            checkpoint_dirs = [
                d for d in self.checkpoint_dir.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint_")
            ]
            
            if len(checkpoint_dirs) <= keep_last_n:
                return
            
            # Sort by creation time, oldest first
            checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest checkpoints
            for old_checkpoint in checkpoint_dirs[:-keep_last_n]:
                print(f"Removing old checkpoint: {old_checkpoint.name}")
                shutil.rmtree(old_checkpoint)
            
        except Exception as e:
            print(f"Warning: Failed to cleanup old checkpoints: {e}")


def safe_checkpoint_save(
    checkpointer: FSDP2Checkpointer,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Dict[str, Any],
    max_retries: int = 3,
) -> bool:
    """Safely save checkpoint with retries and error handling"""
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    for attempt in range(max_retries):
        try:
            # Add attempt info to metadata
            attempt_metadata = metadata.copy()
            attempt_metadata["save_attempt"] = attempt + 1
            
            # Try to save
            success = checkpointer.save(model, optimizer, attempt_metadata)
            
            if success:
                return True
            
            if rank == 0:
                print(f"Checkpoint save attempt {attempt + 1} failed, retrying...")
            
            # Wait before retry
            time.sleep(2 ** attempt)  # Exponential backoff
            
        except Exception as e:
            if rank == 0:
                print(f"Checkpoint save attempt {attempt + 1} failed with error: {e}")
            
            if attempt == max_retries - 1:
                if rank == 0:
                    print("All checkpoint save attempts failed")
                return False
            
            time.sleep(2 ** attempt)
    
    return False


def verify_checkpoint_integrity(checkpoint_path: Path) -> bool:
    """Verify that a checkpoint directory contains all required files"""
    
    required_paths = [
        checkpoint_path / "model",
        checkpoint_path / "optimizer",
        checkpoint_path / "metadata.json",
    ]
    
    for path in required_paths:
        if not path.exists():
            return False
    
    # Check if model and optimizer directories have content
    model_files = list((checkpoint_path / "model").iterdir())
    optimizer_files = list((checkpoint_path / "optimizer").iterdir())
    
    return len(model_files) > 0 and len(optimizer_files) > 0


def list_available_checkpoints(checkpoint_dir: str) -> list:
    """List all available checkpoints in the directory"""
    
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return []
    
    checkpoints = []
    for d in checkpoint_path.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint_"):
            if verify_checkpoint_integrity(d):
                metadata_path = d / "metadata.json"
                metadata = {}
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                checkpoints.append({
                    "name": d.name,
                    "path": str(d),
                    "timestamp": metadata.get("timestamp", "unknown"),
                    "epoch": metadata.get("epoch", "unknown"),
                    "size_mb": sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024),
                })
    
    # Sort by creation time, newest first
    checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return checkpoints


# Global flag for graceful shutdown awareness
def is_graceful_shutdown():
    """Check if graceful shutdown is in progress"""
    # This should match the variable in train.py
    import sys
    for frame_info in sys._current_frames().values():
        frame_globals = frame_info.f_globals
        if 'graceful_shutdown' in frame_globals:
            return frame_globals['graceful_shutdown']
    return False


def safe_barrier_with_timeout(timeout_seconds=30):
    """Execute barrier with timeout and graceful shutdown awareness"""
    import threading
    
    if not dist.is_initialized():
        return True
    
    rank = dist.get_rank()
    
    # Skip barriers during graceful shutdown
    if is_graceful_shutdown():
        if rank == 0:
            print("Skipping barrier due to graceful shutdown")
        return False
    
    try:
        if rank == 0:
            print(f"Rank {rank} waiting for barrier")
        
        barrier_complete = threading.Event()
        
        def barrier_thread():
            try:
                dist.barrier()
                barrier_complete.set()
            except Exception as e:
                if rank == 0:
                    print(f"Barrier failed: {e}")
                barrier_complete.set()
        
        thread = threading.Thread(target=barrier_thread)
        thread.daemon = True
        thread.start()
        
        if barrier_complete.wait(timeout=timeout_seconds):
            if rank == 0:
                print(f"Rank {rank} barrier passed")
            return True
        else:
            if rank == 0:
                print(f"Barrier timeout after {timeout_seconds} seconds")
            return False
            
    except Exception as e:
        if rank == 0:
            print(f"Barrier error: {e}")
        return False