# z_checkpoint.py

import os
import torch
import torch.distributed as dist
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from torch.distributed.checkpoint.filesystem import (
    FileSystemReader,
    FileSystemWriter,
)

from torch.distributed.checkpoint.state_dict_loader import load as load_state_dict
from torch.distributed.checkpoint.state_dict_saver import save as save_state_dict


class FSDP2Checkpointer:
    """FSDP2-compatible checkpointer for LoRA training"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
    
    def save(self, model, optimizer, metadata: Dict[str, Any]):
        """Save model, optimizer, and metadata using FSDP2 APIs"""
        
        # Create timestamped checkpoint directory
        if self.rank == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestamp}"
            checkpoint_path.mkdir(exist_ok=True)
            print(f"Saving checkpoint to: {checkpoint_path}")
        
        # Ensure all ranks wait for directory creation
        dist.barrier()
        
        # Broadcast checkpoint path to all ranks
        if self.rank == 0:
            latest_dir = max(self.checkpoint_dir.glob("checkpoint_*"))
            checkpoint_path_str = str(latest_dir)
        else:
            checkpoint_path_str = ""
        
        # Broadcast the path
        checkpoint_path_list = [checkpoint_path_str]
        dist.broadcast_object_list(checkpoint_path_list, src=0)
        checkpoint_path = Path(checkpoint_path_list[0])
        
        try:
            # Save model state dict using distributed checkpoint
            model_state_dict = {"model": model.state_dict()}
            save_state_dict(
                state_dict=model_state_dict,
                storage_writer=FileSystemWriter(checkpoint_path / "model"),
            )
            
            # Save optimizer state dict
            optim_state_dict = {"optimizer": optimizer.state_dict()}
            save_state_dict(
                state_dict=optim_state_dict,
                storage_writer=FileSystemWriter(checkpoint_path / "optimizer"),
            )
            
            # Save metadata (only on rank 0)
            if self.rank == 0:
                torch.save(metadata, checkpoint_path / "metadata.pt")
                print(f"✓ Checkpoint saved successfully")
                
        except Exception as e:
            if self.rank == 0:
                print(f"✗ Error saving checkpoint: {e}")
            raise e
    
    def load(self, model, optimizer) -> bool:
        """Load the latest checkpoint"""
        latest_checkpoint = self.get_latest_checkpoint()
        if not latest_checkpoint:
            if self.rank == 0:
                print("No checkpoint found to load")
            return False
        
        if self.rank == 0:
            print(f"Loading checkpoint from: {latest_checkpoint.name}")
        
        try:
            # Check if this is a new format checkpoint (with model/optimizer subdirs)
            if (latest_checkpoint / "model").exists():
                # Load model state dict using distributed checkpoint
                model_state_dict = {"model": model.state_dict()}
                load_state_dict(
                    state_dict=model_state_dict,
                    storage_reader=FileSystemReader(latest_checkpoint / "model"),
                )
                model.load_state_dict(model_state_dict["model"])
                
                # Load optimizer state dict
                optim_state_dict = {"optimizer": optimizer.state_dict()}
                load_state_dict(
                    state_dict=optim_state_dict,
                    storage_reader=FileSystemReader(latest_checkpoint / "optimizer"),
                )
                optimizer.load_state_dict(optim_state_dict["optimizer"])
            else:
                # Old format - load directly from checkpoint files
                if self.rank == 0:
                    print("Loading from old checkpoint format...")
                
                # Load model state dict
                model_path = latest_checkpoint / "lora_weights.pt"
                if model_path.exists():
                    model_state_dict = torch.load(model_path, map_location="cpu")
                    model.load_state_dict(model_state_dict, strict=False)
                
                # Load optimizer state dict
                optim_path = latest_checkpoint / "optimizer.pt"
                if optim_path.exists():
                    optim_state_dict = torch.load(optim_path, map_location="cpu")
                    optimizer.load_state_dict(optim_state_dict)
            
            if self.rank == 0:
                print("✓ Checkpoint loaded successfully")
            
            return True
            
        except Exception as e:
            if self.rank == 0:
                print(f"✗ Error loading checkpoint: {e}")
            return False
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the latest checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda x: x.stat().st_mtime)
    
    def is_empty(self) -> bool:
        """Check if there are any checkpoints"""
        return self.get_latest_checkpoint() is None
    
    @property
    def last_training_time(self) -> Optional[str]:
        """Get the timestamp of the last training session"""
        latest = self.get_latest_checkpoint()
        if latest:
            return latest.name.replace("checkpoint_", "")
        return None
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """Clean up old checkpoints, keeping only the last N"""
        if self.rank != 0:
            return
            
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*"),
            key=lambda x: x.stat().st_mtime
        )
        
        if len(checkpoints) <= keep_last_n:
            return
        
        to_delete = checkpoints[:-keep_last_n]
        for checkpoint_dir in to_delete:
            try:
                import shutil
                shutil.rmtree(checkpoint_dir)
                print(f"Cleaned up old checkpoint: {checkpoint_dir.name}")
            except Exception as e:
                print(f"Warning: Could not delete {checkpoint_dir}: {e}")
    
    def save_lora_weights_only(self, model, filename: str = "lora_weights.pt"):
        """Save only the LoRA weights (for inference)"""
        if self.rank != 0:
            return
            
        try:
            # Extract only LoRA parameters
            lora_state_dict = {}
            for name, param in model.named_parameters():
                if any(lora_key in name for lora_key in ['lora_A', 'lora_B', 'lora_embedding']):
                    lora_state_dict[name] = param.cpu().clone()
            
            save_path = self.checkpoint_dir / filename
            torch.save(lora_state_dict, save_path)
            print(f"✓ LoRA weights saved to: {save_path}")
            
        except Exception as e:
            print(f"✗ Error saving LoRA weights: {e}")
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about available checkpoints"""
        if self.rank != 0:
            return {}
            
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        info = {
            "total_checkpoints": len(checkpoints),
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoints": []
        }
        
        for checkpoint_path in checkpoints:
            try:
                metadata_path = checkpoint_path / "metadata.pt"
                if metadata_path.exists():
                    metadata = torch.load(metadata_path, map_location='cpu')
                else:
                    metadata = {}
                
                checkpoint_info = {
                    "timestamp": checkpoint_path.name.replace("checkpoint_", ""),
                    "path": str(checkpoint_path),
                    "size_mb": sum(
                        f.stat().st_size for f in checkpoint_path.rglob('*') 
                        if f.is_file()
                    ) / (1024 * 1024),
                    "metadata": metadata
                }
                info["checkpoints"].append(checkpoint_info)
                
            except Exception as e:
                print(f"Warning: Could not read checkpoint info for {checkpoint_path}: {e}")
        
        return info


# Utility functions for checkpoint management
def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
    return {}

def clear_memory_cache():
    """Clear CUDA memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()