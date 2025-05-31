# checkpoint.py

import os
import shutil
import torch
import torch.distributed as dist
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import time

from torch.distributed.checkpoint.filesystem import (
    FileSystemReader,
    FileSystemWriter,
)

from torch.distributed.checkpoint.state_dict_loader import load as load_state_dict
from torch.distributed.checkpoint.state_dict_saver import save as save_state_dict


class FSDP2Checkpointer:
    """FSDP2-compatible checkpointer with robust timeout handling"""
    
    def __init__(self, checkpoint_dir: str, save_peft_format: bool = True, checkpoint_timeout: int = 300):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.save_peft_format = save_peft_format
        self.checkpoint_timeout = checkpoint_timeout  # 5 minutes default
    
    def save(self, model, optimizer, metadata: Dict[str, Any]):
        """Save model, optimizer, and metadata using FSDP2 APIs with timeout protection"""
        
        start_time = time.time()
        
        # Create timestamped checkpoint directory
        if self.rank == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestamp}"
            checkpoint_path.mkdir(exist_ok=True)
            print(f"Saving checkpoint to: {checkpoint_path}")
        
        # Ensure all ranks wait for directory creation with timeout
        try:
            dist.barrier()
        except Exception as e:
            if self.rank == 0:
                print(f"Warning: Barrier failed during directory creation: {e}")
            # Continue anyway
        
        # Broadcast checkpoint path to all ranks
        if self.rank == 0:
            latest_dir = max(self.checkpoint_dir.glob("checkpoint_*"))
            checkpoint_path_str = str(latest_dir)
        else:
            checkpoint_path_str = ""
        
        # Broadcast the path with timeout protection
        try:
            checkpoint_path_list = [checkpoint_path_str]
            dist.broadcast_object_list(checkpoint_path_list, src=0)
            checkpoint_path = Path(checkpoint_path_list[0])
        except Exception as e:
            if self.rank == 0:
                print(f"Warning: Broadcast failed, using fallback path: {e}")
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            checkpoint_path.mkdir(exist_ok=True)
        
        # Try saving with timeout protection
        try:
            success = self._save_with_timeout(model, optimizer, metadata, checkpoint_path, start_time)
            if not success:
                # Fallback to simple save
                if self.rank == 0:
                    print("Falling back to simple checkpoint save...")
                self._save_simple_fallback(model, optimizer, metadata, checkpoint_path)
                
        except Exception as e:
            if self.rank == 0:
                print(f"Error during checkpoint save: {e}")
                # Try fallback save
                try:
                    self._save_simple_fallback(model, optimizer, metadata, checkpoint_path)
                except Exception as fallback_error:
                    print(f"Fallback save also failed: {fallback_error}")
                    raise e
    
    def _save_with_timeout(self, model, optimizer, metadata, checkpoint_path, start_time):
        """Save with timeout protection"""
        
        try:
            # Check if we're already close to timeout
            if time.time() - start_time > self.checkpoint_timeout * 0.8:
                if self.rank == 0:
                    print("Approaching timeout, skipping distributed checkpoint save")
                return False
            
            # Clear any CUDA cache before saving
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Save model state dict using distributed checkpoint
            if self.rank == 0:
                print("Saving model state dict...")
            
            model_start = time.time()
            model_state_dict = {"model": model.state_dict()}
            
            # Check timeout before model save
            if time.time() - start_time > self.checkpoint_timeout * 0.6:
                if self.rank == 0:
                    print("Timeout approaching, aborting model save")
                return False
            
            save_state_dict(
                state_dict=model_state_dict,
                storage_writer=FileSystemWriter(checkpoint_path / "model"),
            )
            
            if self.rank == 0:
                print(f"Model saved in {time.time() - model_start:.2f}s")
            
            # Check timeout before optimizer save
            if time.time() - start_time > self.checkpoint_timeout * 0.8:
                if self.rank == 0:
                    print("Timeout approaching, skipping optimizer save")
                # Still save metadata and PEFT
                self._save_metadata_and_peft(model, checkpoint_path, metadata)
                return True
            
            # Save optimizer state dict
            if self.rank == 0:
                print("Saving optimizer state dict...")
            
            optim_start = time.time()
            optim_state_dict = {"optimizer": optimizer.state_dict()}
            save_state_dict(
                state_dict=optim_state_dict,
                storage_writer=FileSystemWriter(checkpoint_path / "optimizer"),
            )
            
            if self.rank == 0:
                print(f"Optimizer saved in {time.time() - optim_start:.2f}s")
            
            # Save metadata and PEFT (only on rank 0)
            self._save_metadata_and_peft(model, checkpoint_path, metadata)
            
            if self.rank == 0:
                total_time = time.time() - start_time
                print(f"Checkpoint saved successfully in {total_time:.2f}s")
            
            return True
            
        except Exception as e:
            if self.rank == 0:
                print(f"Distributed checkpoint save failed: {e}")
            return False
    
    def _save_metadata_and_peft(self, model, checkpoint_path, metadata):
        """Save metadata and PEFT format (rank 0 only)"""
        if self.rank != 0:
            return
        
        try:
            # Save metadata
            torch.save(metadata, checkpoint_path / "metadata.pt")
            
            # Save PEFT format if enabled
            if self.save_peft_format:
                self._save_peft_format_safe(model, checkpoint_path, metadata)
                
        except Exception as e:
            print(f"Warning: Failed to save metadata/PEFT: {e}")
    
    def _save_simple_fallback(self, model, optimizer, metadata, checkpoint_path):
        """Simple fallback save method that avoids distributed checkpoints"""
        
        if self.rank == 0:
            print("Using simple fallback checkpoint save...")
        
        try:
            # Clear cache first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Extract LoRA weights manually (safer for FSDP2)
            lora_weights = self._extract_lora_weights_safe(model)
            
            if self.rank == 0:
                # Save LoRA weights only
                if lora_weights:
                    torch.save(lora_weights, checkpoint_path / "lora_weights.pt")
                    print(f"Saved {len(lora_weights)} LoRA parameters")
                else:
                    print("Warning: No LoRA weights found")
                
                # Save metadata
                torch.save(metadata, checkpoint_path / "metadata.pt")
                
                # Save optimizer state (best effort)
                try:
                    optim_state = optimizer.state_dict()
                    # Only save if state dict is reasonable size
                    state_size = sum(p.numel() for group in optim_state['state'].values() 
                                   for p in group.values() if isinstance(p, torch.Tensor))
                    if state_size < 1e9:  # Less than 1B parameters in optimizer state
                        torch.save(optim_state, checkpoint_path / "optimizer.pt")
                        print("Optimizer state saved")
                    else:
                        print("Optimizer state too large, skipping")
                except Exception as e:
                    print(f"Warning: Could not save optimizer state: {e}")
                
                # Create PEFT format
                if self.save_peft_format and lora_weights:
                    self._save_peft_format_from_weights(lora_weights, checkpoint_path, metadata)
                
                print("Fallback checkpoint save completed")
            
            # Synchronize all ranks
            if dist.is_initialized():
                try:
                    dist.barrier()
                except:
                    pass  # Don't fail if barrier fails
                    
        except Exception as e:
            if self.rank == 0:
                print(f"Fallback save failed: {e}")
            raise e
    
    def _extract_lora_weights_safe(self, model):
        """Safely extract LoRA weights without triggering collective operations"""
        lora_weights = {}
        
        try:
            # Use named_parameters which should be safer than state_dict() for FSDP2
            for name, param in model.named_parameters():
                if any(lora_key in name for lora_key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
                    # Clean up FSDP prefixes
                    clean_name = name
                    for prefix in ['_fsdp_wrapped_module.', 'module.', '_orig_mod.']:
                        clean_name = clean_name.replace(prefix, '')
                    
                    try:
                        # Safely extract parameter data
                        if hasattr(param, 'data'):
                            param_data = param.data.detach()
                            if param_data.is_cuda:
                                param_data = param_data.cpu()
                            lora_weights[clean_name] = param_data.clone()
                    except Exception as e:
                        print(f"Warning: Could not extract parameter {name}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error during LoRA weight extraction: {e}")
            
        return lora_weights
    
    def _save_peft_format_safe(self, model, checkpoint_path, metadata):
        """Save PEFT format with better error handling"""
        
        try:
            peft_dir = checkpoint_path / "peft"
            peft_dir.mkdir(exist_ok=True)
            
            # Extract LoRA weights safely
            lora_weights = self._extract_lora_weights_safe(model)
            
            if lora_weights:
                self._save_peft_format_from_weights(lora_weights, checkpoint_path, metadata)
            else:
                print("Warning: No LoRA weights found for PEFT checkpoint")
                
        except Exception as e:
            print(f"Warning: PEFT save failed: {e}")
    
    def _save_peft_format_from_weights(self, lora_weights, checkpoint_path, metadata):
        """Save PEFT format from extracted weights"""
        
        peft_dir = checkpoint_path / "peft"
        peft_dir.mkdir(exist_ok=True)
        
        # Create adapter config
        adapter_config = {
            "auto_mapping": None,
            "base_model_name_or_path": metadata.get('model_name', 'bigcode/starcoder2-7b'),
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "lora_alpha": float(metadata.get('lora_alpha', 16)),
            "lora_dropout": 0.0,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": int(metadata.get('lora_rank', 16)),
            "revision": None,
            "target_modules": metadata.get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"]),
            "task_type": "CAUSAL_LM"
        }
        
        # Save config
        import json
        with open(peft_dir / "adapter_config.json", 'w') as f:
            json.dump(adapter_config, f, indent=2)
        
        # Save weights
        try:
            from safetensors.torch import save_file
            save_file(lora_weights, peft_dir / "adapter_model.safetensors")
        except ImportError:
            torch.save(lora_weights, peft_dir / "adapter_model.bin")
        
        print(f"PEFT checkpoint saved to: {peft_dir}")
    
    def load(self, model, optimizer) -> bool:
        """Load the latest checkpoint with improved error handling"""
        latest_checkpoint = self.get_latest_checkpoint()
        if not latest_checkpoint:
            if self.rank == 0:
                print("No checkpoint found to load")
            return False
        
        if self.rank == 0:
            print(f"Loading checkpoint from: {latest_checkpoint.name}")
        
        try:
            # Try new format first (with model/optimizer subdirs)
            if (latest_checkpoint / "model").exists():
                return self._load_distributed_checkpoint(model, optimizer, latest_checkpoint)
            else:
                return self._load_simple_checkpoint(model, optimizer, latest_checkpoint)
                
        except Exception as e:
            if self.rank == 0:
                print(f"Error loading checkpoint: {e}")
            return False
    
    def _load_distributed_checkpoint(self, model, optimizer, checkpoint_path):
        """Load distributed checkpoint format"""
        try:
            # Load model state dict
            model_state_dict = {"model": model.state_dict()}
            load_state_dict(
                state_dict=model_state_dict,
                storage_reader=FileSystemReader(checkpoint_path / "model"),
            )
            model.load_state_dict(model_state_dict["model"])
            
            # Load optimizer state dict if it exists
            if (checkpoint_path / "optimizer").exists():
                optim_state_dict = {"optimizer": optimizer.state_dict()}
                load_state_dict(
                    state_dict=optim_state_dict,
                    storage_reader=FileSystemReader(checkpoint_path / "optimizer"),
                )
                optimizer.load_state_dict(optim_state_dict["optimizer"])
            
            if self.rank == 0:
                print("Distributed checkpoint loaded successfully")
            return True
            
        except Exception as e:
            if self.rank == 0:
                print(f"Failed to load distributed checkpoint: {e}")
            return False
    
    def _load_simple_checkpoint(self, model, optimizer, checkpoint_path):
        """Load simple checkpoint format"""
        try:
            # Load LoRA weights
            lora_path = checkpoint_path / "lora_weights.pt"
            if lora_path.exists():
                lora_state_dict = torch.load(lora_path, map_location="cpu")
                model.load_state_dict(lora_state_dict, strict=False)
                if self.rank == 0:
                    print(f"Loaded {len(lora_state_dict)} LoRA parameters")
            
            # Load optimizer state
            optim_path = checkpoint_path / "optimizer.pt"
            if optim_path.exists():
                optim_state_dict = torch.load(optim_path, map_location="cpu")
                optimizer.load_state_dict(optim_state_dict)
                if self.rank == 0:
                    print("Optimizer state loaded")
            
            if self.rank == 0:
                print("Simple checkpoint loaded successfully")
            return True
            
        except Exception as e:
            if self.rank == 0:
                print(f"Failed to load simple checkpoint: {e}")
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
                shutil.rmtree(checkpoint_dir)
                print(f"Cleaned up old checkpoint: {checkpoint_dir.name}")
            except Exception as e:
                print(f"Warning: Could not delete {checkpoint_dir}: {e}")


# Utility function for safe checkpoint saving in training loop
def safe_checkpoint_save(checkpointer, model, optimizer, metadata, max_retries=2):
    """Safely save checkpoint with retries and fallbacks"""
    
    rank = int(os.environ.get("LOCAL_RANK", 0))
    
    for attempt in range(max_retries):
        try:
            if rank == 0:
                print(f"Checkpoint save attempt {attempt + 1}/{max_retries}")
            
            # Clear cache before saving
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Add timeout to metadata
            metadata['save_attempt'] = attempt + 1
            metadata['timestamp'] = datetime.now().isoformat()
            
            checkpointer.save(model, optimizer, metadata)
            
            if rank == 0:
                print("Checkpoint saved successfully")
            return True
            
        except Exception as e:
            if rank == 0:
                print(f"Checkpoint save attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                if rank == 0:
                    print(f"Retrying in 10 seconds...")
                time.sleep(10)
                
                # Clear caches and try again
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Try to reset any hanging distributed state
                try:
                    if dist.is_initialized():
                        dist.barrier()
                except:
                    pass
            else:
                if rank == 0:
                    print("All checkpoint save attempts failed")
                return False
    
    return False