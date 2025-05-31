# checkpoint.py

import os
import shutil
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
    """FSDP2-compatible checkpointer for LoRA training with PEFT format support"""
    
    def __init__(self, checkpoint_dir: str, save_peft_format: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.save_peft_format = save_peft_format
    
    def save(self, model, optimizer, metadata: Dict[str, Any]):
        """Save model, optimizer, and metadata using FSDP2 APIs with optional PEFT format"""
        
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
            # Save model state dict using distributed checkpoint (FSDP2 compatible)
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
                
                # Save PEFT format if enabled
                if self.save_peft_format:
                    self._save_peft_format_fsdp2(model, checkpoint_path, metadata)
                
                print(f"✓ Checkpoint saved successfully")
                
        except Exception as e:
            if self.rank == 0:
                print(f"✗ Error saving checkpoint: {e}")
            raise e
    
    def _save_peft_format_fsdp2(self, model, checkpoint_path: Path, metadata: Dict[str, Any]):
        """Save in PEFT-compatible format using FSDP2 APIs"""
        
        try:
            peft_dir = checkpoint_path / "peft"
            if self.rank == 0:
                peft_dir.mkdir(exist_ok=True)
            
            # Wait for directory creation
            dist.barrier()
            
            # FSDP2 approach: Get state dict and gather on rank 0
            if self.rank == 0:
                # Load the distributed checkpoint we just saved
                state_dict = {}
                load_state_dict(
                    state_dict=state_dict,
                    storage_reader=FileSystemReader(checkpoint_path / "model"),
                )
                
                # Extract the model state dict
                full_state_dict = state_dict.get("model", {})
                
                # Extract LoRA weights from the loaded state dict
                lora_weights = self._extract_lora_weights_from_state_dict(full_state_dict)
                
                if lora_weights:
                    # Create PEFT checkpoint structure
                    self._create_peft_checkpoint_structure(peft_dir, lora_weights, metadata)
                    print(f"✓ PEFT checkpoint saved to: {peft_dir}")
                else:
                    print("Warning: No LoRA weights found for PEFT checkpoint")
                    # Create minimal PEFT structure
                    self._create_peft_checkpoint_manual_safe(model, peft_dir, metadata)
                    
        except Exception as e:
            if self.rank == 0:
                print(f"Warning: Failed to save PEFT format: {e}")
                # Fallback to manual extraction
                self._save_peft_format_fallback(model, checkpoint_path, metadata)
    
    def _save_peft_format_fallback(self, model, checkpoint_path: Path, metadata: Dict[str, Any]):
        """Fallback PEFT save method that works with FSDP2"""
        
        if self.rank != 0:
            return
            
        try:
            peft_dir = checkpoint_path / "peft"
            peft_dir.mkdir(exist_ok=True)
            
            print("Using fallback PEFT save method...")
            
            # Try to extract from named parameters (FSDP2 compatible)
            lora_weights = self._extract_lora_from_named_parameters(model)
            
            if lora_weights:
                self._create_peft_checkpoint_structure(peft_dir, lora_weights, metadata)
                print(f"✓ PEFT checkpoint saved using fallback method")
            else:
                # Create minimal PEFT structure
                self._create_peft_checkpoint_manual_safe(model, peft_dir, metadata)
                print(f"✓ PEFT checkpoint created using safe manual method")
                
        except Exception as e:
            print(f"Warning: All PEFT save methods failed: {e}")
    
    def _extract_lora_from_named_parameters(self, model):
        """Extract LoRA weights from model.named_parameters() - FSDP2 safe"""
        lora_weights = {}
        
        try:
            for name, param in model.named_parameters():
                # Look for LoRA-specific parameter names
                if any(lora_key in name for lora_key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
                    # Clean up FSDP and module prefixes
                    clean_name = name
                    for prefix in ['_fsdp_wrapped_module.', 'module.', '_orig_mod.']:
                        clean_name = clean_name.replace(prefix, '')
                    
                    # For FSDP2, we need to handle sharded parameters carefully
                    if hasattr(param, 'detach'):
                        try:
                            # Detach and move to CPU if possible
                            param_data = param.detach()
                            if param_data.is_cuda:
                                param_data = param_data.cpu()
                            lora_weights[clean_name] = param_data.clone()
                        except Exception as e:
                            print(f"Warning: Could not extract parameter {name}: {e}")
                            continue
                            
        except Exception as e:
            print(f"Error extracting from named_parameters: {e}")
            
        return lora_weights
    
    def _extract_lora_weights_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract LoRA weights from a full state dict"""
        lora_weights = {}
        
        for key, tensor in state_dict.items():
            # Look for LoRA-specific parameter names
            if any(lora_key in key for lora_key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
                # Clean up FSDP and module prefixes
                clean_key = key
                for prefix in ['_fsdp_wrapped_module.', 'module.', '_orig_mod.']:
                    clean_key = clean_key.replace(prefix, '')
                
                # Ensure tensor is on CPU and detached
                if tensor.device != torch.device('cpu'):
                    tensor = tensor.cpu()
                if tensor.requires_grad:
                    tensor = tensor.detach()
                
                lora_weights[clean_key] = tensor.clone()
        
        return lora_weights
    
    def _create_peft_checkpoint_structure(self, peft_dir: Path, lora_weights: Dict[str, torch.Tensor], metadata: Dict[str, Any]):
        """Create PEFT checkpoint structure from LoRA weights"""
        
        if not lora_weights:
            raise RuntimeError("No LoRA weights provided")
        
        # Create adapter_config.json
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
            "lora_dropout": 0.0,  # Set to 0 for inference
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": int(metadata.get('lora_rank', 16)),
            "revision": None,
            "target_modules": metadata.get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"]),
            "task_type": "CAUSAL_LM"
        }
        
        # Save adapter config
        import json
        with open(peft_dir / "adapter_config.json", 'w') as f:
            json.dump(adapter_config, f, indent=2)
        
        # Save adapter weights
        try:
            # Try to save as safetensors (preferred format)
            from safetensors.torch import save_file
            save_file(lora_weights, peft_dir / "adapter_model.safetensors")
        except ImportError:
            # Fallback to PyTorch format
            torch.save(lora_weights, peft_dir / "adapter_model.bin")
    
    def _create_peft_checkpoint_manual_safe(self, model, peft_dir: Path, metadata: Dict[str, Any]):
        """Safely create PEFT checkpoint without accessing distributed tensors directly"""
        
        # Create a minimal PEFT structure with config only
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
        
        # Save adapter config
        import json
        with open(peft_dir / "adapter_config.json", 'w') as f:
            json.dump(adapter_config, f, indent=2)
        
        # Create an empty weights file as placeholder
        empty_weights = {}
        try:
            from safetensors.torch import save_file
            save_file(empty_weights, peft_dir / "adapter_model.safetensors")
        except ImportError:
            torch.save(empty_weights, peft_dir / "adapter_model.bin")
        
        print("Created PEFT config structure. Note: Weights will need to be extracted manually for inference.")
    
    def load(self, model, optimizer) -> bool:
        """Load the latest checkpoint using FSDP2 APIs"""
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
                # Load model state dict using distributed checkpoint (FSDP2 compatible)
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
    
    def extract_peft_weights_from_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Optional[str]:
        """Extract PEFT weights from a distributed checkpoint for inference use"""
        
        if self.rank != 0:
            return None
            
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            
        if not checkpoint_path:
            print("No checkpoint found to extract PEFT weights from")
            return None
        
        try:
            print(f"Extracting PEFT weights from: {checkpoint_path}")
            
            # Check if PEFT checkpoint already exists
            peft_dir = checkpoint_path / "peft"
            if peft_dir.exists() and (peft_dir / "adapter_config.json").exists():
                # Check if weights file exists and is not empty
                weights_file = peft_dir / "adapter_model.safetensors"
                if not weights_file.exists():
                    weights_file = peft_dir / "adapter_model.bin"
                
                if weights_file.exists() and weights_file.stat().st_size > 1000:  # More than 1KB
                    print(f"✓ PEFT checkpoint already exists: {peft_dir}")
                    return str(peft_dir)
            
            # Need to extract weights from distributed checkpoint
            print("Extracting weights from distributed checkpoint...")
            
            # Load the distributed model checkpoint
            state_dict = {}
            load_state_dict(
                state_dict=state_dict,
                storage_reader=FileSystemReader(checkpoint_path / "model"),
            )
            
            # Extract the actual model state dict
            full_state_dict = state_dict.get("model", {})
            
            # Extract LoRA weights
            lora_weights = self._extract_lora_weights_from_state_dict(full_state_dict)
            
            if not lora_weights:
                print("Warning: No LoRA weights found in checkpoint")
                return None
            
            # Load metadata
            metadata_path = checkpoint_path / "metadata.pt"
            if metadata_path.exists():
                metadata = torch.load(metadata_path, map_location='cpu')
            else:
                metadata = {"model_name": "bigcode/starcoder2-7b", "lora_rank": 16, "lora_alpha": 16}
            
            # Create PEFT structure
            peft_dir.mkdir(exist_ok=True)
            self._create_peft_checkpoint_structure(peft_dir, lora_weights, metadata)
            
            print(f"✓ PEFT weights extracted to: {peft_dir}")
            return str(peft_dir)
            
        except Exception as e:
            print(f"✗ Failed to extract PEFT weights: {e}")
            return None
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the latest checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda x: x.stat().st_mtime)
    
    def get_latest_peft_checkpoint(self) -> Optional[str]:
        """Get the path to the latest PEFT checkpoint for inference"""
        latest_checkpoint = self.get_latest_checkpoint()
        if latest_checkpoint:
            peft_dir = latest_checkpoint / "peft"
            if peft_dir.exists() and (peft_dir / "adapter_config.json").exists():
                # Check if weights exist
                weights_file = peft_dir / "adapter_model.safetensors"
                if not weights_file.exists():
                    weights_file = peft_dir / "adapter_model.bin"
                
                if weights_file.exists() and weights_file.stat().st_size > 1000:
                    return str(peft_dir)
                else:
                    # Try to extract weights
                    return self.extract_peft_weights_from_checkpoint(latest_checkpoint)
        return None
    
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
    
    def save_lora_weights_only(self, model, filename: str = "lora_weights.pt"):
        """Save only the LoRA weights (for inference) - FSDP2 compatible legacy method"""
        if self.rank != 0:
            return
            
        try:
            # FSDP2 approach: extract from named_parameters
            lora_state_dict = self._extract_lora_from_named_parameters(model)
            
            if not lora_state_dict:
                print("Warning: No LoRA weights found to save")
                return
            
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
                
                # Check for PEFT checkpoint
                peft_dir = checkpoint_path / "peft"
                has_peft = peft_dir.exists() and (peft_dir / "adapter_config.json").exists()
                
                # Check if PEFT weights exist and are valid
                peft_weights_valid = False
                if has_peft:
                    weights_file = peft_dir / "adapter_model.safetensors"
                    if not weights_file.exists():
                        weights_file = peft_dir / "adapter_model.bin"
                    peft_weights_valid = weights_file.exists() and weights_file.stat().st_size > 1000
                
                checkpoint_info = {
                    "timestamp": checkpoint_path.name.replace("checkpoint_", ""),
                    "path": str(checkpoint_path),
                    "peft_path": str(peft_dir) if has_peft else None,
                    "has_peft": has_peft,
                    "peft_weights_valid": peft_weights_valid,
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