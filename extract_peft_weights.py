# extract_peft_weights.py

"""
Utility script to extract PEFT weights from FSDP2 checkpoints.
This script can be used to retroactively create PEFT-compatible checkpoints
from existing training checkpoints.
"""

import argparse
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional

from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load as load_state_dict


def extract_lora_weights_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
            print(f"Extracted: {clean_key} -> {tensor.shape}")
    
    return lora_weights


def create_peft_checkpoint_structure(peft_dir: Path, lora_weights: Dict[str, torch.Tensor], metadata: Dict[str, Any]):
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
    peft_dir.mkdir(exist_ok=True, parents=True)
    with open(peft_dir / "adapter_config.json", 'w') as f:
        json.dump(adapter_config, f, indent=2)
    
    print(f"Created adapter config: {peft_dir / 'adapter_config.json'}")
    
    # Save adapter weights
    try:
        from safetensors.torch import save_file
        save_file(lora_weights, peft_dir / "adapter_model.safetensors")
        print(f"Saved weights as safetensors: {peft_dir / 'adapter_model.safetensors'}")
    except ImportError:
        torch.save(lora_weights, peft_dir / "adapter_model.bin")
        print(f"Saved weights as PyTorch: {peft_dir / 'adapter_model.bin'}")


def extract_peft_from_checkpoint(checkpoint_path: Path, output_path: Optional[Path] = None) -> bool:
    """Extract PEFT weights from a distributed checkpoint"""
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        return False
    
    # Check if this is a distributed checkpoint
    model_dir = checkpoint_path / "model"
    if not model_dir.exists():
        print(f"Error: No model directory found in checkpoint: {checkpoint_path}")
        return False
    
    print(f"Extracting PEFT weights from: {checkpoint_path}")
    
    try:
        # Load the distributed model checkpoint
        state_dict = {}
        load_state_dict(
            state_dict=state_dict,
            storage_reader=FileSystemReader(model_dir),
        )
        
        # Extract the actual model state dict
        full_state_dict = state_dict.get("model", {})
        print(f"Loaded state dict with {len(full_state_dict)} keys")
        
        # Extract LoRA weights
        lora_weights = extract_lora_weights_from_state_dict(full_state_dict)
        
        if not lora_weights:
            print("Warning: No LoRA weights found in checkpoint")
            return False
        
        print(f"Found {len(lora_weights)} LoRA weight tensors")
        
        # Load metadata
        metadata_path = checkpoint_path / "metadata.pt"
        if metadata_path.exists():
            metadata = torch.load(metadata_path, map_location='cpu')
            print(f"Loaded metadata: {metadata}")
        else:
            print("Warning: No metadata found, using defaults")
            metadata = {
                "model_name": "bigcode/starcoder2-7b", 
                "lora_rank": 16, 
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            }
        
        # Determine output path
        if output_path is None:
            output_path = checkpoint_path / "peft"
        
        # Create PEFT structure
        create_peft_checkpoint_structure(output_path, lora_weights, metadata)
        
        print(f"✓ PEFT weights extracted successfully to: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to extract PEFT weights: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_peft_checkpoint(peft_path: Path) -> bool:
    """Verify that a PEFT checkpoint is valid"""
    
    if not peft_path.exists():
        print(f"Error: PEFT path does not exist: {peft_path}")
        return False
    
    # Check for required files
    config_file = peft_path / "adapter_config.json"
    weights_file_safetensors = peft_path / "adapter_model.safetensors"
    weights_file_pytorch = peft_path / "adapter_model.bin"
    
    if not config_file.exists():
        print(f"Error: Missing adapter_config.json in {peft_path}")
        return False
    
    if not weights_file_safetensors.exists() and not weights_file_pytorch.exists():
        print(f"Error: Missing weight files in {peft_path}")
        return False
    
    try:
        # Verify config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        required_keys = ['peft_type', 'r', 'lora_alpha', 'target_modules', 'task_type']
        for key in required_keys:
            if key not in config:
                print(f"Error: Missing required key '{key}' in adapter_config.json")
                return False
        
        print(f"✓ Config valid: {config['peft_type']} with r={config['r']}, alpha={config['lora_alpha']}")
        
        # Verify weights
        if weights_file_safetensors.exists():
            try:
                from safetensors.torch import load_file
                weights = load_file(weights_file_safetensors)
                print(f"✓ Safetensors weights valid: {len(weights)} tensors")
            except ImportError:
                print("Warning: safetensors not available, cannot verify safetensors file")
        elif weights_file_pytorch.exists():
            weights = torch.load(weights_file_pytorch, map_location='cpu')
            print(f"✓ PyTorch weights valid: {len(weights)} tensors")
        
        return True
        
    except Exception as e:
        print(f"Error verifying PEFT checkpoint: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract PEFT weights from FSDP2 checkpoints")
    
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the checkpoint directory"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for PEFT checkpoint (default: checkpoint_path/peft)"
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing PEFT checkpoint, don't extract"
    )
    
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Process all checkpoints in the directory"
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint_path)
    
    if args.all_checkpoints:
        # Process all checkpoint directories
        if not checkpoint_path.exists():
            print(f"Error: Directory does not exist: {checkpoint_path}")
            return
        
        checkpoint_dirs = list(checkpoint_path.glob("checkpoint_*"))
        if not checkpoint_dirs:
            print(f"No checkpoint directories found in: {checkpoint_path}")
            return
        
        print(f"Found {len(checkpoint_dirs)} checkpoint directories")
        
        success_count = 0
        for ckpt_dir in sorted(checkpoint_dirs):
            print(f"\n--- Processing {ckpt_dir.name} ---")
            
            if args.verify_only:
                peft_dir = ckpt_dir / "peft"
                if verify_peft_checkpoint(peft_dir):
                    success_count += 1
            else:
                output_path = Path(args.output_path) if args.output_path else None
                if extract_peft_from_checkpoint(ckpt_dir, output_path):
                    success_count += 1
        
        print(f"\n✓ Successfully processed {success_count}/{len(checkpoint_dirs)} checkpoints")
        
    else:
        # Process single checkpoint
        if args.verify_only:
            # Verify existing PEFT checkpoint
            peft_path = checkpoint_path / "peft" if args.output_path is None else Path(args.output_path)
            success = verify_peft_checkpoint(peft_path)
        else:
            # Extract PEFT weights
            output_path = Path(args.output_path) if args.output_path else None
            success = extract_peft_from_checkpoint(checkpoint_path, output_path)
        
        if success:
            print("✓ Operation completed successfully")
        else:
            print("✗ Operation failed")
            exit(1)


if __name__ == "__main__":
    main()