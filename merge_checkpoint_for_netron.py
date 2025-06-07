# merge_checkpoint_for_netron.py

import os
import torch
import torch.nn as nn
import json
from pathlib import Path
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load as load_state_dict
import traceback
from safetensors.torch import save_file, load_file
from model import LoRAModel

def merge_fsdp2_checkpoint_for_netron(checkpoint_path: str, output_format: str = "safetensors"):
    """
    Merge FSDP2 distributed checkpoint files into a single file for Netron visualization.
    
    Args:
        checkpoint_path: Path to the checkpoint directory containing model/ subdirectory
        output_format: Format to save merged model ("safetensors", "pytorch", "onnx")
    """
    checkpoint_path_obj = Path(checkpoint_path)
    
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")
    
    model_dir = checkpoint_path_obj.joinpath("model")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} does not exist")
    
    # Load metadata to get model configuration
    metadata_path = checkpoint_path_obj.joinpath("metadata.pt")
    if metadata_path.exists():
        metadata = torch.load(metadata_path, map_location="cpu")
        print(f"Found metadata: {metadata}")
    else:
        print("No metadata found, using defaults")
        metadata = {
            "model_name": "bigcode/starcoder2-7b",
            "lora_rank": 16,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
    
    print("Creating base model...")
    
    # Create the base model structure (this won't load the actual weights)
    base_model = AutoModelForCausalLM.from_pretrained(
        metadata.get("model_name", "bigcode/starcoder2-7b"),
        torch_dtype=torch.float32,  # Use float32 for better compatibility
        trust_remote_code=True,
        use_cache=False,
    )
    
    # Create LoRA wrapper
    lora_wrapper = LoRAModel(
        model_name=metadata.get("model_name", "bigcode/starcoder2-7b"),
        lora_rank=metadata.get("lora_rank", 16),
        lora_alpha=metadata.get("lora_alpha", 16),
        lora_dropout=0.0,  # Set to 0 for inference
        target_modules=metadata.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        base_model=base_model,
    )
    
    model = lora_wrapper.model
    
    print("Loading distributed checkpoint...")
    
    try:
        # Load the distributed checkpoint
        model_state_dict = {"model": model.state_dict()}
        
        # Load from distributed checkpoint
        load_state_dict(
            state_dict=model_state_dict,
            storage_reader=FileSystemReader(model_dir),
        )
        
        # Load the merged state dict into the model
        model.load_state_dict(model_state_dict["model"], strict=False)
        
        print("Successfully loaded distributed checkpoint")
        
    except Exception as e:
        print(f"Error loading distributed checkpoint: {e}")
        traceback.print_exc()
        return False
    
    # Create output directory
    output_dir = checkpoint_path_obj.parent.joinpath(f"netron_ready_{checkpoint_path_obj.name}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Saving merged model to {output_dir}")
    
    # Save in requested format
    if output_format.lower() == "safetensors":
        save_safetensors_format(model, output_dir, metadata)
    elif output_format.lower() == "pytorch":
        save_pytorch_format(model, output_dir, metadata)
    elif output_format.lower() == "onnx":
        save_onnx_format(model, output_dir, metadata)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    print(f"Merged model saved to {output_dir}")
    return True

def save_safetensors_format(model, output_dir, metadata):
    """Save model in Safetensors format (best for Netron)"""
    
    # Get state dict and ensure all tensors are on CPU
    state_dict = model.state_dict()
    cpu_state_dict = {}
    
    for name, tensor in state_dict.items():
        if tensor is not None:
            cpu_state_dict[name] = tensor.detach().cpu().contiguous()
    
    # Save as safetensors
    safetensors_path = output_dir.joinpath("model.safetensors")
    save_file(cpu_state_dict, safetensors_path)
    
    # Save metadata as JSON
    metadata_path = output_dir.joinpath("metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Safetensors model saved: {safetensors_path}")
    print(f"Metadata saved: {metadata_path}")

def save_pytorch_format(model, output_dir, metadata):
    """Save model in PyTorch format"""
    
    # Save full model
    model_path = output_dir.joinpath("model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata,
        'model_config': model.config.to_dict() if hasattr(model, 'config') else {}
    }, model_path)
    
    # Also save just the state dict
    state_dict_path = output_dir.joinpath("state_dict.pth")
    torch.save(model.state_dict(), state_dict_path)
    
    print(f"PyTorch model saved: {model_path}")
    print(f"State dict saved: {state_dict_path}")

def save_onnx_format(model, output_dir, metadata):
    """Save model in ONNX format with proper architecture representation"""
    
    try:
        print("Preparing model for ONNX export...")
        
        # Set model to evaluation mode
        model.eval()
        
        # Get model configuration
        config = getattr(model, 'config', None)
        if config is None:
            # Use default StarCoder2 config
            vocab_size = 49152
            hidden_size = 4608
            max_position_embeddings = 16384
        else:
            vocab_size = getattr(config, 'vocab_size', 49152)
            hidden_size = getattr(config, 'hidden_size', 4608)
            max_position_embeddings = getattr(config, 'max_position_embeddings', 16384)
        
        # Create realistic input tensors for the model
        batch_size = 1
        sequence_length = min(512, max_position_embeddings)  # Use reasonable sequence length
        
        # Create input tensors
        input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length), dtype=torch.long)
        attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
        
        print(f"Using input shape: {input_ids.shape}")
        print(f"Vocabulary size: {vocab_size}")
        
        # Create ONNX export path
        onnx_path = output_dir.joinpath("model.onnx")
        
        print(f"Exporting to ONNX: {onnx_path}")
        
        # Export with proper input/output names and dynamic axes
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
        
        with torch.no_grad():
            torch.onnx.export(
                model,
                args=(input_ids, attention_mask),
                f=onnx_path,
                export_params=True,
                opset_version=14,  # Use newer opset for better compatibility
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes=dynamic_axes,
                verbose=False,
                strip_doc_string=False,
                keep_initializers_as_inputs=False
            )
        
        print(f"Successfully exported full model to ONNX: {onnx_path}")
        return True
        
    except Exception as e:
        print(f"Full model ONNX export failed: {e}")
        print("Falling back to LoRA-only export...")
        
        # Fallback to LoRA-only export
        lora_weights = extract_lora_weights_from_model(model)
        if lora_weights:
            return save_onnx_lora_architecture(lora_weights, output_dir, metadata)
        
        return False

class LoRAArchitectureModel(nn.Module):
    """
    A PyTorch model that represents the LoRA architecture for better Netron visualization.
    This creates a meaningful computational graph instead of a dummy identity function.
    """
    
    def __init__(self, lora_weights, metadata):
        super().__init__()
        
        self.lora_rank = metadata.get('lora_rank', 16)
        self.lora_alpha = metadata.get('lora_alpha', 16)
        self.target_modules = metadata.get('target_modules', ['q_proj', 'v_proj', 'k_proj', 'o_proj'])
        
        # Group LoRA weights by layer and module type
        self.lora_layers = nn.ModuleDict()
        
        # Parse LoRA weights and create representative layers
        layer_groups = self._group_lora_weights(lora_weights)
        
        for layer_name, layer_weights in layer_groups.items():
            layer_modules = nn.ModuleDict()
            
            for module_name, module_weights in layer_weights.items():
                if 'lora_A' in module_weights and 'lora_B' in module_weights:
                    # Create LoRA layer representation
                    lora_A = module_weights['lora_A']
                    lora_B = module_weights['lora_B']
                    
                    # Validate tensors before creating layers
                    try:
                        # Test if tensors are usable
                        _ = lora_A.shape
                        _ = lora_B.shape
                        _ = lora_A.data_ptr()
                        _ = lora_B.data_ptr()
                        
                        # Create the LoRA computation as separate layers
                        layer_modules[f"{module_name}_lora_A"] = nn.Linear(
                            lora_A.shape[1], lora_A.shape[0], bias=False
                        )
                        layer_modules[f"{module_name}_lora_B"] = nn.Linear(
                            lora_B.shape[1], lora_B.shape[0], bias=False
                        )
                        
                        # Load the actual weights and convert to float32 for ONNX compatibility
                        # Make sure tensors are contiguous and properly allocated
                        lora_A_data = lora_A.clone().detach().float().contiguous()
                        lora_B_data = lora_B.clone().detach().float().contiguous()
                        
                        # Verify the cloned data is valid
                        _ = lora_A_data.data_ptr()
                        _ = lora_B_data.data_ptr()
                        
                        layer_modules[f"{module_name}_lora_A"].weight.data = lora_A_data
                        layer_modules[f"{module_name}_lora_B"].weight.data = lora_B_data
                        
                    except Exception as e:
                        print(f"Warning: Could not create layer for {module_name}: {e}")
                        print(f"  lora_A shape: {getattr(lora_A, 'shape', 'unknown')}")
                        print(f"  lora_B shape: {getattr(lora_B, 'shape', 'unknown')}")
                        continue
            
            if layer_modules:
                self.lora_layers[layer_name] = layer_modules
        
        # Create scaling parameter
        self.scaling = self.lora_alpha / self.lora_rank
        
        print(f"Created LoRA architecture model with {len(self.lora_layers)} layers")
        for layer_name, layer_modules in self.lora_layers.items():
            print(f"  Layer {layer_name}: {len(list(layer_modules.parameters()))} parameters")
            
        # Validate all parameters are properly allocated
        total_params = 0
        invalid_params = 0
        for name, param in self.named_parameters():
            try:
                _ = param.data_ptr()
                total_params += 1
            except RuntimeError:
                print(f"Warning: Invalid parameter {name}")
                invalid_params += 1
        
        print(f"Model validation: {total_params} valid parameters, {invalid_params} invalid parameters")
    
    def _group_lora_weights(self, lora_weights):
        """Group LoRA weights by transformer layer and module type"""
        layer_groups = {}
        
        for weight_name, weight_tensor in lora_weights.items():
            # Parse weight name to extract layer and module info
            # Example: "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
            parts = weight_name.split('.')
            
            # Find layer number
            layer_num = None
            module_type = None
            lora_type = None
            
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    layer_num = parts[i + 1]
                elif part in self.target_modules:
                    module_type = part
                elif part in ['lora_A', 'lora_B']:
                    lora_type = part
            
            if layer_num is not None and module_type is not None and lora_type is not None:
                layer_key = f"layer_{layer_num}"
                
                if layer_key not in layer_groups:
                    layer_groups[layer_key] = {}
                
                if module_type not in layer_groups[layer_key]:
                    layer_groups[layer_key][module_type] = {}
                
                layer_groups[layer_key][module_type][lora_type] = weight_tensor
        
        return layer_groups
    
    def forward(self, x):
        """
        Forward pass that demonstrates LoRA computation.
        This creates a meaningful computational graph for Netron.
        """
        # x should be a tensor representing hidden states
        # Shape: (batch_size, sequence_length, hidden_size)
        
        outputs = []
        
        # Process through each LoRA layer
        for layer_name, layer_modules in self.lora_layers.items():
            layer_output = x
            
            # Apply LoRA computations for each module in this layer
            for module_name, module in layer_modules.items():
                if 'lora_A' in module_name:
                    # Ensure input tensor matches module dtype
                    if layer_output.dtype != module.weight.dtype:
                        layer_output = layer_output.to(module.weight.dtype)
                    
                    # LoRA A transformation (down-projection)
                    lora_a_output = module(layer_output)
                    
                    # Find corresponding LoRA B module
                    base_name = module_name.replace('_lora_A', '')
                    lora_b_name = f"{base_name}_lora_B"
                    
                    if lora_b_name in layer_modules:
                        lora_b_module = layer_modules[lora_b_name]
                        # LoRA B transformation (up-projection)
                        lora_output = lora_b_module(lora_a_output)
                        # Apply scaling
                        lora_output = lora_output * self.scaling
                        # Add to residual (simulating the addition to base weights)
                        layer_output = layer_output + lora_output
            
            outputs.append(layer_output)
        
        # Combine outputs from all layers
        if outputs:
            # Simple combination - in practice this would be more complex
            final_output = torch.stack(outputs, dim=0).mean(dim=0)
        else:
            final_output = x
        
        return final_output

def save_onnx_lora_architecture(lora_weights, output_dir, metadata):
    """Save LoRA architecture in ONNX format with proper computational graph"""
    
    try:
        print("Creating LoRA architecture model for ONNX export...")
        
        # First, validate and clean all LoRA weights
        print("Validating and cleaning LoRA weights...")
        validated_lora_weights = {}
        
        for name, weight in lora_weights.items():
            try:
                # Ensure tensor is valid and properly allocated
                clean_weight = extract_raw_tensor_data(weight)
                if clean_weight is not None and clean_weight.numel() > 0:
                    # Make sure it's contiguous and float32
                    clean_weight = clean_weight.float().contiguous()
                    # Verify it's actually usable
                    _ = clean_weight.data_ptr()
                    validated_lora_weights[name] = clean_weight
                    print(f"  Validated: {name} -> shape {clean_weight.shape}")
                else:
                    print(f"  Skipped invalid weight: {name}")
            except Exception as e:
                print(f"  Failed to validate weight {name}: {e}")
                continue
        
        if not validated_lora_weights:
            print("No valid LoRA weights found after validation")
            return False
        
        print(f"Successfully validated {len(validated_lora_weights)} LoRA weights")
        
        # Create the LoRA architecture model with validated weights
        lora_model = LoRAArchitectureModel(validated_lora_weights, metadata)
        lora_model.eval()
        
        # Convert all model parameters to float32 for ONNX compatibility
        print("Converting model to float32 for ONNX compatibility...")
        lora_model = lora_model.float()
        
        # Additional validation - test a forward pass
        print("Testing model forward pass...")
        hidden_size = None
        for weight_name, weight_tensor in validated_lora_weights.items():
            if 'lora_B' in weight_name:
                hidden_size = weight_tensor.shape[0]
                break
        
        if hidden_size is None:
            hidden_size = 4608  # Default StarCoder2 hidden size
        
        # Create test input and verify model works
        batch_size = 1
        sequence_length = 8  # Small test size first
        test_input = torch.randn(batch_size, sequence_length, hidden_size, dtype=torch.float32)
        
        try:
            with torch.no_grad():
                test_output = lora_model(test_input)
                print(f"Test forward pass successful: {test_input.shape} -> {test_output.shape}")
        except Exception as test_error:
            print(f"Model forward pass failed: {test_error}")
            return False
        
        # Now create the actual input for ONNX export
        sequence_length = 128  # Reasonable size for visualization
        dummy_input = torch.randn(batch_size, sequence_length, hidden_size, dtype=torch.float32)
        
        onnx_path = output_dir.joinpath("lora_architecture.onnx")
        
        print(f"Exporting LoRA architecture to ONNX: {onnx_path}")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Input dtype: {dummy_input.dtype}")
        print(f"Hidden size: {hidden_size}")
        
        # Export with dynamic axes for flexibility
        dynamic_axes = {
            'hidden_states': {0: 'batch_size', 1: 'sequence_length'},
            'lora_output': {0: 'batch_size', 1: 'sequence_length'}
        }
        
        with torch.no_grad():
            torch.onnx.export(
                lora_model,
                args=(dummy_input,),
                f=onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['hidden_states'],
                output_names=['lora_output'],
                dynamic_axes=dynamic_axes,
                verbose=False,
                strip_doc_string=False
            )
        
        # Also save a simpler version for basic weight visualization
        simple_path = output_dir.joinpath("lora_weights.onnx")
        save_simple_lora_onnx(validated_lora_weights, simple_path)
        
        print(f"Successfully exported LoRA architecture to ONNX: {onnx_path}")
        print(f"Also created simple weights visualization: {simple_path}")
        return True
        
    except Exception as e:
        print(f"LoRA architecture ONNX export failed: {e}")
        traceback.print_exc()
        return False

def save_simple_lora_onnx(lora_weights, onnx_path):
    """Create a simple ONNX model that just contains the LoRA weights as parameters"""
    
    class SimpleLoRAWeights(nn.Module):
        def __init__(self, lora_weights):
            super().__init__()
            
            # Create parameters for each LoRA weight
            for name, weight in lora_weights.items():
                # Clean the parameter name for PyTorch
                clean_name = name.replace('.', '_').replace('-', '_').replace('/', '_')
                clean_tensor = extract_raw_tensor_data(weight)
                if clean_tensor is not None and clean_tensor.numel() > 0:
                    self.register_parameter(clean_name, nn.Parameter(clean_tensor))
        
        def forward(self, x):
            # Simple forward that shows the weights exist
            return x.sum() * 0 + sum(p.sum() for p in self.parameters())
    
    try:
        model = SimpleLoRAWeights(lora_weights)
        model.eval()
        
        # Create minimal input
        dummy_input = torch.tensor([1.0])
        
        with torch.no_grad():
            torch.onnx.export(
                model,
                args=(dummy_input,),
                f=onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                verbose=False
            )
        
        return True
        
    except Exception as e:
        print(f"Simple ONNX export failed: {e}")
        return False

def extract_lora_weights_from_model(model):
    """Extract LoRA weights from a loaded model"""
    lora_weights = {}
    
    for name, param in model.named_parameters():
        if any(lora_key in name for lora_key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
            try:
                clean_tensor = extract_raw_tensor_data(param)
                if clean_tensor is not None:
                    lora_weights[name] = clean_tensor
            except Exception as e:
                print(f"Warning: Could not extract parameter {name}: {e}")
                continue
    
    return lora_weights

def extract_raw_tensor_data(tensor):
    """Extract raw tensor data, bypassing wrapper subclasses and ensuring float32 dtype"""
    try:
        # Handle tensor subclasses that don't support .numpy()
        if hasattr(tensor, 'detach'):
            tensor = tensor.detach()
        
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu()
        
        # Check if tensor has valid storage
        try:
            _ = tensor.data_ptr()
        except RuntimeError:
            print(f"Warning: Tensor has invalid storage, attempting to reconstruct...")
            # Try to reconstruct tensor from shape and dtype
            if hasattr(tensor, 'shape') and hasattr(tensor, 'dtype'):
                # Create a new tensor with the same shape and dtype, filled with zeros
                reconstructed = torch.zeros(tensor.shape, dtype=tensor.dtype)
                print(f"Reconstructed tensor with shape {tensor.shape} and dtype {tensor.dtype}")
                tensor = reconstructed
            else:
                return None
        
        # Try direct conversion first (for regular tensors)
        try:
            if tensor.dtype in [torch.bfloat16, torch.float16]:
                clean_tensor = tensor.float()
            else:
                clean_tensor = tensor.clone()
            
            # Verify the tensor is actually usable
            _ = clean_tensor.data_ptr()
            return clean_tensor
            
        except (RuntimeError, AttributeError):
            # Fallback: try numpy conversion for tensor subclasses
            try:
                numpy_data = tensor.numpy()
                clean_tensor = torch.from_numpy(numpy_data.copy())
                
                # Convert to float32 for better ONNX compatibility
                if clean_tensor.dtype in [torch.bfloat16, torch.float16]:
                    clean_tensor = clean_tensor.float()
                
                return clean_tensor
                
            except (RuntimeError, AttributeError) as e:
                if "not supported for tensor subclasses" in str(e):
                    # For tensor subclasses, try to access underlying data
                    try:
                        # Try to get the underlying tensor data
                        if hasattr(tensor, '_tensor'):
                            underlying = tensor._tensor
                        elif hasattr(tensor, 'data'):
                            underlying = tensor.data
                        else:
                            # Last resort: reconstruct from shape and dtype
                            print(f"Reconstructing tensor from shape {tensor.shape}")
                            underlying = torch.zeros(tensor.shape, dtype=tensor.dtype)
                        
                        return extract_raw_tensor_data(underlying)
                        
                    except Exception as inner_e:
                        print(f"Failed to extract from tensor subclass: {inner_e}")
                        return None
                else:
                    raise e
        
    except Exception as e:
        print(f"Failed to extract raw tensor data: {e}")
        return None

def safe_state_dict_for_save(state_dict):
    """Recursively ensure all tensors in a state dict are on CPU, contiguous, cloned, and not meta."""
    safe_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, dict):
            safe_dict[k] = safe_state_dict_for_save(v)
        elif isinstance(v, torch.Tensor):
            if getattr(v, 'is_meta', False):
                print(f"Warning: Skipping meta tensor '{k}' in state_dict")
                continue
            try:
                # First try the raw tensor extraction to bypass wrapper classes
                clean_tensor = extract_raw_tensor_data(v)
                if clean_tensor is not None:
                    safe_dict[k] = clean_tensor
                    continue
                
                # Fallback to original method
                _ = v.data_ptr()
                
                if v.device.type != 'cpu':
                    v = v.cpu()
                if not v.is_contiguous():
                    v = v.contiguous()
                safe_dict[k] = v.clone()
                
            except RuntimeError as e:
                if "invalid python storage" in str(e):
                    print(f"Warning: Invalid storage for '{k}', attempting numpy conversion...")
                    try:
                        # Try numpy conversion as last resort
                        clean_tensor = extract_raw_tensor_data(v)
                        if clean_tensor is not None:
                            safe_dict[k] = clean_tensor
                        else:
                            print(f"Warning: Could not convert tensor '{k}', skipping")
                            continue
                    except Exception as fix_error:
                        print(f"Warning: Could not fix tensor '{k}': {fix_error}, skipping")
                        continue
                else:
                    print(f"Warning: Could not process tensor '{k}': {e}, skipping")
                    continue
        else:
            safe_dict[k] = v
    return safe_dict

def extract_lora_weights_only(checkpoint_path: str, output_format: str = "safetensors"):
    """
    Extract only LoRA weights from checkpoint for focused visualization.
    """
    checkpoint_path_obj = Path(checkpoint_path)
    peft_dir = checkpoint_path_obj.joinpath("peft")
    
    if peft_dir.exists():
        print(f"Found existing PEFT format from {peft_dir}")
        
        # Create output directory
        output_dir = checkpoint_path_obj.parent.joinpath(f"netron_lora_{checkpoint_path_obj.name}")
        output_dir.mkdir(exist_ok=True)
        
        # Load the adapter model weights
        adapter_bin_path = peft_dir.joinpath("adapter_model.bin")
        adapter_safetensors_path = peft_dir.joinpath("adapter_model.safetensors")
        
        lora_weights = None
        
        # Try to load from either .bin or .safetensors
        if adapter_safetensors_path.exists():
            print(f"Loading LoRA weights from {adapter_safetensors_path}")
            from safetensors.torch import load_file
            lora_weights = load_file(adapter_safetensors_path)
        elif adapter_bin_path.exists():
            print(f"Loading LoRA weights from {adapter_bin_path}")
            lora_weights = torch.load(adapter_bin_path, map_location="cpu")
        else:
            print("No adapter model weights found in PEFT directory")
            return False
        
        if lora_weights is None:
            print("Failed to load LoRA weights")
            return False
        
        # Load configuration for metadata
        config_path = peft_dir.joinpath("adapter_config.json")
        metadata = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                metadata = json.load(f)
            import shutil
            shutil.copy2(config_path, output_dir.joinpath("adapter_config.json"))
        
        # Apply the safe tensor fix
        print("Applying safe tensor fix for storage issues...")
        safe_lora_weights = safe_state_dict_for_save(lora_weights)
        
        if len(safe_lora_weights) == 0:
            print("Error: No valid tensors after applying safety fixes")
            return False
        
        print(f"Successfully processed {len(safe_lora_weights)} tensors")
        
        # Try multiple output formats for maximum compatibility
        successful_formats = []
        
        # Save weights in requested format
        if output_format.lower() == "safetensors":
            output_file = output_dir.joinpath("lora_weights.safetensors")
            print(f"Converting to Safetensors format: {output_file}")
            
            try:
                save_file(safe_lora_weights, output_file)
                print(f"Successfully saved Safetensors file: {output_file}")
                successful_formats.append("safetensors")
                
            except Exception as safetensors_error:
                print(f"Safetensors save failed: {safetensors_error}")
                
        elif output_format.lower() == "onnx":
            # Create LoRA architecture for ONNX
            if save_onnx_lora_architecture(safe_lora_weights, output_dir, metadata):
                successful_formats.append("onnx")
        
        # Always try PyTorch format as fallback
        try:
            output_file = output_dir.joinpath("lora_weights.pth")
            print(f"Saving PyTorch format: {output_file}")
            torch.save(safe_lora_weights, output_file)
            print(f"Successfully saved PyTorch file: {output_file}")
            successful_formats.append("pytorch")
        except Exception as pytorch_error:
            print(f"PyTorch save failed: {pytorch_error}")
        
        # Try ONNX format as additional option (if not already requested)
        if output_format.lower() != "onnx":
            if save_onnx_lora_architecture(safe_lora_weights, output_dir, metadata):
                successful_formats.append("onnx")
        
        if not successful_formats:
            print("Error: All save formats failed")
            return False
        
        # Save metadata
        weight_metadata = {
            'lora_parameters': list(safe_lora_weights.keys()),
            'parameter_count': sum(p.numel() for p in safe_lora_weights.values() if p is not None),
            'requested_format': output_format,
            'successful_formats': successful_formats,
            'source': 'peft_adapter',
            'safe_tensors_applied': True,
            'adapter_config': metadata
        }
        
        with open(output_dir.joinpath("lora_metadata.json"), 'w') as f:
            json.dump(weight_metadata, f, indent=2, default=str)
        
        print(f"\nLoRA weights converted and saved to {output_dir}")
        print(f"Found {len(safe_lora_weights)} LoRA parameters")
        print(f"Total parameters: {weight_metadata['parameter_count']:,}")
        print(f"Successful formats: {', '.join(successful_formats)}")
        
        # Provide specific recommendations
        if "onnx" in successful_formats:
            print(f"\nRECOMMENDED: Try the ONNX architecture file first - it shows LoRA computation graph")
            print(f"   netron {output_dir.joinpath('lora_architecture.onnx')}")
            print(f"Alternative: Simple weights visualization")
            print(f"   netron {output_dir.joinpath('lora_weights.onnx')}")
        elif "safetensors" in successful_formats:
            print(f"\nRECOMMENDED: Try the Safetensors file")
            print(f"   netron {output_dir.joinpath('lora_weights.safetensors')}")
        elif "pytorch" in successful_formats:
            print(f"\nTry the PyTorch file (experimental Netron support)")
            print(f"   netron {output_dir.joinpath('lora_weights.pth')}")
        
        return True
    
    else:
        print("No PEFT directory found, extracting LoRA weights from distributed checkpoint...")
        
        # Load and extract LoRA weights only
        model_dir = checkpoint_path_obj.joinpath("model")
        if not model_dir.exists():
            print("No model directory found")
            return False
        
        # Create a minimal model to load LoRA weights
        metadata_path = checkpoint_path_obj.joinpath("metadata.pt")
        if metadata_path.exists():
            metadata = torch.load(metadata_path, map_location="cpu")
        else:
            metadata = {
                "model_name": "bigcode/starcoder2-7b",
                "lora_rank": 16,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            }
        
        try:
            # Create minimal LoRA model structure
            base_model = AutoModelForCausalLM.from_pretrained(
                metadata.get("model_name", "bigcode/starcoder2-7b"),
                torch_dtype=torch.float32,
                trust_remote_code=True,
                use_cache=False,
            )
            
            lora_wrapper = LoRAModel(
                model_name=metadata.get("model_name", "bigcode/starcoder2-7b"),
                lora_rank=metadata.get("lora_rank", 16),
                lora_alpha=metadata.get("lora_alpha", 16),
                lora_dropout=0.0,
                target_modules=metadata.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
                base_model=base_model,
            )
            
            model = lora_wrapper.model
            
            # Load distributed checkpoint
            model_state_dict = {"model": model.state_dict()}
            load_state_dict(
                state_dict=model_state_dict,
                storage_reader=FileSystemReader(model_dir),
            )
            
            # Extract only LoRA parameters
            lora_state_dict = {}
            for name, param in model_state_dict["model"].items():
                if any(lora_key in name for lora_key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
                    clean_tensor = extract_raw_tensor_data(param)
                    if clean_tensor is not None:
                        lora_state_dict[name] = clean_tensor
            
            # Save LoRA weights
            output_dir = checkpoint_path_obj.parent.joinpath(f"netron_lora_{checkpoint_path_obj.name}")
            output_dir.mkdir(exist_ok=True)
            
            # Apply safe tensor fix
            safe_lora_weights = safe_state_dict_for_save(lora_state_dict)
            
            successful_formats = []
            
            if output_format.lower() == "safetensors":
                try:
                    save_file(safe_lora_weights, output_dir.joinpath("lora_weights.safetensors"))
                    successful_formats.append("safetensors")
                except Exception as e:
                    print(f"Safetensors save failed: {e}")
            elif output_format.lower() == "onnx":
                if save_onnx_lora_architecture(safe_lora_weights, output_dir, metadata):
                    successful_formats.append("onnx")
            else:
                try:
                    torch.save(safe_lora_weights, output_dir.joinpath("lora_weights.pth"))
                    successful_formats.append("pytorch")
                except Exception as e:
                    print(f"PyTorch save failed: {e}")
            
            # Save metadata
            with open(output_dir.joinpath("lora_metadata.json"), 'w') as f:
                json.dump({
                    'lora_parameters': list(safe_lora_weights.keys()),
                    'parameter_count': sum(p.numel() for p in safe_lora_weights.values()),
                    'metadata': metadata,
                    'successful_formats': successful_formats
                }, f, indent=2, default=str)
            
            print(f"LoRA weights extracted to {output_dir}")
            print(f"Found {len(safe_lora_weights)} LoRA parameters")
            
            return True
            
        except Exception as e:
            print(f"Error extracting LoRA weights: {e}")
            traceback.print_exc()
            return False

def launch_netron(model_path):
    """Launch Netron with the model file"""
    try:
        import netron
        print(f"Launching Netron with {model_path}...")
        netron.start(str(model_path))
        return True
    except ImportError:
        print("Netron not installed. Install with: pip install netron")
        return False
    except Exception as e:
        print(f"Failed to launch Netron: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Merge FSDP2 checkpoint for Netron visualization")
    parser.add_argument("checkpoint_path", help="Path to checkpoint directory")
    parser.add_argument("--format", choices=["safetensors", "pytorch", "onnx"], 
                       default="safetensors", help="Output format for Netron")
    parser.add_argument("--lora-only", action="store_true", 
                       help="Extract only LoRA weights for focused visualization")
    parser.add_argument("--launch", action="store_true",
                       help="Automatically launch Netron after conversion")
    parser.add_argument("--no-launch", action="store_true",
                       help="Don't launch Netron, just convert")
    
    args = parser.parse_args()
    
    print(f"Processing checkpoint: {args.checkpoint_path}")
    print(f"Output format: {args.format}")
    
    if args.lora_only:
        success = extract_lora_weights_only(args.checkpoint_path, args.format)
        output_dir = Path(args.checkpoint_path).parent.joinpath(f"netron_lora_{Path(args.checkpoint_path).name}")
    else:
        success = merge_fsdp2_checkpoint_for_netron(args.checkpoint_path, args.format)
        output_dir = Path(args.checkpoint_path).parent.joinpath(f"netron_ready_{Path(args.checkpoint_path).name}")
    
    if success:
        print("\n" + "="*50)
        print("SUCCESS!")
        print("="*50)
        
        # Find the best model file to launch
        model_file = None
        
        if args.lora_only:
            # Priority order for LoRA visualization
            candidates = [
                output_dir.joinpath("lora_architecture.onnx"),  # Best - shows computation graph
                output_dir.joinpath("lora_weights.onnx"),       # Good - simple weights
                output_dir.joinpath("lora_weights.safetensors"), # Okay - just weights
                output_dir.joinpath("lora_weights.pth")         # Fallback
            ]
        else:
            # Priority order for full model
            if args.format == "safetensors":
                candidates = [output_dir.joinpath("model.safetensors")]
            elif args.format == "pytorch":
                candidates = [output_dir.joinpath("model.pth"), output_dir.joinpath("state_dict.pth")]
            elif args.format == "onnx":
                candidates = [output_dir.joinpath("model.onnx")]
            else:
                candidates = []
        
        # Find the first existing file
        for candidate in candidates:
            if candidate.exists():
                model_file = candidate
                break
        
        if model_file:
            print(f"Model file created: {model_file}")
            print(f"File size: {model_file.stat().st_size / 1024 / 1024:.1f} MB")
            
            # Launch Netron if requested
            if args.launch and not args.no_launch:
                if launch_netron(model_file):
                    print("Netron launched successfully!")
                else:
                    print("Failed to launch Netron automatically")
            
            # Provide usage instructions
            print(f"\nTo visualize the model:")
            print(f"1. Command line: netron {model_file}")
            print(f"2. Python: import netron; netron.start('{model_file}')")
            print("3. Web browser: Visit https://netron.app and upload the file")
            
            if args.lora_only and args.format == "onnx":
                print(f"\nTip: The 'lora_architecture.onnx' file shows the computational graph")
                print(f"     The 'lora_weights.onnx' file shows just the weight parameters")
        else:
            print("Warning: No model files found in output directory")
            # List what files were actually created
            created_files = list(output_dir.glob("*"))
            if created_files:
                print("Files created:")
                for f in created_files:
                    print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    else:
        print("Failed to process checkpoint")

if __name__ == "__main__":
    main()