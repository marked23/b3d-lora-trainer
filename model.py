# model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Any, Union
from peft import PeftModel, PeftMixedModel


class LoRAModel(nn.Module):
    """Wrapper for StarCoder2 model with LoRA."""
    
    def __init__(
        self,
        model_name: str = "bigcode/starcoder2-7b",
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
        base_model: Any = None,
    ):
        super().__init__()
        
        # Use provided base model or load new one
        if base_model is not None:
            self.base_model = base_model
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        
        # Configure LoRA
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        # Apply LoRA
        self.model: Union[PeftModel, PeftMixedModel] = get_peft_model(self.base_model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def generate(self, **kwargs):
        return self.model.generate(**kwargs)
    
    def save_pretrained(self, save_path):
        """Save only the LoRA weights."""
        self.model.save_pretrained(save_path)
    
    def merge_and_save(self, save_path):
        """Merge LoRA weights with base model and save."""
        merged_model: AutoModelForCausalLM = self.model.merge_and_unload() # type: ignore
        merged_model.save_pretrained(save_path) # type: ignore