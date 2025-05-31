import torch
from torch.utils.data import Dataset
import json
import re
from typing import List, Dict, Tuple, Optional, Any, Union, TypedDict
from dataclasses import dataclass
import random


class SampleDict(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    method_name: str
    context: str


@dataclass
class ParameterExample:
    """Represents a single parameter usage example"""
    method_name: str
    context: str  # e.g., "BuildPart", "BuildSketch"
    parameters: Dict[str, Any]
    code_snippet: str
    completion_target: str  # what should be completed
    valid_params: List[str]  # valid parameter names
    invalid_params: List[str]  # parameters that don't exist


class CodeParametersDataset(Dataset):
    """
    Dataset for training LoRA on build123d parameter accuracy.
    Focuses on preventing parameter hallucination by showing correct API usage.
    """
    
    def __init__(
        self, 
        parameter_matrix_path: str,
        tokenizer,
        max_length: int = 512,
        completion_ratio: float = 0.7,
        include_context: bool = True,
        augment_negatives: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.completion_ratio = completion_ratio
        self.include_context = include_context
        self.augment_negatives = augment_negatives
        
        # Parse the parameter matrix
        self.examples = self._parse_parameter_matrix(parameter_matrix_path)
        
        # Generate completion examples
        self.training_samples = self._generate_training_samples()
        
    def _parse_parameter_matrix(self, matrix_path: str) -> List[ParameterExample]:
        """Parse the parameter matrix markdown into structured examples"""
        examples = []
        
        # Load the parameter matrix
        with open(matrix_path, 'r') as f:
            content = f.read()
        
        # Extract method definitions using regex
        method_sections = re.findall(
            r'### (\w+)\n```python\n(.*?)\n```', 
            content, 
            re.DOTALL
        )
        
        for method_name, code_block in method_sections:
            examples.extend(self._parse_method_section(method_name, code_block))
            
        return examples
    
    def _parse_method_section(self, method_name: str, code_block: str) -> List[ParameterExample]:
        """Parse a single method section into multiple examples"""
        examples = []
        lines = code_block.strip().split('\n')
        
        # Extract signature comment
        signature_line = None
        valid_params = []
        invalid_params = []
        
        for line in lines:
            if line.startswith('# EXACT SIGNATURE:'):
                signature_line = line
                # Extract parameter names from signature
                sig_match = re.search(r'\((.*?)\)', signature_line)
                if sig_match:
                    params = sig_match.group(1).split(',')
                    valid_params = [p.split('=')[0].strip() for p in params if p.strip()]
                    
            elif line.startswith('# DOES NOT EXIST:'):
                # Extract invalid parameters
                invalid_part = line.replace('# DOES NOT EXIST:', '').strip()
                invalid_params = [p.strip() for p in invalid_part.split(',')]
        
        # Extract code examples
        for line in lines:
            if not line.startswith('#') and method_name in line and line.strip():
                # Determine context from the line
                context = self._infer_context(line)
                
                # Parse parameters from the line
                params = self._extract_parameters(line)
                
                example = ParameterExample(
                    method_name=method_name,
                    context=context,
                    parameters=params,
                    code_snippet=line.strip(),
                    completion_target=line.strip(),
                    valid_params=valid_params,
                    invalid_params=invalid_params
                )
                examples.append(example)
                
        return examples
    
    def _infer_context(self, code_line: str) -> str:
        """Infer the build context from the code line"""
        if any(obj in code_line for obj in ['Box', 'Cylinder', 'Sphere', 'Cone']):
            return 'BuildPart'
        elif any(obj in code_line for obj in ['Circle', 'Rectangle', 'RegularPolygon']):
            return 'BuildSketch'
        elif any(obj in code_line for obj in ['Line', 'Arc', 'Spline']):
            return 'BuildLine'
        else:
            return 'BuildPart'  # default
    
    def _extract_parameters(self, code_line: str) -> Dict[str, Any]:
        """Extract parameter dictionary from code line"""
        params = {}
        
        # Simple regex to extract parameter assignments
        param_matches = re.findall(r'(\w+)=([^,)]+)', code_line)
        for param_name, param_value in param_matches:
            params[param_name] = param_value.strip()
            
        return params
    
    def _generate_training_samples(self) -> List[Dict[str, str]]:
        """Generate training samples for parameter completion"""
        samples = []
        
        for example in self.examples:
            # Generate completion samples
            samples.extend(self._create_completion_samples(example))
            
            # Generate context samples
            if self.include_context:
                samples.extend(self._create_context_samples(example))
            
            # Generate negative samples (showing what NOT to do)
            if self.augment_negatives:
                samples.extend(self._create_negative_samples(example))
        
        return samples
    
    def _create_completion_samples(self, example: ParameterExample) -> List[Dict[str, str]]:
        """Create parameter completion training samples"""
        samples = []
        
        # Basic completion: partial parameters -> complete parameters
        code = example.code_snippet
        
        # Create variations by truncating at different points
        variations = [
            # Just method name
            f"{example.method_name}(",
            # Method name with first parameter
            self._truncate_at_param(code, 1),
            # Method name with first two parameters
            self._truncate_at_param(code, 2),
        ]
        
        for variation in variations:
            if variation and variation != code:
                samples.append({
                    'input': variation,
                    'target': code,
                    'context': example.context,
                    'method': example.method_name
                })
        
        return samples
    
    def _create_context_samples(self, example: ParameterExample) -> List[Dict[str, str]]:
        """Create context-aware samples showing proper usage within builders"""
        samples = []
        
        context_templates = {
            'BuildPart': [
                f"with BuildPart() as part:\n    {example.code_snippet}",
                f"with BuildPart() as component:\n    Box(10, 10, 5)\n    {example.code_snippet}",
            ],
            'BuildSketch': [
                f"with BuildSketch() as sketch:\n    {example.code_snippet}",
                f"with BuildPart() as part:\n    with BuildSketch() as sketch:\n        {example.code_snippet}",
            ],
            'BuildLine': [
                f"with BuildLine() as line:\n    {example.code_snippet}",
            ]
        }
        
        templates = context_templates.get(example.context, [])
        for template in templates:
            # Create completion from context
            lines = template.split('\n')
            partial = '\n'.join(lines[:-1]) + '\n    ' + example.method_name + '('
            
            samples.append({
                'input': partial,
                'target': template,
                'context': example.context,
                'method': example.method_name
            })
        
        return samples
    
    def _create_negative_samples(self, example: ParameterExample) -> List[Dict[str, str]]:
        """Create negative samples showing incorrect parameter usage"""
        samples = []
        
        if not example.invalid_params:
            return samples
        
        # Create examples with invalid parameters
        base_call = f"{example.method_name}("
        
        for invalid_param in example.invalid_params[:2]:  # Limit to 2 invalid params
            # Create a "wrong" completion with invalid parameter
            wrong_completion = f"{example.method_name}({invalid_param}=10"
            correct_completion = example.code_snippet
            
            # The model should learn to prefer the correct completion
            samples.append({
                'input': base_call,
                'target': correct_completion,
                'negative': wrong_completion,
                'context': example.context,
                'method': example.method_name
            })
        
        return samples
    
    def _truncate_at_param(self, code: str, param_index: int) -> str:
        """Truncate code at the nth parameter for completion training"""
        if '(' not in code:
            return code
            
        method_part, params_part = code.split('(', 1)
        params_part = params_part.rstrip(')')
        
        if not params_part:
            return method_part + '('
        
        # Split parameters (simple split, might need more sophisticated parsing)
        params = []
        paren_depth = 0
        current_param = ""
        
        for char in params_part:
            if char == ',' and paren_depth == 0:
                params.append(current_param.strip())
                current_param = ""
            else:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                current_param += char
        
        if current_param.strip():
            params.append(current_param.strip())
        
        if param_index >= len(params):
            return code
        
        # Return truncated version
        truncated_params = params[:param_index]
        if truncated_params:
            return method_part + '(' + ', '.join(truncated_params) + ', '
        else:
            return method_part + '('
    
    def __len__(self) -> int:
        return len(self.training_samples)
    
    def __getitem__(self, idx: int) -> SampleDict:
        sample = self.training_samples[idx]
        
        # Tokenize input and target
        input_text = sample['input']
        target_text = sample['target']
        
        # For causal language modeling, we want input + target as one sequence
        full_text = input_text + target_text[len(input_text):]
        
        # Tokenize
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = tokens['input_ids'].clone()
        
        # Mask the input portion so loss is only computed on completion
        input_tokens = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        input_length = len(input_tokens['input_ids'][0])
        labels[0, :input_length] = -100  # Ignore loss for input portion
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'method_name': sample.get('method', ''),
            'context': sample.get('context', '')
        }


class CodeParametersDatasetFromDict(Dataset):
    """
    Alternative dataset class for when you have pre-parsed parameter data
    """
    
    def __init__(
        self,
        parameter_data: List[Dict],
        tokenizer,
        max_length: int = 512,
        template_style: str = "completion"  # "completion" or "context"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_style = template_style
        self.samples = self._process_parameter_data(parameter_data)
    
    def _process_parameter_data(self, data: List[Dict]) -> List[Dict]:
        """Process raw parameter data into training samples"""
        samples = []
        
        for item in data:
            if self.template_style == "completion":
                samples.extend(self._create_completion_variations(item))
            elif self.template_style == "context":
                samples.extend(self._create_context_variations(item))
        
        return samples
    
    def _create_completion_variations(self, item: Dict) -> List[Dict]:
        """Create parameter completion variations"""
        method = item.get('method', '')
        params = item.get('parameters', {})
        context = item.get('context', 'BuildPart')
        
        variations = []
        
        # Full method call
        param_strs = [f"{k}={v}" for k, v in params.items()]
        full_call = f"{method}({', '.join(param_strs)})"
        
        # Create truncated versions
        for i in range(len(param_strs) + 1):
            if i == 0:
                input_text = f"{method}("
            else:
                partial_params = ', '.join(param_strs[:i])
                input_text = f"{method}({partial_params}, "
            
            variations.append({
                'input': input_text,
                'target': full_call,
                'method': method,
                'context': context
            })
        
        return variations
    
    def _create_context_variations(self, item: Dict) -> List[Dict]:
        """Create context-aware variations"""
        # Similar to above but with build context
        variations = self._create_completion_variations(item)
        
        context_variations = []
        for var in variations:
            context_template = f"with {item.get('context', 'BuildPart')}() as obj:\n    {var['input']}"
            context_target = f"with {item.get('context', 'BuildPart')}() as obj:\n    {var['target']}"
            
            context_variations.append({
                'input': context_template,
                'target': context_target,
                'method': var['method'],
                'context': var['context']
            })
        
        return context_variations
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        input_text = sample['input']
        target_text = sample['target']
        
        # Create the full sequence for causal LM
        completion_part = target_text[len(input_text):]
        full_text = input_text + completion_part
        
        # Tokenize
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels
        labels = tokens['input_ids'].clone()
        
        # Mask input portion
        input_tokens = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        input_length = len(input_tokens['input_ids'][0])
        labels[0, :input_length] = -100
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }


# # Usage example
# if __name__ == "__main__":
#     from transformers import AutoTokenizer
    
#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
#     tokenizer.pad_token = tokenizer.eos_token
    
#     # Create dataset from parameter matrix
#     dataset = CodeParametersDataset(
#         parameter_matrix_path="build123d_parameter_matrix.md",
#         tokenizer=tokenizer,
#         max_length=256,
#         include_context=True,
#         augment_negatives=True
#     )
    
#     print(f"Dataset size: {len(dataset)}")
    
#     # Check a sample
#     sample = dataset[0]
#     print("Sample input_ids shape:", sample['input_ids'].shape)
#     print("Sample method:", sample['method_name'])
#     print("Sample context:", sample['context'])
    
#     # Alternative: create from pre-parsed data
#     parameter_data = [
#         {
#             'method': 'Box',
#             'parameters': {'length': 10, 'width': 5, 'height': 3},
#             'context': 'BuildPart'
#         },
#         {
#             'method': 'Circle',
#             'parameters': {'radius': 10},
#             'context': 'BuildSketch'
#         }
#     ]
    
#     dataset2 = CodeParametersDatasetFromDict(
#         parameter_data=parameter_data,
#         tokenizer=tokenizer,
#         template_style="completion"
#     )
    
#     print(f"Alternative dataset size: {len(dataset2)}")