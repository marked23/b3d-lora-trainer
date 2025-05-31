# inference.py

import argparse
import torch
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

class LoRAInference:
    """Inference class for LoRA fine-tuned StarCoder2 model."""
    
    def __init__(
        self,
        base_model_name: str = "bigcode/starcoder2-7b",
        lora_checkpoint_path: Optional[str] = None,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype
        
        print(f"Loading tokenizer: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading base model: {base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
            use_cache=True,
        )
        
        # Load LoRA weights if provided
        if lora_checkpoint_path:
            print(f"Loading LoRA weights from: {lora_checkpoint_path}")
            self.model = PeftModel.from_pretrained(
                self.base_model,
                lora_checkpoint_path,
                torch_dtype=torch_dtype,
            )
        else:
            print("Using base model without LoRA")
            self.model = self.base_model
        
        self.model.eval()
        print(f"Model loaded on device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def generate_code(
        self,
        prompt: str,
        max_length: int = 512,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate code completion from prompt."""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length - max_new_tokens,
        ).to(self.device)
        
        # Set up stop criteria
        if stop_sequences is None:
            stop_sequences = ["\n\n", "```", "# Test", "# Example"]
        
        # Generate
        with torch.no_grad():
            start_time = time.time()
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            generation_time = time.time() - start_time
        
        # Decode outputs
        generated_texts = []
        input_length = inputs["input_ids"].shape[1]
        
        for output in outputs:
            # Extract only the generated part
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Apply stop sequences
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
            
            generated_texts.append(generated_text.strip())
        
        print(f"Generation time: {generation_time:.2f}s")
        return generated_texts
    
    def compare_models(
        self,
        prompts: List[str],
        lora_checkpoint_path: str,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """Compare base model vs LoRA fine-tuned model."""
        
        results = {
            "prompts": prompts,
            "base_model_outputs": [],
            "lora_model_outputs": [],
            "generation_config": generation_kwargs
        }
        
        print("Generating with base model...")
        for prompt in prompts:
            base_output = self.generate_code(prompt, **generation_kwargs)
            results["base_model_outputs"].append(base_output[0])
        
        # Load LoRA model
        print(f"Loading LoRA model from: {lora_checkpoint_path}")
        lora_model = PeftModel.from_pretrained(
            self.base_model,
            lora_checkpoint_path,
            torch_dtype=self.torch_dtype,
        )
        lora_model.eval()
        
        # Temporarily replace model
        original_model = self.model
        self.model = lora_model
        
        print("Generating with LoRA model...")
        for prompt in prompts:
            lora_output = self.generate_code(prompt, **generation_kwargs)
            results["lora_model_outputs"].append(lora_output[0])
        
        # Restore original model
        self.model = original_model
        
        return results

def load_test_prompts() -> List[str]:
    """Load test prompts for build123d parameter evaluation."""
    return [
        # Box-related prompts
        "# Create a box with length 20, width 15, height 10\nwith BuildPart() as part:\n    ",
        
        # Cylinder-related prompts  
        "# Create a cylinder with radius 8 and height 12\nwith BuildPart() as part:\n    ",
        
        # Hole-related prompts
        "# Create a box and add a hole with radius 3\nwith BuildPart() as part:\n    Box(20, 15, 10)\n    ",
        
        # Complex operations
        "# Create a box, add a cylinder, then fillet the edges\nwith BuildPart() as part:\n    Box(20, 15, 10)\n    Cylinder(radius=5, height=15)\n    ",
        
        # 2D sketch operations
        "# Create a rectangle and a circle in a sketch\nwith BuildSketch() as sketch:\n    ",
        
        # Counter operations
        "# Create a counterbore hole with radius 4, counterbore radius 8, depth 3\nwith BuildPart() as part:\n    Box(30, 20, 15)\n    ",
        
        # Location contexts
        "# Create multiple cylinders using polar locations\nwith BuildPart() as part:\n    with PolarLocations(radius=20, count=6):\n        ",
        
        # Selectors and operations
        "# Create a box and fillet specific edges\nwith BuildPart() as part:\n    Box(25, 20, 8)\n    fillet(",
    ]

def analyze_parameter_accuracy(generated_code: str) -> Dict[str, Any]:
    """Analyze the accuracy of parameters in generated code."""
    
    # Define correct parameter patterns for build123d
    correct_patterns = {
        "Box": r"Box\s*\(\s*(?:length\s*=\s*)?\d+(?:\.\d+)?\s*,\s*(?:width\s*=\s*)?\d+(?:\.\d+)?\s*,\s*(?:height\s*=\s*)?\d+(?:\.\d+)?",
        "Cylinder": r"Cylinder\s*\(\s*(?:radius\s*=\s*)?\d+(?:\.\d+)?\s*,\s*(?:height\s*=\s*)?\d+(?:\.\d+)?",
        "Hole": r"Hole\s*\(\s*(?:radius\s*=\s*)?\d+(?:\.\d+)?(?:\s*,\s*(?:depth\s*=\s*)?\d+(?:\.\d+)?)?\s*\)",
        "Circle": r"Circle\s*\(\s*(?:radius\s*=\s*)?\d+(?:\.\d+)?\s*\)",
        "Rectangle": r"Rectangle\s*\(\s*(?:width\s*=\s*)?\d+(?:\.\d+)?\s*,\s*(?:height\s*=\s*)?\d+(?:\.\d+)?\s*\)",
    }
    
    # Common incorrect parameters to detect
    incorrect_patterns = {
        "Box_wrong": r"Box\s*\([^)]*(?:center|material|thickness|depth|size|x|y|z)\s*=",
        "Cylinder_wrong": r"Cylinder\s*\([^)]*(?:diameter|center|length|width|size)\s*=",
        "Hole_wrong": r"Hole\s*\([^)]*(?:diameter|center_x|center_y|thread|size)\s*=",
        "Circle_wrong": r"Circle\s*\([^)]*(?:diameter|center|size|radius_x|radius_y)\s*=",
        "Rectangle_wrong": r"Rectangle\s*\([^)]*(?:length|size|x|y|center)\s*=",
    }
    
    analysis = {
        "correct_usage": [],
        "incorrect_usage": [],
        "hallucinated_parameters": [],
        "accuracy_score": 0.0
    }
    
    total_constructs = 0
    correct_constructs = 0
    
    # Check for correct patterns
    for construct, pattern in correct_patterns.items():
        matches = re.findall(pattern, generated_code, re.IGNORECASE)
        if matches:
            total_constructs += len(matches)
            correct_constructs += len(matches)
            analysis["correct_usage"].extend([(construct, match) for match in matches])
    
    # Check for incorrect patterns
    for construct, pattern in incorrect_patterns.items():
        matches = re.findall(pattern, generated_code, re.IGNORECASE)
        if matches:
            total_constructs += len(matches)
            base_construct = construct.replace("_wrong", "")
            analysis["incorrect_usage"].extend([(base_construct, match) for match in matches])
    
    # Calculate accuracy score
    if total_constructs > 0:
        analysis["accuracy_score"] = correct_constructs / total_constructs
    
    return analysis

def run_evaluation_suite(
    inference_engine: LoRAInference,
    test_prompts: List[str],
    lora_checkpoint_path: Optional[str] = None,
    output_file: str = "evaluation_results.json"
) -> Dict[str, Any]:
    """Run comprehensive evaluation suite."""
    
    print("=" * 60)
    print("STARTING LORA EVALUATION SUITE")
    print("=" * 60)
    
    generation_config = {
        "max_new_tokens": 150,
        "temperature": 0.3,  # Lower temperature for more deterministic output
        "top_p": 0.9,
        "do_sample": True,
        "num_return_sequences": 1,
    }
    
    results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_config": {
            "base_model": "bigcode/starcoder2-7b",
            "lora_checkpoint": lora_checkpoint_path,
            "generation_config": generation_config,
        },
        "test_results": []
    }
    
    if lora_checkpoint_path:
        # Compare base vs LoRA
        comparison_results = inference_engine.compare_models(
            test_prompts, 
            lora_checkpoint_path, 
            **generation_config
        )
        
        for i, prompt in enumerate(test_prompts):
            base_output = comparison_results["base_model_outputs"][i]
            lora_output = comparison_results["lora_model_outputs"][i]
            
            base_analysis = analyze_parameter_accuracy(base_output)
            lora_analysis = analyze_parameter_accuracy(lora_output)
            
            test_result = {
                "prompt": prompt,
                "base_model": {
                    "output": base_output,
                    "analysis": base_analysis
                },
                "lora_model": {
                    "output": lora_output,
                    "analysis": lora_analysis
                },
                "improvement": lora_analysis["accuracy_score"] - base_analysis["accuracy_score"]
            }
            
            results["test_results"].append(test_result)
            
            print(f"\n--- Test {i+1} ---")
            print(f"Prompt: {prompt[:50]}...")
            print(f"Base accuracy: {base_analysis['accuracy_score']:.2f}")
            print(f"LoRA accuracy: {lora_analysis['accuracy_score']:.2f}")
            print(f"Improvement: {test_result['improvement']:+.2f}")
    
    else:
        # Single model evaluation
        for i, prompt in enumerate(test_prompts):
            output = inference_engine.generate_code(prompt, **generation_config)[0]
            analysis = analyze_parameter_accuracy(output)
            
            test_result = {
                "prompt": prompt,
                "output": output,
                "analysis": analysis
            }
            
            results["test_results"].append(test_result)
            
            print(f"\n--- Test {i+1} ---")
            print(f"Prompt: {prompt[:50]}...")
            print(f"Output: {output[:100]}...")
            print(f"Accuracy: {analysis['accuracy_score']:.2f}")
    
    # Calculate overall statistics
    if lora_checkpoint_path:
        base_scores = [r["base_model"]["analysis"]["accuracy_score"] for r in results["test_results"]]
        lora_scores = [r["lora_model"]["analysis"]["accuracy_score"] for r in results["test_results"]]
        
        results["summary"] = {
            "average_base_accuracy": sum(base_scores) / len(base_scores),
            "average_lora_accuracy": sum(lora_scores) / len(lora_scores),
            "average_improvement": sum(lora_scores) / len(lora_scores) - sum(base_scores) / len(base_scores),
            "tests_with_improvement": sum(1 for r in results["test_results"] if r["improvement"] > 0),
            "total_tests": len(results["test_results"])
        }
        
        print(f"\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Average Base Accuracy: {results['summary']['average_base_accuracy']:.3f}")
        print(f"Average LoRA Accuracy: {results['summary']['average_lora_accuracy']:.3f}")
        print(f"Average Improvement: {results['summary']['average_improvement']:+.3f}")
        print(f"Tests with Improvement: {results['summary']['tests_with_improvement']}/{results['summary']['total_tests']}")
    
    else:
        scores = [r["analysis"]["accuracy_score"] for r in results["test_results"]]
        results["summary"] = {
            "average_accuracy": sum(scores) / len(scores),
            "total_tests": len(results["test_results"])
        }
        
        print(f"\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Average Accuracy: {results['summary']['average_accuracy']:.3f}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return results

def interactive_demo(inference_engine: LoRAInference):
    """Interactive demonstration of the model."""
    
    print("\n" + "=" * 60)
    print("INTERACTIVE DEMO MODE")
    print("=" * 60)
    print("Enter build123d prompts to see the model's completions.")
    print("Type 'quit' to exit, 'examples' to see sample prompts.")
    
    while True:
        user_input = input("\nPrompt: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.lower() == 'examples':
            print("\nExample prompts:")
            for i, prompt in enumerate(load_test_prompts()[:5], 1):
                print(f"{i}. {prompt}")
            continue
        
        if not user_input:
            continue
        
        print("\nGenerating...")
        try:
            outputs = inference_engine.generate_code(
                user_input,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True
            )
            
            print(f"\nCompletion:")
            print("-" * 40)
            print(outputs[0])
            print("-" * 40)
            
            # Analyze the output
            analysis = analyze_parameter_accuracy(outputs[0])
            print(f"\nAccuracy Score: {analysis['accuracy_score']:.2f}")
            if analysis['correct_usage']:
                print("✓ Correct parameter usage found")
            if analysis['incorrect_usage']:
                print("✗ Incorrect parameter usage detected")
                
        except Exception as e:
            print(f"Error during generation: {e}")

def main():
    parser = argparse.ArgumentParser(description="LoRA Inference and Evaluation")
    
    parser.add_argument(
        "--base-model", 
        type=str, 
        default="bigcode/starcoder2-7b",
        help="Base model name or path"
    )
    parser.add_argument(
        "--lora-checkpoint", 
        type=str,
        help="Path to LoRA checkpoint directory"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["evaluate", "interactive", "compare"],
        default="evaluate",
        help="Inference mode"
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        default="evaluation_results.json",
        help="Output file for evaluation results"
    )
    parser.add_argument(
        "--custom-prompts", 
        type=str,
        help="Path to custom prompts JSON file"
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference_engine = LoRAInference(
        base_model_name=args.base_model,
        lora_checkpoint_path=args.lora_checkpoint if args.mode != "compare" else None,
        device=args.device,
    )
    
    # Load test prompts
    if args.custom_prompts:
        with open(args.custom_prompts, 'r') as f:
            test_prompts = json.load(f)
    else:
        test_prompts = load_test_prompts()
    
    # Run based on mode
    if args.mode == "evaluate":
        run_evaluation_suite(
            inference_engine, 
            test_prompts,
            output_file=args.output_file
        )
    
    elif args.mode == "compare":
        if not args.lora_checkpoint:
            raise ValueError("--lora-checkpoint required for compare mode")
        
        run_evaluation_suite(
            inference_engine, 
            test_prompts,
            lora_checkpoint_path=args.lora_checkpoint,
            output_file=args.output_file
        )
    
    elif args.mode == "interactive":
        interactive_demo(inference_engine)

if __name__ == "__main__":
    main()