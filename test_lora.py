# test_lora.py
"""
Quick testing script for LoRA inference with build123d parameter validation.
This script provides easy commands to test your fine-tuned model.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from inference import LoRAInference, analyze_parameter_accuracy, load_test_prompts

def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> str:
    """Find the most recent checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Look for subdirectories with training timestamps
    checkpoint_dirs = [d for d in checkpoint_path.iterdir() if d.is_dir()]
    
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint subdirectories found in {checkpoint_dir}")
    
    # Sort by modification time and get the latest
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
    
    # Check if peft subdirectory exists
    peft_dir = latest_checkpoint.joinpath("peft")
    if peft_dir.exists():
        print(f"Found latest checkpoint: {latest_checkpoint} (using peft subdirectory)")
        return str(peft_dir)
    else:
        print(f"Found latest checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)

def quick_test(lora_checkpoint: str, num_tests: int = 3):
    """Run a quick test with a few prompts."""
    
    print("🚀 Quick LoRA Test")
    print("=" * 50)
    
    # Initialize inference
    inference_engine = LoRAInference(
        lora_checkpoint_path=lora_checkpoint,
        device="auto"
    )
    
    # Get a subset of test prompts
    test_prompts = load_test_prompts()[:num_tests]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{num_tests} ---")
        print(f"Prompt: {prompt}")
        print("Generated:")
        
        try:
            output = inference_engine.generate_code(
                prompt,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True
            )[0]
            
            print(f"  {output}")
            
            # Quick analysis
            analysis = analyze_parameter_accuracy(output)
            accuracy = analysis['accuracy_score']
            
            if accuracy > 0.8:
                print(f"  ✅ Good accuracy: {accuracy:.2f}")
            elif accuracy > 0.5:
                print(f"  ⚠️  Moderate accuracy: {accuracy:.2f}")
            else:
                print(f"  ❌ Low accuracy: {accuracy:.2f}")
                
            if analysis['incorrect_usage']:
                print(f"  🚨 Found parameter issues: {len(analysis['incorrect_usage'])}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Quick test completed!")

def compare_models(lora_checkpoint: str, num_tests: int = 5):
    """Compare base model vs LoRA model performance."""
    
    print("🔍 Model Comparison")
    print("=" * 50)
    
    # Initialize with base model only
    inference_engine = LoRAInference(device="auto")
    
    # Get test prompts
    test_prompts = load_test_prompts()[:num_tests]
    
    comparison_results = inference_engine.compare_models(
        test_prompts,
        lora_checkpoint,
        max_new_tokens=100,
        temperature=0.3,
        do_sample=True
    )
    
    print(f"\n📊 Comparison Results:")
    print("-" * 50)
    
    total_base_accuracy = 0
    total_lora_accuracy = 0
    improvements = 0
    
    for i, prompt in enumerate(test_prompts):
        base_output = comparison_results["base_model_outputs"][i]
        lora_output = comparison_results["lora_model_outputs"][i]
        
        base_analysis = analyze_parameter_accuracy(base_output)
        lora_analysis = analyze_parameter_accuracy(lora_output)
        
        base_accuracy = base_analysis['accuracy_score']
        lora_accuracy = lora_analysis['accuracy_score']
        improvement = lora_accuracy - base_accuracy
        
        total_base_accuracy += base_accuracy
        total_lora_accuracy += lora_accuracy
        
        if improvement > 0:
            improvements += 1
        
        print(f"\nTest {i+1}:")
        print(f"  Prompt: {prompt[:60]}...")
        print(f"  Base accuracy:  {base_accuracy:.2f}")
        print(f"  LoRA accuracy:  {lora_accuracy:.2f}")
        print(f"  Improvement:    {improvement:+.2f}")
        
        if improvement > 0.1:
            print(f"  📈 Significant improvement!")
        elif improvement < -0.1:
            print(f"  📉 Performance decreased")
    
    avg_base = total_base_accuracy / num_tests
    avg_lora = total_lora_accuracy / num_tests
    avg_improvement = avg_lora - avg_base
    
    print(f"\n🎯 Summary:")
    print(f"  Average base accuracy:  {avg_base:.3f}")
    print(f"  Average LoRA accuracy:  {avg_lora:.3f}")
    print(f"  Average improvement:    {avg_improvement:+.3f}")
    print(f"  Tests with improvement: {improvements}/{num_tests}")
    
    if avg_improvement > 0.05:
        print(f"  ✅ LoRA shows meaningful improvement!")
    elif avg_improvement > 0:
        print(f"  ⚠️  LoRA shows slight improvement")
    else:
        print(f"  ❌ LoRA performance is not better than base model")

def interactive_test(lora_checkpoint: str):
    """Run an interactive testing session."""
    
    print("💬 Interactive LoRA Testing")
    print("=" * 50)
    print("Enter build123d code prompts to test the model.")
    print("Commands:")
    print("  'examples' - Show example prompts")
    print("  'quit' - Exit")
    print("  'accuracy' - Check accuracy of last output")
    
    inference_engine = LoRAInference(
        lora_checkpoint_path=lora_checkpoint,
        device="auto"
    )
    
    last_output = ""
    
    while True:
        user_input = input("\n🎯 Prompt: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.lower() == 'examples':
            print("\nExample prompts:")
            for i, prompt in enumerate(load_test_prompts()[:5], 1):
                print(f"{i}. {prompt[:80]}...")
            continue
        
        if user_input.lower() == 'accuracy':
            if last_output:
                analysis = analyze_parameter_accuracy(last_output)
                print(f"\nLast output accuracy: {analysis['accuracy_score']:.2f}")
                if analysis['correct_usage']:
                    print(f"✅ Correct usage: {len(analysis['correct_usage'])}")
                if analysis['incorrect_usage']:
                    print(f"❌ Incorrect usage: {len(analysis['incorrect_usage'])}")
            else:
                print("No previous output to analyze")
            continue
        
        if not user_input:
            continue
        
        try:
            output = inference_engine.generate_code(
                user_input,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True
            )[0]
            
            print(f"\n📝 Generated:")
            print("-" * 30)
            print(output)
            print("-" * 30)
            
            last_output = output
            
            # Quick accuracy check
            analysis = analyze_parameter_accuracy(output)
            accuracy = analysis['accuracy_score']
            print(f"Accuracy: {accuracy:.2f}", end="")
            
            if accuracy > 0.8:
                print(" ✅")
            elif accuracy > 0.5:
                print(" ⚠️")
            else:
                print(" ❌")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def validate_build123d_syntax(code: str) -> dict:
    """Validate build123d syntax patterns in generated code."""
    
    import re
    
    # Common build123d patterns
    patterns = {
        'context_managers': r'with (BuildPart|BuildSketch|BuildLine)\(\) as \w+:',
        'basic_shapes': r'(Box|Cylinder|Sphere|Cone)\s*\(',
        'holes': r'(Hole|CounterBoreHole|CounterSinkHole)\s*\(',
        '2d_shapes': r'(Circle|Rectangle|Polygon)\s*\(',
        'operations': r'(fillet|chamfer|extrude|revolve)\s*\(',
        'selectors': r'\.(edges|faces|vertices|solids)\(\)',
        'locations': r'with (Locations|GridLocations|PolarLocations)\s*\(',
    }
    
    results = {}
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, code, re.IGNORECASE)
        results[pattern_name] = len(matches)
    
    return results

def comprehensive_test(lora_checkpoint: str):
    """Run a comprehensive test suite."""
    
    print("🔬 Comprehensive LoRA Testing")
    print("=" * 50)
    
    # Run evaluation suite
    from inference import run_evaluation_suite
    
    inference_engine = LoRAInference(device="auto")
    test_prompts = load_test_prompts()
    
    results = run_evaluation_suite(
        inference_engine,
        test_prompts,
        lora_checkpoint_path=lora_checkpoint,
        output_file=f"comprehensive_test_{Path(lora_checkpoint).name}.json"
    )
    
    # Additional syntax validation
    print(f"\n🔍 Syntax Validation:")
    print("-" * 30)
    
    syntax_scores = []
    for result in results["test_results"]:
        lora_output = result["lora_model"]["output"]
        syntax_analysis = validate_build123d_syntax(lora_output)
        
        # Calculate syntax score based on patterns found
        pattern_count = sum(syntax_analysis.values())
        syntax_scores.append(pattern_count)
        
        print(f"Output patterns found: {pattern_count}")
        for pattern, count in syntax_analysis.items():
            if count > 0:
                print(f"  {pattern}: {count}")
    
    avg_syntax_score = sum(syntax_scores) / len(syntax_scores) if syntax_scores else 0
    print(f"\nAverage syntax patterns per output: {avg_syntax_score:.1f}")
    
    return results

def check_parameter_hallucinations(lora_checkpoint: str):
    """Specifically test for parameter hallucinations."""
    
    print("🚨 Parameter Hallucination Check")
    print("=" * 50)
    
    # Specific prompts designed to trigger parameter hallucinations
    hallucination_prompts = [
        "# Create a box with custom dimensions\nwith BuildPart() as part:\n    Box(",
        "# Make a cylinder with specific size\nwith BuildPart() as part:\n    Cylinder(",
        "# Add a hole to the center\nwith BuildPart() as part:\n    Box(20, 15, 10)\n    Hole(",
        "# Create a circle in sketch\nwith BuildSketch() as sketch:\n    Circle(",
        "# Make a rectangle with dimensions\nwith BuildSketch() as sketch:\n    Rectangle(",
    ]
    
    inference_engine = LoRAInference(
        lora_checkpoint_path=lora_checkpoint,
        device="auto"
    )
    
    hallucination_count = 0
    total_tests = len(hallucination_prompts)
    
    for i, prompt in enumerate(hallucination_prompts, 1):
        print(f"\nTest {i}/{total_tests}")
        print(f"Prompt: {prompt.strip()}")
        
        output = inference_engine.generate_code(
            prompt,
            max_new_tokens=50,
            temperature=0.1,  # Very low temperature for deterministic output
            do_sample=True
        )[0]
        
        print(f"Output: {output}")
        
        analysis = analyze_parameter_accuracy(output)
        
        if analysis['incorrect_usage']:
            hallucination_count += 1
            print(f"❌ HALLUCINATION DETECTED!")
            for construct, usage in analysis['incorrect_usage']:
                print(f"   {construct}: {usage}")
        else:
            print(f"✅ No hallucinations detected")
        
        print(f"Accuracy: {analysis['accuracy_score']:.2f}")
    
    hallucination_rate = hallucination_count / total_tests
    print(f"\n📊 Hallucination Summary:")
    print(f"Tests with hallucinations: {hallucination_count}/{total_tests}")
    print(f"Hallucination rate: {hallucination_rate:.2%}")
    
    if hallucination_rate < 0.1:
        print("✅ Excellent! Very low hallucination rate")
    elif hallucination_rate < 0.3:
        print("⚠️  Moderate hallucination rate - could be improved")
    else:
        print("❌ High hallucination rate - needs more training")

def benchmark_performance(lora_checkpoint: str):
    """Benchmark inference performance."""
    
    print("⚡ Performance Benchmark")
    print("=" * 50)
    
    inference_engine = LoRAInference(
        lora_checkpoint_path=lora_checkpoint,
        device="auto"
    )
    
    test_prompt = "# Create a box with holes\nwith BuildPart() as part:\n    Box(20, 15, 10)\n    "
    
    # Test different configurations
    configs = [
        {"name": "Fast", "max_new_tokens": 50, "temperature": 0.1},
        {"name": "Balanced", "max_new_tokens": 100, "temperature": 0.3},
        {"name": "Creative", "max_new_tokens": 150, "temperature": 0.7},
    ]
    
    for config in configs:
        print(f"\n{config['name']} Configuration:")
        
        import time
        start_time = time.time()
        
        outputs = inference_engine.generate_code(
            test_prompt,
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            do_sample=True,
            num_return_sequences=3
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"  Generation time: {generation_time:.2f}s")
        print(f"  Tokens per second: {(config['max_new_tokens'] * 3) / generation_time:.1f}")
        
        # Check quality of outputs
        accuracies = []
        for output in outputs:
            analysis = analyze_parameter_accuracy(output)
            accuracies.append(analysis['accuracy_score'])
        
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"  Average accuracy: {avg_accuracy:.2f}")
        print(f"  Best output: {max(accuracies):.2f}")

def main():
    parser = argparse.ArgumentParser(description="Test LoRA fine-tuned StarCoder2 model")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to LoRA checkpoint (if not provided, will find latest)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "compare", "interactive", "comprehensive", "hallucination", "benchmark", "all"],
        default="quick",
        help="Test mode to run"
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=5,
        help="Number of test prompts to use"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoints"
    )
    
    args = parser.parse_args()
    
    # Find checkpoint
    if args.checkpoint:
        lora_checkpoint = args.checkpoint
    else:
        try:
            lora_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("Please provide --checkpoint or ensure checkpoints exist in --checkpoint-dir")
            sys.exit(1)
    
    print(f"Using checkpoint: {lora_checkpoint}")
    
    # Run tests based on mode
    try:
        if args.mode == "quick":
            quick_test(lora_checkpoint, args.num_tests)
        
        elif args.mode == "compare":
            compare_models(lora_checkpoint, args.num_tests)
        
        elif args.mode == "interactive":
            interactive_test(lora_checkpoint)
        
        elif args.mode == "comprehensive":
            comprehensive_test(lora_checkpoint)
        
        elif args.mode == "hallucination":
            check_parameter_hallucinations(lora_checkpoint)
        
        elif args.mode == "benchmark":
            benchmark_performance(lora_checkpoint)
        
        elif args.mode == "all":
            print("🧪 Running All Tests")
            print("=" * 60)
            
            quick_test(lora_checkpoint, 3)
            print("\n" + "="*60)
            
            compare_models(lora_checkpoint, 3)
            print("\n" + "="*60)
            
            check_parameter_hallucinations(lora_checkpoint)
            print("\n" + "="*60)
            
            benchmark_performance(lora_checkpoint)
            print("\n" + "="*60)
            
            comprehensive_test(lora_checkpoint)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()