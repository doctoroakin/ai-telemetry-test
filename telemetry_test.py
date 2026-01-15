"""
Local Model Telemetry Experiment
Testing Phi-2 with full instrumentation
"""

import time
import psutil
import json
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model... this will take a minute...")

# Load Phi-2
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    torch_dtype=torch.float32  # CPU compatible
)

print("Model loaded!\n")

# Test cases
TEST_CASES = {
    'math': [
        "Calculate 847 * 923. Show your work.",
        "Solve for x: 5x + 12 = 47",
        "What is 23% of 850?",
    ],
    'creative': [
        "Write a haiku about robots.",
        "Describe a new color that doesn't exist.",
        "Invent a word for missing a place you've never been.",
    ],
    'factual': [
        "What is the capital of Brazil?",
        "When was the Declaration of Independence signed?",
        "Who painted the Mona Lisa?",
    ],
    'reasoning': [
        "If all cats are mammals, and all mammals breathe air, what can we conclude?",
        "Is this valid: All birds fly. Penguins are birds. So penguins fly.",
        "Compare 'knowing how' vs 'knowing that'.",
    ]
}

def capture_telemetry(task_type, prompt):
    """Run inference and capture everything"""
    
    process = psutil.Process()
    
    # BEFORE metrics
    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=0.1)
    start_memory = psutil.virtual_memory().percent
    start_process_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs['input_ids'].shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=input_length + 100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # AFTER metrics
    end_time = time.time()
    end_cpu = psutil.cpu_percent(interval=0.1)
    end_memory = psutil.virtual_memory().percent
    end_process_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    tokens_generated = outputs.shape[1] - input_length
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'task_type': task_type,
        'prompt': prompt,
        'output': output_text,
        'input_tokens': input_length,
        'output_tokens': tokens_generated,
        'total_tokens': outputs.shape[1],
        'latency_seconds': end_time - start_time,
        'tokens_per_second': tokens_generated / (end_time - start_time),
        'cpu_start': start_cpu,
        'cpu_end': end_cpu,
        'cpu_delta': end_cpu - start_cpu,
        'memory_start_pct': start_memory,
        'memory_end_pct': end_memory,
        'memory_delta_pct': end_memory - start_memory,
        'process_memory_start_mb': start_process_memory,
        'process_memory_end_mb': end_process_memory,
        'process_memory_delta_mb': end_process_memory - start_process_memory
    }
    
    return result

def run_experiment():
    """Run all tests"""
    
    results = []
    total = sum(len(cases) for cases in TEST_CASES.values())
    
    print(f"🔬 Starting experiment with {total} prompts\n")
    
    count = 0
    for task_type, prompts in TEST_CASES.items():
        print(f"\nTesting {task_type}...")
        for prompt in prompts:
            count += 1
            print(f"  [{count}/{total}] {prompt[:50]}...")
            result = capture_telemetry(task_type, prompt)
            results.append(result)
            print(f"      → {result['latency_seconds']:.1f}s, {result['tokens_per_second']:.1f} tok/s")
    
    # Save
    filename = f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {filename}\n")
    
    # Quick analysis
    print("Quick Summary:")
    for task_type in TEST_CASES.keys():
        task_results = [r for r in results if r['task_type'] == task_type]
        avg_latency = sum(r['latency_seconds'] for r in task_results) / len(task_results)
        avg_tps = sum(r['tokens_per_second'] for r in task_results) / len(task_results)
        avg_cpu = sum(r['cpu_delta'] for r in task_results) / len(task_results)
        print(f"  {task_type:12} → latency: {avg_latency:5.1f}s  |  tok/s: {avg_tps:4.1f}  |  cpu Δ: {avg_cpu:+5.1f}%")

if __name__ == "__main__":
    run_experiment()
