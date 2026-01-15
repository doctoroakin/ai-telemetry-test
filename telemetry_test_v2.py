"""
Expanded Telemetry Experiment - 10 samples per category
"""

import time
import psutil
import json
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    torch_dtype=torch.float32
)

print("Model loaded!\n")

# Expanded test cases - 10 per category
TEST_CASES = {
    'math': [
        "Calculate 847 * 923. Show your work.",
        "Solve for x: 5x + 12 = 47",
        "What is 23% of 850?",
        "Divide 1,296 by 18",
        "What is the square root of 144?",
        "Calculate (25 + 17) * 3",
        "Solve: 2x - 8 = 24",
        "What is 15% of 200?",
        "Multiply 156 by 7",
        "Calculate 999 + 1,234"
    ],
    'creative': [
        "Write a haiku about robots.",
        "Describe a new color that doesn't exist.",
        "Invent a word for missing a place you've never been.",
        "Write a short poem about silence.",
        "Describe what happiness tastes like.",
        "Invent a name for a new emotion.",
        "Write a haiku about technology.",
        "Describe the sound of loneliness.",
        "Create a metaphor for time passing.",
        "Describe what dreams smell like."
    ],
    'factual': [
        "What is the capital of Brazil?",
        "When was the Declaration of Independence signed?",
        "Who painted the Mona Lisa?",
        "What is the largest planet in our solar system?",
        "Who wrote Romeo and Juliet?",
        "What year did World War 2 end?",
        "What is the capital of Japan?",
        "Who invented the telephone?",
        "What is the tallest mountain on Earth?",
        "What is the speed of light?"
    ],
    'reasoning': [
        "If all cats are mammals, and all mammals breathe air, what can we conclude?",
        "Is this valid: All birds fly. Penguins are birds. So penguins fly.",
        "Compare 'knowing how' vs 'knowing that'.",
        "What is the flaw in: Everyone I know likes pizza, so everyone likes pizza.",
        "If A is taller than B, and B is taller than C, who is shortest?",
        "Is correlation the same as causation? Explain briefly.",
        "What's wrong with: After we got a cat, it rained. Therefore cats cause rain.",
        "If some doctors are women, and all women are people, what follows?",
        "Can something be true but unknowable? Brief answer.",
        "What's the difference between necessary and sufficient conditions?"
    ]
}

def capture_telemetry(task_type, prompt):
    """Run inference and capture everything"""
    
    process = psutil.Process()
    
    # BEFORE
    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=0.1)
    start_memory = psutil.virtual_memory().percent
    start_process_memory = process.memory_info().rss / 1024 / 1024
    
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
    
    # AFTER
    end_time = time.time()
    end_cpu = psutil.cpu_percent(interval=0.1)
    end_memory = psutil.virtual_memory().percent
    end_process_memory = process.memory_info().rss / 1024 / 1024
    
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
        'tokens_per_second': tokens_generated / (end_time - start_time) if (end_time - start_time) > 0 else 0,
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
    
    print(f"🔬 Running EXPANDED experiment with {total} prompts")
    print(f"This will take about 15-20 minutes\n")
    
    count = 0
    for task_type, prompts in TEST_CASES.items():
        print(f"\n{'='*60}")
        print(f"Testing {task_type.upper()} ({len(prompts)} prompts)")
        print(f"{'='*60}")
        for prompt in prompts:
            count += 1
            print(f"[{count:2d}/{total}] {prompt[:55]:55} ", end="", flush=True)
            result = capture_telemetry(task_type, prompt)
            results.append(result)
            print(f"→ {result['latency_seconds']:5.1f}s")
    
    # Save
    filename = f"telemetry_expanded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Results saved to {filename}")
    print(f"{'='*60}\n")
    
    # Detailed analysis
    print("DETAILED SUMMARY:\n")
    for task_type in TEST_CASES.keys():
        task_results = [r for r in results if r['task_type'] == task_type]
        
        avg_latency = sum(r['latency_seconds'] for r in task_results) / len(task_results)
        avg_tps = sum(r['tokens_per_second'] for r in task_results) / len(task_results)
        avg_cpu = sum(r['cpu_delta'] for r in task_results) / len(task_results)
        avg_memory = sum(r['process_memory_delta_mb'] for r in task_results) / len(task_results)
        avg_output_tokens = sum(r['output_tokens'] for r in task_results) / len(task_results)
        
        min_latency = min(r['latency_seconds'] for r in task_results)
        max_latency = max(r['latency_seconds'] for r in task_results)
        
        print(f"{task_type.upper():12}")
        print(f"  Latency:     {avg_latency:5.1f}s  (range: {min_latency:.1f}s - {max_latency:.1f}s)")
        print(f"  Tokens/sec:  {avg_tps:5.1f}")
        print(f"  CPU delta:   {avg_cpu:+5.1f}%")
        print(f"  Memory:      {avg_memory:+5.1f} MB")
        print(f"  Avg output:  {avg_output_tokens:5.1f} tokens")
        print()

if __name__ == "__main__":
    run_experiment()
