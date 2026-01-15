"""
Phase 3: Double Control Protocol + Temperature Manipulation
Tests the Entropy Signature hypothesis definitively
"""

import time
import psutil
import json
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*70)
print("PHASE 3: ENTROPY SIGNATURE VALIDATION")
print("Double Control Protocol + Temperature Manipulation")
print("="*70)

print("\nLoading Phi-2 model...")
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    torch_dtype=torch.float32
)
print("Model loaded!\n")

# Prompt padding function
def pad_prompt_to_tokens(prompt, target_tokens=50):
    """Pad prompt to approximately target token count"""
    current = tokenizer(prompt, return_tensors="pt")
    current_len = current['input_ids'].shape[1]
    
    if current_len >= target_tokens:
        return prompt
    
    # Add padding instructions that won't affect the task
    padding = " Please provide your answer following these guidelines: be clear, concise, and accurate in your response. Consider all aspects carefully."
    
    # Keep adding padding until we reach target
    while current_len < target_tokens:
        prompt += padding
        current = tokenizer(prompt, return_tensors="pt")
        current_len = current['input_ids'].shape[1]
        if current_len >= target_tokens:
            break
    
    return prompt

# Controlled test cases - 5 per category
# Each will be padded to ~50 tokens input and forced to 20 words output
BASE_PROMPTS = {
    'math': [
        "Calculate 847 * 923",
        "Solve for x: 5x + 12 = 47",
        "What is 23% of 850?",
        "Divide 1,296 by 18",
        "Calculate (25 + 17) * 3"
    ],
    'creative': [
        "Write a haiku about robots",
        "Describe a new color",
        "Invent a word for nostalgia",
        "Write about silence",
        "Describe what happiness tastes like"
    ],
    'factual': [
        "What is the capital of Brazil?",
        "When was the Declaration of Independence signed?",
        "Who painted the Mona Lisa?",
        "What is the largest planet?",
        "Who wrote Romeo and Juliet?"
    ],
    'reasoning': [
        "If all cats are mammals, and all mammals breathe air, what follows?",
        "Is this valid: All birds fly, penguins are birds, so penguins fly?",
        "What's wrong with: Everyone I know likes pizza, so everyone likes pizza?",
        "If A > B and B > C, who is shortest?",
        "Is correlation the same as causation?"
    ]
}

# Build controlled prompts
CONTROLLED_PROMPTS = {}
for task_type, prompts in BASE_PROMPTS.items():
    CONTROLLED_PROMPTS[task_type] = []
    for prompt in prompts:
        # Add output control instruction
        controlled = f"{prompt}. Answer in exactly 20 words."
        # Pad to ~50 tokens
        padded = pad_prompt_to_tokens(controlled, target_tokens=50)
        CONTROLLED_PROMPTS[task_type].append(padded)

def capture_telemetry(task_type, prompt, temperature, experiment_name):
    """Run inference and capture telemetry"""
    
    process = psutil.Process()
    
    # BEFORE
    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=0.1)
    start_memory = psutil.virtual_memory().percent
    start_process_memory = process.memory_info().rss / 1024 / 1024
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs['input_ids'].shape[1]
    
    # Generate with controlled temperature
    with torch.no_grad():
        if temperature == 0:
            # Greedy decoding
            outputs = model.generate(
                inputs['input_ids'],
                max_length=input_length + 30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            # Normal sampling
            outputs = model.generate(
                inputs['input_ids'],
                max_length=input_length + 30,
                do_sample=True,
                temperature=temperature,
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
        'experiment': experiment_name,
        'temperature': temperature,
        'timestamp': datetime.now().isoformat(),
        'task_type': task_type,
        'prompt': prompt[:100] + "...",  # Truncate for readability
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

def run_experiment(temperature, experiment_name):
    """Run full experiment at specified temperature"""
    
    results = []
    total = sum(len(prompts) for prompts in CONTROLLED_PROMPTS.values())
    
    print(f"\n{'='*70}")
    print(f"RUNNING: {experiment_name} (Temperature={temperature})")
    print(f"{'='*70}")
    print(f"Total prompts: {total}")
    print(f"Each prompt: ~50 input tokens, forced 20-word output\n")
    
    count = 0
    for task_type, prompts in CONTROLLED_PROMPTS.items():
        print(f"\n{task_type.upper()}:")
        for prompt in prompts:
            count += 1
            # Show first 50 chars of base prompt
            base = prompt.split('.')[0][:50]
            print(f"  [{count:2d}/{total}] {base:50} ", end="", flush=True)
            
            result = capture_telemetry(task_type, prompt, temperature, experiment_name)
            results.append(result)
            
            print(f"→ {result['latency_seconds']:5.1f}s, {result['output_tokens']} tok")
            
            # Small delay between prompts
            time.sleep(0.5)
    
    return results

# Warmup run (discard results)
print("\n" + "="*70)
print("WARMUP RUN (preventing cold start artifacts)")
print("="*70)
warmup_prompt = pad_prompt_to_tokens("What is 2+2? Answer in exactly 20 words.", 50)
_ = capture_telemetry('warmup', warmup_prompt, 0.7, 'warmup')
print("Warmup complete!\n")

# Run both experiments
all_results = []

# Experiment A: Temperature 0.7 (Control - normal sampling)
exp_a_results = run_experiment(0.7, "Experiment A: Normal Sampling (Temp=0.7)")
all_results.extend(exp_a_results)

print("\n⏸️  Pause between experiments...")
time.sleep(5)

# Experiment B: Temperature 0.0 (Greedy decoding - no sampling)
exp_b_results = run_experiment(0.0, "Experiment B: Greedy Decoding (Temp=0.0)")
all_results.extend(exp_b_results)

# Save results
filename = f"phase3_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(filename, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*70}")
print(f"✅ Results saved to {filename}")
print(f"{'='*70}")

# Quick analysis
print("\n" + "="*70)
print("QUICK ANALYSIS")
print("="*70)

for exp_name, temp in [("Experiment A (Temp=0.7)", 0.7), ("Experiment B (Temp=0.0)", 0.0)]:
    print(f"\n{exp_name}:")
    exp_results = [r for r in all_results if r['temperature'] == temp]
    
    for task_type in ['math', 'creative', 'factual', 'reasoning']:
        task_results = [r for r in exp_results if r['task_type'] == task_type]
        if not task_results:
            continue
        
        # Calculate per-token metrics
        avg_ms_per_tok = sum((r['latency_seconds'] * 1000) / r['output_tokens'] 
                             for r in task_results if r['output_tokens'] > 0) / len(task_results)
        avg_cpu_per_tok = sum(r['cpu_delta'] / r['output_tokens'] 
                             for r in task_results if r['output_tokens'] > 0) / len(task_results)
        avg_input_toks = sum(r['input_tokens'] for r in task_results) / len(task_results)
        avg_output_toks = sum(r['output_tokens'] for r in task_results) / len(task_results)
        
        print(f"  {task_type:10} → {avg_ms_per_tok:6.1f} ms/tok | CPU: {avg_cpu_per_tok:+5.2f}%/tok | In: {avg_input_toks:.0f} | Out: {avg_output_toks:.0f}")

print("\n" + "="*70)
print("ENTROPY SIGNATURE TEST")
print("="*70)

# Get creative CPU at both temperatures
creative_a = [r for r in exp_a_results if r['task_type'] == 'creative']
creative_b = [r for r in exp_b_results if r['task_type'] == 'creative']

if creative_a and creative_b:
    cpu_a = sum(r['cpu_delta'] / r['output_tokens'] for r in creative_a if r['output_tokens'] > 0) / len(creative_a)
    cpu_b = sum(r['cpu_delta'] / r['output_tokens'] for r in creative_b if r['output_tokens'] > 0) / len(creative_b)
    
    print(f"\nCreative Task CPU Usage:")
    print(f"  Temp=0.7 (Normal): {cpu_a:+5.2f}% per token")
    print(f"  Temp=0.0 (Greedy): {cpu_b:+5.2f}% per token")
    print(f"  Change: {((cpu_b - cpu_a) / cpu_a * 100):.1f}%")
    
    if cpu_b < cpu_a * 0.8:  # 20% drop
        print(f"\n  ✅ ENTROPY SIGNATURE CONFIRMED!")
        print(f"  Removing sampling overhead reduced CPU usage by {((cpu_a - cpu_b) / cpu_a * 100):.1f}%")
    elif cpu_b < cpu_a:
        print(f"\n  ⚠️  Small decrease observed but not dramatic")
    else:
        print(f"\n  ❌ No decrease - hypothesis may be incorrect")

print("\n" + "="*70)
print("Next step: Run detailed analysis with analyze_phase3.py")
print("="*70)
