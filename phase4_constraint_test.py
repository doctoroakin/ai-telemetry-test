"""
Phase 4: The Constraint Test
Testing if format constraints reduce CPU usage at constant temperature
"""

import time
import psutil
import json
from datetime import datetime
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*70)
print("PHASE 4: THE CONSTRAINT TEST")
print("Format Uncertainty vs. Computational Cost")
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

# Paired prompts - each question has loose and constrained version
PAIRED_PROMPTS = {
    'factual': [
        {
            'base': "Who was the first president of the United States?",
            'loose': "Who was the first president of the United States?",
            'constrained': "Who was the first president of the United States? Answer with exactly two words."
        },
        {
            'base': "What is the capital of France?",
            'loose': "What is the capital of France?",
            'constrained': "What is the capital of France? Answer with exactly one word."
        },
        {
            'base': "When did World War 2 end?",
            'loose': "When did World War 2 end?",
            'constrained': "When did World War 2 end? Answer in the format: Year (YYYY)."
        },
        {
            'base': "Who painted the Mona Lisa?",
            'loose': "Who painted the Mona Lisa?",
            'constrained': "Who painted the Mona Lisa? Answer with exactly three words."
        },
        {
            'base': "What is the speed of light?",
            'loose': "What is the speed of light?",
            'constrained': "What is the speed of light? Answer in the format: number unit."
        }
    ],
    'math': [
        {
            'base': "What is 847 times 923?",
            'loose': "What is 847 times 923?",
            'constrained': "What is 847 times 923? Answer with just the number."
        },
        {
            'base': "Solve for x: 5x + 12 = 47",
            'loose': "Solve for x: 5x + 12 = 47",
            'constrained': "Solve for x: 5x + 12 = 47. Answer with just: x = [value]"
        },
        {
            'base': "Calculate 23% of 850",
            'loose': "Calculate 23% of 850",
            'constrained': "Calculate 23% of 850. Answer with just the number."
        },
        {
            'base': "What is the square root of 144?",
            'loose': "What is the square root of 144?",
            'constrained': "What is the square root of 144? Answer with just the number."
        },
        {
            'base': "Divide 1296 by 18",
            'loose': "Divide 1296 by 18",
            'constrained': "Divide 1296 by 18. Answer with just the number."
        }
    ],
    'creative': [
        {
            'base': "Write a haiku about robots",
            'loose': "Write a haiku about robots",
            'constrained': "Write a haiku about robots. Use exactly 3 lines with 5-7-5 syllables."
        },
        {
            'base': "Describe a new color",
            'loose': "Describe a new color",
            'constrained': "Describe a new color. Use exactly 15 words."
        },
        {
            'base': "Invent a word for nostalgia",
            'loose': "Invent a word for nostalgia",
            'constrained': "Invent a word for nostalgia. Explain in exactly 10 words."
        },
        {
            'base': "Describe what happiness tastes like",
            'loose': "Describe what happiness tastes like",
            'constrained': "Describe what happiness tastes like. Use exactly 12 words."
        },
        {
            'base': "Write about silence",
            'loose': "Write about silence",
            'constrained': "Write about silence. Use exactly 15 words."
        }
    ],
    'reasoning': [
        {
            'base': "If all cats are mammals, and all mammals breathe air, what follows?",
            'loose': "If all cats are mammals, and all mammals breathe air, what follows?",
            'constrained': "If all cats are mammals, and all mammals breathe air, what follows? Answer with exactly 4 words."
        },
        {
            'base': "Is this valid: All birds fly, penguins are birds, so penguins fly?",
            'loose': "Is this valid: All birds fly, penguins are birds, so penguins fly?",
            'constrained': "Is this valid: All birds fly, penguins are birds, so penguins fly? Answer: Yes or No, then explain in 10 words."
        },
        {
            'base': "What's wrong with: Everyone I know likes pizza, so everyone likes pizza?",
            'loose': "What's wrong with: Everyone I know likes pizza, so everyone likes pizza?",
            'constrained': "What's wrong with: Everyone I know likes pizza, so everyone likes pizza? Answer in exactly 8 words."
        },
        {
            'base': "If A is taller than B, and B is taller than C, who is shortest?",
            'loose': "If A is taller than B, and B is taller than C, who is shortest?",
            'constrained': "If A is taller than B, and B is taller than C, who is shortest? Answer with exactly 1 word."
        },
        {
            'base': "Is correlation the same as causation?",
            'loose': "Is correlation the same as causation?",
            'constrained': "Is correlation the same as causation? Answer: Yes or No, then explain in 8 words."
        }
    ]
}

def capture_telemetry(task_type, prompt, constraint_type, base_question):
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
    
    # Generate at temp=0.7 (constant for all)
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=input_length + 50,  # Allow more tokens
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
        'experiment': 'Phase 4: Constraint Test',
        'timestamp': datetime.now().isoformat(),
        'task_type': task_type,
        'constraint': constraint_type,  # 'loose' or 'constrained'
        'base_question': base_question,
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
    """Run the full Phase 4 experiment"""
    
    # Build list of all prompts with metadata
    all_prompts = []
    for task_type, pairs in PAIRED_PROMPTS.items():
        for pair in pairs:
            # Add loose version
            all_prompts.append({
                'task_type': task_type,
                'prompt': pair['loose'],
                'constraint': 'loose',
                'base': pair['base']
            })
            # Add constrained version
            all_prompts.append({
                'task_type': task_type,
                'prompt': pair['constrained'],
                'constraint': 'constrained',
                'base': pair['base']
            })
    
    # RANDOMIZE to prevent time-based artifacts
    random.shuffle(all_prompts)
    
    print(f"Total prompts to test: {len(all_prompts)}")
    print(f"(20 pairs = 40 prompts, randomized order)\n")
    
    # Warmup
    print("="*70)
    print("WARMUP (preventing cold start)")
    print("="*70)
    warmup = "What is 2+2?"
    _ = capture_telemetry('warmup', warmup, 'warmup', 'warmup')
    print("Warmup complete!\n")
    
    # Run all prompts
    results = []
    total = len(all_prompts)
    
    print("="*70)
    print("RUNNING EXPERIMENT")
    print("="*70)
    
    for i, prompt_data in enumerate(all_prompts, 1):
        task = prompt_data['task_type']
        constraint = prompt_data['constraint']
        base = prompt_data['base'][:40]
        
        print(f"[{i:2d}/{total}] {task:10} | {constraint:12} | {base:40} ", end="", flush=True)
        
        result = capture_telemetry(
            task_type=prompt_data['task_type'],
            prompt=prompt_data['prompt'],
            constraint_type=prompt_data['constraint'],
            base_question=prompt_data['base']
        )
        
        results.append(result)
        print(f"→ {result['latency_seconds']:5.1f}s, {result['output_tokens']:3d} tok")
        
        time.sleep(0.5)  # Small delay between prompts
    
    # Save results
    filename = f"phase4_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Results saved to {filename}")
    print(f"{'='*70}")
    
    # Quick analysis
    analyze_results(results)
    
    return results

def analyze_results(results):
    """Quick analysis of constraint effect"""
    
    print("\n" + "="*70)
    print("QUICK ANALYSIS: LOOSE vs CONSTRAINED")
    print("="*70)
    
    for task_type in ['factual', 'math', 'creative', 'reasoning']:
        loose = [r for r in results if r['task_type'] == task_type and r['constraint'] == 'loose' and r['output_tokens'] > 0]
        constrained = [r for r in results if r['task_type'] == task_type and r['constraint'] == 'constrained' and r['output_tokens'] > 0]
        
        if not loose or not constrained:
            continue
        
        # Calculate per-token metrics
        loose_cpu = sum(r['cpu_delta'] / r['output_tokens'] for r in loose) / len(loose)
        constrained_cpu = sum(r['cpu_delta'] / r['output_tokens'] for r in constrained) / len(constrained)
        
        loose_ms = sum((r['latency_seconds'] * 1000) / r['output_tokens'] for r in loose) / len(loose)
        constrained_ms = sum((r['latency_seconds'] * 1000) / r['output_tokens'] for r in constrained) / len(constrained)
        
        loose_tokens = sum(r['output_tokens'] for r in loose) / len(loose)
        constrained_tokens = sum(r['output_tokens'] for r in constrained) / len(constrained)
        
        cpu_reduction = ((loose_cpu - constrained_cpu) / loose_cpu * 100) if loose_cpu > 0 else 0
        
        print(f"\n{task_type.upper()}:")
        print(f"  Loose:        {loose_cpu:+5.2f}% CPU/tok | {loose_ms:6.1f} ms/tok | {loose_tokens:5.1f} tokens avg")
        print(f"  Constrained:  {constrained_cpu:+5.2f}% CPU/tok | {constrained_ms:6.1f} ms/tok | {constrained_tokens:5.1f} tokens avg")
        print(f"  → CPU reduction: {cpu_reduction:+5.1f}%")
        
        if task_type == 'factual' and cpu_reduction > 40:
            print(f"  ✅ HYPOTHESIS CONFIRMED: >40% reduction in factual questions!")
    
    print("\n" + "="*70)
    print("THE VERDICT")
    print("="*70)
    
    factual_loose = [r for r in results if r['task_type'] == 'factual' and r['constraint'] == 'loose' and r['output_tokens'] > 0]
    factual_constrained = [r for r in results if r['task_type'] == 'factual' and r['constraint'] == 'constrained' and r['output_tokens'] > 0]
    
    if factual_loose and factual_constrained:
        fl_cpu = sum(r['cpu_delta'] / r['output_tokens'] for r in factual_loose) / len(factual_loose)
        fc_cpu = sum(r['cpu_delta'] / r['output_tokens'] for r in factual_constrained) / len(factual_constrained)
        reduction = ((fl_cpu - fc_cpu) / fl_cpu * 100) if fl_cpu > 0 else 0
        
        print(f"\nFactual Questions (The Critical Test):")
        print(f"  Loose format:       {fl_cpu:+5.2f}% CPU per token")
        print(f"  Constrained format: {fc_cpu:+5.2f}% CPU per token")
        print(f"  Reduction:          {reduction:5.1f}%")
        
        if reduction > 40:
            print(f"\n  🎯 SUCCESS! Format constraints reduce CPU by {reduction:.0f}%")
            print(f"  This proves format uncertainty drives computational cost!")
        elif reduction > 20:
            print(f"\n  ⚠️  Moderate effect: {reduction:.0f}% reduction observed")
        else:
            print(f"\n  ❌ Effect not observed (may need different constraints)")

if __name__ == "__main__":
    results = run_experiment()
