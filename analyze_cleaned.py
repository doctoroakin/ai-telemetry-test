"""
Clean Data Analysis - Remove Outliers
"""

import json
import statistics

# Load data
filename = "telemetry_expanded_20251122_111944.json"
with open(filename, 'r') as f:
    data = json.load(f)

print("="*70)
print("CLEANED DATA ANALYSIS - OUTLIER REMOVAL")
print("="*70)

# First pass: calculate stats to identify outliers
by_task_raw = {
    'math': [],
    'creative': [],
    'factual': [],
    'reasoning': []
}

for result in data:
    task = result['task_type']
    if result['output_tokens'] >= 5:  # Skip anomalies
        by_task_raw[task].append(result)

# Calculate means and std devs for memory
print("\nSTEP 1: IDENTIFYING OUTLIERS")
print("-" * 70)

for task in ['math', 'creative', 'factual', 'reasoning']:
    results = by_task_raw[task]
    memory_values = [r['process_memory_delta_mb'] for r in results]
    
    mean_mem = statistics.mean(memory_values)
    std_mem = statistics.stdev(memory_values) if len(memory_values) > 1 else 0
    
    print(f"\n{task.upper()}:")
    print(f"  Mean memory: {mean_mem:.2f} MB")
    print(f"  Std dev:     {std_mem:.2f} MB")
    print(f"  Threshold:   {mean_mem + 2*std_mem:.2f} MB (mean + 2σ)")
    
    # Find outliers
    outliers = [r for r in results if r['process_memory_delta_mb'] > mean_mem + 2*std_mem]
    if outliers:
        print(f"  🚨 OUTLIERS FOUND: {len(outliers)}")
        for o in outliers:
            print(f"     - {o['process_memory_delta_mb']:.1f} MB: '{o['prompt'][:50]}...'")

# Second pass: filter outliers and recalculate
by_task_clean = {
    'math': [],
    'creative': [],
    'factual': [],
    'reasoning': []
}

outliers_removed = 0

for result in data:
    task = result['task_type']
    
    # Skip short outputs
    if result['output_tokens'] < 5:
        continue
    
    # Calculate task-specific outlier threshold
    task_results = by_task_raw[task]
    memory_values = [r['process_memory_delta_mb'] for r in task_results]
    mean_mem = statistics.mean(memory_values)
    std_mem = statistics.stdev(memory_values) if len(memory_values) > 1 else 0
    threshold = mean_mem + 2 * std_mem
    
    # Filter outliers
    if result['process_memory_delta_mb'] > threshold:
        outliers_removed += 1
        continue
    
    # Calculate normalized metrics
    normalized = {
        'ms_per_token': (result['latency_seconds'] * 1000) / result['output_tokens'],
        'memory_mb_per_token': result['process_memory_delta_mb'] / result['output_tokens'],
        'cpu_delta_per_token': result['cpu_delta'] / result['output_tokens'],
        'output_tokens': result['output_tokens'],
        'latency': result['latency_seconds'],
        'memory': result['process_memory_delta_mb'],
        'cpu': result['cpu_delta']
    }
    
    by_task_clean[task].append(normalized)

print(f"\n{'='*70}")
print(f"STEP 2: CLEANED DATASET")
print(f"{'='*70}")
print(f"\nTotal outliers removed: {outliers_removed}")

# Show comparison
print("\n" + "="*70)
print("BEFORE vs AFTER CLEANING")
print("="*70)

for task in ['math', 'creative', 'factual', 'reasoning']:
    raw_n = len(by_task_raw[task])
    clean_n = len(by_task_clean[task])
    removed = raw_n - clean_n
    
    print(f"\n{task.upper()}: {raw_n} samples → {clean_n} samples ({removed} removed)")

print("\n" + "="*70)
print("NORMALIZED METRICS (CLEANED DATA)")
print("="*70)

normalized_results = {}

for task in ['math', 'creative', 'factual', 'reasoning']:
    results = by_task_clean[task]
    n = len(results)
    
    if n == 0:
        print(f"\n{task.upper()}: No data after cleaning!")
        continue
    
    # Calculate averages
    ms_per_tok = statistics.mean(r['ms_per_token'] for r in results)
    mem_per_tok = statistics.mean(r['memory_mb_per_token'] for r in results)
    cpu_per_tok = statistics.mean(r['cpu_delta_per_token'] for r in results)
    
    # Calculate standard deviations
    ms_std = statistics.stdev(r['ms_per_token'] for r in results) if n > 1 else 0
    mem_std = statistics.stdev(r['memory_mb_per_token'] for r in results) if n > 1 else 0
    cpu_std = statistics.stdev(r['cpu_delta_per_token'] for r in results) if n > 1 else 0
    
    normalized_results[task] = {
        'ms_per_token': ms_per_tok,
        'mem_per_token': mem_per_tok,
        'cpu_per_token': cpu_per_tok,
        'ms_std': ms_std,
        'mem_std': mem_std,
        'cpu_std': cpu_std,
        'n': n
    }
    
    print(f"\n{task.upper()} (n={n}):")
    print(f"  ms/token:     {ms_per_tok:7.1f} (±{ms_std:.1f})")
    print(f"  MB/token:     {mem_per_tok:7.4f} (±{mem_std:.4f})")
    print(f"  CPU%/token:   {cpu_per_tok:+7.2f} (±{cpu_std:.2f})")

print("\n" + "="*70)
print("KEY FINDINGS (CLEANED DATA)")
print("="*70)

tasks = [t for t in ['math', 'creative', 'factual', 'reasoning'] if t in normalized_results]

ms_values = [(task, normalized_results[task]['ms_per_token']) for task in tasks]
mem_values = [(task, normalized_results[task]['mem_per_token']) for task in tasks]
cpu_values = [(task, normalized_results[task]['cpu_per_token']) for task in tasks]

ms_sorted = sorted(ms_values, key=lambda x: x[1])
mem_sorted = sorted(mem_values, key=lambda x: x[1])
cpu_sorted = sorted(cpu_values, key=lambda x: x[1])

print("\n⏱️  SPEED (ms per token):")
for i, (task, val) in enumerate(ms_sorted, 1):
    print(f"  {i}. {task:10} {val:6.1f} ms/token")

print("\n💾 MEMORY (MB per token):")
for i, (task, val) in enumerate(mem_sorted, 1):
    print(f"  {i}. {task:10} {val:7.4f} MB/token")

print("\n🔥 CPU (% per token):")
for i, (task, val) in enumerate(cpu_sorted, 1):
    print(f"  {i}. {task:10} {val:+6.2f}% /token")

# Calculate variance
ms_range = ms_sorted[-1][1] - ms_sorted[0][1]
mem_range = mem_sorted[-1][1] - mem_sorted[0][1]
cpu_range = cpu_sorted[-1][1] - cpu_sorted[0][1]

print("\n" + "="*70)
print("VARIANCE ANALYSIS")
print("="*70)
print(f"\nSpeed variance:  {ms_range:.1f} ms/token ({ms_range/ms_sorted[0][1]*100:.1f}% difference)")
print(f"Memory variance: {mem_range:.4f} MB/token ({mem_range/mem_sorted[0][1]*100 if mem_sorted[0][1] > 0 else 0:.1f}% difference)")
print(f"CPU variance:    {cpu_range:.2f}%/token ({abs(cpu_range/cpu_sorted[0][1])*100 if cpu_sorted[0][1] != 0 else 0:.1f}% difference)")

print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

# Check for CPU pattern
creative_cpu = normalized_results.get('creative', {}).get('cpu_per_token', 0)
reasoning_cpu = normalized_results.get('reasoning', {}).get('cpu_per_token', 0)
math_cpu = normalized_results.get('math', {}).get('cpu_per_token', 0)

print("\n🎯 ENTROPY SIGNATURE TEST:")
if creative_cpu > math_cpu and creative_cpu > reasoning_cpu:
    print(f"   ✅ CONFIRMED: Creative tasks use {creative_cpu/math_cpu:.1f}x more CPU per token than Math")
    print(f"   This supports the 'flat probability distribution' hypothesis!")
else:
    print(f"   ❌ Pattern not clear after cleaning")

print("\n" + "="*70)
