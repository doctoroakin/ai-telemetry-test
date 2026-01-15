filename = "telemetry_expanded_20251122_111944.json""""
Normalize Telemetry Data - Control for Output Length
"""

import json
import statistics

# Load your data
filename = "telemetry_expanded_20251122_111944.json"
with open(filename, 'r') as f:
    data = json.load(f)

print("="*70)
print("NORMALIZED TELEMETRY ANALYSIS")
print("Controlling for Output Length")
print("="*70)

# Group by task type
by_task = {
    'math': [],
    'creative': [],
    'factual': [],
    'reasoning': []
}

for result in data:
    task = result['task_type']
    
    # Skip if output is too short (anomalies)
    if result['output_tokens'] < 5:
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
    
    by_task[task].append(normalized)

print("\n" + "="*70)
print("RAW METRICS (Before Normalization)")
print("="*70)

for task in ['math', 'creative', 'factual', 'reasoning']:
    results = by_task[task]
    n = len(results)
    
    avg_latency = statistics.mean(r['latency'] for r in results)
    avg_tokens = statistics.mean(r['output_tokens'] for r in results)
    avg_memory = statistics.mean(r['memory'] for r in results)
    avg_cpu = statistics.mean(r['cpu'] for r in results)
    
    print(f"\n{task.upper()} (n={n}):")
    print(f"  Latency:      {avg_latency:6.1f}s")
    print(f"  Output tokens: {avg_tokens:6.1f}")
    print(f"  Memory delta:  {avg_memory:6.1f} MB")
    print(f"  CPU delta:     {avg_cpu:+6.1f}%")

print("\n" + "="*70)
print("NORMALIZED METRICS (Per Token)")
print("="*70)

# Store for comparison
normalized_results = {}

for task in ['math', 'creative', 'factual', 'reasoning']:
    results = by_task[task]
    n = len(results)
    
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
    print(f"  MB/token:     {mem_per_tok:7.3f} (±{mem_std:.3f})")
    print(f"  CPU%/token:   {cpu_per_tok:+7.2f} (±{cpu_std:.2f})")

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

# Find max/min for each metric
tasks = ['math', 'creative', 'factual', 'reasoning']

ms_values = [(task, normalized_results[task]['ms_per_token']) for task in tasks]
mem_values = [(task, normalized_results[task]['mem_per_token']) for task in tasks]
cpu_values = [(task, normalized_results[task]['cpu_per_token']) for task in tasks]

ms_sorted = sorted(ms_values, key=lambda x: x[1])
mem_sorted = sorted(mem_values, key=lambda x: x[1])
cpu_sorted = sorted(cpu_values, key=lambda x: x[1])

print("\n⏱️  SPEED (ms per token) - Lower is faster:")
for i, (task, val) in enumerate(ms_sorted, 1):
    print(f"  {i}. {task:10} {val:6.1f} ms/token")

print("\n💾 MEMORY (MB per token) - Higher means more memory used:")
for i, (task, val) in enumerate(mem_sorted, 1):
    print(f"  {i}. {task:10} {val:6.3f} MB/token")

print("\n🔥 CPU (% per token) - Higher means more CPU intensive:")
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
print(f"Memory variance: {mem_range:.3f} MB/token ({mem_range/mem_sorted[0][1]*100 if mem_sorted[0][1] > 0 else 0:.1f}% difference)")
print(f"CPU variance:    {cpu_range:.2f}%/token ({abs(cpu_range/cpu_sorted[0][1])*100 if cpu_sorted[0][1] != 0 else 0:.1f}% difference)")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

# Determine if there's signal
significant_speed = ms_range / ms_sorted[0][1] > 0.1  # >10% difference
significant_memory = mem_range / mem_sorted[0][1] > 0.5 if mem_sorted[0][1] > 0 else False  # >50% difference
significant_cpu = abs(cpu_range / cpu_sorted[0][1]) > 0.1 if cpu_sorted[0][1] != 0 else False  # >10% difference

print("\nAfter controlling for output length:")
print(f"  Speed differences:  {'✅ SIGNIFICANT' if significant_speed else '❌ Not significant'}")
print(f"  Memory differences: {'✅ SIGNIFICANT' if significant_memory else '❌ Not significant'}")
print(f"  CPU differences:    {'✅ SIGNIFICANT' if significant_cpu else '❌ Not significant'}")

if significant_memory:
    print(f"\n🎯 KEY FINDING: {mem_sorted[-1][0].upper()} uses {mem_sorted[-1][1]/mem_sorted[0][1]:.1f}x more memory per token than {mem_sorted[0][0].upper()}")

print("\n" + "="*70)
