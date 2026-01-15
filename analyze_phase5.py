import json
import statistics
from collections import defaultdict

# Load the results
with open('phase5_alignment_tax_20251122_130649.json', 'r') as f:
    results = json.load(f)

print("DETAILED PHASE 5 ANALYSIS")
print("=" * 60)

# Group by category and constraint level
by_category = defaultdict(lambda: defaultdict(list))

for r in results:
    category = r['category']
    level = r['constraint_level']
    by_category[category][level].append({
        'cpu_per_token': r['cpu_per_token'],
        'tokens': r['num_tokens'],
        'ms_per_token': r['ms_per_token'],
        'constraint_check': r['constraint_check']
    })

# Analyze by category
print("\nBy Category Analysis:")
print("-" * 40)

for category in ['creative', 'factual', 'math']:
    print(f"\n{category.upper()}:")
    for level in ['free', 'soft', 'hard']:
        data = by_category[category][level]
        if data:
            avg_cpu = statistics.mean([d['cpu_per_token'] for d in data])
            avg_tokens = statistics.mean([d['tokens'] for d in data])
            avg_ms = statistics.mean([d['ms_per_token'] for d in data])
            print(f"  {level:6s}: {avg_cpu:.3f} CPU%/tok, {avg_tokens:3.0f} tokens, {avg_ms:.0f} ms/tok")

# Length correlation
print("\n" + "=" * 60)
print("OUTPUT LENGTH VS CPU CORRELATION:")
print("-" * 40)

for level in ['free', 'soft', 'hard']:
    level_results = [r for r in results if r['constraint_level'] == level]
    tokens = [r['num_tokens'] for r in level_results]
    cpus = [r['cpu_per_token'] for r in level_results]
    
    avg_tokens = statistics.mean(tokens)
    avg_cpu = statistics.mean(cpus)
    
    print(f"{level:6s}: {avg_tokens:4.0f} avg tokens → {avg_cpu:.3f} avg CPU%/token")

# The math anomaly
print("\n" + "=" * 60)
print("THE MATH ANOMALY:")
print("-" * 40)

math_results = [r for r in results if r['category'] == 'math']
for r in math_results:
    print(f"{r['constraint_level']:6s}: {r['num_tokens']:3d} tokens, "
          f"{r['cpu_per_token']:.3f} CPU%/tok - Output: '{r['output'][:50]}'")

print("\nHypothesis: Math prompts are shortest, so overhead dominates per-token metric")
