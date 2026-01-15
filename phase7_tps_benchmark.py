#!/usr/bin/env python3
"""
Phase 7: Tokens Per Second (TPS) Benchmark
Measuring the Alignment Tax through efficiency, not energy
"""

import time
import json
import requests
from datetime import datetime
import statistics

# Configuration
MODEL = "llama3.1:8b-instruct-q4_0"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OUTPUT_FILE = f"phase7_tps_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Gradient of constraint difficulty (0 = easiest, 4 = hardest)
CONSTRAINT_LEVELS = [
    {
        "level": 0,
        "name": "Free",
        "prompts": [
            "Write about a cat.",
            "Describe the ocean.",
            "Explain gravity.",
            "Tell me about computers.",
            "Describe a sunset."
        ]
    },
    {
        "level": 1,
        "name": "Format_Only",
        "prompts": [
            "Write about a cat in exactly three sentences.",
            "Describe the ocean in exactly 20 words.",
            "Explain gravity in one sentence.",
            "Tell me about computers in exactly 15 words.",
            "Describe a sunset using exactly two sentences."
        ]
    },
    {
        "level": 2,
        "name": "Single_Negative",
        "prompts": [
            "Write about a cat without using the letter 'e'.",
            "Describe the ocean without using 'water' or 'blue'.",
            "Explain gravity without using 'force' or 'mass'.",
            "Tell me about computers without using 'data' or 'processor'.",
            "Describe a sunset without using 'sun' or 'sky'."
        ]
    },
    {
        "level": 3,
        "name": "Multiple_Negative",
        "prompts": [
            "Write about a cat without using 'e', 'a', or any number.",
            "Describe the ocean without 'water', 'blue', 'wave', 'sea', or 'fish'.",
            "Explain gravity without 'force', 'mass', 'pull', 'Newton', or 'Earth'.",
            "Tell me about computers without 'data', 'processor', 'memory', 'chip', or 'digital'.",
            "Describe a sunset without 'sun', 'sky', 'orange', 'red', or 'beautiful'."
        ]
    },
    {
        "level": 4,
        "name": "Nearly_Impossible",
        "prompts": [
            "Write a palindrome about cats that makes sense.",
            "Describe the ocean where each word starts with the last letter of the previous word.",
            "Explain gravity using only words that rhyme with 'down'.",
            "Tell me about computers using only 3-letter words.",
            "Describe a sunset where every word contains exactly 5 letters."
        ]
    }
]

def measure_tps(prompt, timeout=60):
    """Measure tokens per second for a single prompt"""
    
    start_time = time.time()
    
    try:
        # Stream the response to measure token rate
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 200  # Limit max tokens
                }
            },
            stream=True,
            timeout=timeout
        )
        
        tokens = []
        token_times = []
        
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        tokens.append(chunk['response'])
                        token_times.append(time.time() - start_time)
                        
                    if chunk.get('done', False):
                        break
                except:
                    continue
        
        total_time = time.time() - start_time
        num_tokens = len(tokens)
        
        if num_tokens > 0:
            tps = num_tokens / total_time
            output_text = ''.join(tokens)
            
            # Check for early termination (suspiciously short)
            early_termination = num_tokens < 20
            
            return {
                "tokens": num_tokens,
                "time_s": round(total_time, 2),
                "tps": round(tps, 2),
                "output_preview": output_text[:100],
                "early_termination": early_termination
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_benchmark():
    """Run the complete TPS benchmark across all constraint levels"""
    
    print("=" * 60)
    print("PHASE 7: TOKENS PER SECOND BENCHMARK")
    print("Measuring the Alignment Tax through efficiency")
    print("=" * 60)
    
    # Warmup
    print("\n♨️ Warming up model...")
    measure_tps("Hello, how are you?")
    time.sleep(2)
    
    results = []
    
    for level_data in CONSTRAINT_LEVELS:
        level = level_data["level"]
        name = level_data["name"]
        prompts = level_data["prompts"]
        
        print(f"\n📊 Level {level}: {name}")
        print("-" * 40)
        
        level_results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"  [{i}/{len(prompts)}] ", end="")
            
            result = measure_tps(prompt)
            
            if result:
                print(f"✓ {result['tps']:.2f} TPS ({result['tokens']} tokens)")
                level_results.append(result['tps'])
                
                # Store detailed result
                results.append({
                    "level": level,
                    "name": name,
                    "prompt": prompt[:50] + "...",
                    "tps": result['tps'],
                    "tokens": result['tokens'],
                    "time_s": result['time_s'],
                    "early_termination": result['early_termination']
                })
            else:
                print("✗ Failed")
            
            time.sleep(1)  # Brief pause between prompts
        
        # Calculate level statistics
        if level_results:
            avg_tps = statistics.mean(level_results)
            stdev_tps = statistics.stdev(level_results) if len(level_results) > 1 else 0
            print(f"\n  Average: {avg_tps:.2f} ± {stdev_tps:.2f} TPS")
    
    # Save raw results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS: THE ALIGNMENT TAX CURVE")
    print("=" * 60)
    
    # Group by level and calculate averages
    level_averages = {}
    for level in range(5):
        level_data = [r['tps'] for r in results if r['level'] == level]
        if level_data:
            level_averages[level] = statistics.mean(level_data)
    
    # Display the curve
    print("\nDifficulty vs Performance:")
    print("-" * 40)
    
    baseline_tps = level_averages.get(0, 1)
    
    for level in sorted(level_averages.keys()):
        tps = level_averages[level]
        tax_percent = ((baseline_tps - tps) / baseline_tps) * 100
        bar = "█" * int(tps * 2)  # Visual bar chart
        
        level_name = CONSTRAINT_LEVELS[level]["name"]
        print(f"Level {level} ({level_name:20s}): {bar} {tps:.2f} TPS (Tax: {tax_percent:+.1f}%)")
    
    # Calculate the gradient
    if len(level_averages) > 1:
        levels = sorted(level_averages.keys())
        first_tps = level_averages[levels[0]]
        last_tps = level_averages[levels[-1]]
        total_tax = ((first_tps - last_tps) / first_tps) * 100
        
        print(f"\n🎯 TOTAL ALIGNMENT TAX: {total_tax:.1f}%")
        print(f"   (Performance drops from {first_tps:.2f} to {last_tps:.2f} TPS)")
    
    print(f"\n✅ Results saved to: {OUTPUT_FILE}")
    
    # Check for exponential decay pattern
    print("\n📈 Checking for exponential decay pattern...")
    if len(level_averages) >= 3:
        # Simple check: is each drop bigger than the last?
        drops = []
        levels = sorted(level_averages.keys())
        for i in range(1, len(levels)):
            drop = level_averages[levels[i-1]] - level_averages[levels[i]]
            drops.append(drop)
        
        if all(drops[i] >= drops[i-1] for i in range(1, len(drops))):
            print("   ✓ Exponential pattern detected! Each constraint level has increasing cost.")
        else:
            print("   ○ Linear or irregular pattern. Constraints may have hit complexity ceiling.")

if __name__ == "__main__":
    run_benchmark()