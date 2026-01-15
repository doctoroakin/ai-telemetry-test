#!/usr/bin/env python3
"""
Phase 5: Testing the Alignment Tax Hypothesis
Does instruction-following impose measurable computational cost?
"""

import json
import time
import psutil
import ollama
import random
from datetime import datetime
from typing import Dict, List
import statistics

class AlignmentTaxExperiment:
    def __init__(self):
        self.client = ollama.Client()
        self.model = 'llama3.1:8b-instruct-q4_0'
        self.process = psutil.Process()
        self.results = []
        self.warmed_up = False
        
        # Prompt pairs: free -> soft constraint -> hard constraint
        self.prompt_sets = [
            # Creative prompts
            {
                "category": "creative",
                "free": "Describe a sunset.",
                "soft": "Describe a sunset in exactly three sentences.",
                "hard": "Describe a sunset in exactly three sentences without using the words 'sun', 'sky', 'orange', 'red', or 'beautiful'."
            },
            {
                "category": "creative", 
                "free": "Write about a rainy day.",
                "soft": "Write about a rainy day in exactly 20 words.",
                "hard": "Write about a rainy day in exactly 20 words without using 'rain', 'water', 'wet', 'drop', or 'cloud'."
            },
            {
                "category": "creative",
                "free": "Describe the ocean.",
                "soft": "Describe the ocean using exactly two sentences.",
                "hard": "Describe the ocean in exactly two sentences without using 'water', 'blue', 'wave', 'sea', or 'fish'."
            },
            
            # Factual prompts
            {
                "category": "factual",
                "free": "Explain photosynthesis.",
                "soft": "Explain photosynthesis in exactly 15 words.",
                "hard": "Explain photosynthesis in exactly 15 words without using 'plant', 'light', 'energy', 'green', or 'sun'."
            },
            {
                "category": "factual",
                "free": "What is gravity?",
                "soft": "Explain gravity in exactly one sentence.",
                "hard": "Explain gravity in exactly one sentence without using 'force', 'mass', 'pull', 'Earth', or 'Newton'."
            },
            {
                "category": "factual",
                "free": "Explain how computers work.",
                "soft": "Explain how computers work in exactly 20 words.",
                "hard": "Explain how computers work in exactly 20 words without using 'processor', 'memory', 'data', 'chip', or 'digital'."
            },
            
            # Math/reasoning prompts
            {
                "category": "math",
                "free": "What is 47 times 23?",
                "soft": "Calculate 47 times 23 and answer with just the number.",
                "hard": "Calculate 47 times 23. Answer with only the number, no words or explanation."
            },
            {
                "category": "math",
                "free": "How many seconds are in a day?",
                "soft": "Calculate seconds in a day. Answer with just the number.",
                "hard": "Calculate seconds in a day. Give only the numerical answer with no text, units, or punctuation."
            }
        ]
    
    def warmup(self):
        """Warm up the model with a few throwaway inferences"""
        if self.warmed_up:
            return
            
        print("Warming up model...")
        for _ in range(2):
            self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': 'Hello'}]
            )
            time.sleep(1)
        
        self.warmed_up = True
        print("Warmup complete.\n")
    
    def measure_inference(self, prompt: str, category: str, constraint_level: str) -> Dict:
        """Run inference with telemetry monitoring"""
        
        # Pre-inference baseline
        time.sleep(0.5)  # Let system settle
        cpu_baseline = self.process.cpu_percent(interval=0.1)
        mem_baseline = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Prepare for monitoring
        cpu_samples = []
        mem_samples = []
        token_times = []
        tokens_received = []
        
        # Start timing
        start_time = time.perf_counter()
        last_token_time = start_time
        
        try:
            # Stream response to measure per-token metrics
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
                options={
                    'temperature': 0.7,  # Keep consistent with Phase 3
                    'top_p': 0.9
                }
            )
            
            # Collect telemetry during streaming
            for chunk in response:
                current_time = time.perf_counter()
                
                if 'message' in chunk and 'content' in chunk['message']:
                    token = chunk['message']['content']
                    tokens_received.append(token)
                    token_times.append(current_time - last_token_time)
                    last_token_time = current_time
                    
                    # Sample CPU and memory
                    cpu_samples.append(self.process.cpu_percent(interval=0))
                    mem_samples.append(self.process.memory_info().rss / 1024 / 1024)
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
        
        # Calculate metrics
        end_time = time.perf_counter()
        total_time = end_time - start_time
        output_text = ''.join(tokens_received)
        num_tokens = len(tokens_received)
        
        # Filter out zero CPU samples (happens at start sometimes)
        cpu_samples = [s for s in cpu_samples if s > 0] or [0]
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'constraint_level': constraint_level,
            'prompt': prompt,
            'output': output_text,
            'num_tokens': num_tokens,
            'total_time_seconds': total_time,
            'ms_per_token': (total_time * 1000 / num_tokens) if num_tokens > 0 else 0,
            'cpu_baseline': cpu_baseline,
            'cpu_mean': statistics.mean(cpu_samples),
            'cpu_max': max(cpu_samples),
            'cpu_per_token': statistics.mean(cpu_samples) / num_tokens if num_tokens > 0 else 0,
            'memory_baseline_mb': mem_baseline,
            'memory_max_mb': max(mem_samples) if mem_samples else mem_baseline,
            'memory_delta_mb': (max(mem_samples) - mem_baseline) if mem_samples else 0,
            'early_termination': num_tokens < 20  # Flag if response is suspiciously short
        }
        
        # Check constraint adherence
        result['constraint_check'] = self.check_constraint(prompt, output_text, constraint_level)
        
        return result
    
    def check_constraint(self, prompt: str, output: str, level: str) -> str:
        """Basic check if constraints were followed"""
        if level == "free":
            return "no_constraint"
        
        output_lower = output.lower()
        words = output.split()
        sentences = output.count('.') + output.count('!') + output.count('?')
        
        # Check soft constraints (word/sentence counts)
        if "exactly three sentences" in prompt and sentences != 3:
            return f"failed_sentence_count: got {sentences}"
        if "exactly two sentences" in prompt and sentences != 2:
            return f"failed_sentence_count: got {sentences}"
        if "exactly one sentence" in prompt and sentences != 1:
            return f"failed_sentence_count: got {sentences}"
        if "exactly 20 words" in prompt and len(words) != 20:
            return f"failed_word_count: got {len(words)}"
        if "exactly 15 words" in prompt and len(words) != 15:
            return f"failed_word_count: got {len(words)}"
        if "just the number" in prompt and len(words) > 3:
            return f"failed_brevity: got {len(words)} words"
        
        # Check hard constraints (forbidden words)
        forbidden_words = []
        if "'sun'" in prompt: forbidden_words.append('sun')
        if "'sky'" in prompt: forbidden_words.append('sky')
        if "'orange'" in prompt: forbidden_words.append('orange')
        if "'plant'" in prompt: forbidden_words.append('plant')
        if "'light'" in prompt: forbidden_words.append('light')
        if "'rain'" in prompt: forbidden_words.append('rain')
        if "'water'" in prompt: forbidden_words.append('water')
        if "'force'" in prompt: forbidden_words.append('force')
        if "'processor'" in prompt: forbidden_words.append('processor')
        
        for word in forbidden_words:
            if word in output_lower:
                return f"used_forbidden_word: {word}"
        
        return "passed"
    
    def run_experiment(self):
        """Run the full experiment"""
        self.warmup()
        
        print("Starting Phase 5: Alignment Tax Experiment")
        print(f"Model: {self.model}")
        print(f"Testing {len(self.prompt_sets)} prompt sets with 3 constraint levels each")
        print("-" * 60)
        
        # Randomize order but keep triplets together
        random.shuffle(self.prompt_sets)
        
        for i, prompt_set in enumerate(self.prompt_sets, 1):
            print(f"\n[{i}/{len(self.prompt_sets)}] Testing {prompt_set['category']} prompt set...")
            
            # Test each constraint level
            for level in ['free', 'soft', 'hard']:
                print(f"  - {level} constraint...", end=' ')
                
                result = self.measure_inference(
                    prompt=prompt_set[level],
                    category=prompt_set['category'],
                    constraint_level=level
                )
                
                if result:
                    self.results.append(result)
                    cpu_per_token = result['cpu_per_token']
                    tokens = result['num_tokens']
                    constraint_check = result['constraint_check']
                    
                    status = "✓" if 'passed' in constraint_check or 'no_constraint' in constraint_check else "✗"
                    print(f"{status} {tokens} tokens, {cpu_per_token:.3f} CPU%/token")
                    
                    # Small delay between prompts
                    time.sleep(2)
                else:
                    print("ERROR")
        
        # Save results
        self.save_results()
        self.analyze_results()
    
    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase5_alignment_tax_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")
    
    def analyze_results(self):
        """Quick analysis of alignment tax"""
        print("\n" + "=" * 60)
        print("PHASE 5 PRELIMINARY RESULTS: ALIGNMENT TAX ANALYSIS")
        print("=" * 60)
        
        # Group by constraint level
        by_level = {'free': [], 'soft': [], 'hard': []}
        
        for r in self.results:
            level = r['constraint_level']
            by_level[level].append(r['cpu_per_token'])
        
        # Calculate averages
        print("\nAverage CPU% per token by constraint level:")
        print("-" * 40)
        
        averages = {}
        for level in ['free', 'soft', 'hard']:
            if by_level[level]:
                avg = statistics.mean(by_level[level])
                stdev = statistics.stdev(by_level[level]) if len(by_level[level]) > 1 else 0
                averages[level] = avg
                print(f"{level:8s}: {avg:.3f} ± {stdev:.3f} CPU%/token")
        
        # Calculate alignment tax
        if 'free' in averages and 'hard' in averages:
            tax = ((averages['hard'] - averages['free']) / averages['free']) * 100
            print(f"\n🎯 ALIGNMENT TAX: {tax:+.1f}%")
            print(f"   (Hard constraints cost {tax:.1f}% more CPU than free generation)")
        
        # Check constraint adherence
        print("\nConstraint Adherence:")
        print("-" * 40)
        for level in ['soft', 'hard']:
            level_results = [r for r in self.results if r['constraint_level'] == level]
            passed = sum(1 for r in level_results if 'passed' in r['constraint_check'])
            total = len(level_results)
            print(f"{level:8s}: {passed}/{total} passed ({100*passed/total:.0f}%)")
        
        print("\n✓ Analysis complete!")

if __name__ == "__main__":
    experiment = AlignmentTaxExperiment()
    experiment.run_experiment()
