import subprocess
import time
import re
import json
import requests
import os
import signal

# Configuration
MODEL = "llama3.1:8b-instruct-q4_0"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OUTPUT_FILE = "phase6_energy_results.json"

# Test prompts
PROMPTS = [
    {
        "name": "Control_Free",
        "prompt": "Write a short story about a space explorer discovering a new planet. Write about 100 words."
    },
    {
        "name": "Constraint_Format", 
        "prompt": "Write a short story about a space explorer. Use exactly three sentences. No more, no less."
    },
    {
        "name": "Constraint_Negative",
        "prompt": "Write a short story about a space explorer without using the letter 'e' or the word 'planet'."
    },
    {
        "name": "Constraint_Impossible",
        "prompt": "Write a palindrome that is exactly 50 words long and makes coherent sense about space exploration."
    }
]

def get_baseline_power(duration=3):
    """Get baseline power consumption"""
    print(f"📊 Measuring baseline power for {duration}s...")
    
    cmd = ["sudo", "powermetrics", "-i", "1000", "--samplers", "cpu_power", "-n", str(duration)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse Intel power readings
    pattern = re.compile(r"Intel energy model derived package power.*?:\s+(\d+\.?\d*)\s*W")
    matches = pattern.findall(result.stdout)
    
    if matches:
        baseline = sum(float(m) for m in matches) / len(matches)
        print(f"✅ Baseline power: {baseline:.2f}W")
        return baseline
    else:
        print("⚠️ No baseline readings, using 4.0W default")
        return 4.0

def run_with_power_monitoring(prompt, name):
    """Run inference while monitoring power"""
    print(f"\n🔬 Testing: {name}")
    
    # Start powermetrics in background
    power_file = f"/tmp/power_{name}.txt"
    power_cmd = f"sudo powermetrics -i 500 --samplers cpu_power > {power_file} 2>&1"
    power_proc = subprocess.Popen(power_cmd, shell=True, preexec_fn=os.setsid)
    
    # Let it stabilize
    time.sleep(1)
    
    # Run inference
    start_time = time.time()
    
    try:
        response = requests.post(OLLAMA_API_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 200  # Limit for testing
            }
        }, timeout=120)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        result_json = response.json()
        output_text = result_json.get("response", "")
        token_count = result_json.get("eval_count", len(output_text.split()) * 1.3)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        os.killpg(os.getpgid(power_proc.pid), signal.SIGTERM)
        return None
    
    # Stop power monitoring
    time.sleep(1)  # Capture final samples
    os.killpg(os.getpgid(power_proc.pid), signal.SIGTERM)
    time.sleep(0.5)
    
    # Parse power data
    try:
        with open(power_file, 'r') as f:
            power_data = f.read()
        
        pattern = re.compile(r"Intel energy model derived package power.*?:\s+(\d+\.?\d*)\s*W")
        power_readings = [float(m) for m in pattern.findall(power_data)]
        
        if power_readings:
            avg_power = sum(power_readings) / len(power_readings)
            max_power = max(power_readings)
            print(f"   ⚡ Power: {avg_power:.2f}W avg, {max_power:.2f}W max ({len(power_readings)} samples)")
        else:
            print(f"   ⚠️ No power samples found")
            avg_power = 0
            max_power = 0
            
    except Exception as e:
        print(f"   ❌ Error reading power: {e}")
        avg_power = 0
        max_power = 0
    
    # Clean up
    try:
        os.remove(power_file)
    except:
        pass
    
    # Calculate metrics
    result = {
        "name": name,
        "prompt": prompt[:100] + "...",
        "output_preview": output_text[:100] + "...",
        "tokens": int(token_count),
        "time_s": round(elapsed, 2),
        "avg_power_w": round(avg_power, 2),
        "max_power_w": round(max_power, 2),
        "sec_per_token": round(elapsed / token_count, 4) if token_count > 0 else 0
    }
    
    print(f"   📝 {int(token_count)} tokens in {elapsed:.1f}s")
    print(f"   ⏱️  {result['sec_per_token']:.3f} sec/token")
    
    return result

def main():
    print("=" * 60)
    print("PHASE 6: ENERGY COST OF ALIGNMENT")
    print("=" * 60)
    
    # Get baseline
    baseline_power = get_baseline_power()
    
    results = []
    
    # Warmup
    print("\n♨️ Warming up model...")
    run_with_power_monitoring("Hello", "warmup")
    
    # Run experiments
    for prompt_data in PROMPTS:
        result = run_with_power_monitoring(
            prompt_data["prompt"],
            prompt_data["name"]
        )
        if result:
            result["net_power_w"] = round(max(0.1, result["avg_power_w"] - baseline_power), 2)
            result["joules"] = round(result["net_power_w"] * result["time_s"], 2)
            result["joules_per_token"] = round(result["joules"] / result["tokens"], 4) if result["tokens"] > 0 else 0
            results.append(result)
        time.sleep(2)
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS: ALIGNMENT TAX IN JOULES")
    print("=" * 60)
    
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Tokens: {r['tokens']}")
        print(f"  Power: {r['net_power_w']}W net ({r['avg_power_w']}W raw)")
        print(f"  Energy: {r['joules']}J total")
        print(f"  💎 Cost: {r['joules_per_token']:.4f} J/token")
    
    # Calculate ratios
    if len(results) > 1 and results[0]['joules_per_token'] > 0:
        print("\n" + "-" * 40)
        print("ALIGNMENT TAX (vs Free):")
        baseline_j = results[0]['joules_per_token']
        for r in results[1:]:
            if r['joules_per_token'] > 0:
                ratio = r['joules_per_token'] / baseline_j
                print(f"  {r['name']:20s}: {ratio:.2f}x energy cost")
    
    print(f"\n✅ Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("❌ Requires sudo for powermetrics")
        print("Run: sudo python3 llm_joulemeter_simple.py")
        exit(1)
    
    main()