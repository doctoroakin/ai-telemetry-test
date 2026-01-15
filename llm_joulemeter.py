import subprocess
import time
import re
import csv
import threading
import queue
import requests
import json

# --- CONFIGURATION ---
MODEL = "llama3.1:8b-instruct-q4_0"  # Your actual model
OLLAMA_API_URL = "http://localhost:11434/api/generate"
SAMPLE_INTERVAL_MS = 500  # 500ms for Intel (powermetrics is slower)
OUTPUT_FILE = "phase6_energy_results.csv"

# Test cases (The "Gradient of Constraint")
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

# --- POWER MONITORING CLASS FOR INTEL ---
class IntelPowerMonitor(threading.Thread):
    def __init__(self, interval_ms=500):
        super().__init__()
        self.interval_ms = interval_ms
        self.stop_event = threading.Event()
        self.data_queue = queue.Queue()
        self.baseline_watts = 0.0

    def run(self):
        # Intel-specific powermetrics command
        cmd = [
            "sudo", "powermetrics",
            "-i", str(self.interval_ms),
            "--samplers", "cpu_power",
            "-n", "0"  # Run continuously
        ]
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )

        # Intel pattern: "Intel energy model derived package power (CPUs+GT+SA): X.XXW"
        power_pattern = re.compile(r"Intel energy model derived package power.*?:\s+(\d+\.?\d*)\s*W")
        
        buffer = ""
        while not self.stop_event.is_set():
            try:
                char = process.stdout.read(1)
                if not char:
                    break
                    
                buffer += char
                
                # Look for newline to process lines
                if char == '\n':
                    # Check for power match
                    match = power_pattern.search(buffer)
                    if match:
                        watts = float(match.group(1))
                        self.data_queue.put((time.time(), watts))
                    
                    # Keep last 1000 chars to ensure we catch multi-line patterns
                    if len(buffer) > 1000:
                        buffer = buffer[-500:]
                        
            except Exception as e:
                print(f"Monitor error: {e}")
                break

        process.terminate()

    def stop(self):
        self.stop_event.set()

    def calibrate_baseline(self, duration_sec=3):
        print(f"🔌 Calibrating baseline power for {duration_sec}s... (Don't touch anything!)")
        self.start()
        time.sleep(duration_sec)
        
        readings = []
        while not self.data_queue.empty():
            readings.append(self.data_queue.get()[1])
        
        if readings:
            self.baseline_watts = sum(readings) / len(readings)
            print(f"✅ Baseline set: {self.baseline_watts:.2f} W")
        else:
            print("⚠️ No baseline readings. Setting to 3.0W default for Intel Mac")
            self.baseline_watts = 3.0
        
        # Clear queue for actual run
        with self.data_queue.mutex:
            self.data_queue.queue.clear()

# --- MAIN EXECUTION ---
def run_experiment():
    # Initialize Intel Power Monitor
    monitor = IntelPowerMonitor(interval_ms=SAMPLE_INTERVAL_MS)
    monitor.calibrate_baseline(5)  # Longer calibration for Intel

    results = []
    print(f"\n🚀 Starting Phase 6: {MODEL} Energy Benchmark\n")

    for task in PROMPTS:
        print(f"Testing: {task['name']}...")
        
        # Clear previous power data
        with monitor.data_queue.mutex:
            monitor.data_queue.queue.clear()
        
        # Wait for system to settle
        time.sleep(1)
        
        # Mark start time
        start_time = time.time()
        
        # Call Ollama (Blocking)
        try:
            response = requests.post(OLLAMA_API_URL, json={
                "model": MODEL,
                "prompt": task["prompt"],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500  # Limit max tokens for testing
                }
            }, timeout=120)  # 2 minute timeout
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Process Response
            result_json = response.json()
            output_text = result_json.get("response", "")
            token_count = result_json.get("eval_count", 0)
            
            if token_count == 0:
                # Estimate tokens if not provided
                token_count = len(output_text.split()) * 1.3  # Rough estimate
                print(f"   ⚠️ Estimated tokens: {int(token_count)}")
            
            # Collect Power Data
            time.sleep(0.5)  # Let final samples arrive
            power_samples = []
            while not monitor.data_queue.empty():
                t, watts = monitor.data_queue.get()
                if start_time <= t <= end_time:
                    power_samples.append(watts)

            if not power_samples:
                print("   ❌ No power samples captured")
                continue

            # Calculate metrics
            avg_power_raw = sum(power_samples) / len(power_samples)
            avg_power_net = max(0.1, avg_power_raw - monitor.baseline_watts)  # Minimum 0.1W
            
            # Energy calculation
            total_energy_joules = avg_power_net * elapsed
            joules_per_token = total_energy_joules / token_count if token_count > 0 else 0
            sec_per_token = elapsed / token_count if token_count > 0 else 0

            print(f"   ⏱️  Time: {elapsed:.2f}s")
            print(f"   📝 Tokens: {int(token_count)}")
            print(f"   ⚡ Net Power: {avg_power_net:.2f} W (raw: {avg_power_raw:.2f} W)")
            print(f"   🔋 Total Energy: {total_energy_joules:.2f} J")
            print(f"   💎 Cost: {joules_per_token:.4f} J/token")
            print(f"   📄 Output preview: {output_text[:100]}...")
            print("-" * 40)

            results.append({
                "Task": task["name"],
                "Tokens": int(token_count),
                "Time_s": round(elapsed, 2),
                "Avg_Watts_Raw": round(avg_power_raw, 2),
                "Avg_Watts_Net": round(avg_power_net, 2),
                "Total_Joules": round(total_energy_joules, 2),
                "Joules_Per_Token": round(joules_per_token, 4),
                "Sec_Per_Token": round(sec_per_token, 4)
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
        
        # Cool down between runs
        time.sleep(3)

    # Stop Monitor
    monitor.stop()
    monitor.join()

    # Save to CSV if we have results
    if results:
        keys = results[0].keys()
        with open(OUTPUT_FILE, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        
        print(f"\n✅ Done. Results saved to {OUTPUT_FILE}")
        
        # Quick analysis
        print("\n📊 QUICK ANALYSIS:")
        print("-" * 40)
        for r in results:
            print(f"{r['Task']:20s}: {r['Joules_Per_Token']:.4f} J/token")
        
        # Calculate ratios
        if len(results) > 1:
            baseline = results[0]['Joules_Per_Token']
            print("\nAlignment Tax Ratios (vs Free):")
            for r in results[1:]:
                ratio = r['Joules_Per_Token'] / baseline if baseline > 0 else 0
                print(f"  {r['Task']:20s}: {ratio:.2f}x energy cost")
    else:
        print("\n❌ No results collected. Check Ollama is running.")

if __name__ == "__main__":
    import os
    if os.geteuid() != 0:
        print("❌ This script requires sudo for powermetrics")
        print("👉 Run: sudo python3 llm_joulemeter_intel.py")
        exit(1)
    
    run_experiment()