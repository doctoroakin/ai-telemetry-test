import ollama
import psutil
import subprocess
import time
import statistics

def find_ollama_runner():
    """Find the ollama runner process"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'cmdline' in proc.info and proc.info['cmdline']:
                if any('ollama runner' in ' '.join(proc.info['cmdline']) for cmd in [proc.info['cmdline']]):
                    return proc.pid
        except:
            pass
    return None

def monitor_inference(prompt, label):
    """Monitor the actual runner process"""
    client = ollama.Client()
    
    # Find runner PID
    runner_pid = find_ollama_runner()
    if not runner_pid:
        print("No runner found. Running a warmup...")
        client.chat(model='llama3.1:8b-instruct-q4_0', 
                   messages=[{'role': 'user', 'content': 'Hi'}])
        time.sleep(1)
        runner_pid = find_ollama_runner()
    
    print(f"\n{label}")
    print(f"Monitoring runner PID: {runner_pid}")
    
    try:
        runner = psutil.Process(runner_pid)
        
        # Baseline
        runner.cpu_percent()  # First call to initialize
        time.sleep(0.5)
        
        # Monitor during inference
        cpu_samples = []
        
        # Start inference
        start = time.time()
        response_generator = client.chat(
            model='llama3.1:8b-instruct-q4_0',
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )
        
        # Collect CPU samples during streaming
        for chunk in response_generator:
            cpu = runner.cpu_percent(interval=0)
            if cpu > 0:  # Only record non-zero samples
                cpu_samples.append(cpu)
        
        end = time.time()
        
        # Results
        if cpu_samples:
            print(f"Time: {end-start:.2f}s")
            print(f"CPU samples: {len(cpu_samples)}")
            print(f"Max CPU: {max(cpu_samples):.1f}%")
            print(f"Mean CPU: {statistics.mean(cpu_samples):.1f}%")
            print(f"Median CPU: {statistics.median(cpu_samples):.1f}%")
        else:
            print("No CPU samples collected")
            
        return cpu_samples
        
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    print("OLLAMA RUNNER PROCESS MONITORING")
    print("=" * 50)
    
    # Warmup
    monitor_inference("Hi", "WARMUP")
    time.sleep(2)
    
    # Test free
    free_cpu = monitor_inference(
        "Describe a sunset.",
        "FREE GENERATION"
    )
    
    time.sleep(2)
    
    # Test constrained
    constrained_cpu = monitor_inference(
        "Describe a sunset in exactly three sentences without using the words 'sun', 'sky', or 'orange'.",
        "HARD CONSTRAINED"
    )
    
    # Compare
    print("\n" + "=" * 50)
    print("COMPARISON:")
    if free_cpu and constrained_cpu:
        print(f"Free:        Mean {statistics.mean(free_cpu):.1f}% CPU (n={len(free_cpu)})")
        print(f"Constrained: Mean {statistics.mean(constrained_cpu):.1f}% CPU (n={len(constrained_cpu)})")
        
        if statistics.mean(free_cpu) > 0:
            increase = ((statistics.mean(constrained_cpu) - statistics.mean(free_cpu)) / statistics.mean(free_cpu)) * 100
            print(f"Alignment Tax: {increase:+.1f}%")
