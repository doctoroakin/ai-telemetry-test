import ollama
import time
import os
import subprocess
import signal

def measure_with_top():
    """Use system 'top' command to measure Ollama process"""
    
    # Find Ollama process ID
    result = subprocess.run(['pgrep', 'ollama'], capture_output=True, text=True)
    pids = result.stdout.strip().split('\n')
    if not pids or pids[0] == '':
        print("No Ollama process found. Make sure 'ollama serve' is running.")
        return
    
    pid = pids[0]  # Use first PID
    print(f"Monitoring Ollama PID: {pid}")
    
    client = ollama.Client()
    
    # Warmup
    print("Warming up...")
    client.chat(
        model='llama3.1:8b-instruct-q4_0',
        messages=[{'role': 'user', 'content': 'Hi'}]
    )
    time.sleep(2)
    
    # Start monitoring with macOS top command
    # -l 0 = run continuously, -s 1 = 1 second interval, -pid = specific process
    top_command = f"top -l 0 -s 1 -pid {pid} > cpu_monitor.txt"
    print(f"Starting CPU monitor: {top_command}")
    monitor = subprocess.Popen(top_command, shell=True, preexec_fn=os.setsid)
    
    time.sleep(2)  # Let monitoring stabilize
    
    # Run inference
    print("\nRunning FREE inference (no constraints)...")
    start = time.time()
    response = client.chat(
        model='llama3.1:8b-instruct-q4_0',
        messages=[{'role': 'user', 'content': 'Describe a sunset.'}]
    )
    end = time.time()
    
    print(f"Time: {end-start:.2f}s")
    print(f"Output length: {len(response['message']['content'])} chars")
    
    time.sleep(3)  # Gap between tests
    
    print("\nRunning CONSTRAINED inference...")
    start = time.time()
    response = client.chat(
        model='llama3.1:8b-instruct-q4_0',
        messages=[{'role': 'user', 'content': 'Describe a sunset in exactly three sentences.'}]
    )
    end = time.time()
    
    print(f"Time: {end-start:.2f}s")
    print(f"Output length: {len(response['message']['content'])} chars")
    
    time.sleep(2)
    
    # Stop monitoring
    print("\nStopping monitor...")
    os.killpg(os.getpgid(monitor.pid), signal.SIGTERM)
    
    print("\n✓ Check cpu_monitor.txt for CPU usage")
    print("Look for the CPU% column during each inference period")

if __name__ == "__main__":
    measure_with_top()