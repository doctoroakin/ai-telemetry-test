import subprocess
import time
import re

def debug_powermetrics():
    """See exactly what powermetrics is outputting"""
    print("Starting powermetrics debug...")
    
    cmd = [
        "sudo", "powermetrics",
        "-i", "1000",  # 1 second intervals
        "--samplers", "cpu_power",
        "-n", "3"  # Just 3 samples
    ]
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    stdout, stderr = process.communicate()
    
    print("=" * 60)
    print("RAW OUTPUT:")
    print(stdout)
    print("=" * 60)
    
    # Try to find power readings
    intel_pattern = re.compile(r"Intel energy model derived package power.*?:\s+(\d+\.?\d*)\s*W")
    matches = intel_pattern.findall(stdout)
    
    if matches:
        print(f"\n✅ Found {len(matches)} power readings: {matches}")
    else:
        print("\n❌ No power readings found with current pattern")
        
        # Try alternative patterns
        alt_patterns = [
            (r"Package Power.*?:\s+(\d+\.?\d*)\s*W", "Package Power"),
            (r"CPU Power.*?:\s+(\d+\.?\d*)\s*W", "CPU Power"),
            (r"(\d+\.?\d*)\s*W", "Any Watts"),
        ]
        
        for pattern, name in alt_patterns:
            alt_matches = re.compile(pattern).findall(stdout)
            if alt_matches:
                print(f"   Found with '{name}' pattern: {alt_matches[:3]}...")
                break

if __name__ == "__main__":
    debug_powermetrics()
