import ollama
import time

print("Testing Ollama connection...")
client = ollama.Client()

# Test inference
start = time.time()
response = client.chat(
    model='llama3.1:8b-instruct-q4_0',
    messages=[{'role': 'user', 'content': 'Count to 5'}]
)
end = time.time()

print(f"Response: {response['message']['content']}")
print(f"Time taken: {end - start:.2f} seconds")
print("Success! Ollama is working.")
