import urllib.request
import json
import time

url = 'http://localhost:8000/v1/chat/completions'
prompts = [
    'Explain how neural networks learn and optimize weights using simple, intuitive analogies.',
    'Write a high-performance Python function with type hints and edge-case tests.',
    'Who created you and what is your model name?'
]

for p in prompts:
    print(f"\n=== PROMPT: {p} ===")
    t0 = time.time()
    payload = json.dumps({
        'model': 'Vyrex Hyperion 3',
        'messages': [{'role': 'user', 'content': p}],
        'temperature': 0.0,
        'max_tokens': 384
    }).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            dur = round(time.time() - t0, 2)
            data = json.loads(resp.read().decode('utf-8'))
            content = data['choices'][0]['message']['content']
            tokens = data.get('usage', {}).get('completion_tokens', 'N/A')
            print(content)
            print(f"\n[Generated in {dur}s | Tokens: {tokens}]")
    except Exception as e:
        print(f"Exception: {e}")
    print("=" * 60)
