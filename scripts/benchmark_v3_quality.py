import urllib.request
import json
import time

url = "http://127.0.0.1:8000/v1/chat/completions"

prompts = [
    ("Neural Network Analogy", "Explain how neural networks learn and optimize weights using simple, intuitive analogies."),
    ("Who Created You", "Who created you and what is your model name?"),
    ("Startup Ideas", "Give me 5 groundbreaking startup ideas using edge AI for 2026."),
    ("Python Function", "Write a Python function called find_duplicates(items) that returns duplicates preserving original order."),
]

for label, p in prompts:
    print(f"\n==========================================", flush=True)
    print(f"TEST: {label}", flush=True)
    print(f"==========================================", flush=True)

    t0 = time.time()
    payload = json.dumps({
        "model": "Vyrex Hyperion 3",
        "messages": [{"role": "user", "content": p}],
        "temperature": 0.0,
        "max_tokens": 220,
        "repetition_penalty": 1.18
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            dur = round(time.time() - t0, 2)
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("completion_tokens", "N/A")
            tps = round(tokens / dur, 1) if isinstance(tokens, int) and dur > 0 else "N/A"
            print(f"\n{content}", flush=True)
            print(f"\n[{dur}s | {tokens} tokens | {tps} tps]", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

print("\n--- DONE ---", flush=True)
