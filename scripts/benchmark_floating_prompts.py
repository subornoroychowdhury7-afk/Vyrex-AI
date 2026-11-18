import urllib.request
import json
import time

url = "http://127.0.0.1:8000/v1/chat/completions"

prompts = [
    ("p2: Code & Architecture", "Write a high-performance Python function with type hints and edge-case tests."),
    ("p3: Write & Refine", "Draft an engaging and persuasive product launch announcement for an AI platform."),
    ("p5: FastAPI Backend", "Show me an idiomatic async FastAPI route with request validation and clean error handling."),
    ("p6: Agentic UI/UX", "What are the most exciting UI/UX patterns for modern conversational AI agents?"),
    ("p7: Transformer Attention", "How does multi-head self-attention work mathematically in transformers?"),
    ("p8: QLoRA Article Intro", "Write a compelling introduction for an article about fine-tuning LLMs with QLoRA."),
]

results = []

for label, p in prompts:
    print(f"\n{'='*50}", flush=True)
    print(f"TESTING: {label}", flush=True)
    print(f"PROMPT: {p}", flush=True)
    print(f"{'='*50}", flush=True)

    t0 = time.time()
    payload = json.dumps({
        "model": "Vyrex Hyperion 3",
        "messages": [{"role": "user", "content": p}],
        "temperature": 0.0,
        "max_tokens": 180,
        "repetition_penalty": 1.18
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            dur = round(time.time() - t0, 2)
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            tps = round(tokens / dur, 1) if dur > 0 else 0
            print(f"\nRESPONSE:\n{content}\n", flush=True)
            print(f"[{dur}s | {tokens} tokens | {tps} tps]", flush=True)
            results.append((label, dur, tokens, tps, "PASS" if tokens > 10 else "FAIL"))
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        results.append((label, 0, 0, 0, f"ERR: {e}"))

print(f"\n{'='*50}\nSUMMARY\n{'='*50}")
for label, dur, tokens, tps, status in results:
    print(f"{label}: {status} in {dur}s ({tokens} tokens, {tps} tps)")
