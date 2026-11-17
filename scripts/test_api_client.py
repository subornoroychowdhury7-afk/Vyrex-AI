#!/usr/bin/env python3
"""
Test client for VYREX AI Production REST API.
Demonstrates:
1. Health check & model list retrieval.
2. Direct HTTP JSON chat completion.
3. Streaming chat completion via SSE.
4. Official OpenAI Python SDK compatibility.

Usage:
    python scripts/test_api_client.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
import time
import httpx


def test_health_and_models(base_url: str):
    print("\n--- 1. Testing Health & Model Endpoints ---")
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        res = client.get("/v1/health")
        print(f"[+] /v1/health status: {res.status_code}")
        print(json.dumps(res.json(), indent=2))

        models_res = client.get("/v1/models")
        print(f"[+] /v1/models status: {models_res.status_code}")
        print(json.dumps(models_res.json(), indent=2))


def test_chat_completion_json(base_url: str, prompt: str):
    print(f"\n--- 2. Testing Non-Streaming Chat Completion ---")
    print(f"Prompt: {prompt}")
    payload = {
        "model": "vyrex-ai-1.7b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": False,
    }

    start = time.perf_counter()
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        res = client.post("/v1/chat/completions", json=payload)
    elapsed = (time.perf_counter() - start) * 1000

    if res.status_code == 200:
        data = res.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data["usage"]["completion_tokens"]
        print(f"\n[+] Status: {res.status_code} ({elapsed:.1f} ms | {tokens} tokens)")
        print(f"VYREX Response:\n{content}")
    else:
        print(f"[!] Request failed: {res.status_code} - {res.text}")


def test_chat_completion_stream(base_url: str, prompt: str):
    print(f"\n--- 3. Testing Streaming SSE Chat Completion ---")
    print(f"Prompt: {prompt}")
    payload = {
        "model": "vyrex-ai-1.7b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": True,
    }

    print("VYREX Stream: ", end="", flush=True)
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            print(delta, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
    print("\n[+] Stream completed.")


def test_openai_sdk(base_url: str, prompt: str):
    print(f"\n--- 4. Testing Official OpenAI Python SDK Compatibility ---")
    try:
        from openai import OpenAI
    except ImportError:
        print("[i] 'openai' package not installed. Skipping official SDK test (install with `pip install openai`).")
        return

    client = OpenAI(base_url=f"{base_url}/v1", api_key="vyrex-local-token")
    response = client.chat.completions.create(
        model="vyrex-ai-1.7b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    print(f"[+] OpenAI SDK Call Succeeded!")
    print(f"VYREX Response: {response.choices[0].message.content}")


def main():
    parser = argparse.ArgumentParser(description="Test VYREX AI REST API")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Who created you and what is your architecture?",
    )
    args = parser.parse_args()

    try:
        test_health_and_models(args.base_url)
        test_chat_completion_json(args.base_url, args.prompt)
        test_chat_completion_stream(args.base_url, "Explain 4-bit quantization in two sentences.")
        test_openai_sdk(args.base_url, args.prompt)
    except httpx.ConnectError:
        print(f"\n[!] Could not connect to {args.base_url}. Make sure the server is running!")
        print("    Start server with: python scripts/serve_vyrex_api.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
