#!/usr/bin/env python3
"""
vLLM High-Throughput Production Server for VYREX AI.
Launches the official vLLM OpenAI-compatible server with PagedAttention and continuous batching.

Usage:
    python scripts/serve_vllm.py --port 8000
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = ROOT / "pretrained" / "qwen3_1.7b" / "original"
ADAPTER = ROOT / "pretrained" / "qwen3_1.7b" / "adapters" / "vyrex_1_7b_qlora_v3" / "best_checkpoint"
MERGED_MODEL = ROOT / "pretrained" / "vyrex_1_7b_merged"


def main():
    parser = argparse.ArgumentParser(description="Serve VYREX AI with vLLM")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    # Determine model path
    if MERGED_MODEL.is_dir() and any(MERGED_MODEL.glob("*.safetensors")):
        model_target = str(MERGED_MODEL)
        lora_args = []
        print(f"[*] Detected merged standalone model at: {model_target}")
    else:
        model_target = str(BASE_MODEL)
        lora_args = [
            "--enable-lora",
            "--lora-modules",
            f"vyrex-ai-1.7b={ADAPTER}",
        ]
        print(f"[*] Serving base model with dynamic LoRA adapter: {ADAPTER}")

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_target,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--trust-remote-code",
        *lora_args,
    ]

    print("\n" + "=" * 65)
    print("         VYREX AI: HIGH-THROUGHPUT vLLM PRODUCTION SERVER")
    print("=" * 65)
    print(f"Command: {' '.join(cmd)}")
    print(f"Serving: http://{args.host}:{args.port}")
    print("=" * 65 + "\n")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[!] Error: vLLM is not installed or not in PATH.")
        print("    Install vLLM on Linux / WSL / Docker with: pip install vllm")


if __name__ == "__main__":
    main()
