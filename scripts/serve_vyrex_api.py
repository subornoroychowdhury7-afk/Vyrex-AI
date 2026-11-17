#!/usr/bin/env python3
"""
Launcher script for VYREX AI Production OpenAI-Compatible REST API Server.
Usage:
    python scripts/serve_vyrex_api.py --port 8000 --host 0.0.0.0 --backend hf
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to pythonpath
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description="VYREX AI Production API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument(
        "--backend",
        type=str,
        default="hf",
        choices=["hf", "vllm"],
        help="Inference engine backend ('hf' for HuggingFace QLoRA 4-bit, 'vllm' for PagedAttention vLLM)",
    )
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn workers")
    args = parser.parse_args()

    os.environ["VYREX_BACKEND"] = args.backend
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)

    import uvicorn
    print("\n" + "=" * 65)
    print("            VYREX AI PRODUCTION API SERVER")
    print("=" * 65)
    print(f"Creator: Suborno Roy Chowdhury | VYREX AI Project")
    print(f"Backend: {args.backend.upper()}")
    print(f"Listening: http://{args.host}:{args.port}")
    print(f"API Docs:  http://{args.host}:{args.port}/docs")
    print(f"Health:    http://{args.host}:{args.port}/v1/health")
    print(f"Endpoint:  http://{args.host}:{args.port}/v1/chat/completions")
    print("=" * 65 + "\n")

    uvicorn.run("vyrex_ai.serving.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
