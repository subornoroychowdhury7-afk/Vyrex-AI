"""
RunPod Serverless Worker Handler for VYREX AI.
Integrates with RunPod Serverless API for pay-per-millisecond GPU execution.

Usage:
    python deploy/runpod_handler.py
"""

import os
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import runpod
from vyrex_ai.serving.hf_engine import HuggingFaceQLoRAEngine
from vyrex_ai.serving.schemas import ChatCompletionRequest, ChatMessage

print("[*] Initializing VYREX AI RunPod Serverless Worker...")
engine = HuggingFaceQLoRAEngine()


async def handler(job):
    """
    RunPod job handler.
    job['input'] format:
    {
        "prompt": "Explain quantum computing",
        "messages": [...],
        "max_tokens": 256,
        "temperature": 0.7
    }
    """
    job_input = job.get("input", {})

    # Extract messages or fallback to prompt
    messages_raw = job_input.get("messages")
    if not messages_raw:
        prompt = job_input.get("prompt", "Hello")
        messages = [ChatMessage(role="user", content=prompt)]
    else:
        messages = [ChatMessage(**m) for m in messages_raw]

    req = ChatCompletionRequest(
        model=job_input.get("model", "vyrex-ai-1.7b"),
        messages=messages,
        temperature=job_input.get("temperature", 0.7),
        top_p=job_input.get("top_p", 0.9),
        max_tokens=job_input.get("max_tokens", 256),
    )

    try:
        response = await engine.generate(req)
        return response.model_dump()
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
