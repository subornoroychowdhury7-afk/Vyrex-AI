"""
Serverless Cloud GPU Deployment for VYREX AI on Modal.

Runs the fine-tuned Vyrex Hyperion 3 (Qwen3-1.7B + QLoRA v3) on an NVIDIA T4 GPU
with auto-scaling to zero when idle — giving you a 24/7 worldwide endpoint
without keeping your laptop on.

Deploy:
    .\.venv\Scripts\modal deploy deploy/modal_app.py

Your permanent URL will look like:
    https://<username>--vyrex-ai-production-fastapi-app.modal.run
"""

import modal

# ---------------------------------------------------------------------------
# Container Image
# ---------------------------------------------------------------------------
# Builds once, cached by Modal. Re-built only if pip deps change.
vyrex_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core ML stack
        "torch>=2.4.0",
        "transformers>=4.48.0",
        "peft>=0.14.0",
        "accelerate>=1.2.0",
        "bitsandbytes>=0.45.0",
        # API server
        "fastapi>=0.115.0",
        "uvicorn>=0.32.0",
        "pydantic>=2.10.0",
        "sse-starlette>=2.2.0",
        "httpx>=0.28.0",
    )
    # Upload your source code into the container
    .add_local_dir("src", remote_path="/app/src")
    # Upload your model weights (Qwen3-1.7B base + QLoRA adapter) into the container
    # This is what makes YOUR weights run in the cloud, not some generic LLM
    .add_local_dir("pretrained", remote_path="/app/pretrained")
)

app = modal.App("vyrex-ai-production", image=vyrex_image)


# ---------------------------------------------------------------------------
# Serverless GPU Function
# ---------------------------------------------------------------------------
@app.function(
    gpu="T4",                   # NVIDIA T4 — 16 GB VRAM, well within budget ($30 free/mo)
    scaledown_window=300,       # Stay warm 5 min after last request; scale to zero after
    timeout=600,                # Allow up to 10 min for cold-start model loading
    _allow_background_volume_commits=True,
)
@modal.asgi_app()
def fastapi_app():
    import sys
    import os

    # Ensure /app/src is on the Python path so `vyrex_ai` package is importable
    sys.path.insert(0, "/app/src")

    # Tell the HF engine to use HF backend (not vllm)
    os.environ.setdefault("VYREX_BACKEND", "hf")

    # Import and return the production FastAPI app (defined in src/vyrex_ai/serving/api.py)
    from vyrex_ai.serving.api import app as vyrex_api
    return vyrex_api
