"""
Serverless Cloud GPU Deployment for VYREX AI on Modal.
Runs on an NVIDIA A10G or T4 GPU with auto-scaling to zero when idle.

Deploy:
    modal deploy deploy/modal_app.py
"""

import modal

# Define container image with all dependencies
vyrex_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.48.0",
        "peft>=0.14.0",
        "accelerate>=1.2.0",
        "bitsandbytes>=0.45.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.32.0",
        "pydantic>=2.10.0",
        "sse-starlette>=2.2.0",
        "httpx>=0.28.0",
    )
    .add_local_dir("src", remote_path="/app/src")
    .add_local_dir("pretrained", remote_path="/app/pretrained")
    .add_local_dir("scripts", remote_path="/app/scripts")
)

app = modal.App("vyrex-ai-production", image=vyrex_image)


@app.function(
    gpu="A10G",  # or "T4" for budget tier
    scaledown_window=300,  # Keep warm for 5 minutes
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/app/src")
    from vyrex_ai.serving.api import app as vyrex_api
    return vyrex_api
