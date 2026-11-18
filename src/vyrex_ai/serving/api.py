"""
Production FastAPI Application for VYREX AI.
Implements the OpenAI-compatible REST API specification.
"""

from contextlib import asynccontextmanager
import json
import os
import time
import uuid
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from vyrex_ai.serving.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelListResponse,
    HealthResponse,
    ChatMessage,
)
from vyrex_ai.serving.hf_engine import HuggingFaceQLoRAEngine
try:
    from vyrex_ai.serving.vllm_engine import VLLMInferenceEngine
except ImportError:
    VLLMInferenceEngine = None  # vllm not installed (e.g. on Modal HF-only deploy)
from vyrex_ai.serving.engine_base import BaseInferenceEngine

# Global engine instance
engine: Optional[BaseInferenceEngine] = None
BACKEND_TYPE = os.getenv("VYREX_BACKEND", "hf").lower()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print(f"[*] Starting VYREX AI Production API Server (Backend: {BACKEND_TYPE.upper()})...")
    if BACKEND_TYPE == "vllm":
        if VLLMInferenceEngine is None:
            print("[!] vllm not installed — falling back to HF engine")
            engine = HuggingFaceQLoRAEngine()
        else:
            engine = VLLMInferenceEngine()
    else:
        engine = HuggingFaceQLoRAEngine()

    try:
        await engine.load()
        print("[+] VYREX AI model engine initialized and ready for inference.")
    except Exception as e:
        print(f"[!] Warning during model load: {e}")
        # Allows health checks and lazy initialization
    yield
    print("[*] Shutting down VYREX AI API Server...")


app = FastAPI(
    title="VYREX AI Production Inference API",
    version="1.0.0",
    description=(
        "Production-grade, OpenAI-compatible REST API serving the fine-tuned VYREX AI "
        "(Qwen3-1.7B + QLoRA V3) conversational model. Architected by Suborno Roy Chowdhury."
    ),
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_and_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", f"req-{uuid.uuid4().hex[:12]}")
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{process_time:.2f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": str(exc),
                "type": "internal_server_error",
                "param": None,
                "code": 500,
            }
        },
    )


@app.get("/", tags=["General"])
async def root():
    return {
        "service": "VYREX AI Production API",
        "creator": "Suborno Roy Chowdhury",
        "docs_url": "/docs",
        "health_url": "/v1/health",
        "chat_url": "/v1/chat/completions",
    }


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
@app.get("/v1/health", response_model=HealthResponse, tags=["Observability"])
async def health():
    if engine is None:
        return HealthResponse(status="initializing")
    metrics = engine.get_health_metrics()
    return HealthResponse(**metrics)


@app.get("/v1/models", response_model=ModelListResponse, tags=["OpenAI Compatible"])
async def list_models():
    return ModelListResponse(
        data=[
            ModelCard(id="Vyrex Hyperion 3"),
            ModelCard(id="vyrex-hyperion-3"),
            ModelCard(id="vyrex-ai-1.7b"),
        ]
    )


@app.post("/v1/chat/completions", tags=["OpenAI Compatible"])
async def chat_completions(request: ChatCompletionRequest):
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not yet initialized.",
        )

    if request.stream:
        async def event_generator() -> AsyncIterator[dict]:
            async for chunk in engine.stream_generate(request):
                yield {"data": chunk.model_dump_json()}
            yield {"data": "[DONE]"}

        return EventSourceResponse(event_generator())

    response = await engine.generate(request)
    return response


@app.post("/v1/completions", tags=["OpenAI Compatible"])
async def text_completions(request: dict):
    """Legacy completions endpoint wrapping the prompt into chat format."""
    prompt = request.get("prompt", "")
    req = ChatCompletionRequest(
        model=request.get("model", "vyrex-ai-1.7b"),
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=request.get("temperature", 0.7),
        top_p=request.get("top_p", 0.9),
        max_tokens=request.get("max_tokens", 256),
    )
    res = await engine.generate(req)
    text = res.choices[0].message.content
    return {
        "id": res.id,
        "object": "text_completion",
        "created": res.created,
        "model": res.model,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": res.usage.model_dump(),
    }


def main():
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    print(f"[*] Starting VYREX AI Server on http://{host}:{port}")
    uvicorn.run("vyrex_ai.serving.api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
