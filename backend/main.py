"""
VYREX AI Production Backend Server for Render.
Provides OpenAI-compatible REST API endpoints with SSE streaming and CORS for Vercel.
"""

from contextlib import asynccontextmanager
import time
import uuid

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

import httpx

from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthResponse,
    ModelCard,
    ModelListResponse,
)
from engine import VyrexBackendEngine
from config import HOST, PORT, FALLBACK_API_KEY, FALLBACK_API_BASE, FALLBACK_MODEL, REMOTE_GPU_URL

engine = VyrexBackendEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Starting VYREX AI Render Backend Service...")
    await engine.initialize()
    status_info = engine.get_status()
    print(f"[+] Engine initialized. Mode: {status_info['backend_mode']} | Device: {status_info['device']}")
    yield
    print("[*] Shutting down VYREX AI Render Backend Service...")


app = FastAPI(
    title="VYREX AI Render Backend",
    version="1.0.0",
    description="Production REST API serving VYREX AI for Vercel frontend deployments.",
    lifespan=lifespan,
)

# Robust CORS middleware for Vercel frontend and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Vercel preview URLs, production domains, and localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
)


@app.middleware("http")
async def add_metadata_headers(request: Request, call_next):
    request_id = request.headers.get("x-request-id", f"req-{uuid.uuid4().hex[:12]}")
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    return response


@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
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


@app.get("/")
async def root():
    return {
        "service": "VYREX AI Production API",
        "creator": "Vyrex AI Project",
        "docs_url": "/docs",
        "health_url": "/v1/health",
        "models_url": "/v1/models",
        "chat_url": "/v1/chat/completions",
    }


@app.get("/health", response_model=HealthResponse)
@app.get("/v1/health", response_model=HealthResponse)
async def health():
    info = engine.get_status()
    return HealthResponse(**info)


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    return ModelListResponse(
        data=[
            ModelCard(id="Vyrex Hyperion 3"),
            ModelCard(id="vyrex-hyperion-3"),
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        async def event_generator():
            async for chunk in engine.stream_generate(request):
                yield {"data": chunk.model_dump_json()}
            yield {"data": "[DONE]"}

        return EventSourceResponse(event_generator())

    response = await engine.generate(request)
    return response


@app.get("/debug/groq")
async def debug_groq():
    """Live diagnostic: tests Groq API key directly from this Render instance."""
    import os
    key_raw = os.getenv("FALLBACK_API_KEY") or os.getenv("GROQ_API_KEY") or ""
    key = key_raw.strip().strip('"').strip("'")
    result = {
        "key_env_name": "FALLBACK_API_KEY" if os.getenv("FALLBACK_API_KEY") else ("GROQ_API_KEY" if os.getenv("GROQ_API_KEY") else "NOT_SET"),
        "key_length": len(key),
        "key_prefix": key[:8] + "..." if len(key) > 8 else "(empty)",
        "api_base": FALLBACK_API_BASE,
        "model": FALLBACK_MODEL,
        "engine_mode": engine.mode,
        "remote_gpu_url": REMOTE_GPU_URL or "(not set)",
    }
    if not key:
        result["groq_test"] = "SKIP — no API key configured on this Render instance"
        return result
    try:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": FALLBACK_MODEL,
            "messages": [{"role": "user", "content": "Say: OK"}],
            "max_tokens": 10,
            "stream": False,
        }
        url = f"{FALLBACK_API_BASE.rstrip('/')}/chat/completions"
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, headers=headers, json=payload)
            result["groq_http_status"] = r.status_code
            result["groq_response_body"] = r.text[:600]
            if r.status_code == 200:
                data = r.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                result["groq_content"] = content
                result["groq_test"] = "PASS" if content else "FAIL — empty content"
            else:
                result["groq_test"] = f"FAIL — HTTP {r.status_code}"
    except Exception as e:
        result["groq_test"] = f"ERROR — {e}"
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
