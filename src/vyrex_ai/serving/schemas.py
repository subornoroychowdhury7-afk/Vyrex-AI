"""
OpenAI-compatible Pydantic v2 schemas for VYREX AI REST API.
Adheres strictly to the OpenAI /v1/chat/completions specification.
"""

from typing import List, Optional, Union, Dict, Any, Literal
import time
import uuid
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="Vyrex Hyperion 3", description="Model identifier")
    messages: List[ChatMessage] = Field(..., description="List of messages in conversation")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=256, ge=1, le=4096)
    stream: Optional[bool] = Field(default=False, description="Whether to stream response chunks via SSE")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Optional stop sequences")
    repetition_penalty: Optional[float] = Field(default=1.1, ge=1.0, le=2.0)
    boundary_bias: Optional[float] = Field(default=4.8, description="VYREX calibrated stop boundary bias")


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "Vyrex Hyperion 3"
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[Literal["system", "user", "assistant"]] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "Vyrex Hyperion 3"
    choices: List[ChatCompletionChunkChoice]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "Suborno Roy Chowdhury | VYREX AI Project"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    model_name: str = "Vyrex Hyperion 3"
    backend: str = "qwen3-1.7b-qlora-v3"
    quantization: str = "nf4-double-quant-bf16"
    device: str = "cuda"
    uptime_seconds: float = 0.0
    memory_allocated_mb: Optional[float] = None
