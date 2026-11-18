"""
Pydantic v2 Schemas for VYREX AI Backend.
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
    model: str = Field(default="vyrex-ai-1.7b", description="Model identifier")
    messages: List[ChatMessage] = Field(..., description="List of messages in conversation")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    stream: Optional[bool] = Field(default=False, description="Whether to stream response chunks via SSE")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Optional stop sequences")


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
    model: str = "vyrex-ai-1.7b"
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
    model: str = "vyrex-ai-1.7b"
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
    model_name: str = "vyrex-ai-1.7b"
    backend_mode: str
    device: str
    uptime_seconds: float
    platform: str = "Render"
    creator: str = "Suborno Roy Chowdhury"
