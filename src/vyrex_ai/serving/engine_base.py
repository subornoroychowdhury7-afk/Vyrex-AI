"""
Abstract Base Class for VYREX AI Inference Engines.
Allows seamless switching between native HuggingFace QLoRA and high-throughput vLLM.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, List
from vyrex_ai.serving.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
)


class BaseInferenceEngine(ABC):
    @abstractmethod
    async def load(self) -> None:
        """Load model and initialize weights."""
        pass

    @abstractmethod
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate complete response non-streaming."""
        pass

    @abstractmethod
    async def stream_generate(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream response chunks via Server-Sent Events (SSE)."""
        pass

    @abstractmethod
    def get_health_metrics(self) -> Dict[str, Any]:
        """Return memory, device, and load status."""
        pass
