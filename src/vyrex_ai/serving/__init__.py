"""
VYREX AI Serving Package.
OpenAI-compatible inference engines, schemas, and REST API.
"""

from vyrex_ai.serving.api import app
from vyrex_ai.serving.schemas import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
)

__all__ = ["app", "ChatMessage", "ChatCompletionRequest", "ChatCompletionResponse"]
