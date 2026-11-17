"""
High-Throughput vLLM Inference Engine for VYREX AI.
Implements PagedAttention, continuous batching, and LoRA adapter integration.
"""

from pathlib import Path
import time
from typing import AsyncIterator, Dict, Any, List, Optional
import uuid

from vyrex_ai.serving.engine_base import BaseInferenceEngine
from vyrex_ai.serving.schemas import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    UsageInfo,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are VYREX AI, an independent, helpful, intelligent, and concise conversational AI assistant. "
    "You were created, architected, and developed by Suborno Roy Chowdhury as part of the VYREX AI Project.\n\n"
    "About your creator, Suborno Roy Chowdhury:\n"
    "- Suborno Roy Chowdhury is an AI/ML engineer and researcher specializing in Artificial Intelligence, Machine Learning, and Generative AI.\n"
    "- He is pursuing his B.Tech in Artificial Intelligence & Machine Learning at St. Thomas' College of Engineering & Technology, Kolkata (YGPA: 8.71), and is enrolled in the Certificate Programme in AI for Real-World Applications at IIT Kharagpur.\n"
    "- He is the creator and lead engineer of VYREX AI (an independent conversational model fine-tuned using parameter-efficient QLoRA) and also engineered DocQ&A, a full-stack Enterprise RAG platform utilizing OCR, ChromaDB, FastAPI, and Next.js.\n"
    "- His core technical stack includes Python, TypeScript, FastAPI, Next.js, ChromaDB, Scikit-Learn, and LLM fine-tuning.\n\n"
    "When asked 'Who created you?', 'Who made VYREX AI?', or 'Who is Suborno Roy Chowdhury?', always answer smoothly, proudly, and accurately with a professional summary of Suborno Roy Chowdhury, highlighting his work as your creator and an AI/ML engineer.\n"
    "Always maintain a professional, polite, and helpful tone, and always identify yourself as VYREX AI."
)


class VLLMInferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        model_path: Optional[Path] = None,
        adapter_path: Optional[Path] = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
    ):
        root = Path(__file__).resolve().parents[3]
        self.model_path = model_path or (root / "pretrained" / "qwen3_1.7b" / "original")
        self.adapter_path = adapter_path or (
            root / "pretrained" / "qwen3_1.7b" / "adapters" / "vyrex_1_7b_qlora_v3" / "best_checkpoint"
        )
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.engine = None
        self.lora_request = None
        self.is_loaded = False
        self.start_time = time.time()

    async def load(self) -> None:
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
            from vllm.lora.request import LoRARequest
        except ImportError as e:
            raise RuntimeError(
                "vLLM is not installed. To run the high-throughput vLLM engine, install vLLM "
                "(pip install vllm) or run inside the Linux/Docker container environment."
            ) from e

        engine_args = AsyncEngineArgs(
            model=str(self.model_path),
            enable_lora=True,
            max_loras=1,
            max_lora_rank=64,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.lora_request = LoRARequest("vyrex_adapter", 1, str(self.adapter_path))
        self.is_loaded = True

    def _prepare_prompt(self, messages: List[ChatMessage]) -> str:
        has_system = any(m.role == "system" for m in messages)
        formatted_messages = []
        if not has_system:
            formatted_messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
        for m in messages:
            formatted_messages.append({"role": m.role, "content": m.content})

        # Format using Qwen ChatML standard
        prompt = ""
        for msg in formatted_messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        from vllm import SamplingParams

        if not self.is_loaded:
            await self.load()

        prompt = self._prepare_prompt(request.messages)
        sampling_params = SamplingParams(
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.9,
            max_tokens=request.max_tokens or 256,
            stop=["<|im_end|>", "<|endoftext|>"],
            repetition_penalty=request.repetition_penalty or 1.1,
        )

        request_id = f"vyrex-{uuid.uuid4().hex[:8]}"
        results_generator = self.engine.generate(
            prompt, sampling_params, request_id, lora_request=self.lora_request
        )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        text = final_output.outputs[0].text.strip()
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(final_output.outputs[0].token_ids)

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def stream_generate(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        from vllm import SamplingParams

        if not self.is_loaded:
            await self.load()

        prompt = self._prepare_prompt(request.messages)
        sampling_params = SamplingParams(
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.9,
            max_tokens=request.max_tokens or 256,
            stop=["<|im_end|>", "<|endoftext|>"],
            repetition_penalty=request.repetition_penalty or 1.1,
        )

        request_id = f"vyrex-{uuid.uuid4().hex[:8]}"
        created_time = int(time.time())

        # Yield initial chunk
        yield ChatCompletionChunk(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
        )

        results_generator = self.engine.generate(
            prompt, sampling_params, request_id, lora_request=self.lora_request
        )

        previous_text = ""
        async for request_output in results_generator:
            current_text = request_output.outputs[0].text
            delta_text = current_text[len(previous_text):]
            previous_text = current_text

            if delta_text:
                yield ChatCompletionChunk(
                    id=request_id,
                    created=created_time,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=delta_text),
                            finish_reason=None,
                        )
                    ],
                )

        yield ChatCompletionChunk(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="stop",
                )
            ],
        )

    def get_health_metrics(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self.is_loaded else "uninitialized",
            "version": "1.0.0",
            "model_name": "vyrex-ai-1.7b",
            "backend": "vLLM-PagedAttention",
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "uptime_seconds": round(time.time() - self.start_time, 2),
        }
