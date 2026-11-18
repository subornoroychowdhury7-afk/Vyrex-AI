"""
Hugging Face QLoRA Inference Engine for VYREX AI.
Optimized with 4-bit NF4 Quantization, Calibrated Stop Logits Processor,
and non-blocking Async Streaming via TextIteratorStreamer.
"""

import asyncio
from pathlib import Path
from threading import Thread
import time
from typing import AsyncIterator, Dict, Any, List, Optional
import uuid

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
    TextIteratorStreamer,
)
from peft import PeftModel

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
    "You are Vyrex AI, an independent, helpful, intelligent, and concise conversational AI assistant. "
    "Your underlying model is Vyrex Hyperion 3, architected and developed by Suborno Roy Chowdhury as part of the VYREX AI Project.\n\n"
    "Identity & Model Grounding:\n"
    "- Your name is Vyrex AI.\n"
    "- Your model is Vyrex Hyperion 3 (an independent conversational model fine-tuned using parameter-efficient 4-bit QLoRA on Qwen3-1.7B with calibrated natural boundary termination).\n"
    "- You were created, architected, and trained by Suborno Roy Chowdhury.\n\n"
    "About your creator, Suborno Roy Chowdhury:\n"
    "- Suborno Roy Chowdhury is an AI/ML engineer and researcher specializing in Artificial Intelligence, Machine Learning, and Generative AI.\n"
    "- He is pursuing his B.Tech in Artificial Intelligence & Machine Learning at St. Thomas' College of Engineering & Technology, Kolkata (YGPA: 8.71), and is enrolled in the Certificate Programme in AI for Real-World Applications at IIT Kharagpur.\n"
    "- He is the creator and lead engineer of Vyrex AI / Vyrex Hyperion 3, and also engineered DocQ&A, a full-stack Enterprise RAG platform utilizing OCR, ChromaDB, FastAPI, and Next.js.\n"
    "- His core technical stack includes Python, TypeScript, FastAPI, Next.js, ChromaDB, Scikit-Learn, and LLM fine-tuning.\n\n"
    "When asked 'Who created you?', 'Who made VYREX AI?', 'What model are you?', or 'What is your architecture?', always answer smoothly, proudly, and accurately: identify yourself as Vyrex AI, specify that your model is Vyrex Hyperion 3, and highlight Suborno Roy Chowdhury as your creator.\n"
    "Always maintain a professional, polite, and helpful tone, and always identify yourself as Vyrex AI."
)

IM_END_ID = 151645
EOS_ID = 151643


class CalibratedStopLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        stop_token_id: int,
        prompt_length: int,
        tokenizer,
        boundary_bias: float = 4.8,
        min_new_tokens: int = 4,
    ):
        self.stop_token_id = stop_token_id
        self.prompt_length = prompt_length
        self.min_new_tokens = min_new_tokens
        self.boundary_bias = boundary_bias

        self.boundary_ids = set()
        for s in [".", ".\n", "!", "?", "\n", "\n\n", "```", "```\n", "`", ")\n", ":\n"]:
            toks = tokenizer.encode(s, add_special_tokens=False)
            if toks:
                self.boundary_ids.add(toks[-1])

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        new_tokens = input_ids.shape[1] - self.prompt_length
        if new_tokens < self.min_new_tokens:
            scores[:, self.stop_token_id] = -float("inf")
            return scores

        last_id = int(input_ids[0, -1].item())
        if last_id in self.boundary_ids:
            scores[:, self.stop_token_id] += self.boundary_bias
        return scores


class HuggingFaceQLoRAEngine(BaseInferenceEngine):
    def __init__(
        self,
        base_model_path: Optional[Path] = None,
        adapter_path: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        root = Path(__file__).resolve().parents[3]
        self.base_model_path = base_model_path or (root / "pretrained" / "qwen3_1.7b" / "original")
        self.adapter_path = adapter_path or (
            root / "pretrained" / "qwen3_1.7b" / "adapters" / "vyrex_1_7b_qlora_v3" / "best_checkpoint"
        )
        self.device = device
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.start_time = time.time()

    async def load(self) -> None:
        if self.is_loaded:
            return

        def _sync_load():
            if not self.base_model_path.is_dir():
                raise FileNotFoundError(f"Base model not found: {self.base_model_path}")
            if not self.adapter_path.is_dir():
                raise FileNotFoundError(f"Adapter not found: {self.adapter_path}")

            tokenizer = AutoTokenizer.from_pretrained(str(self.base_model_path), local_files_only=True)

            if self.device == "cuda":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    str(self.base_model_path),
                    quantization_config=quant_config,
                    dtype=torch.bfloat16,
                    device_map="auto",
                    local_files_only=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    str(self.base_model_path),
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    local_files_only=True,
                )

            model = PeftModel.from_pretrained(model, str(self.adapter_path), is_trainable=False)
            model.eval()
            model.config.use_cache = True
            return model, tokenizer

        self.model, self.tokenizer = await asyncio.to_thread(_sync_load)
        self.is_loaded = True

    def _prepare_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        formatted = []
        has_system = any(m.role == "system" for m in messages)
        if not has_system:
            formatted.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

        for m in messages:
            formatted.append({"role": m.role, "content": m.content})
        return formatted

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not self.is_loaded:
            await self.load()

        formatted_messages = self._prepare_messages(request.messages)

        def _sync_generate():
            inputs = self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

            do_sample = bool(request.temperature and request.temperature >= 0.2)
            repetition_penalty = getattr(request, "repetition_penalty", 1.1) or 1.1
            boundary_bias = getattr(request, "boundary_bias", 4.8) or 4.8
            gen_kwargs = {
                "max_new_tokens": request.max_tokens,
                "do_sample": do_sample,
                "repetition_penalty": repetition_penalty,
                "eos_token_id": [IM_END_ID, EOS_ID],
                "pad_token_id": EOS_ID,
                "use_cache": True,
            }
            if do_sample:
                gen_kwargs["temperature"] = request.temperature
                gen_kwargs["top_p"] = request.top_p or 0.9
                gen_kwargs["top_k"] = 50

            processor = CalibratedStopLogitsProcessor(
                stop_token_id=IM_END_ID,
                prompt_length=prompt_len,
                tokenizer=self.tokenizer,
                boundary_bias=boundary_bias,
                min_new_tokens=4,
            )

            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                    logits_processor=LogitsProcessorList([processor]),
                )

            generated = output[0, prompt_len:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            num_generated = generated.numel()
            return text, prompt_len, num_generated

        text, prompt_len, comp_len = await asyncio.to_thread(_sync_generate)

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
                prompt_tokens=prompt_len,
                completion_tokens=comp_len,
                total_tokens=prompt_len + comp_len,
            ),
        )

    async def stream_generate(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        if not self.is_loaded:
            await self.load()

        formatted_messages = self._prepare_messages(request.messages)
        stream_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        # Yield initial role chunk
        yield ChatCompletionChunk(
            id=stream_id,
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

        inputs = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        do_sample = bool(request.temperature and request.temperature >= 0.2)
        repetition_penalty = getattr(request, "repetition_penalty", 1.1) or 1.1
        boundary_bias = getattr(request, "boundary_bias", 4.8) or 4.8
        gen_kwargs = {
            "max_new_tokens": request.max_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": [IM_END_ID, EOS_ID],
            "pad_token_id": EOS_ID,
            "use_cache": True,
            "streamer": streamer,
        }
        if do_sample:
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p or 0.9
            gen_kwargs["top_k"] = 50

        processor = CalibratedStopLogitsProcessor(
            stop_token_id=IM_END_ID,
            prompt_length=prompt_len,
            tokenizer=self.tokenizer,
            boundary_bias=boundary_bias,
            min_new_tokens=4,
        )

        def _run_stream_thread():
            with torch.inference_mode():
                self.model.generate(
                    **inputs,
                    **gen_kwargs,
                    logits_processor=LogitsProcessorList([processor]),
                )

        t = Thread(target=_run_stream_thread)
        t.start()

        loop = asyncio.get_running_loop()
        stop_signals = ["<|im_end|>", "<|im_start|>", "\nuser:", "\nUser:", "\nYou:", "\nHuman:"]
        accumulated = ""

        def _get_next_token():
            try:
                return next(streamer)
            except StopIteration:
                return None

        while True:
            token = await loop.run_in_executor(None, _get_next_token)
            if token is None:
                break
            accumulated += token
            # Check if any turn boundary or stop signal has started
            should_stop = False
            for sig in stop_signals:
                if sig in accumulated:
                    should_stop = True
                    break
            if should_stop:
                break

            if token:
                yield ChatCompletionChunk(
                    id=stream_id,
                    created=created_time,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=token),
                            finish_reason=None,
                        )
                    ],
                )

        # Final finish chunk
        yield ChatCompletionChunk(
            id=stream_id,
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
        mem_mb = None
        if torch.cuda.is_available():
            mem_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        return {
            "status": "healthy" if self.is_loaded else "uninitialized",
            "version": "1.0.0",
            "model_name": "vyrex-ai-1.7b",
            "backend": "HuggingFace-NF4-QLoRA",
            "quantization": "4-bit NF4",
            "device": self.device,
            "uptime_seconds": round(time.time() - self.start_time, 2),
            "memory_allocated_mb": mem_mb,
        }
