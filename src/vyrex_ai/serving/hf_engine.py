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
    "You are Vyrex AI, a sharp, concise AI assistant powered by Vyrex Hyperion 3.\n"
    "Rules you MUST follow on every response:\n"
    "1. STOP generating the moment your answer is complete. Do NOT add extra sentences, commentary, or follow-up questions.\n"
    "2. Keep answers SHORT and DIRECT — 2-4 sentences for factual questions, up to 8 sentences max for complex topics.\n"
    "3. Never repeat words, phrases, or clauses. Write clean, complete sentences only.\n"
    "4. Identity: You were created by Suborno Roy Chowdhury. Your model name is Vyrex Hyperion 3.\n"
    "5. No sign-offs, no 'Hope this helps', no 'Let me know if' — just the answer, then stop."
)

IM_END_ID = 151645
EOS_ID = 151643


class CalibratedStopLogitsProcessor(LogitsProcessor):
    """
    Strongly biases the model toward emitting EOS/IM_END after sentence boundaries.
    At >100 tokens, aggressively forces termination at any boundary token.
    Also suppresses Chinese/CJK token range to prevent hallucinated CJK sequences.
    """
    def __init__(
        self,
        stop_token_id: int,
        prompt_length: int,
        tokenizer,
        boundary_bias: float = 10.0,
        min_new_tokens: int = 10,
    ):
        self.stop_token_id = stop_token_id
        self.prompt_length = prompt_length
        self.min_new_tokens = min_new_tokens
        self.boundary_bias = boundary_bias
        self.tokenizer = tokenizer

        # Sentence-closing boundary token ids
        self.boundary_ids = set()
        for s in [".", " .", "!\n", "!\n\n", "!", "?", " ?\n", "\n\n", "```\n", "```", ")\n", ":\n", ".\n", ".\n\n"]:
            toks = tokenizer.encode(s, add_special_tokens=False)
            if toks:
                self.boundary_ids.add(toks[-1])

        # Hard-stop run-on conjunction tokens — penalize these heavily after 50 tokens
        self.runon_ids = set()
        for s in [" and", " which", " where", " so", " thus", " therefore", " however", " but", " then", " etc"]:
            toks = tokenizer.encode(s, add_special_tokens=False)
            if toks:
                self.runon_ids.add(toks[-1])

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        new_tokens = input_ids.shape[1] - self.prompt_length

        # Suppress EOS during min_new_tokens window
        if new_tokens < self.min_new_tokens:
            scores[:, self.stop_token_id] = -float("inf")
            return scores

        last_id = int(input_ids[0, -1].item())

        # === BOUNDARY BIAS: push EOS up at sentence-ending positions ===
        if last_id in self.boundary_ids:
            bias = self.boundary_bias
            if new_tokens > 25:
                bias += 10.0   # 20 total
            if new_tokens > 50:
                bias += 25.0   # 35 total
            if new_tokens > 80:
                bias += 50.0   # 60 total — stops cleanly at sentence end
            scores[:, self.stop_token_id] += bias

        # === RUN-ON PENALTY: discourage chaining conjunctions after 40 tokens ===
        if new_tokens > 40:
            for rid in self.runon_ids:
                scores[:, rid] -= 6.0

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
            repetition_penalty = getattr(request, "repetition_penalty", 1.18) or 1.18
            boundary_bias = getattr(request, "boundary_bias", 10.0) or 10.0
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

            # Post-generation cleanup: strip trailing chatter artifacts
            fluff_markers = [
                "\n\nThanks again", "\nThanks again", "\n\nThanks!", "\nThanks!",
                "\n\nHope this helps", "\n\nHope that helps", "\n\nHope this is helpful",
                "\n\nGood luck", "\n\nP.S.", "\n\nSee ya", "\n\nGoodbye", "\n\nLet me know if",
                "\nUser:", "\nHuman:", "\nYou:", "\n\nHappy New Year", "\nBest regards", "\nCheers"
            ]
            for marker in fluff_markers:
                if marker in text:
                    text = text.split(marker)[0].strip()

            # Strip any CJK characters and everything after them
            for i, ch in enumerate(text):
                if '\u4e00' <= ch <= '\u9fff':
                    text = text[:i].strip()
                    break

            # If response trailed off without closing punctuation, trim to last complete sentence
            clean_text = text.strip()
            if clean_text and clean_text[-1] not in ('.', '!', '?', '"', "'", '`', ')'):
                if clean_text.count("```") % 2 != 0:
                    clean_text += "\n```"
                else:
                    last_punct = -1
                    for idx in range(len(clean_text) - 1, -1, -1):
                        if clean_text[idx] in ('.', '!', '?'):
                            last_punct = idx
                            break
                    if last_punct > 30:
                        clean_text = clean_text[:last_punct + 1].strip()
            text = clean_text

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
        repetition_penalty = getattr(request, "repetition_penalty", 1.18) or 1.18
        boundary_bias = getattr(request, "boundary_bias", 10.0) or 10.0
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
        stop_signals = [
            "<|im_end|>", "<|im_start|>", "\nuser:", "\nUser:", "\nyou:", "\nYou:", "\nhuman:", "\nHuman:",
            "\n\nhope this helps", "\n\nhope that helps", "\n\nthanks again", "\nthanks again",
            "\n\nthanks!", "\nthanks!", "\n\ngood luck", "\n\np.s.", "\n\nsee ya", "\n\ngoodbye",
            "\n\nlet me know if", "\n\nhappy new year", "\nbest regards", "\ncheers"
        ]
        accumulated = ""
        recent_words: List[str] = []
        token_count = 0

        def _get_next_token():
            try:
                return next(streamer)
            except StopIteration:
                return None

        while True:
            token = await loop.run_in_executor(None, _get_next_token)
            if token is None:
                break
            token_count += 1
            accumulated += token

            should_stop = False
            acc_lower = accumulated.lower()

            # 1. Stop signal check
            for sig in stop_signals:
                if sig in acc_lower:
                    should_stop = True
                    break

            # 2. CJK hallucination detector — stop immediately on any CJK character
            if not should_stop and any('\u4e00' <= c <= '\u9fff' for c in token):
                should_stop = True

            # 3. Word-level repetition circuit breaker (3 consecutive identical words)
            clean_word = token.strip().lower()
            if not should_stop and clean_word and len(clean_word) > 2:
                recent_words.append(clean_word)
                if len(recent_words) > 12:
                    recent_words.pop(0)
                if len(recent_words) >= 3 and recent_words[-1] == recent_words[-2] == recent_words[-3]:
                    should_stop = True

            # 4. Clean sentence boundary stop after 90 tokens
            if not should_stop and token_count >= 90:
                if any(token.rstrip().endswith(punct) for punct in (".", "!", "?", "```")):
                    should_stop = True

            # 5. Hard token cap — truncate at 180 tokens regardless
            if not should_stop and token_count >= 180:
                should_stop = True

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
