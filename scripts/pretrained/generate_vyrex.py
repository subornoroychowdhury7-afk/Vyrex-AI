from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_0.6b/original"
ADAPTER_DIR = ROOT / "pretrained/qwen3_0.6b/adapters/oasst1_lora_v1"

CHUNK_TOKENS = 256
DEFAULT_MAX_TOTAL_TOKENS = 512


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VYREX inference.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This inference configuration requires BF16 support.")

    for path in (MODEL_DIR, ADAPTER_DIR):
        if not path.exists():
            raise FileNotFoundError(path)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=False,
    )

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    ).to("cuda")

    base.config.use_cache = True

    model = PeftModel.from_pretrained(
        base,
        ADAPTER_DIR,
        is_trainable=False,
    )
    model.eval()

    assistant_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id = tokenizer.eos_token_id

    if assistant_end_id is None or eos_id is None:
        raise RuntimeError("Required Qwen3 stopping tokens are unavailable.")

    stop_ids = list(dict.fromkeys([assistant_end_id, eos_id]))

    print("VYREX Qwen3-0.6B loaded.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Adapter: OASST1 LoRA V1")
    print("Stop token IDs:", stop_ids)
    print("Maximum generated tokens:", DEFAULT_MAX_TOTAL_TOKENS)

    return model, tokenizer, stop_ids


def generate_response(model, tokenizer, stop_ids, prompt, max_total_tokens, repetition_penalty, no_repeat_ngram, sampling, temperature, top_p):
    messages = [{"role": "user", "content": prompt}]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ) + "<think>\n\n</think>\n\n"

    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    input_length = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        output = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=max_total_tokens,
            do_sample=sampling,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram,
            temperature=temperature if sampling else None,
            top_p=top_p if sampling else None,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids,
        )

    generated_ids = output[0, input_length:]
    generated_count = generated_ids.shape[0]

    naturally_stopped = (
        generated_count > 0
        and generated_ids[-1].item() in stop_ids
    )

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    if naturally_stopped:
        status = "COMPLETE"
        reason = "Recognized end-of-turn token generated."
    elif generated_count >= max_total_tokens:
        status = "INCOMPLETE"
        reason = "Maximum generation limit reached; answer may be truncated."
    else:
        status = "INCOMPLETE"
        reason = "Generation returned without a recognized end-of-turn token."

    result = {
        "response": response,
        "status": status,
        "reason": reason,
        "generated_tokens": generated_count,
    }

    del encoded, output, generated_ids
    gc.collect()

    return result

def main():
    parser = argparse.ArgumentParser(
        description="VYREX bounded Qwen3 inference controller."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOTAL_TOKENS,
        help="Hard maximum total generated tokens (default: 512).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional single prompt. Otherwise starts interactive mode.",
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram", type=int, default=0)
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    if args.repetition_penalty < 1.0:
        parser.error("--repetition-penalty must be >= 1.0.")
    if args.no_repeat_ngram < 0:
        parser.error("--no-repeat-ngram must be >= 0.")
    if args.temperature <= 0:
        parser.error("--temperature must be positive.")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1].")

    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive.")

    model, tokenizer, stop_ids = load_model()

    try:
        if args.prompt is not None:
            prompts = [args.prompt]
        else:
            print("\nVYREX ready. Type /exit to quit.\n")
            prompts = None

        while True:
            if prompts is not None:
                if not prompts:
                    break
                prompt = prompts.pop()
            else:
                try:
                    prompt = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting VYREX.")
                    break

            if not prompt:
                if prompts is not None:
                    break
                continue

            if prompt.lower() in {"/exit", "/quit"}:
                break

            result = generate_response(
                model,
                tokenizer,
                stop_ids,
                prompt,
                args.max_tokens,
                args.repetition_penalty,
                args.no_repeat_ngram,
                args.sampling,
                args.temperature,
                args.top_p,
            )

            print("\nVYREX:", result["response"])
            print("\nStatus:", result["status"])
            print("Reason:", result["reason"])
            print("Generated tokens:", result["generated_tokens"])
            print("-" * 60, flush=True)

    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()



