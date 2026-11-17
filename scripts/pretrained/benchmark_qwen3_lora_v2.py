from __future__ import annotations

import gc
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_0.6b/original"
ADAPTER_DIR = ROOT / "pretrained/qwen3_0.6b/adapters/oasst1_lora_v1"
OUTPUT_DIR = ROOT / "pretrained/qwen3_0.6b/evaluations"
OUTPUT_FILE = OUTPUT_DIR / "oasst1_lora_v1_generation_v2.json"

MAX_NEW_TOKENS = 128

PROMPTS = [
    "Explain artificial intelligence in simple English. Give a practical example.",
    "What is the difference between machine learning and deep learning? Explain clearly.",
    "Write a Python function that checks whether an integer is prime. Include code and explain its time complexity.",
    "A train travels 120 kilometers in 2 hours. What is its average speed? Explain your calculation.",
    "Explain three practical ways to reduce overfitting when training a neural network.",
]


def repetition_metrics(text):
    words = re.findall(r"\b[\w']+\b", text.lower())

    def repeated_ngram_ratio(n):
        ngrams = [
            tuple(words[i:i + n])
            for i in range(max(0, len(words) - n + 1))
        ]
        if not ngrams:
            return 0.0
        return round(
            (len(ngrams) - len(set(ngrams))) / len(ngrams), 4
        )

    return {
        "word_count": len(words),
        "repeated_3gram_ratio": repeated_ngram_ratio(3),
        "repeated_4gram_ratio": repeated_ngram_ratio(4),
    }

def generate(model, tokenizer, device, prompts, model_name):
    model.eval()
    results = []

    for index, prompt in enumerate(prompts, 1):
        messages = [{"role": "user", "content": prompt}]

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generation_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        # Warm-up: deliberately excluded from timing.
        with torch.inference_mode():
            model.generate(**generation_args)

        torch.cuda.synchronize()

        speeds = []
        elapsed_times = []
        first_output = None

        # Three identical timed repetitions.
        for repetition in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.inference_mode():
                output = model.generate(**generation_args)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            generated_ids = output[0, input_ids.shape[1]:]
            generated_count = generated_ids.numel()

            speeds.append(generated_count / max(elapsed, 1e-9))
            elapsed_times.append(elapsed)

            if repetition == 0:
                raw_text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=False,
                )
                decoded_text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                ).strip()

                first_output = {
                    "output": decoded_text,
                    "generated_tokens": generated_count,
                    "ended_with_eos": (
                        generated_count > 0
                        and generated_ids[-1].item()
                        == tokenizer.eos_token_id
                    ),
                    "thinking_tokens_present": (
                        "<think>" in raw_text
                        or "</think>" in raw_text
                    ),
                    **repetition_metrics(decoded_text),
                }

            del output, generated_ids

        median_speed = sorted(speeds)[len(speeds) // 2]

        result = {
            "prompt": prompt,
            **first_output,
            "median_tokens_per_second": round(median_speed, 2),
            "median_generation_seconds": round(
                sorted(elapsed_times)[len(elapsed_times) // 2], 4
            ),
            "repetition_speeds": [
                round(speed, 2) for speed in speeds
            ],
        }

        results.append(result)

        print(f"\n[{model_name}] Prompt {index}/{len(prompts)}")
        print("Median speed:", result["median_tokens_per_second"], "tokens/sec")
        print("Repeated 3-gram:", result["repeated_3gram_ratio"])
        print("Repeated 4-gram:", result["repeated_4gram_ratio"])
        print("Output:", result["output"], flush=True)

        del input_ids, attention_mask, encoded
        gc.collect()

    return results

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    for path in (MODEL_DIR, ADAPTER_DIR):
        if not path.exists():
            raise FileNotFoundError(path)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {OUTPUT_FILE}"
        )

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=False,
    )

    print("=== VYREX GENERATION BENCHMARK ===")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Prompts:", len(PROMPTS))
    print("Max new tokens:", MAX_NEW_TOKENS)

    torch.cuda.reset_peak_memory_stats()

    print("\nLoading original Qwen3 base...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    ).to(device)

    base.config.use_cache = True

    torch.cuda.reset_peak_memory_stats()
    base_results = generate(
        base, tokenizer, device, PROMPTS, "BASE"
    )
    base_peak_memory = round(
        torch.cuda.max_memory_allocated() / 1024**3, 3
    )

    del base
    gc.collect()
    torch.cuda.empty_cache()

    print("\nLoading best Vyrex adapter...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    ).to(device)

    base.config.use_cache = True

    vyrex = PeftModel.from_pretrained(
        base,
        ADAPTER_DIR,
        is_trainable=False,
    )
    vyrex.eval()

    torch.cuda.reset_peak_memory_stats()
    vyrex_results = generate(
        vyrex, tokenizer, device, PROMPTS, "VYREX"
    )
    vyrex_peak_memory = round(
        torch.cuda.max_memory_allocated() / 1024**3, 3
    )

    def averages(results):
        speeds = sorted(
            r["median_tokens_per_second"] for r in results
        )
        middle = len(speeds) // 2
        median_speed = (
            speeds[middle] if len(speeds) % 2
            else (speeds[middle - 1] + speeds[middle]) / 2
        )

        return {
            "median_tokens_per_second": round(median_speed, 2),
            "average_repeated_3gram_ratio": round(
                sum(r["repeated_3gram_ratio"] for r in results)
                / len(results), 4
            ),
            "average_repeated_4gram_ratio": round(
                sum(r["repeated_4gram_ratio"] for r in results)
                / len(results), 4
            ),
            "eos_termination_rate": round(
                sum(r["ended_with_eos"] for r in results)
                / len(results), 4
            ),
            "total_generated_tokens": sum(
                r["generated_tokens"] for r in results
            ),
        }

    report = {
        "experiment": "oasst1_lora_v1_generation",
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "adapter": str(ADAPTER_DIR.relative_to(ROOT)),
        "precision": "BF16",
        "decoding": {
            "strategy": "greedy",
            "max_new_tokens": MAX_NEW_TOKENS,
            "chat_template": "Qwen3",
        },
        "base": {
            "summary": averages(base_results),
            "results": base_results,
        },
        "vyrex": {
            "summary": averages(vyrex_results),
            "results": vyrex_results,
        },
        "base_peak_gpu_memory_gb": base_peak_memory,
        "vyrex_peak_gpu_memory_gb": vyrex_peak_memory,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== GENERATION COMPARISON ===")
    print("Base:", averages(base_results))
    print("Vyrex:", averages(vyrex_results))
    print("Base peak GPU memory:", base_peak_memory, "GB")
    print("Vyrex peak GPU memory:", vyrex_peak_memory, "GB")
    print("Saved:", OUTPUT_FILE)

    del vyrex, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()







