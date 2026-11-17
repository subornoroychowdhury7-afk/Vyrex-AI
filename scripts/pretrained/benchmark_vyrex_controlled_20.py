from __future__ import annotations

import gc
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_0.6b/original"
ADAPTERS = {
    "base": None,
    "vyrex_v1": ROOT / "pretrained/qwen3_0.6b/adapters/oasst1_lora_v1",
    "weighted_stop": ROOT / "pretrained/qwen3_0.6b/adapters/oasst1_lora_weighted_stop_v1",
}
OUTPUT = ROOT / "pretrained/qwen3_0.6b/evaluations/controlled_20_prompt_v1.json"
MAX_NEW_TOKENS = 256

PROMPTS = [
    {"category": "arithmetic", "prompt": "What is 17 multiplied by 24? Show the calculation.", "criteria": "408; correct multiplication"},
    {"category": "arithmetic", "prompt": "A car travels 150 km in 2.5 hours. Calculate its average speed in km/h.", "criteria": "60 km/h; distance divided by time"},
    {"category": "arithmetic", "prompt": "A product costs 800 rupees and has a 15% discount. What is the final price?", "criteria": "680 rupees; discount calculation"},
    {"category": "reasoning", "prompt": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain.", "criteria": "No; insufficient information; logical distinction"},
    {"category": "reasoning", "prompt": "A farmer has 17 sheep. All but 9 run away. How many remain? Explain briefly.", "criteria": "9"},
    {"category": "reasoning", "prompt": "If today is Wednesday, what day will it be 100 days from today? Explain.", "criteria": "Friday; 100 modulo 7 equals 2"},
    {"category": "general_knowledge", "prompt": "What is the role of mitochondria in a eukaryotic cell?", "criteria": "ATP; cellular energy; respiration"},
    {"category": "general_knowledge", "prompt": "Explain the difference between a compiler and an interpreter.", "criteria": "translation; execution; compiled versus interpreted"},
    {"category": "general_knowledge", "prompt": "What causes the seasons on Earth?", "criteria": "Earth's axial tilt; orbit around the Sun"},
    {"category": "python", "prompt": "Write a Python function that returns the factorial of a non-negative integer. Include input validation and time complexity.", "criteria": "factorial; base case; loop or recursion; O(n)"},
    {"category": "python", "prompt": "Write a Python function to find the second-largest distinct number in a list without sorting. Explain its time complexity.", "criteria": "distinct; second largest; O(n)"},
    {"category": "python", "prompt": "Find and fix the bug: `def add(a, b): return a - b`. The function should add two numbers. Show corrected code.", "criteria": "return a + b"},
    {"category": "explanation", "prompt": "Explain overfitting in machine learning and give three ways to reduce it.", "criteria": "training versus validation; regularization; early stopping; data augmentation or dropout"},
    {"category": "explanation", "prompt": "Explain gradient descent using a simple intuitive example.", "criteria": "loss; parameters; iterative updates; learning rate"},
    {"category": "explanation", "prompt": "What is the difference between supervised and unsupervised learning? Give one example of each.", "criteria": "labeled data; unlabeled data; examples"},
    {"category": "instruction_following", "prompt": "Answer in exactly three bullet points. Give three benefits of unit testing.", "criteria": "exactly three bullets; benefits of unit testing"},
    {"category": "instruction_following", "prompt": "Explain what an API is in no more than 40 words.", "criteria": "40 words maximum; software communication; interface"},
    {"category": "instruction_following", "prompt": "Return only valid JSON with keys `name` and `language`, using values `Vyrex` and `Python`. Do not add explanation or markdown.", "criteria": "valid JSON; name Vyrex; language Python; no surrounding prose"},
    {"category": "coding_reasoning", "prompt": "What is the time complexity of binary search, and why must the input be sorted?", "criteria": "O(log n); sorted input; halves search space"},
    {"category": "factual", "prompt": "What are the three primary colors of light, and what color does combining all three produce?", "criteria": "red; green; blue; white"},
]


def repetition_metrics(text: str) -> dict:
    words = re.findall(r"\b[\w']+\b", text.lower())

    def ratio(n: int) -> float:
        grams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if not grams:
            return 0.0
        return round((len(grams) - len(set(grams))) / len(grams), 4)

    return {
        "word_count": len(words),
        "repeated_3gram_ratio": ratio(3),
        "repeated_4gram_ratio": ratio(4),
    }


def load_model(adapter_dir, device):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    ).to(device)
    base.config.use_cache = True

    if adapter_dir is None:
        model = base
    else:
        model = PeftModel.from_pretrained(
            base, adapter_dir, is_trainable=False
        )
    model.eval()
    return model


def generate_one(model, tokenizer, device, item):
    messages = [{"role": "user", "content": item["prompt"]}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + "<think>\n\n</think>\n\n"

    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id = tokenizer.eos_token_id
    stop_ids = [end_id, eos_id]

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    generated = output[0, encoded["input_ids"].shape[1]:]
    count = generated.numel()
    raw = tokenizer.decode(generated, skip_special_tokens=False)
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    ended = count > 0 and generated[-1].item() in stop_ids

    result = {
        "category": item["category"],
        "prompt": item["prompt"],
        "expected_answer_criteria": item["criteria"],
        "output": answer,
        "generated_tokens": count,
        "word_count": repetition_metrics(answer)["word_count"],
        "repeated_3gram_ratio": repetition_metrics(answer)["repeated_3gram_ratio"],
        "repeated_4gram_ratio": repetition_metrics(answer)["repeated_4gram_ratio"],
        "ended_with_stop_token": ended,
        "termination": "natural_stop" if ended else "token_limit_or_unexpected_end",
        "generation_seconds": round(elapsed, 4),
        "tokens_per_second": round(count / max(elapsed, 1e-9), 2),
        "thinking_tokens_present": "<think>" in raw or "</think>" in raw,
    }

    del output, generated, encoded
    return result


def summarize(results):
    n = len(results)
    return {
        "prompts": n,
        "natural_stop_count": sum(r["ended_with_stop_token"] for r in results),
        "natural_stop_rate": round(
            sum(r["ended_with_stop_token"] for r in results) / n, 4
        ),
        "mean_generated_tokens": round(
            sum(r["generated_tokens"] for r in results) / n, 2
        ),
        "mean_words": round(sum(r["word_count"] for r in results) / n, 2),
        "mean_repeated_3gram_ratio": round(
            sum(r["repeated_3gram_ratio"] for r in results) / n, 4
        ),
        "mean_repeated_4gram_ratio": round(
            sum(r["repeated_4gram_ratio"] for r in results) / n, 4
        ),
        "median_tokens_per_second": round(
            sorted(r["tokens_per_second"] for r in results)[n // 2], 2
        ),
        "token_limit_hits": sum(
            r["generated_tokens"] >= MAX_NEW_TOKENS for r in results
        ),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")

    for path in [MODEL_DIR, *[p for p in ADAPTERS.values() if p is not None]]:
        if not path.exists():
            raise FileNotFoundError(path)

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=False
    )
    print("GPU:", torch.cuda.get_device_name(0))
    print("Prompts:", len(PROMPTS), "| Models:", len(ADAPTERS))
    print("Maximum new tokens:", MAX_NEW_TOKENS)

    report = {
        "experiment": "controlled_20_prompt_v1",
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "precision": "BF16",
        "decoding": {
            "strategy": "greedy",
            "max_new_tokens": MAX_NEW_TOKENS,
            "chat_template": "Qwen3 with empty think block",
            "stop_token_ids": [
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
                tokenizer.eos_token_id,
            ],
        },
        "prompt_count": len(PROMPTS),
        "models": {},
    }

    for model_name, adapter_dir in ADAPTERS.items():
        print(f"\n=== Loading {model_name} ===", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = load_model(adapter_dir, device)

        print(f"[{model_name}] Warm-up...", flush=True)
        generate_one(model, tokenizer, device, PROMPTS[0])

        results = []
        for index, item in enumerate(PROMPTS, 1):
            result = generate_one(model, tokenizer, device, item)
            results.append(result)
            print(
                f"[{model_name}] {index}/{len(PROMPTS)} "
                f"{item['category']} | tokens={result['generated_tokens']} "
                f"| stop={result['ended_with_stop_token']} "
                f"| speed={result['tokens_per_second']} tok/s",
                flush=True,
            )

        peak = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        report["models"][model_name] = {
            "adapter": str(adapter_dir.relative_to(ROOT)) if adapter_dir else None,
            "summary": summarize(results),
            "peak_gpu_memory_gb": peak,
            "results": results,
        }

        print("SUMMARY:", json.dumps(report["models"][model_name]["summary"]))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== CONTROLLED EVALUATION COMPLETE ===")
    for name, data in report["models"].items():
        print(name, json.dumps(data["summary"]))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
