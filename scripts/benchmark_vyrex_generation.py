
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from vyrex_ai.models.transformer import VyrexConfig, VyrexTransformer
from vyrex_ai.tokenizer import VyrexTokenizer
from vyrex_ai.training import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/oanc-production-v1.yaml"
MANIFEST_PATH = ROOT / "data/processed/oasst1_sft_v1/manifest.json"
CHECKPOINT = ROOT / "checkpoints/oasst1_sft_phase2/best.pt"
TOKENIZER_PATH = ROOT / "artifacts/tokenizer/oanc_production_v1/tokenizer.json"
OUTPUT = ROOT / "artifacts/benchmarks/vyrex_phase2_generation.json"

MAX_NEW_TOKENS = 48
TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.08
SEED = 2026

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

PROMPTS = [
    {"category": "conversation", "prompt": "Hello! How are you today?"},
    {"category": "conversation", "prompt": "What are some practical ways to become more productive?"},
    {"category": "english", "prompt": "Explain the difference between affect and effect."},
    {"category": "english", "prompt": "Write a short paragraph about the importance of education."},
    {"category": "reasoning", "prompt": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain."},
    {"category": "reasoning", "prompt": "A train travels 120 kilometers in 2 hours. What is its average speed?"},
    {"category": "mathematics", "prompt": "A product costs 500 rupees and has a 20% discount. What is its final price?"},
    {"category": "mathematics", "prompt": "Solve the equation 3x + 7 = 22. Show your steps."},
    {"category": "coding", "prompt": "Write a Python function to calculate the factorial of a non-negative integer."},
    {"category": "coding", "prompt": "Explain the difference between a Python list and a tuple."},
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_top_p(logits: torch.Tensor, generator: torch.Generator) -> int:
    logits = logits.float() / TEMPERATURE
    probabilities = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(
        probabilities, descending=True
    )
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    remove = cumulative > TOP_P
    remove[1:] = remove[:-1].clone()
    remove[0] = False

    sorted_probs[remove] = 0.0
    filtered = torch.zeros_like(probabilities)
    filtered.scatter_(0, sorted_indices, sorted_probs)
    filtered = filtered / filtered.sum()

    return int(torch.multinomial(
        filtered, num_samples=1, generator=generator
    ).item())


def generate(
    model: VyrexTransformer,
    tokenizer: VyrexTokenizer,
    prompt: str,
    mode: str,
    device: torch.device,
    generator: torch.Generator,
    max_seq_len: int,
) -> dict:
    formatted_prompt = f"User: {prompt}\nAssistant: "
    encoded = tokenizer.encode(formatted_prompt, add_special_tokens=True)

    if encoded and encoded[-1] == EOS_ID:
        encoded = encoded[:-1]

    if not encoded or encoded[0] != BOS_ID:
        raise ValueError("Tokenizer did not produce the expected BOS token.")

    input_ids = torch.tensor(
        [encoded], dtype=torch.long, device=device
    )

    generated = []
    stop_reason = "max_new_tokens"

    for _ in range(MAX_NEW_TOKENS):
        context = input_ids[:, -max_seq_len:]

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(context)
            else:
                logits, _ = model(context)

        next_logits = logits[0, -1, :].float()

        if mode == "greedy":
            next_token = int(torch.argmax(next_logits).item())

        elif mode == "top_p":
            if generated:
                previous = torch.tensor(
                    list(set(generated)),
                    dtype=torch.long,
                    device=device,
                )
                selected = next_logits[previous]
                adjusted = torch.where(
                    selected < 0,
                    selected * REPETITION_PENALTY,
                    selected / REPETITION_PENALTY,
                )
                next_logits[previous] = adjusted

            next_token = sample_top_p(next_logits, generator)

        else:
            raise ValueError(f"Unknown decoding mode: {mode}")

        if next_token == EOS_ID:
            stop_reason = "eos"
            break
        if next_token == PAD_ID:
            stop_reason = "pad"
            break
        if next_token == BOS_ID:
            stop_reason = "bos"
            break

        generated.append(next_token)
        next_tensor = torch.tensor(
            [[next_token]], dtype=torch.long, device=device
        )
        input_ids = torch.cat((input_ids, next_tensor), dim=1)

    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    token_count = len(generated)
    unique_count = len(set(generated))
    repeated_token_ratio = (
        1.0 - unique_count / token_count if token_count else 0.0
    )

    bigrams = list(zip(generated, generated[1:]))
    repeated_bigram_ratio = (
        1.0 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    )

    return {
        "answer": answer,
        "generated_tokens": token_count,
        "stop_reason": stop_reason,
        "ended_with_eos": stop_reason == "eos",
        "empty": not bool(answer),
        "repeated_token_ratio": round(repeated_token_ratio, 4),
        "repeated_bigram_ratio": round(repeated_bigram_ratio, 4),
    }


def main() -> None:
    for path in (CONFIG_PATH, MANIFEST_PATH, CHECKPOINT, TOKENIZER_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing benchmark: {OUTPUT}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    model_values = raw_config["model"]
    config = VyrexConfig(
        vocab_size=model_values["vocab_size"],
        hidden_size=model_values["hidden_size"],
        num_layers=model_values["num_layers"],
        num_heads=model_values["num_heads"],
        intermediate_size=model_values["intermediate_size"],
        max_seq_len=model_values["max_seq_len"],
        dropout=model_values["dropout"],
        tie_embeddings=model_values["tie_embeddings"],
    )

    tokenizer = VyrexTokenizer.load(TOKENIZER_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is unavailable; benchmark requires BF16.")

    model = VyrexTransformer(config).to(device)
    metadata = load_checkpoint(
        CHECKPOINT,
        model,
        map_location=device,
        restore_rng=False,
    )
    model.eval()

    parameter_count = sum(p.numel() for p in model.parameters())

    print("=== VYREX GENERATION BENCHMARK ===")
    print("Checkpoint:", CHECKPOINT)
    print("Checkpoint step:", metadata.get("step", "unknown"))
    print("Device:", device)
    print("Parameters:", f"{parameter_count:,}")
    print("Prompts:", len(PROMPTS))
    print("Decoding modes: greedy, top_p")
    print()

    results = []

    for index, item in enumerate(PROMPTS, start=1):
        for mode in ("greedy", "top_p"):
            generator_device = device.type
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(SEED + index * 100 + (mode == "top_p"))

            result = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=item["prompt"],
                mode=mode,
                device=device,
                generator=generator,
                max_seq_len=config.max_seq_len,
            )

            record = {
                "id": index,
                "category": item["category"],
                "prompt": item["prompt"],
                "mode": mode,
                **result,
            }
            results.append(record)

            print(f"[{index}/10] {item['category']} | {mode}")
            print("Prompt:", item["prompt"])
            print("Answer:", result["answer"] or "<EMPTY>")
            print(
                "Tokens:", result["generated_tokens"],
                "| Stop:", result["stop_reason"],
                "| Repeated tokens:", f"{result['repeated_token_ratio']:.3f}",
                "| Repeated bigrams:", f"{result['repeated_bigram_ratio']:.3f}",
            )
            print("-" * 72)

    total = len(results)
    empty_count = sum(item["empty"] for item in results)
    eos_count = sum(item["ended_with_eos"] for item in results)

    summary = {
        "total_generations": total,
        "empty_generations": empty_count,
        "empty_rate": empty_count / total if total else 0.0,
        "eos_terminations": eos_count,
        "mean_repeated_token_ratio": (
            sum(x["repeated_token_ratio"] for x in results) / total
        ),
        "mean_repeated_bigram_ratio": (
            sum(x["repeated_bigram_ratio"] for x in results) / total
        ),
    }

    report = {
        "benchmark": "VYREX Phase 2 generation benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_step": metadata.get("step"),
        "tokenizer": str(TOKENIZER_PATH.relative_to(ROOT)),
        "tokenizer_sha256": sha256_file(TOKENIZER_PATH),
        "config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "intermediate_size": config.intermediate_size,
            "max_seq_len": config.max_seq_len,
            "parameter_count": parameter_count,
        },
        "runtime": {
            "python": sys.version,
            "pytorch": torch.__version__,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(0)
                if device.type == "cuda" else None
            ),
        },
        "decoding": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "repetition_penalty": REPETITION_PENALTY,
            "seed": SEED,
        },
        "summary": summary,
        "results": results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== BENCHMARK SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nReport saved:", OUTPUT)


if __name__ == "__main__":
    main()
