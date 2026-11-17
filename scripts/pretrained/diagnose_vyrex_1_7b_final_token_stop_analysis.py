from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_1.7b/original"
ADAPTER_DIR = ROOT / "pretrained/qwen3_1.7b/adapters/vyrex_1_7b_qlora_full_v1/epoch_01_best"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
OUTPUT = ROOT / "pretrained/qwen3_1.7b/evaluations/final_token_stop_analysis_full384.json"

STOP_ID = 151645
PAD_ID = 151643
BATCH_SIZE = 2
TOP_K = 5

def rank_of(scores, token_id):
    return int((scores > scores[token_id]).sum().item()) + 1

def category(token_text):
    text = token_text

    if not text.strip():
        return "whitespace"
    if any(char in text for char in "{}[]`\\"):
        return "code_or_bracket"
    if text.strip() in {".", "?", "!", ",", ":", ";", "。", "！", "？"}:
        return "punctuation"
    if text.strip()[-1:] in {".", "?", "!", ",", ":", ";"}:
        return "punctuation"
    return "other"

def summarize(rows):
    ranks = np.asarray([r["stop_rank"] for r in rows], dtype=np.float64)
    probs = np.asarray([r["stop_probability"] for r in rows], dtype=np.float64)
    final_ranks = np.asarray([r["final_token_rank"] for r in rows], dtype=np.float64)

    return {
        "examples": len(rows),
        "stop_rank_mean": float(ranks.mean()),
        "stop_rank_median": float(np.median(ranks)),
        "stop_rank_p90": float(np.percentile(ranks, 90)),
        "stop_probability_mean": float(probs.mean()),
        "stop_probability_median": float(np.median(probs)),
        "stop_top1_accuracy": float(np.mean(ranks == 1)),
        "stop_top100_percentage": float(100 * np.mean(ranks <= 100)),
        "final_token_top1_accuracy": float(np.mean(final_ranks == 1)),
        "final_token_top5_accuracy": float(np.mean(final_ranks <= 5)),
        "final_token_rank_median": float(np.median(final_ranks)),
    }

def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")

    if not ADAPTER_DIR.is_dir():
        raise FileNotFoundError(ADAPTER_DIR)

    input_ids = np.load(DATA_DIR / "validation_input_ids.npy", mmap_mode="r")
    labels = np.load(DATA_DIR / "validation_labels.npy", mmap_mode="r")

    if input_ids.shape != labels.shape:
        raise ValueError("Input and label shapes do not match")

    examples = []

    for i in range(len(input_ids)):
        positions = np.flatnonzero(labels[i] == STOP_ID)

        if len(positions) != 1:
            raise ValueError(f"Invalid STOP count at example {i}")

        stop_position = int(positions[0])
        final_position = stop_position - 1

        if final_position < 1:
            raise ValueError(f"Invalid final-token position at {i}")

        final_token_id = int(input_ids[i, final_position])

        examples.append({
            "index": i,
            "stop_position": stop_position,
            "final_position": final_position,
            "final_token_id": final_token_id,
            "input_ids": np.array(input_ids[i], dtype=np.int64, copy=True),
        })

    print("Validation examples:", len(examples))
    print("Sequence shape:", input_ids.shape)
    print("Loading tokenizer and NF4 base...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(
        base,
        ADAPTER_DIR,
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()

    device = torch.device("cuda:0")
    rows = []

    print("Scoring final-token and STOP predictions...")

    with torch.inference_mode():
        for start in range(0, len(examples), BATCH_SIZE):
            batch = examples[start:start + BATCH_SIZE]

            x = torch.from_numpy(
                np.stack([item["input_ids"] for item in batch])
            ).to(device)

            attention = (x != PAD_ID).long()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=x,
                    attention_mask=attention,
                    use_cache=False,
                )

            for j, item in enumerate(batch):
                final_scores = output.logits[
                    j, item["final_position"] - 1
                ].float()

                stop_scores = output.logits[
                    j, item["stop_position"] - 1
                ].float()

                final_id = item["final_token_id"]

                final_logz = torch.logsumexp(final_scores, dim=-1)
                stop_logz = torch.logsumexp(stop_scores, dim=-1)

                final_prob = torch.exp(
                    final_scores[final_id] - final_logz
                ).item()

                stop_prob = torch.exp(
                    stop_scores[STOP_ID] - stop_logz
                ).item()

                final_rank = rank_of(final_scores, final_id)
                stop_rank = rank_of(stop_scores, STOP_ID)

                final_token_text = tokenizer.decode(
                    [final_id], skip_special_tokens=False
                )

                top_values, top_ids = torch.topk(stop_scores, TOP_K)

                top5 = []
                for token_id, value in zip(
                    top_ids.tolist(), top_values.tolist()
                ):
                    top5.append({
                        "token_id": int(token_id),
                        "decoded": tokenizer.decode(
                            [int(token_id)], skip_special_tokens=False
                        ),
                        "probability": float(
                            torch.exp(
                                torch.tensor(value, device=device) - stop_logz
                            ).item()
                        ),
                    })

                rows.append({
                    "example_index": item["index"],
                    "final_token_id": final_id,
                    "final_token": final_token_text,
                    "final_token_category": category(final_token_text),
                    "final_token_probability": float(final_prob),
                    "final_token_rank": final_rank,
                    "stop_probability": float(stop_prob),
                    "stop_rank": stop_rank,
                    "top5_stop_competitors": top5,
                })

            del x, attention, output

            if (start // BATCH_SIZE + 1) % 25 == 0:
                print(f"Scored {len(rows)}/{len(examples)}")

    grouped = defaultdict(list)

    for row in rows:
        grouped[row["final_token_category"]].append(row)

    category_results = {
        name: summarize(group)
        for name, group in sorted(grouped.items())
    }

    stop_ranks = np.asarray([r["stop_rank"] for r in rows], dtype=np.float64)
    final_probs = np.asarray(
        [r["final_token_probability"] for r in rows],
        dtype=np.float64,
    )

    report = {
        "model": "Qwen3-1.7B NF4 + VYREX full384 QLoRA",
        "adapter": str(ADAPTER_DIR.relative_to(ROOT)),
        "dataset": str(DATA_DIR.relative_to(ROOT)),
        "examples": len(rows),
        "stop_token_id": STOP_ID,
        "overall": summarize(rows),
        "final_token_categories": category_results,
        "spearman_final_probability_vs_stop_rank": float(
            __import__("scipy").stats.spearmanr(
                final_probs, stop_ranks
            ).statistic
        ),
        "per_example": rows,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== FINAL TOKEN / STOP ANALYSIS ===")
    print(json.dumps(report["overall"], indent=2))
    print("\n=== ENDING TOKEN CATEGORIES ===")
    print(json.dumps(category_results, indent=2))
    print(
        "\nSpearman(final-token probability, STOP rank):",
        report["spearman_final_probability_vs_stop_rank"],
    )
    print("\nReport:", OUTPUT)

if __name__ == "__main__":
    main()
