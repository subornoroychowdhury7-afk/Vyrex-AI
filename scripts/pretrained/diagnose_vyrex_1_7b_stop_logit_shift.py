from pathlib import Path
import gc
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_1.7b/original"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
ADAPTER_DIR = ROOT / "pretrained/qwen3_1.7b/adapters/vyrex_1_7b_qlora_full_v1/epoch_01_best"
OUTPUT = ROOT / "pretrained/qwen3_1.7b/evaluations/stop_logit_shift_base_vs_v1.json"

STOP_ID = 151645
EOS_ID = 151643
BATCH_SIZE = 2

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def load_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=QUANT_CONFIG,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )


def evaluate(model, inputs, labels, metadata, name):
    model.eval()
    records = []
    device = torch.device("cuda")

    with torch.inference_mode():
        for start in range(0, len(metadata), BATCH_SIZE):
            batch = metadata[start:start + BATCH_SIZE]
            indices = [r["index"] for r in batch]

            x = torch.as_tensor(
                np.asarray(inputs[indices], dtype=np.int64),
                device=device,
            )
            y = torch.as_tensor(
                np.asarray(labels[indices], dtype=np.int64),
                device=device,
            )

            positions = torch.tensor(
                [r["stop_pos"] for r in batch],
                device=device,
            )

            idx = torch.arange(x.shape[1], device=device).unsqueeze(0)
            last_valid = idx.masked_fill(y.eq(-100), -1).max(dim=1).values
            mask = (idx <= last_valid.unsqueeze(1)).long()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=x,
                    attention_mask=mask,
                    use_cache=False,
                )

            logits = output.logits.float()
            rows = torch.arange(len(batch), device=device)

            boundary = logits[rows, positions - 1]
            final_logits = logits[rows, positions - 2]

            stop_logits = boundary[:, STOP_ID]
            competitor_logits = boundary.clone()
            competitor_logits[:, STOP_ID] = -torch.inf

            best_logits, best_ids = competitor_logits.max(dim=-1)
            margins = stop_logits - best_logits

            logsum = torch.logsumexp(boundary, dim=-1)
            stop_probs = torch.exp(stop_logits - logsum)
            stop_ranks = (boundary > stop_logits[:, None]).sum(dim=-1) + 1

            final_targets = torch.tensor(
                [r["final_token_id"] for r in batch],
                device=device,
            )
            final_correct = final_logits.argmax(dim=-1).eq(final_targets)

            for j, meta in enumerate(batch):
                records.append({
                    "index": meta["index"],
                    "length": meta["length"],
                    "stop_logit": float(stop_logits[j].item()),
                    "best_competitor_id": int(best_ids[j].item()),
                    "best_competitor_logit": float(best_logits[j].item()),
                    "stop_margin": float(margins[j].item()),
                    "stop_probability": float(stop_probs[j].item()),
                    "stop_rank": int(stop_ranks[j].item()),
                    "final_token_correct": bool(final_correct[j].item()),
                })

            del x, y, mask, output, logits, boundary, final_logits

            if (start + BATCH_SIZE) % 50 == 0 or start + BATCH_SIZE >= len(metadata):
                print(f"{name}: {min(start + BATCH_SIZE, len(metadata))}/{len(metadata)}", flush=True)

    return records


def summarize(records):
    def mean(key):
        return float(np.mean([r[key] for r in records]))

    def median(key):
        return float(np.median([r[key] for r in records]))

    return {
        "examples": len(records),
        "stop_logit_mean": mean("stop_logit"),
        "stop_margin_mean": mean("stop_margin"),
        "stop_margin_median": median("stop_margin"),
        "stop_probability_mean": mean("stop_probability"),
        "stop_probability_median": median("stop_probability"),
        "stop_rank_mean": mean("stop_rank"),
        "stop_rank_median": median("stop_rank"),
        "stop_top1_accuracy": float(np.mean([r["stop_rank"] == 1 for r in records])),
        "final_token_accuracy": mean("final_token_correct"),
    }


def main():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required.")

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    inputs = np.load(DATA_DIR / "validation_input_ids.npy", mmap_mode="r")
    labels = np.load(DATA_DIR / "validation_labels.npy", mmap_mode="r")

    metadata = []
    for i, row in enumerate(labels):
        positions = np.flatnonzero(row == STOP_ID)
        if len(positions) != 1:
            raise RuntimeError(f"Invalid STOP count at row {i}")

        p = int(positions[0])
        if p < 2 or row[p - 1] == -100:
            raise RuntimeError(f"Invalid final token at row {i}")

        metadata.append({
            "index": i,
            "stop_pos": p,
            "length": int(np.count_nonzero(row != -100) - 2),
            "final_token_id": int(row[p - 1]),
        })

    print("Examples:", len(metadata))
    print("GPU:", torch.cuda.get_device_name(0))
    print("Comparison: base vs V1 full384")

    base = load_model()
    base_records = evaluate(base, inputs, labels, metadata, "BASE")
    del base
    gc.collect()
    torch.cuda.empty_cache()

    base = load_model()
    model = PeftModel.from_pretrained(base, ADAPTER_DIR, is_trainable=False)
    adapted_records = evaluate(model, inputs, labels, metadata, "V1_FULL384")

    if [r["index"] for r in base_records] != [r["index"] for r in adapted_records]:
        raise RuntimeError("Example alignment mismatch.")

    paired = []
    for b, a in zip(base_records, adapted_records):
        paired.append({
            "index": b["index"],
            "base_stop_logit": b["stop_logit"],
            "adapted_stop_logit": a["stop_logit"],
            "stop_logit_shift": a["stop_logit"] - b["stop_logit"],
            "base_stop_margin": b["stop_margin"],
            "adapted_stop_margin": a["stop_margin"],
            "stop_margin_shift": a["stop_margin"] - b["stop_margin"],
            "base_stop_rank": b["stop_rank"],
            "adapted_stop_rank": a["stop_rank"],
            "base_stop_probability": b["stop_probability"],
            "adapted_stop_probability": a["stop_probability"],
        })

    report = {
        "dataset": "OASST1 Qwen3 SFT V1 validation",
        "examples": len(metadata),
        "base": summarize(base_records),
        "v1_full384": summarize(adapted_records),
        "paired_shift": {
            "mean_stop_logit_shift": float(np.mean([r["stop_logit_shift"] for r in paired])),
            "median_stop_logit_shift": float(np.median([r["stop_logit_shift"] for r in paired])),
            "mean_stop_margin_shift": float(np.mean([r["stop_margin_shift"] for r in paired])),
            "median_stop_margin_shift": float(np.median([r["stop_margin_shift"] for r in paired])),
            "examples_with_lower_stop_logit": int(sum(r["stop_logit_shift"] < 0 for r in paired)),
            "examples_with_higher_stop_logit": int(sum(r["stop_logit_shift"] > 0 for r in paired)),
            "examples_with_improved_stop_rank": int(sum(r["adapted_stop_rank"] < r["base_stop_rank"] for r in paired)),
            "examples_with_worsened_stop_rank": int(sum(r["adapted_stop_rank"] > r["base_stop_rank"] for r in paired)),
        },
        "paired_examples": paired,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== BASE ===")
    print(json.dumps(report["base"], indent=2))
    print("\n=== V1 FULL384 ===")
    print(json.dumps(report["v1_full384"], indent=2))
    print("\n=== PAIRED SHIFT ===")
    print(json.dumps(report["paired_shift"], indent=2))
    print("\nSaved:", OUTPUT)

    del model, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
