from pathlib import Path
import gc
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_0.6b/original"
ADAPTER_DIR = ROOT / "pretrained/qwen3_0.6b/adapters/oasst1_lora_v1"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
OUTPUT = ROOT / "pretrained/qwen3_0.6b/evaluations/oasst1_termination_v2.json"

STOP_ID = 151645
EOS_ID = 151643
NEWLINE_ID = 198
PAD_ID = 151643
BATCH_SIZE = 2

BUCKETS = [
    (1, 64), (65, 128), (129, 256),
    (257, 384), (385, 512),
]

def summarize(records):
    result = {}

    for name, rows in records.items():
        if not rows:
            continue

        result[name] = {
            "examples": len(rows),
            "stop_probability_mean": float(np.mean([r["stop_probability"] for r in rows])),
            "stop_probability_median": float(np.median([r["stop_probability"] for r in rows])),
            "stop_rank_mean": float(np.mean([r["stop_rank"] for r in rows])),
            "stop_rank_median": float(np.median([r["stop_rank"] for r in rows])),
            "eos_probability_mean": float(np.mean([r["eos_probability"] for r in rows])),
            "newline_probability_mean": float(np.mean([r["newline_probability"] for r in rows])),
            "final_token_probability_mean": float(np.mean([r["final_token_probability"] for r in rows])),
            "final_token_rank_mean": float(np.mean([r["final_token_rank"] for r in rows])),
            "final_token_top1_accuracy": float(np.mean([r["final_token_correct"] for r in rows])),
            "stop_top1_accuracy": float(np.mean([r["stop_correct"] for r in rows])),
        }

    return result


def evaluate(model, inputs, labels, metadata, name):
    model.eval()
    collected = {key: [] for key in ["overall"] + [
        f"{low}-{high}" for low, high in BUCKETS
    ]}

    device = torch.device("cuda")
    total = len(metadata)

    with torch.inference_mode():
        for start in range(0, total, BATCH_SIZE):
            batch = metadata[start:start + BATCH_SIZE]
            indices = [r["index"] for r in batch]
            positions = torch.tensor(
                [r["stop_pos"] for r in batch], device=device
            )

            x = torch.as_tensor(
                np.asarray(inputs[indices], dtype=np.int64),
                device=device,
            )
            label_batch = torch.as_tensor(
                np.asarray(labels[indices], dtype=np.int64),
                device=x.device,
            )
            positions_idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            last_valid = positions_idx.masked_fill(label_batch.eq(-100), -1).max(dim=1).values
            if torch.any(last_valid < 0):
                raise RuntimeError("Batch contains a row without supervised tokens.")
            mask = (positions_idx <= last_valid.unsqueeze(1)).long()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=x,
                    attention_mask=mask,
                    use_cache=False,
                )

            logits = output.logits.float()
            rows = torch.arange(len(batch), device=device)

            # Predict assistant end marker from the preceding position.
            boundary_logits = logits[rows, positions - 1]
            boundary_logsum = torch.logsumexp(boundary_logits, dim=-1)

            stop_prob = torch.exp(
                boundary_logits[:, STOP_ID] - boundary_logsum
            )
            eos_prob = torch.exp(
                boundary_logits[:, EOS_ID] - boundary_logsum
            )
            newline_prob = torch.exp(
                boundary_logits[:, NEWLINE_ID] - boundary_logsum
            )

            stop_rank = (
                (boundary_logits > boundary_logits[:, STOP_ID:STOP_ID + 1])
                .sum(dim=-1) + 1
            )
            stop_correct = boundary_logits.argmax(dim=-1).eq(STOP_ID)

            # Predict the final ordinary assistant-response token.
            final_logits = logits[rows, positions - 2]
            final_targets = torch.tensor(
                [r["final_token_id"] for r in batch],
                device=device,
            )
            final_logsum = torch.logsumexp(final_logits, dim=-1)
            final_prob = torch.exp(
                final_logits.gather(1, final_targets[:, None]).squeeze(1)
                - final_logsum
            )
            final_rank = (
                (final_logits > final_logits.gather(
                    1, final_targets[:, None]
                )).sum(dim=-1) + 1
            )
            final_correct = final_logits.argmax(dim=-1).eq(final_targets)

            values = zip(
                [r["length"] for r in batch],
                stop_prob.cpu().tolist(),
                stop_rank.cpu().tolist(),
                eos_prob.cpu().tolist(),
                newline_prob.cpu().tolist(),
                final_prob.cpu().tolist(),
                final_rank.cpu().tolist(),
                final_correct.cpu().tolist(),
                stop_correct.cpu().tolist(),
            )

            for length, sp, sr, ep, np_, fp, fr, fc, sc in values:
                record = {
                    "stop_probability": sp,
                    "stop_rank": sr,
                    "eos_probability": ep,
                    "newline_probability": np_,
                    "final_token_probability": fp,
                    "final_token_rank": fr,
                    "final_token_correct": fc,
                    "stop_correct": sc,
                }
                collected["overall"].append(record)

                for low, high in BUCKETS:
                    if low <= length <= high:
                        collected[f"{low}-{high}"].append(record)
                        break

            del x, mask, output, logits
            del boundary_logits, final_logits

            if (start + BATCH_SIZE) % 50 == 0 or start + BATCH_SIZE >= total:
                print(f"{name}: {min(start + BATCH_SIZE, total)}/{total}", flush=True)

    return {
        "overall": summarize({"all": collected["overall"]})["all"],
        "length_buckets": summarize({
            key: value for key, value in collected.items()
            if key != "overall"
        }),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable.")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    inputs = np.load(DATA_DIR / "test_input_ids.npy", mmap_mode="r")
    labels = np.load(DATA_DIR / "test_labels.npy", mmap_mode="r")

    metadata = []
    for i in range(len(labels)):
        row = labels[i]
        positions = np.flatnonzero(row == STOP_ID)
        supervised = np.flatnonzero(row != -100)

        if len(positions) != 1:
            raise RuntimeError(f"Invalid stop marker in example {i}")

        pos = int(positions[0])
        if pos < 2 or pos - 1 not in supervised:
            raise RuntimeError(f"Invalid final response token in example {i}")

        metadata.append({
            "index": i,
            "stop_pos": pos,
            "length": len(supervised) - 2,
            "final_token_id": int(row[pos - 1]),
        })

    print("Test examples:", len(metadata))
    print("Batch size:", BATCH_SIZE)
    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True
    )

    print("\nLoading base model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cuda")

    base_result = evaluate(base, inputs, labels, metadata, "BASE")

    print("\nLoading Vyrex adapter...", flush=True)
    model = PeftModel.from_pretrained(
        base, ADAPTER_DIR, is_trainable=False
    )
    model.eval()

    vyrex_result = evaluate(model, inputs, labels, metadata, "VYREX")

    report = {
        "dataset": "OASST1 Qwen3 SFT v1",
        "test_examples": len(metadata),
        "stop_token_id": STOP_ID,
        "eos_token_id": EOS_ID,
        "newline_token_id": NEWLINE_ID,
        "batch_size": BATCH_SIZE,
        "base": base_result,
        "vyrex": vyrex_result,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== BASE OVERALL ===")
    print(json.dumps(base_result["overall"], indent=2))
    print("\n=== VYREX OVERALL ===")
    print(json.dumps(vyrex_result["overall"], indent=2))
    print("\n=== VYREX LENGTH BUCKETS ===")
    print(json.dumps(vyrex_result["length_buckets"], indent=2))
    print("\nSaved:", OUTPUT)
    print("Peak GPU memory:", round(
        torch.cuda.max_memory_allocated() / 1024**3, 3
    ), "GB")


if __name__ == "__main__":
    main()
