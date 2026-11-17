import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import PeftModel


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_0.6b/original"
ADAPTER_DIR = ROOT / "pretrained/qwen3_0.6b/adapters/oasst1_lora_v1"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
OUTPUT_DIR = ROOT / "pretrained/qwen3_0.6b/evaluations"
OUTPUT_FILE = OUTPUT_DIR / "oasst1_stop_cross_entropy_v1.json"

STOP_ID = 151645
IGNORE_ID = -100


def evaluate(model, inputs, labels, device, name):
    model.eval()

    stop_losses = []
    stop_probabilities = []
    stop_ranks = []
    stop_margins = []
    non_stop_loss_sum = 0.0
    non_stop_token_count = 0

    start = time.perf_counter()

    with torch.inference_mode():
        for i in range(len(inputs)):
            x_np = np.array(inputs[i], dtype=np.int64, copy=True)
            y_np = np.array(labels[i], dtype=np.int64, copy=True)

            stop_positions = np.flatnonzero(y_np == STOP_ID)

            if len(stop_positions) != 1:
                raise RuntimeError(
                    f"{name}: example {i} has {len(stop_positions)} STOP labels."
                )

            pos = int(stop_positions[0])

            if pos == 0 or y_np[pos - 1] == IGNORE_ID:
                raise RuntimeError(
                    f"{name}: invalid STOP prediction position in example {i}."
                )

            x = torch.as_tensor(x_np, device=device).unsqueeze(0)

            valid_positions = np.flatnonzero(y_np != IGNORE_ID)
            if len(valid_positions) == 0:
                raise RuntimeError(f"{name}: example {i} has no labels.")

            last_valid = int(valid_positions[-1])

            positions = torch.arange(
                x.shape[1], device=device
            ).unsqueeze(0)

            mask = (positions <= last_valid).long()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=x,
                    attention_mask=mask,
                    use_cache=False,
                )

            # Causal LM: logits at position t predict label at t+1.
            shifted_logits = output.logits[0, :-1, :]
            shifted_labels = torch.as_tensor(
                y_np[1:], dtype=torch.long, device=device
            )

            # STOP prediction.
            stop_logits = shifted_logits[pos - 1].float()

            stop_ce = F.cross_entropy(
                stop_logits.unsqueeze(0),
                torch.tensor([STOP_ID], device=device),
            )

            log_probs = F.log_softmax(stop_logits, dim=-1)
            stop_probability = log_probs[STOP_ID].exp().item()

            stop_logit = stop_logits[STOP_ID]
            competing_max = stop_logits.max().item()
            stop_margin = stop_logit.item() - max(
                stop_logits[j].item()
                for j in range(stop_logits.numel())
                if j != STOP_ID
            )

            stop_rank = int(
                (stop_logits > stop_logit).sum().item()
            ) + 1

            # All other supervised response tokens.
            valid = shifted_labels.ne(IGNORE_ID)
            non_stop = valid & shifted_labels.ne(STOP_ID)

            selected_logits = shifted_logits[non_stop].float()
            selected_labels = shifted_labels[non_stop]

            if selected_labels.numel() > 0:
                losses = F.cross_entropy(
                    selected_logits,
                    selected_labels,
                    reduction="sum",
                )

                non_stop_loss_sum += losses.item()
                non_stop_token_count += selected_labels.numel()

            stop_losses.append(stop_ce.item())
            stop_probabilities.append(stop_probability)
            stop_ranks.append(stop_rank)
            stop_margins.append(stop_margin)

            del (
                x, mask, output, shifted_logits, shifted_labels,
                stop_logits, stop_ce, log_probs,
                selected_logits, selected_labels
            )

            if (i + 1) % 50 == 0:
                print(
                    f"{name}: {i + 1}/{len(inputs)}",
                    flush=True,
                )

    if not stop_losses or non_stop_token_count == 0:
        raise RuntimeError(f"{name}: no valid evaluation results.")

    stop_losses_np = np.asarray(stop_losses, dtype=np.float64)

    return {
        "examples_scored": len(stop_losses),
        "stop_mean_cross_entropy": float(stop_losses_np.mean()),
        "stop_median_cross_entropy": float(np.median(stop_losses_np)),
        "stop_std_cross_entropy": float(stop_losses_np.std()),
        "stop_mean_probability": float(np.mean(stop_probabilities)),
        "stop_median_probability": float(np.median(stop_probabilities)),
        "stop_mean_rank": float(np.mean(stop_ranks)),
        "stop_median_rank": float(np.median(stop_ranks)),
        "stop_top1_accuracy": float(np.mean(np.asarray(stop_ranks) == 1)),
        "stop_mean_logit_margin": float(np.mean(stop_margins)),
        "non_stop_mean_cross_entropy": (
            non_stop_loss_sum / non_stop_token_count
        ),
        "non_stop_supervised_tokens": non_stop_token_count,
        "seconds": round(time.perf_counter() - start, 2),
        "per_example_stop_ce": [float(v) for v in stop_losses],
        "per_example_stop_probability": [
            float(v) for v in stop_probabilities
        ],
        "per_example_stop_rank": [int(v) for v in stop_ranks],
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    for path in (MODEL_DIR, ADAPTER_DIR, DATA_DIR):
        if not path.exists():
            raise FileNotFoundError(path)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT_FILE}"
        )

    inputs = np.load(
        DATA_DIR / "test_input_ids.npy", mmap_mode="r"
    )
    labels = np.load(
        DATA_DIR / "test_labels.npy", mmap_mode="r"
    )

    if inputs.shape != labels.shape or inputs.ndim != 2:
        raise ValueError("Input/label shape mismatch.")

    if len(inputs) != 312:
        raise RuntimeError(
            f"Expected 312 test examples, found {len(inputs)}."
        )

    stop_counts = np.sum(labels == STOP_ID, axis=1)

    if not np.all(stop_counts == 1):
        raise RuntimeError(
            "Every test example must contain exactly one STOP label."
        )

    if np.any(labels[:, 1:] == STOP_ID):
        shifted_stop_count = int(np.sum(labels[:, 1:] == STOP_ID))
        if shifted_stop_count != len(inputs):
            raise RuntimeError("Unexpected shifted STOP count.")

    device = torch.device("cuda")

    print("=== STOP CROSS-ENTROPY EVALUATION ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Test examples: {len(inputs)}")
    print(f"Sequence length: {inputs.shape[1]}")
    print("STOP integrity: PASS (one per example)")

    print("\nLoading Base model...", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    ).to(device)

    base.config.use_cache = False

    base_result = evaluate(
        base, inputs, labels, device, "BASE"
    )

    print(
        "BASE:",
        {
            k: v for k, v in base_result.items()
            if not k.startswith("per_example")
        },
        flush=True,
    )

    del base
    gc.collect()
    torch.cuda.empty_cache()

    print("\nLoading VYREX V1...", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    ).to(device)

    base.config.use_cache = False

    model = PeftModel.from_pretrained(
        base,
        ADAPTER_DIR,
        is_trainable=False,
    )
    model.eval()

    vyrex_result = evaluate(
        model, inputs, labels, device, "VYREX"
    )

    stop_ce_improvement = (
        base_result["stop_mean_cross_entropy"]
        - vyrex_result["stop_mean_cross_entropy"]
    )

    stop_ce_improvement_percent = (
        stop_ce_improvement
        / base_result["stop_mean_cross_entropy"]
        * 100
    )

    non_stop_improvement = (
        base_result["non_stop_mean_cross_entropy"]
        - vyrex_result["non_stop_mean_cross_entropy"]
    )

    report = {
        "experiment": "oasst1_lora_v1_stop_cross_entropy",
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "adapter": str(ADAPTER_DIR.relative_to(ROOT)),
        "dataset": "OASST1 Qwen3 SFT v1",
        "split": "test",
        "examples": len(inputs),
        "stop_token_id": STOP_ID,
        "precision": "BF16 inference / FP32 loss",
        "base": base_result,
        "vyrex_v1": vyrex_result,
        "stop_ce_improvement_nats": stop_ce_improvement,
        "stop_ce_improvement_percent": stop_ce_improvement_percent,
        "non_stop_ce_improvement_nats": non_stop_improvement,
        "interpretation": {
            "lower_stop_ce_is_better": True,
            "lower_non_stop_ce_is_better": True,
            "stop_ce_improvement_positive": stop_ce_improvement > 0,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== FINAL COMPARISON ===")
    print(
        f"Base STOP CE:       "
        f"{base_result['stop_mean_cross_entropy']:.6f}"
    )
    print(
        f"VYREX STOP CE:      "
        f"{vyrex_result['stop_mean_cross_entropy']:.6f}"
    )
    print(
        f"STOP CE improvement: "
        f"{stop_ce_improvement:+.6f} nats "
        f"({stop_ce_improvement_percent:+.2f}%)"
    )
    print(
        f"Base non-STOP CE:   "
        f"{base_result['non_stop_mean_cross_entropy']:.6f}"
    )
    print(
        f"VYREX non-STOP CE:  "
        f"{vyrex_result['non_stop_mean_cross_entropy']:.6f}"
    )
    print(f"Saved: {OUTPUT_FILE}")

    del model, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
