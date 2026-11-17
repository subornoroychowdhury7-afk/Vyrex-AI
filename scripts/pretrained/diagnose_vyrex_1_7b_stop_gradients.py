from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "pretrained/qwen3_1.7b/original"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"

ADAPTER_DIR = (
    ROOT
    / "pretrained/qwen3_1.7b/adapters/"
    / "vyrex_1_7b_qlora_full_v1/epoch_01_best"
)

OUTPUT = (
    ROOT
    / "pretrained/qwen3_1.7b/evaluations/"
    / "stop_gradient_analysis_1_7b_full384.json"
)

STOP_ID = 151645
STOP_WEIGHT = 5.0
GRADIENT_SAMPLE_COUNT = 16


def attention_mask(labels):
    positions = torch.arange(
        labels.shape[1], device=labels.device
    ).unsqueeze(0)

    last_valid = positions.masked_fill(
        labels.eq(-100), -1
    ).max(dim=1).values

    if torch.any(last_valid < 0):
        raise RuntimeError("Example has no supervised tokens.")

    return (positions <= last_valid.unsqueeze(1)).long()


def gradient_norm(grads):
    total = torch.zeros((), device="cuda", dtype=torch.float64)

    for grad in grads:
        if grad is not None:
            total += grad.detach().double().pow(2).sum()

    return total.sqrt()


def gradient_dot(grads_a, grads_b):
    total = torch.zeros((), device="cuda", dtype=torch.float64)

    for a, b in zip(grads_a, grads_b):
        if a is not None and b is not None:
            total += (
                a.detach().double() * b.detach().double()
            ).sum()

    return total


def summarize(values):
    values = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required.")

    if OUTPUT.exists():
        raise FileExistsError(f"Report already exists: {OUTPUT}")

    for path in (MODEL_DIR, DATA_DIR, ADAPTER_DIR):
        if not path.exists():
            raise FileNotFoundError(path)

    print("=== VYREX 1.7B STOP GRADIENT DIAGNOSTIC ===")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Adapter:", ADAPTER_DIR.name)

    inputs = np.load(
        DATA_DIR / "validation_input_ids.npy",
        mmap_mode="r",
    )
    labels = np.load(
        DATA_DIR / "validation_labels.npy",
        mmap_mode="r",
    )

    if inputs.shape != labels.shape:
        raise RuntimeError("Validation array shape mismatch.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=False,
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nLoading NF4 base model...")

    torch.cuda.reset_peak_memory_stats()

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )

    base.config.use_cache = False

    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
    )

    print("Loading existing full-epoch adapter...")

    model = PeftModel.from_pretrained(
        base,
        ADAPTER_DIR,
        is_trainable=True,
    )

    model.eval()

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable:
        raise RuntimeError("No trainable adapter parameters.")

    trainable_count = sum(
        parameter.numel() for parameter in trainable
    )

    print("Trainable parameters:", trainable_count)
    print("Validation examples:", len(inputs))

    # ---------------------------------------------------------
    # PART 1: Full validation token-level loss decomposition
    # ---------------------------------------------------------

    print("\n=== PART 1: VALIDATION LOSS DECOMPOSITION ===")

    stop_loss_sum = 0.0
    ordinary_loss_sum = 0.0
    stop_count = 0
    ordinary_count = 0

    model.eval()

    with torch.inference_mode():
        for i in range(len(inputs)):
            x = torch.as_tensor(
                np.array(inputs[i], dtype=np.int64, copy=True),
                device="cuda",
            ).unsqueeze(0)

            y = torch.as_tensor(
                np.array(labels[i], dtype=np.int64, copy=True),
                device="cuda",
            ).unsqueeze(0)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=x,
                    attention_mask=attention_mask(y),
                    use_cache=False,
                )

            logits = output.logits[:, :-1, :]
            targets = y[:, 1:]

            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).view_as(targets)

            stop_mask = targets.eq(STOP_ID)
            ordinary_mask = targets.ne(-100) & ~stop_mask

            stop_loss_sum += (
                losses.masked_select(stop_mask).float().sum().item()
            )

            ordinary_loss_sum += (
                losses.masked_select(ordinary_mask).float().sum().item()
            )

            stop_count += int(stop_mask.sum().item())
            ordinary_count += int(ordinary_mask.sum().item())

            del x, y, output, logits, targets, losses

            if (i + 1) % 50 == 0:
                print(
                    f"Validation: {i + 1}/{len(inputs)}",
                    flush=True,
                )

    if stop_count != len(inputs):
        raise RuntimeError(
            f"Expected one STOP per example, found {stop_count}."
        )

    stop_mean = stop_loss_sum / stop_count
    ordinary_mean = ordinary_loss_sum / ordinary_count

    weighted_denominator = (
        ordinary_count + STOP_WEIGHT * stop_count
    )

    weighted_validation_loss = (
        ordinary_loss_sum + STOP_WEIGHT * stop_loss_sum
    ) / weighted_denominator

    unweighted_validation_loss = (
        ordinary_loss_sum + stop_loss_sum
    ) / (ordinary_count + stop_count)

    print("STOP count:", stop_count)
    print("Ordinary token count:", ordinary_count)
    print("STOP mean CE:", round(stop_mean, 6))
    print("Ordinary mean CE:", round(ordinary_mean, 6))
    print("Unweighted validation loss:", round(
        unweighted_validation_loss, 6
    ))
    print("5x weighted validation loss:", round(
        weighted_validation_loss, 6
    ))

    # ---------------------------------------------------------
    # PART 2: Representative per-example LoRA gradient analysis
    # ---------------------------------------------------------

    print("\n=== PART 2: GRADIENT CONTRIBUTION ANALYSIS ===")

    lengths = []

    for i in range(len(labels)):
        targets = labels[i, 1:]
        valid = targets != -100
        lengths.append(int(np.count_nonzero(valid)))

    # Select examples spread across the full response-length range.
    sorted_indices = np.argsort(np.asarray(lengths))
    positions = np.linspace(
        0,
        len(sorted_indices) - 1,
        min(GRADIENT_SAMPLE_COUNT, len(sorted_indices)),
        dtype=int,
    )
    selected_indices = sorted_indices[positions].tolist()

    results = []

    model.eval()

    for sample_number, index in enumerate(selected_indices, start=1):
        print(
            f"Gradient sample {sample_number}/"
            f"{len(selected_indices)} (example {index})",
            flush=True,
        )

        x = torch.as_tensor(
            np.array(inputs[index], dtype=np.int64, copy=True),
            device="cuda",
        ).unsqueeze(0)

        y = torch.as_tensor(
            np.array(labels[index], dtype=np.int64, copy=True),
            device="cuda",
        ).unsqueeze(0)

        model.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=x,
                attention_mask=attention_mask(y),
                use_cache=False,
            )

        logits = output.logits[:, :-1, :]
        targets = y[:, 1:]

        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(targets)

        stop_mask = targets.eq(STOP_ID)
        ordinary_mask = targets.ne(-100) & ~stop_mask

        n_stop = int(stop_mask.sum().item())
        n_ordinary = int(ordinary_mask.sum().item())

        if n_stop != 1 or n_ordinary == 0:
            raise RuntimeError(
                f"Unexpected masks in example {index}: "
                f"STOP={n_stop}, ordinary={n_ordinary}"
            )

        denominator = n_ordinary + STOP_WEIGHT * n_stop

        # These are the exact two components of the training
        # objective for this example.
        stop_component = (
            losses.masked_select(stop_mask).sum()
            * STOP_WEIGHT / denominator
        )

        ordinary_component = (
            losses.masked_select(ordinary_mask).sum()
            / denominator
        )

        stop_grads = torch.autograd.grad(
            stop_component,
            trainable,
            retain_graph=True,
            allow_unused=True,
        )

        ordinary_grads = torch.autograd.grad(
            ordinary_component,
            trainable,
            retain_graph=False,
            allow_unused=True,
        )

        stop_norm = gradient_norm(stop_grads)
        ordinary_norm = gradient_norm(ordinary_grads)
        dot = gradient_dot(stop_grads, ordinary_grads)

        stop_norm_value = stop_norm.item()
        ordinary_norm_value = ordinary_norm.item()
        dot_value = dot.item()

        cosine = dot_value / max(
            stop_norm_value * ordinary_norm_value,
            1e-30,
        )

        combined_squared = max(
            stop_norm_value ** 2
            + ordinary_norm_value ** 2
            + 2.0 * dot_value,
            0.0,
        )

        combined_norm = combined_squared ** 0.5

        results.append({
            "example_index": int(index),
            "ordinary_tokens": n_ordinary,
            "stop_tokens": n_stop,
            "stop_component_gradient_norm": stop_norm_value,
            "ordinary_component_gradient_norm": ordinary_norm_value,
            "stop_to_ordinary_gradient_norm_ratio": (
                stop_norm_value / max(ordinary_norm_value, 1e-30)
            ),
            "stop_gradient_fraction_of_component_norms": (
                stop_norm_value
                / max(stop_norm_value + ordinary_norm_value, 1e-30)
            ),
            "gradient_cosine_similarity": cosine,
            "combined_gradient_norm": combined_norm,
            "stop_component_loss": float(stop_component.detach().item()),
            "ordinary_component_loss": float(
                ordinary_component.detach().item()
            ),
        })

        del (
            x, y, output, logits, targets, losses,
            stop_component, ordinary_component,
            stop_grads, ordinary_grads,
            stop_norm, ordinary_norm, dot,
        )

        model.zero_grad(set_to_none=True)

        if sample_number % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # PART 3: Aggregate and save
    # ---------------------------------------------------------

    stop_norms = [
        item["stop_component_gradient_norm"] for item in results
    ]
    ordinary_norms = [
        item["ordinary_component_gradient_norm"] for item in results
    ]
    ratios = [
        item["stop_to_ordinary_gradient_norm_ratio"]
        for item in results
    ]
    cosines = [
        item["gradient_cosine_similarity"] for item in results
    ]
    fractions = [
        item["stop_gradient_fraction_of_component_norms"]
        for item in results
    ]

    report = {
        "experiment": "Vyrex Qwen3-1.7B full384 STOP gradient diagnostic",
        "adapter": str(ADAPTER_DIR.relative_to(ROOT)),
        "base_quantization": "NF4 double quantization",
        "compute_dtype": "BF16",
        "stop_token_id": STOP_ID,
        "stop_token_weight": STOP_WEIGHT,
        "validation": {
            "examples": len(inputs),
            "stop_tokens": stop_count,
            "ordinary_tokens": ordinary_count,
            "stop_mean_cross_entropy": stop_mean,
            "ordinary_mean_cross_entropy": ordinary_mean,
            "stop_to_ordinary_mean_ce_ratio": (
                stop_mean / max(ordinary_mean, 1e-30)
            ),
            "unweighted_loss": unweighted_validation_loss,
            "weighted_loss": weighted_validation_loss,
        },
        "gradient_analysis": {
            "sample_count": len(results),
            "selection": "16 examples distributed across sorted response lengths",
            "trainable_parameters": trainable_count,
            "stop_component_gradient_norm": summarize(stop_norms),
            "ordinary_component_gradient_norm": summarize(ordinary_norms),
            "stop_to_ordinary_gradient_norm_ratio": summarize(ratios),
            "stop_gradient_fraction_of_component_norms": summarize(
                fractions
            ),
            "gradient_cosine_similarity": summarize(cosines),
            "per_example": results,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "peak_allocated_gb": round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
            "peak_reserved_gb": round(
                torch.cuda.max_memory_reserved() / 1024**3, 3
            ),
        },
        "weights_updated": False,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== GRADIENT SUMMARY ===")
    print("STOP gradient norm:", summarize(stop_norms))
    print("Ordinary gradient norm:", summarize(ordinary_norms))
    print("STOP/ordinary gradient ratio:", summarize(ratios))
    print("STOP gradient fraction:", summarize(fractions))
    print("Gradient cosine similarity:", summarize(cosines))

    print("\nPeak allocated VRAM:", report["gpu"]["peak_allocated_gb"], "GB")
    print("Peak reserved VRAM:", report["gpu"]["peak_reserved_gb"], "GB")
    print("Weights updated: False")
    print("Report:", OUTPUT)


if __name__ == "__main__":
    main()
