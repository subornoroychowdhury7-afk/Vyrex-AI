from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_1.7b/original"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
EVAL_DIR = ROOT / "pretrained/qwen3_1.7b/evaluations"

OUTPUT_FILE = EVAL_DIR / "vyrex_1_7b_qlora_comparison_test.json"

ADAPTERS = {
    "pilot_100_steps": (
        ROOT / "pretrained/qwen3_1.7b/adapters/"
        "vyrex_1_7b_qlora_v1/epoch_01_best"
    ),
    "full_epoch_384_steps": (
        ROOT / "pretrained/qwen3_1.7b/adapters/"
        "vyrex_1_7b_qlora_full_v1/epoch_01_best"
    ),
}


def evaluate(model, inputs, labels):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    start = time.perf_counter()

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

            positions = torch.arange(
                x.shape[1], device=x.device
            ).unsqueeze(0)

            last_valid = positions.masked_fill(
                y.eq(-100), -1
            ).max(dim=1).values

            if torch.any(last_valid < 0):
                raise RuntimeError(
                    f"No supervised tokens in sample {i}"
                )

            mask = (
                positions <= last_valid.unsqueeze(1)
            ).long()

            token_count = int(y[:, 1:].ne(-100).sum().item())

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=x,
                    attention_mask=mask,
                    labels=y,
                    use_cache=False,
                )

            if not torch.isfinite(output.loss):
                raise RuntimeError(
                    f"Non-finite loss at sample {i}"
                )

            total_loss += output.loss.item() * token_count
            total_tokens += token_count

            del x, y, mask, output

            if (i + 1) % 50 == 0:
                print(
                    f"  Evaluated {i + 1}/{len(inputs)}",
                    flush=True,
                )

    if total_tokens == 0:
        raise RuntimeError("No supervised test tokens.")

    elapsed = time.perf_counter() - start
    loss = total_loss / total_tokens

    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "supervised_tokens": total_tokens,
        "examples": len(inputs),
        "seconds": round(elapsed, 2),
        "tokens_per_second": round(total_tokens / elapsed, 2),
    }


def load_base(quant_config):
    return AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )


def run_evaluation(name, adapter_path, inputs, labels, quant_config):
    print(f"\n=== EVALUATING: {name} ===", flush=True)

    if adapter_path is not None and not adapter_path.is_dir():
        raise FileNotFoundError(adapter_path)

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    model = load_base(quant_config)

    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )

    model.eval()
    model.config.use_cache = False

    result = evaluate(model, inputs, labels)

    result["peak_gpu_allocated_gb"] = round(
        torch.cuda.max_memory_allocated() / 1024**3, 3
    )
    result["peak_gpu_reserved_gb"] = round(
        torch.cuda.max_memory_reserved() / 1024**3, 3
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"{name}: loss={result['loss']:.6f}, "
        f"ppl={result['perplexity']:.4f}, "
        f"peak={result['peak_gpu_allocated_gb']} GB",
        flush=True,
    )

    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 required.")

    for path in (MODEL_DIR, DATA_DIR):
        if not path.is_dir():
            raise FileNotFoundError(path)

    if OUTPUT_FILE.exists():
        raise FileExistsError(OUTPUT_FILE)

    for path in ADAPTERS.values():
        if not path.is_dir():
            raise FileNotFoundError(path)

    inputs = np.load(
        DATA_DIR / "test_input_ids.npy", mmap_mode="r"
    )
    labels = np.load(
        DATA_DIR / "test_labels.npy", mmap_mode="r"
    )

    if inputs.shape != labels.shape or inputs.ndim != 2:
        raise ValueError("Test input/label shape mismatch.")

    print("=== VYREX QWEN3-1.7B TEST COMPARISON ===")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Examples:", len(inputs))
    print("Sequence length:", inputs.shape[1])

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    results = {}

    results["base_nf4"] = run_evaluation(
        "Qwen3-1.7B Base NF4",
        None,
        inputs,
        labels,
        quant_config,
    )

    for name, adapter_path in ADAPTERS.items():
        results[name] = run_evaluation(
            name,
            adapter_path,
            inputs,
            labels,
            quant_config,
        )

    base_loss = results["base_nf4"]["loss"]

    for name, result in results.items():
        result["improvement_vs_base_percent"] = (
            (base_loss - result["loss"]) / base_loss * 100
        )

    report = {
        "experiment": "vyrex_qwen3_1.7b_qlora_test_comparison",
        "dataset": "OASST1 Qwen3 SFT v1",
        "split": "test",
        "examples": len(inputs),
        "sequence_length": inputs.shape[1],
        "supervised_tokens": results["base_nf4"]["supervised_tokens"],
        "base_model": "Qwen/Qwen3-1.7B-Base",
        "quantization": "NF4 4-bit double quantization",
        "compute_dtype": "BF16",
        "gpu": torch.cuda.get_device_name(0),
        "results": results,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n========== FINAL TEST COMPARISON ==========")

    for name, result in results.items():
        print(
            f"{name:24s} "
            f"loss={result['loss']:.6f} "
            f"ppl={result['perplexity']:.4f} "
            f"improvement={result['improvement_vs_base_percent']:.2f}%"
        )

    print("\nReport:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
