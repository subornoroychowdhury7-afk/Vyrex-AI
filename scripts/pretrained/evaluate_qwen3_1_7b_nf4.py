from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "pretrained/qwen3_1.7b/original"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
OUTPUT_DIR = ROOT / "pretrained/qwen3_1.7b/evaluations"
OUTPUT_FILE = OUTPUT_DIR / "oasst1_1_7b_nf4_test.json"


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

            with torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
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

            if (i + 1) % 25 == 0:
                print(
                    f"Evaluated {i + 1}/{len(inputs)} examples",
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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    for path in (MODEL_DIR, DATA_DIR):
        if not path.exists():
            raise FileNotFoundError(path)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT_FILE}"
        )

    print("=== QWEN3-1.7B NF4 TEST EVALUATION ===")
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())

    inputs = np.load(
        DATA_DIR / "test_input_ids.npy", mmap_mode="r"
    )
    labels = np.load(
        DATA_DIR / "test_labels.npy", mmap_mode="r"
    )

    if inputs.shape != labels.shape or inputs.ndim != 2:
        raise ValueError("Test input/label shape mismatch.")

    print("Test examples:", len(inputs))
    print("Sequence length:", inputs.shape[1])

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nLoading Qwen3-1.7B in NF4...", flush=True)

    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )

    model.eval()
    model.config.use_cache = False

    print(
        "Allocated GB:",
        round(torch.cuda.memory_allocated() / 1024**3, 3),
        flush=True,
    )

    result = evaluate(model, inputs, labels)

    peak_allocated = round(
        torch.cuda.max_memory_allocated() / 1024**3, 3
    )
    peak_reserved = round(
        torch.cuda.max_memory_reserved() / 1024**3, 3
    )

    report = {
        "experiment": "oasst1_1_7b_nf4",
        "model": "Qwen/Qwen3-1.7B-Base",
        "dataset": "OASST1 Qwen3 SFT v1",
        "split": "test",
        "quantization": "NF4 4-bit double quantization",
        "compute_dtype": "BF16",
        "base_model_precision_comparison": (
            "Compare cautiously against 0.6B BF16 results "
            "because quantization differs."
        ),
        "gpu": torch.cuda.get_device_name(0),
        "result": result,
        "peak_gpu_allocated_gb": peak_allocated,
        "peak_gpu_reserved_gb": peak_reserved,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n========== FINAL RESULT ==========")
    print(f"Test loss:       {result['loss']:.6f}")
    print(f"Perplexity:      {result['perplexity']:.4f}")
    print(f"Supervised:      {result['supervised_tokens']}")
    print(f"Evaluation time: {result['seconds']} seconds")
    print(f"Throughput:      {result['tokens_per_second']} tokens/sec")
    print(f"Peak allocated:  {peak_allocated} GB")
    print(f"Peak reserved:   {peak_reserved} GB")
    print("Report:", OUTPUT_FILE)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
