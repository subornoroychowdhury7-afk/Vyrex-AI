from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_0.6b/original"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
ADAPTER_DIR = ROOT / "pretrained/qwen3_0.6b/adapters/oasst1_lora_v1"
OUTPUT_DIR = ROOT / "pretrained/qwen3_0.6b/evaluations"
OUTPUT_FILE = OUTPUT_DIR / "oasst1_lora_v1_test.json"

PAD_ID = 151643


def evaluate(model, inputs, labels, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    start = time.perf_counter()

    with torch.inference_mode():
        for i in range(len(inputs)):
            x = torch.as_tensor(
                np.array(inputs[i], dtype=np.int64, copy=True),
                device=device,
            ).unsqueeze(0)

            y = torch.as_tensor(
                np.array(labels[i], dtype=np.int64, copy=True),
                device=device,
            ).unsqueeze(0)

            positions_idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            last_valid = positions_idx.masked_fill(y.eq(-100), -1).max(dim=1).values
            if torch.any(last_valid < 0):
                raise RuntimeError("Batch contains a row without supervised tokens.")
            mask = (positions_idx <= last_valid.unsqueeze(1)).long()
            token_count = int(y[:, 1:].ne(-100).sum().item())

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=x,
                    attention_mask=mask,
                    labels=y,
                    use_cache=False,
                )

            if not torch.isfinite(output.loss):
                raise RuntimeError(f"Non-finite loss at test sample {i}")

            total_loss += output.loss.item() * token_count
            total_tokens += token_count

            del x, y, mask, output

    if total_tokens == 0:
        raise RuntimeError("Test dataset contains no supervised tokens.")

    loss = total_loss / total_tokens

    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "supervised_tokens": total_tokens,
        "examples": len(inputs),
        "seconds": round(time.perf_counter() - start, 2),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    for path in (MODEL_DIR, ADAPTER_DIR, DATA_DIR):
        if not path.exists():
            raise FileNotFoundError(path)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {OUTPUT_FILE}"
        )

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    print("=== VYREX QWEN3 TEST EVALUATION ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    inputs = np.load(DATA_DIR / "test_input_ids.npy", mmap_mode="r")
    labels = np.load(DATA_DIR / "test_labels.npy", mmap_mode="r")

    if inputs.shape != labels.shape or inputs.ndim != 2:
        raise ValueError("Test input/label shape mismatch.")

    print(f"Test examples: {len(inputs)}", flush=True)
    print(f"Sequence length: {inputs.shape[1]}", flush=True)

    print("\nLoading original Qwen3 base...", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    ).to(device)

    base.config.use_cache = False
    base_result = evaluate(base, inputs, labels, device)

    print(
        f"BASE: loss={base_result['loss']:.6f} "
        f"ppl={base_result['perplexity']:.4f}",
        flush=True,
    )

    print("\nLoading best Vyrex LoRA adapter...", flush=True)

    model = PeftModel.from_pretrained(
        base,
        ADAPTER_DIR,
        is_trainable=False,
    )
    model.eval()

    adapter_result = evaluate(model, inputs, labels, device)

    print(
        f"VYREX: loss={adapter_result['loss']:.6f} "
        f"ppl={adapter_result['perplexity']:.4f}",
        flush=True,
    )

    improvement = (
        (base_result["loss"] - adapter_result["loss"])
        / base_result["loss"]
        * 100
    )

    peak_memory = round(
        torch.cuda.max_memory_allocated() / (1024 ** 3), 3
    )

    report = {
        "experiment": "oasst1_lora_v1",
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "adapter": str(ADAPTER_DIR.relative_to(ROOT)),
        "dataset": "OASST1 Qwen3 SFT v1",
        "split": "test",
        "precision": "BF16",
        "base": base_result,
        "vyrex_best_adapter": adapter_result,
        "test_loss_improvement_percent": improvement,
        "peak_gpu_allocated_gb": peak_memory,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== FINAL TEST COMPARISON ===")
    print(f"Base loss:       {base_result['loss']:.6f}")
    print(f"Vyrex loss:      {adapter_result['loss']:.6f}")
    print(f"Improvement:     {improvement:+.2f}%")
    print(f"Peak GPU memory: {peak_memory:.3f} GB")
    print(f"Report: {OUTPUT_FILE}")

    del model, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
