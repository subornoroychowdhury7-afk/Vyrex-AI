from pathlib import Path
import gc
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_0.6b/original"
ADAPTER_DIR = ROOT / "pretrained/qwen3_0.6b/adapters/oasst1_lora_v1"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
OUTPUT_FILE = ROOT / "pretrained/qwen3_0.6b/evaluations/oasst1_end_probability.json"

PAD_ID = 151643
STOP_ID = 151645


def inspect_model(model, inputs, labels, tokenizer, device, name):
    model.eval()
    results = []

    with torch.inference_mode():
        for i in range(len(inputs)):
            positions = np.flatnonzero(labels[i] == STOP_ID)

            if len(positions) != 1:
                continue

            pos = int(positions[0])
            if pos == 0:
                continue

            x = torch.as_tensor(
                np.array(inputs[i], dtype=np.int64, copy=True),
                device=device,
            ).unsqueeze(0)

            valid_positions = np.flatnonzero(np.asarray(labels[i]) != -100)
            if len(valid_positions) == 0:
                raise RuntimeError(f"Example {i} has no supervised tokens.")
            last_valid = int(valid_positions[-1])
            mask = (torch.arange(x.shape[1], device=x.device).unsqueeze(0) <= last_valid).long()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=x,
                    attention_mask=mask,
                    use_cache=False,
                )

            logits = output.logits[0, pos - 1].float()
            target_logit = logits[STOP_ID]
            log_probs = torch.log_softmax(logits, dim=-1)
            probability = log_probs[STOP_ID].exp().item()
            rank = int((logits > target_logit).sum().item()) + 1

            top_ids = torch.topk(logits, 5).indices.tolist()

            results.append({
                "example": i,
                "stop_probability": probability,
                "stop_rank": rank,
                "top5_ids": top_ids,
                "top5_tokens": tokenizer.convert_ids_to_tokens(top_ids),
            })

            del x, mask, output, logits, log_probs

            if (i + 1) % 50 == 0:
                print(f"{name}: {i + 1}/{len(inputs)}", flush=True)

    if not results:
        raise RuntimeError("No valid assistant-end targets found.")

    probabilities = [r["stop_probability"] for r in results]
    ranks = [r["stop_rank"] for r in results]

    return {
        "examples_scored": len(results),
        "mean_stop_probability": float(np.mean(probabilities)),
        "median_stop_probability": float(np.median(probabilities)),
        "mean_stop_rank": float(np.mean(ranks)),
        "median_stop_rank": float(np.median(ranks)),
        "top1_stop_accuracy": float(np.mean([r == 1 for r in ranks])),
        "examples": results,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    if OUTPUT_FILE.exists():
        raise FileExistsError(OUTPUT_FILE)

    device = torch.device("cuda")

    inputs = np.load(DATA_DIR / "test_input_ids.npy", mmap_mode="r")
    labels = np.load(DATA_DIR / "test_labels.npy", mmap_mode="r")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=False,
    )

    actual_stop_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if actual_stop_id != STOP_ID:
        raise RuntimeError(f"Unexpected stop token ID: {actual_stop_id}")

    print("Loading original Qwen3 base...", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    ).to(device)

    base_result = inspect_model(
        base, inputs, labels, tokenizer, device, "BASE"
    )

    print("Loading Vyrex adapter...", flush=True)

    model = PeftModel.from_pretrained(
        base, ADAPTER_DIR, is_trainable=False
    )
    model.eval()

    vyrex_result = inspect_model(
        model, inputs, labels, tokenizer, device, "VYREX"
    )

    report = {
        "stop_token": "<|im_end|>",
        "stop_token_id": STOP_ID,
        "base": base_result,
        "vyrex": vyrex_result,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nBASE:", {
        k: v for k, v in base_result.items() if k != "examples"
    })
    print("\nVYREX:", {
        k: v for k, v in vyrex_result.items() if k != "examples"
    })
    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
