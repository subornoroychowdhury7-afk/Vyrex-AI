from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vyrex_ai.config import load_config
from vyrex_ai.models.transformer import VyrexConfig, VyrexTransformer
from vyrex_ai.training import load_checkpoint

DATA_DIR = ROOT / "data" / "processed" / "oasst1_sft_v1"
BASE_CHECKPOINT = ROOT / "checkpoints" / "oanc_production_v1" / "best.pt"
SFT_CHECKPOINT = ROOT / "checkpoints" / "oasst1_sft_v1" / "best.pt"
REPORT = ROOT / "checkpoints" / "oasst1_sft_v1" / "test_comparison.json"
BATCH_SIZE = 4


def load_test_data():
    x = np.load(DATA_DIR / "test_input_ids.npy")
    y = np.load(DATA_DIR / "test_labels.npy")

    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 129:
        raise ValueError("Invalid test array dimensions")

    return TensorDataset(
        torch.from_numpy(x.copy()),
        torch.from_numpy(y.copy()),
    )


def evaluate(model, loader, device, use_bf16):
    model.eval()
    loss_sum = 0.0
    token_count = 0

    with torch.inference_mode():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)

            inputs = input_ids[:, :-1]
            targets = labels[:, 1:]
            count = int((targets != -100).sum().item())

            if count == 0:
                continue

            if use_bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(inputs, targets)
            else:
                _, loss = model(inputs, targets)

            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Invalid evaluation loss")

            loss_sum += loss.item() * count
            token_count += count

    if token_count == 0:
        raise RuntimeError("No supervised test tokens")

    average_loss = loss_sum / token_count

    return {
        "test_loss": average_loss,
        "test_perplexity": math.exp(average_loss),
        "supervised_tokens": token_count,
    }


def main():
    if not BASE_CHECKPOINT.is_file():
        raise FileNotFoundError(BASE_CHECKPOINT)
    if not SFT_CHECKPOINT.is_file():
        raise FileNotFoundError(SFT_CHECKPOINT)
    if REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite {REPORT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    config = load_config(ROOT / "configs" / "oanc-production-v1.yaml")
    m = config.model

    model_config = VyrexConfig(
        vocab_size=m.vocab_size,
        hidden_size=m.hidden_size,
        num_layers=m.num_layers,
        num_heads=m.num_heads,
        intermediate_size=m.intermediate_size,
        max_seq_len=m.max_seq_len,
        dropout=m.dropout,
        tie_embeddings=m.tie_embeddings,
    )

    model = VyrexTransformer(model_config).to(device)
    loader = DataLoader(
        load_test_data(),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print("=== CONTROLLED TEST COMPARISON ===")
    print("Device:", device)
    print("Precision:", "BF16" if use_bf16 else "FP32")
    print("Test examples:", len(loader.dataset))
    print("Evaluation batches:", len(loader))

    results = {}

    checkpoints = [
        ("pretrained", BASE_CHECKPOINT),
        ("sft", SFT_CHECKPOINT),
    ]

    for name, path in checkpoints:
        print(f"\nEvaluating {name}: {path.name}", flush=True)

        metadata = load_checkpoint(
            path,
            model,
            map_location=device,
            restore_rng=False,
        )

        result = evaluate(model, loader, device, use_bf16)
        result["checkpoint_step"] = metadata["step"]
        result["checkpoint_best_validation_loss"] = (
            metadata["best_validation_loss"]
        )
        results[name] = result

        print("Checkpoint step:", result["checkpoint_step"])
        print("Test loss:", f"{result['test_loss']:.6f}")
        print("Test perplexity:", f"{result['test_perplexity']:.4f}")
        print("Supervised tokens:", result["supervised_tokens"])

    base_loss = results["pretrained"]["test_loss"]
    sft_loss = results["sft"]["test_loss"]
    improvement = (base_loss - sft_loss) / base_loss * 100

    report = {
        "evaluation": "same held-out OASST1 test split",
        "test_examples": len(loader.dataset),
        "results": results,
        "relative_test_loss_improvement_percent": improvement,
        "lower_test_loss_is_better": True,
    }

    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== FINAL COMPARISON ===")
    print(f"Pretrained test loss: {base_loss:.6f}")
    print(f"SFT test loss:        {sft_loss:.6f}")
    print(f"Relative improvement: {improvement:+.2f}%")
    print("Report:", REPORT)


if __name__ == "__main__":
    main()
