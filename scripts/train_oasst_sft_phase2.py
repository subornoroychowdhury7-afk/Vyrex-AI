from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vyrex_ai.config import load_config
from vyrex_ai.models.transformer import VyrexConfig, VyrexTransformer
from vyrex_ai.training import load_checkpoint, save_checkpoint

DATA_DIR = ROOT / "data" / "processed" / "oasst1_sft_v1"
BASE_CONFIG = ROOT / "configs" / "oanc-production-v1.yaml"
BASE_CHECKPOINT = ROOT / "checkpoints" / "oasst1_sft_v1" / "best.pt"
OUTPUT_DIR = ROOT / "checkpoints" / "oasst1_sft_phase2"

MAX_STEPS = 600
ACCUMULATION = 8
EVAL_EVERY = 100
TRAIN_LR = 2.0e-6
MIN_LR = 2.0e-7
WARMUP_STEPS = 20
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def autocast_context(device, use_bf16):
    if device.type == "cuda" and use_bf16:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def learning_rate(step):
    if step <= WARMUP_STEPS:
        return TRAIN_LR * step / WARMUP_STEPS
    progress = min(1.0, (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + cosine * (TRAIN_LR - MIN_LR)


def load_arrays(split):
    x = np.load(DATA_DIR / f"{split}_input_ids.npy")
    y = np.load(DATA_DIR / f"{split}_labels.npy")

    if x.dtype != np.int32 or y.dtype != np.int32:
        raise ValueError(f"{split}: expected int32 arrays")
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 129:
        raise ValueError(f"{split}: invalid array shape")
    if not np.any(y[:, 1:] != -100):
        raise ValueError(f"{split}: no supervised tokens")
    if np.any(x < 0) or np.any(y < -100):
        raise ValueError(f"{split}: invalid token IDs")

    return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def evaluate(model, loader, device, use_bf16):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device, dtype=torch.long, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            inputs = input_ids[:, :-1]
            targets = labels[:, 1:]
            count = int((targets != -100).sum().item())

            if count == 0:
                continue

            with autocast_context(device, use_bf16):
                _, loss = model(inputs, targets)

            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss")

            total_loss += loss.item() * count
            total_tokens += count

    model.train()

    if total_tokens == 0:
        raise RuntimeError("No validation tokens")

    return total_loss / total_tokens, total_tokens


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")

    manifest = json.loads(
        (DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    tokenizer_path = ROOT / manifest["tokenizer"]

    if sha256_file(tokenizer_path) != manifest["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer SHA256 mismatch")

    if not BASE_CHECKPOINT.is_file():
        raise FileNotFoundError(BASE_CHECKPOINT)

    print("=== DATASET VERIFIED ===")
    print("Dataset:", manifest["dataset"])
    print("Tokenizer hash: verified")
    print("Train chunks:", manifest["counts"]["splits"]["train"]["training_chunks"])
    print("Validation chunks:", manifest["counts"]["splits"]["validation"]["training_chunks"])
    print("Test chunks:", manifest["counts"]["splits"]["test"]["training_chunks"])

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("high")

    train_x, train_y = load_arrays("train")
    val_x, val_y = load_arrays("validation")

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    config = load_config(BASE_CONFIG)
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

    print("\n=== LOADING PHASE 1 BEST VYREX SFT WEIGHTS ===")
    metadata = load_checkpoint(
        BASE_CHECKPOINT,
        model,
        map_location=device,
        restore_rng=False,
    )
    print("Base checkpoint step:", metadata["step"])
    print("Base validation loss:", metadata["best_validation_loss"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    metrics_path = OUTPUT_DIR / "metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            ["step", "train_loss", "validation_loss", "validation_tokens", "lr"]
        )

    print("\n=== SFT CONFIGURATION ===")
    print("Device:", device)
    print("Precision:", "BF16" if use_bf16 else "FP32")
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Optimizer steps:", MAX_STEPS)
    print("Gradient accumulation:", ACCUMULATION)
    print("Learning rate:", TRAIN_LR)
    print("Output:", OUTPUT_DIR)

    model.train()
    iterator = iter(train_loader)
    best_validation_loss = float("inf")
    accumulated_loss = 0.0
    accumulated_tokens = 0

    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss_sum = 0.0
        step_tokens = 0

        for _ in range(ACCUMULATION):
            try:
                input_ids, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                input_ids, labels = next(iterator)

            input_ids = input_ids.to(device, dtype=torch.long, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            inputs = input_ids[:, :-1]
            targets = labels[:, 1:]
            count = int((targets != -100).sum().item())

            if count == 0:
                continue

            with autocast_context(device, use_bf16):
                _, loss = model(inputs, targets)

            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}")

            (loss * count).backward()
            step_loss_sum += loss.item() * count
            step_tokens += count

        if step_tokens == 0:
            raise RuntimeError("No supervised tokens in optimizer step")

        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(step_tokens)

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.step()

        accumulated_loss += step_loss_sum
        accumulated_tokens += step_tokens

        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            val_loss, val_tokens = evaluate(
                model, val_loader, device, use_bf16
            )
            train_loss = accumulated_loss / accumulated_tokens

            print(
                f"Step {step}/{MAX_STEPS} | "
                f"train_loss={train_loss:.4f} | "
                f"validation_loss={val_loss:.4f} | lr={lr:.7e}",
                flush=True,
            )

            with metrics_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    [step, train_loss, val_loss, val_tokens, lr]
                )

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                save_checkpoint(
                    OUTPUT_DIR / "best.pt",
                    model,
                    optimizer,
                    step,
                    best_validation_loss=best_validation_loss,
                )
                print("  BEST CHECKPOINT UPDATED", flush=True)

            save_checkpoint(
                OUTPUT_DIR / "latest.pt",
                model,
                optimizer,
                step,
                best_validation_loss=best_validation_loss,
            )

            accumulated_loss = 0.0
            accumulated_tokens = 0

    print("\n=== SFT TRAINING COMPLETE ===")
    print("Completed steps:", MAX_STEPS)
    print("Best validation loss:", best_validation_loss)
    print("Best checkpoint:", OUTPUT_DIR / "best.pt")
    print("Latest checkpoint:", OUTPUT_DIR / "latest.pt")
    print("Metrics:", metrics_path)
    print("Test split was not used for training or model selection.")


if __name__ == "__main__":
    main()
