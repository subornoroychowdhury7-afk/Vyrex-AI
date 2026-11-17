from __future__ import annotations
import argparse
import gc
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_1.7b/original"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/vyrex_v3_sft"
ARTIFACT_DIR = ROOT / "pretrained/qwen3_1.7b"

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

RANK = 16
ALPHA = 32
DROPOUT = 0.05

ACCUMULATION = 16
LEARNING_RATE = 1.2e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

STOP_TOKEN_ID = 151645
STOP_TOKEN_WEIGHT = 6.0
SEED = 42


def load_arrays():
    arrays = {}
    for split in ("train", "validation"):
        x = np.load(DATA_DIR / f"{split}_input_ids.npy", mmap_mode="r")
        y = np.load(DATA_DIR / f"{split}_labels.npy", mmap_mode="r")
        if x.shape != y.shape or x.ndim != 2:
            raise ValueError(f"Invalid {split} dataset dimensions.")
        arrays[split] = (x, y)
    return arrays


def make_attention_mask(y):
    positions = torch.arange(y.shape[1], device=y.device).unsqueeze(0)
    last_valid = positions.masked_fill(y.eq(-100), -1).max(dim=1).values
    if torch.any(last_valid < 0):
        raise RuntimeError("Example contains no supervised tokens.")
    return (positions <= last_valid.unsqueeze(1)).long()


def weighted_training_loss(model, x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=x,
            attention_mask=make_attention_mask(y),
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

    weights = torch.where(
        targets.eq(-100),
        0.0,
        torch.where(
            targets.eq(STOP_TOKEN_ID),
            STOP_TOKEN_WEIGHT,
            1.0,
        ),
    )

    weighted_count = weights.sum()
    if weighted_count.item() <= 0:
        raise RuntimeError("Example has no weighted targets.")

    loss = (losses * weights).sum() / weighted_count
    return loss, float(weighted_count.item())


def evaluate(model, inputs, labels):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

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
                    attention_mask=make_attention_mask(y),
                    use_cache=False,
                )

            logits = output.logits[:, :-1, :]
            targets = y[:, 1:]

            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
                ignore_index=-100,
            )

            count = int(targets.ne(-100).sum().item())
            if not torch.isfinite(loss_sum):
                raise RuntimeError(f"Invalid validation loss at {i}.")

            total_loss += loss_sum.item()
            total_tokens += count
            del x, y, output, logits, targets, loss_sum

    model.train()
    if not total_tokens:
        raise RuntimeError("Validation set has no supervised tokens.")
    return total_loss / total_tokens


def save_adapter(model, tokenizer, directory):
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="vyrex_1_7b_qlora_v3")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise ValueError("Invalid run name.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required.")

    run_dir = ARTIFACT_DIR / "runs" / args.run_name
    adapter_dir = ARTIFACT_DIR / "adapters" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("=== VYREX QWEN3-1.7B QLoRA V3 ===", flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Mode:", "SMOKE TEST" if args.smoke_test else f"TRAINING ({args.epochs} EPOCHS)", flush=True)
    print(f"LoRA Config: Rank={RANK}, Alpha={ALPHA}, Targets={TARGET_MODULES}", flush=True)

    arrays = load_arrays()
    train_x, train_y = arrays["train"]
    val_x, val_y = arrays["validation"]

    print("Training examples:", len(train_x), flush=True)
    print("Validation examples:", len(val_x), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nLoading 1.7B NF4 base...", flush=True)
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.train()
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    if args.smoke_test:
        print("\n=== RUNNING 1-STEP GRADIENT TEST ===", flush=True)
        x = torch.as_tensor(np.array(train_x[0], dtype=np.int64, copy=True), device="cuda").unsqueeze(0)
        y = torch.as_tensor(np.array(train_y[0], dtype=np.int64, copy=True), device="cuda").unsqueeze(0)

        optimizer.zero_grad(set_to_none=True)
        loss, weighted_count = weighted_training_loss(model, x, y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        optimizer.step()

        print(f"Smoke Test Loss: {loss.item():.4f}")
        print(f"Smoke Test Grad Norm: {grad_norm.item():.4f}")
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print("\n>>> SMOKE TEST PASSED: V3 Architecture & Dataset Verified. Ready for full training. <<<", flush=True)
        return

    updates_per_epoch = math.ceil(len(train_x) / ACCUMULATION)
    total_steps = updates_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    warmup_steps = max(1, int(total_steps * 0.05))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    print("\n=== INITIAL VALIDATION ===", flush=True)
    initial_val = evaluate(model, val_x, val_y)
    print(f"Initial validation loss: {initial_val:.6f}", flush=True)

    best_val = initial_val
    best_adapter_dir = adapter_dir / "epoch_01_best"
    global_step = 0
    start_time = time.time()

    history = {
        "run_name": args.run_name,
        "base_model": "Qwen3-1.7B-Base",
        "dataset": "vyrex_v3_sft",
        "rank": RANK,
        "alpha": ALPHA,
        "epochs": args.epochs,
        "total_steps": total_steps,
        "initial_validation_loss": initial_val,
        "epochs_data": [],
    }

    indices = np.arange(len(train_x))

    for epoch in range(args.epochs):
        print(f"\n--- STARTING EPOCH {epoch + 1}/{args.epochs} ---", flush=True)
        np.random.shuffle(indices)
        epoch_loss_sum = 0.0
        epoch_tokens = 0

        for step in range(updates_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            batch_slice = indices[step * ACCUMULATION : min((step + 1) * ACCUMULATION, len(train_x))]

            for idx in batch_slice:
                x = torch.as_tensor(np.array(train_x[idx], dtype=np.int64, copy=True), device="cuda").unsqueeze(0)
                y = torch.as_tensor(np.array(train_y[idx], dtype=np.int64, copy=True), device="cuda").unsqueeze(0)

                loss, count = weighted_training_loss(model, x, y)
                scaled_loss = loss / len(batch_slice)
                scaled_loss.backward()

                epoch_loss_sum += loss.item() * count
                epoch_tokens += count
                del x, y, loss, scaled_loss

            torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step == 1 or global_step % 20 == 0:
                print(
                    f"Epoch {epoch + 1} | Step {global_step}/{total_steps} | "
                    f"Loss: {epoch_loss_sum / max(epoch_tokens, 1):.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                    flush=True,
                )

            if args.max_steps and global_step >= args.max_steps:
                break

        val_loss = evaluate(model, val_x, val_y)
        train_loss = epoch_loss_sum / max(epoch_tokens, 1)
        print(f"\n>>> EPOCH {epoch + 1} COMPLETE: Train Loss = {train_loss:.5f} | Val Loss = {val_loss:.5f} <<<", flush=True)

        epoch_save_dir = adapter_dir / f"epoch_{epoch + 1:02d}"
        save_adapter(model, tokenizer, epoch_save_dir)

        if val_loss < best_val:
            best_val = val_loss
            best_adapter_dir = adapter_dir / "best_checkpoint"
            save_adapter(model, tokenizer, best_adapter_dir)
            print(f"Saved new best adapter to: {best_adapter_dir}", flush=True)

        history["epochs_data"].append({
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        (run_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if args.max_steps and global_step >= args.max_steps:
            break

    print("\n=== V3 TRAINING COMPLETE ===", flush=True)
    print("Best Validation Loss:", best_val, flush=True)
    print("Best Adapter:", best_adapter_dir, flush=True)
    print("Peak VRAM:", f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
