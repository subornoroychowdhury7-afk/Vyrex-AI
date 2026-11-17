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
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v2"
ARTIFACT_DIR = ROOT / "pretrained/qwen3_1.7b"

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

RANK = 8
ALPHA = 16
DROPOUT = 0.05

ACCUMULATION = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

STOP_TOKEN_ID = 151645
STOP_TOKEN_WEIGHT = 5.0
SEED = 42


def load_arrays():
    arrays = {}

    for split in ("train", "validation"):
        x = np.load(
            DATA_DIR / f"{split}_input_ids.npy",
            mmap_mode="r",
        )
        y = np.load(
            DATA_DIR / f"{split}_labels.npy",
            mmap_mode="r",
        )

        if x.shape != y.shape or x.ndim != 2:
            raise ValueError(f"Invalid {split} dataset dimensions.")

        arrays[split] = (x, y)

    return arrays


def make_attention_mask(y):
    positions = torch.arange(
        y.shape[1], device=y.device
    ).unsqueeze(0)

    last_valid = positions.masked_fill(
        y.eq(-100), -1
    ).max(dim=1).values

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
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        default="vyrex_1_7b_qlora_v1",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise ValueError("Invalid run name.")

    if args.max_steps < 0 or args.epochs < 1:
        raise ValueError("Invalid training limits.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required.")

    for path in (MODEL_DIR, DATA_DIR):
        if not path.is_dir():
            raise FileNotFoundError(path)

    run_dir = ARTIFACT_DIR / "runs" / args.run_name
    adapter_dir = ARTIFACT_DIR / "adapters" / args.run_name

    if not args.smoke_test:
        if run_dir.exists() or adapter_dir.exists():
            raise FileExistsError(
                f"Existing run artifacts: {args.run_name}"
            )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("=== VYREX QWEN3-1.7B QLoRA ===", flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("BF16:", torch.cuda.is_bf16_supported(), flush=True)
    print("Mode:", "SMOKE TEST" if args.smoke_test else "TRAINING", flush=True)

    arrays = load_arrays()

    train_x, train_y = arrays["train"]
    val_x, val_y = arrays["validation"]

    print("Training examples:", len(train_x), flush=True)
    print("Validation examples:", len(val_x), flush=True)
    print("Sequence length:", train_x.shape[1], flush=True)

    manifest = json.loads(
        (DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=False,
    )

    if tokenizer.pad_token_id != manifest["tokenizer_pad_token_id"]:
        raise RuntimeError("Tokenizer PAD ID mismatch.")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nLoading 1.7B NF4 base...", flush=True)

    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
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

    trainable = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    if not trainable:
        raise RuntimeError("No trainable LoRA parameters.")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "Initial allocated VRAM:",
        round(torch.cuda.memory_allocated() / 1024**3, 3),
        "GB",
        flush=True,
    )

    if args.smoke_test:
        print("\n=== ONE-STEP GRADIENT TEST ===", flush=True)

        x = torch.as_tensor(
            np.array(train_x[0], dtype=np.int64, copy=True),
            device="cuda",
        ).unsqueeze(0)

        y = torch.as_tensor(
            np.array(train_y[0], dtype=np.int64, copy=True),
            device="cuda",
        ).unsqueeze(0)

        optimizer.zero_grad(set_to_none=True)

        loss, weighted_count = weighted_training_loss(model, x, y)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite smoke-test loss.")

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, MAX_GRAD_NORM
        )

        if not torch.isfinite(grad_norm):
            raise RuntimeError("Non-finite gradient norm.")

        gradients_present = any(
            p.grad is not None
            and torch.isfinite(p.grad).all().item()
            for p in trainable
        )

        if not gradients_present:
            raise RuntimeError("No valid LoRA gradients.")

        optimizer.step()

        print("Loss:", round(loss.item(), 6), flush=True)
        print("Weighted tokens:", weighted_count, flush=True)
        print("Gradient norm:", round(grad_norm.item(), 6), flush=True)
        print("Optimizer update: SUCCESS", flush=True)
        print(
            "Peak allocated VRAM:",
            round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "GB",
            flush=True,
        )
        print(
            "Peak reserved VRAM:",
            round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "GB",
            flush=True,
        )
        print("No training artifacts were saved.", flush=True)

        del x, y, loss
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return

    updates_per_epoch = math.ceil(
        len(train_x) / ACCUMULATION
    )

    total_steps = updates_per_epoch * args.epochs

    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    warmup_steps = max(1, int(total_steps * 0.03))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (step - warmup_steps) / max(
            total_steps - warmup_steps, 1
        )

        return 0.5 * (
            1.0 + math.cos(math.pi * min(progress, 1.0))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_factor
    )

    print("\n=== INITIAL VALIDATION ===", flush=True)

    initial_val = evaluate(model, val_x, val_y)

    print(
        f"Initial validation loss: {initial_val:.6f}",
        flush=True,
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    

    history = {
        "experiment": "vyrex_qwen3_1.7b_qlora",
        "base_model": "Qwen/Qwen3-1.7B-Base",
        "dataset": "OASST1 Qwen3 SFT v1",
        "quantization": "NF4 double quantization",
        "compute_dtype": "BF16",
        "lora_rank": RANK,
        "lora_alpha": ALPHA,
        "lora_dropout": DROPOUT,
        "target_modules": TARGET_MODULES,
        "sequence_length": int(train_x.shape[1]),
        "micro_batch_size": 1,
        "gradient_accumulation": ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "stop_token_weight": STOP_TOKEN_WEIGHT,
        "epochs_requested": args.epochs,
        "max_steps": args.max_steps,
        "initial_validation_loss": initial_val,
        "epochs": [],
    }

    best_val = float("inf")
    best_adapter_dir = None
    global_step = 0
    start_time = time.time()

    print("\n=== TRAINING STARTED ===", flush=True)

    for epoch in range(args.epochs):
        model.train()

        permutation = np.random.permutation(len(train_x))

        epoch_loss_sum = 0.0
        epoch_tokens = 0

        for group_start in range(0, len(permutation), ACCUMULATION):
            if args.max_steps and global_step >= args.max_steps:
                break

            group = permutation[
                group_start:group_start + ACCUMULATION
            ]

            weighted_counts = []

            for idx in group:
                targets = train_y[int(idx)][1:]
                valid = targets != -100

                ordinary = int(np.count_nonzero(valid))
                stop_count = int(np.count_nonzero(
                    targets[valid] == STOP_TOKEN_ID
                ))

                weighted_counts.append(
                    ordinary + (STOP_TOKEN_WEIGHT - 1.0) * stop_count
                )

            group_weighted_tokens = sum(weighted_counts)

            if group_weighted_tokens <= 0:
                raise RuntimeError("Empty accumulation group.")

            optimizer.zero_grad(set_to_none=True)

            for idx, count in zip(group, weighted_counts):
                i = int(idx)

                x = torch.as_tensor(
                    np.array(train_x[i], dtype=np.int64, copy=True),
                    device="cuda",
                ).unsqueeze(0)

                y = torch.as_tensor(
                    np.array(train_y[i], dtype=np.int64, copy=True),
                    device="cuda",
                ).unsqueeze(0)

                try:
                    loss, actual_count = weighted_training_loss(
                        model, x, y
                    )

                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss.")

                    scaled_loss = (
                        loss * (count / group_weighted_tokens)
                    )

                    scaled_loss.backward()

                    epoch_loss_sum += loss.item() * count
                    epoch_tokens += count

                    del x, y, loss, scaled_loss

                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA OOM during QLoRA training. "
                        "No previous model or adapter was modified."
                    )

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, MAX_GRAD_NORM
            )

            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite gradient norm.")

            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step == 1 or global_step % 10 == 0:
                print(
                    f"epoch={epoch + 1} "
                    f"step={global_step}/{total_steps} "
                    f"train_loss={epoch_loss_sum / max(epoch_tokens, 1):.5f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"VRAM={torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                    flush=True,
                )

        train_loss = epoch_loss_sum / max(epoch_tokens, 1)

        val_loss = evaluate(model, val_x, val_y)

        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "elapsed_seconds": round(time.time() - start_time, 2),
        }

        history["epochs"].append(record)

        print(
            f"\nEPOCH {epoch + 1}: "
            f"train={train_loss:.6f} "
            f"validation={val_loss:.6f}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss

            best_adapter_dir = adapter_dir / f"epoch_{epoch + 1:02d}_best"
            save_adapter(model, tokenizer, best_adapter_dir)

            print("Saved best adapter.", flush=True)

        history["best_validation_loss"] = best_val

        (run_dir / "training_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        if args.max_steps and global_step >= args.max_steps:
            break

    history["final_global_step"] = global_step
    history["final_validation_loss"] = history["epochs"][-1]["validation_loss"]
    history["best_validation_loss"] = best_val
    history["elapsed_seconds"] = round(time.time() - start_time, 2)
    history["peak_gpu_allocated_gb"] = round(
        torch.cuda.max_memory_allocated() / 1024**3, 3
    )
    history["peak_gpu_reserved_gb"] = round(
        torch.cuda.max_memory_reserved() / 1024**3, 3
    )

    (run_dir / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print("\n=== TRAINING COMPLETE ===", flush=True)
    print("Steps:", global_step, flush=True)
    print("Initial validation:", initial_val, flush=True)
    print("Best validation:", best_val, flush=True)
    print("Adapter:", best_adapter_dir, flush=True)
    print("History:", run_dir / "training_history.json", flush=True)
    print("Peak VRAM:", history["peak_gpu_allocated_gb"], "GB", flush=True)


if __name__ == "__main__":
    main()
