from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_0.6b/original"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
ARTIFACT_DIR = ROOT / "pretrained/qwen3_0.6b"

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
EPOCHS = 3
SEED = 42


def evaluate(model, inputs, labels, pad_id, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

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
            token_count = int(y.ne(-100).sum().item())

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=x,
                    attention_mask=mask,
                    labels=y,
                    use_cache=False,
                )

            if not torch.isfinite(output.loss):
                raise RuntimeError(f"Non-finite evaluation loss at sample {i}")

            total_loss += output.loss.item() * token_count
            total_tokens += token_count

            del x, y, mask, output

    model.train()
    return total_loss / max(total_tokens, 1)


def save_adapter(model, tokenizer, directory):
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="oasst1_lora_v1")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise ValueError("Invalid run name.")

    if args.max_steps < 0 or args.epochs < 1:
        raise ValueError("Invalid training limits.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU BF16 support is required.")

    run_dir = ARTIFACT_DIR / "runs" / args.run_name
    adapter_dir = ARTIFACT_DIR / "adapters" / args.run_name

    if run_dir.exists() or adapter_dir.exists():
        raise FileExistsError(
            f"Run artifacts already exist: {args.run_name}. "
            "Choose a new --run-name."
        )

    manifest = json.loads(
        (DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    pad_id = manifest["tokenizer_pad_token_id"]

    arrays = {}
    for split in ("train", "validation"):
        arrays[split] = (
            np.load(DATA_DIR / f"{split}_input_ids.npy", mmap_mode="r"),
            np.load(DATA_DIR / f"{split}_labels.npy", mmap_mode="r"),
        )

    train_x, train_y = arrays["train"]
    val_x, val_y = arrays["validation"]

    if train_x.shape != train_y.shape:
        raise ValueError("Training input/label shape mismatch.")

    if val_x.shape != val_y.shape:
        raise ValueError("Validation input/label shape mismatch.")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda")
    print("=== QWEN3 LORA TRAINING ===", flush=True)
    print("Run:", args.run_name, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Training examples:", len(train_x), flush=True)
    print("Validation examples:", len(val_x), flush=True)
    print("Sequence length:", train_x.shape[1], flush=True)
    print("Precision: BF16", flush=True)
    print("LoRA rank:", RANK, flush=True)

    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=False,
    )

    if tokenizer.pad_token_id != pad_id:
        raise ValueError("Tokenizer padding ID does not match dataset.")

    print("Loading frozen Qwen3 base...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    ).to(device)

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]

    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found.")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    updates_per_epoch = math.ceil(len(train_x) / ACCUMULATION)
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
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_factor
    )

    print("\n=== INITIAL VALIDATION ===", flush=True)
    initial_val = evaluate(
        model, val_x, val_y, pad_id, device
    )
    print(f"Initial validation loss: {initial_val:.6f}", flush=True)

    run_dir.mkdir(parents=True, exist_ok=False)
    adapter_dir.mkdir(parents=True, exist_ok=False)

    history = {
        "run_name": args.run_name,
        "base_model": manifest["base_model"],
        "base_revision": manifest["base_model_revision"],
        "dataset": "OpenAssistant OASST1",
        "lora_rank": RANK,
        "lora_alpha": ALPHA,
        "lora_dropout": DROPOUT,
        "target_modules": TARGET_MODULES,
        "precision": "bfloat16",
        "sequence_length": 512,
        "micro_batch_size": 1,
        "gradient_accumulation": ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "epochs_requested": args.epochs,
        "max_steps": args.max_steps,
        "initial_validation_loss": initial_val,
        "epochs": [],
    }

    best_val = float("inf")
    global_step = 0
    start_time = time.time()
    stop = False

    print("\n=== TRAINING STARTED ===", flush=True)

    for epoch in range(args.epochs):
        model.train()
        permutation = np.random.permutation(len(train_x))
        epoch_loss_sum = 0.0
        epoch_tokens = 0

        for group_start in range(0, len(permutation), ACCUMULATION):
            group = permutation[
                group_start:group_start + ACCUMULATION
            ]

            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break

            token_counts = [
                int(np.count_nonzero(train_y[int(i)] != -100))
                for i in group
            ]
            group_tokens = sum(token_counts)

            optimizer.zero_grad(set_to_none=True)

            for index, count in zip(group, token_counts):
                i = int(index)

                x = torch.as_tensor(
                    np.array(train_x[i], dtype=np.int64, copy=True),
                    device=device,
                ).unsqueeze(0)

                y = torch.as_tensor(
                    np.array(train_y[i], dtype=np.int64, copy=True),
                    device=device,
                ).unsqueeze(0)

                positions_idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
                last_valid = positions_idx.masked_fill(y.eq(-100), -1).max(dim=1).values
                if torch.any(last_valid < 0):
                    raise RuntimeError("Batch contains a row without supervised tokens.")
                mask = (positions_idx <= last_valid.unsqueeze(1)).long()

                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model(
                            input_ids=x,
                            attention_mask=mask,
                            labels=y,
                            use_cache=False,
                        )

                    loss = output.loss

                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite training loss at step {global_step + 1}"
                        )

                    weighted_loss = loss * (count / group_tokens)
                    weighted_loss.backward()

                    epoch_loss_sum += loss.item() * count
                    epoch_tokens += count

                    del x, y, mask, output, loss, weighted_loss

                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA OOM during training. No existing base "
                        "model or checkpoints were modified. Reduce "
                        "sequence length or use a smaller LoRA setup."
                    )

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, MAX_GRAD_NORM
            )

            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite gradient norm.")

            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 25 == 0 or global_step == 1:
                print(
                    f"epoch={epoch + 1} step={global_step} "
                    f"train_loss={epoch_loss_sum / max(epoch_tokens, 1):.5f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"gpu_allocated={torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                    flush=True,
                )

        if epoch_tokens:
            epoch_train_loss = epoch_loss_sum / epoch_tokens
        else:
            epoch_train_loss = float("nan")

        val_loss = evaluate(model, val_x, val_y, pad_id, device)

        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": epoch_train_loss,
            "validation_loss": val_loss,
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        history["epochs"].append(record)

        print(
            f"\nEPOCH {epoch + 1}: "
            f"train_loss={epoch_train_loss:.6f} "
            f"validation_loss={val_loss:.6f}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            save_adapter(model, tokenizer, adapter_dir)
            history["best_validation_loss"] = best_val
            print("Saved new best adapter.", flush=True)

        (run_dir / "training_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        if stop:
            break

        if args.max_steps and global_step >= args.max_steps:
            break

    final_dir = run_dir / "final_adapter"
    save_adapter(model, tokenizer, final_dir)

    history["final_global_step"] = global_step
    history["final_validation_loss"] = history["epochs"][-1]["validation_loss"]
    history["best_validation_loss"] = best_val
    history["elapsed_seconds"] = round(time.time() - start_time, 2)

    (run_dir / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print("\n=== TRAINING COMPLETE ===", flush=True)
    print("Steps:", global_step, flush=True)
    print("Initial validation loss:", initial_val, flush=True)
    print("Best validation loss:", best_val, flush=True)
    print("Best adapter:", adapter_dir, flush=True)
    print("Final adapter:", final_dir, flush=True)
    print(
        "Peak GPU memory:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "GB",
        flush=True,
    )


if __name__ == "__main__":
    main()
