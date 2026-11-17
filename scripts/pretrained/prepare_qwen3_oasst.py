from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_0.6b/original"
SOURCE = ROOT / "data/raw/oasst1/oasst_ready.trees.jsonl.gz"
LICENSE = ROOT / "data/raw/oasst1/LICENSE"
OUTPUT = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"

SEED = 42
MAX_SEQ_LEN = 512
EXPECTED_TREES = 10364
EXPECTED_TARGETS = 7371

BLOCKED_FLAGS = (
    "spam",
    "lang_mismatch",
    "pii",
    "not_appropriate",
    "hate_speech",
    "sexual_content",
)

IM_START = 151644
IM_END = 151645
IGNORE_INDEX = -100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labels_are_safe(message: dict) -> bool:
    labels = message.get("labels") or {}

    for name in BLOCKED_FLAGS:
        entry = labels.get(name)
        if isinstance(entry, dict):
            value = entry.get("value")
            if isinstance(value, bool):
                if value:
                    return False
            elif isinstance(value, (int, float)):
                if not math.isfinite(value) or value > 0:
                    return False

    return True


def message_is_valid(message: dict, *, assistant_target: bool) -> bool:
    if message.get("lang") != "en":
        return False
    if message.get("deleted") is not False:
        return False
    if message.get("review_result") is not True:
        return False
    if message.get("synthetic") is True:
        return False
    if not labels_are_safe(message):
        return False

    if assistant_target:
        quality = (message.get("labels") or {}).get("quality")
        if not isinstance(quality, dict):
            return False

        score = quality.get("value")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return False
        if not math.isfinite(score) or score < 0.75:
            return False

    return isinstance(message.get("text"), str) and bool(
        message["text"].strip()
    )


def split_for_tree(tree_id: str) -> str:
    key = f"{SEED}:{tree_id}".encode("utf-8")
    bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) % 100

    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def to_chat_message(message: dict) -> dict:
    role = message.get("role")

    if role == "prompter":
        role = "user"
    elif role == "assistant":
        role = "assistant"
    else:
        raise ValueError(f"Unexpected OASST role: {role!r}")

    return {
        "role": role,
        "content": message["text"].strip(),
    }


def encode_example(tokenizer, history: list[dict], response: str, stats):
    # OASST history ends with the user message being answered.
    messages = [to_chat_message(m) for m in history]
    messages.append({"role": "assistant", "content": response.strip()})

    if not messages or messages[-2]["role"] != "user":
        stats["invalid_history"] += 1
        return None

    while True:
        history_messages = messages[:-1]

        prefix_text = tokenizer.apply_chat_template(
            history_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Qwen3's default thinking-enabled template inserts this
        # immediately after the final assistant generation prefix.
        prefix_text += "<think>\n\n</think>\n\n"

        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        if not full_text.startswith(prefix_text):
            raise RuntimeError(
                "Qwen3 chat-template prefix alignment failed"
            )

        prefix_ids = tokenizer.encode(
            prefix_text,
            add_special_tokens=False,
        )
        full_ids = tokenizer.encode(
            full_text,
            add_special_tokens=False,
        )

        if full_ids[:len(prefix_ids)] != prefix_ids:
            raise RuntimeError(
                "Tokenized assistant boundary does not match"
            )

        if len(full_ids) <= MAX_SEQ_LEN:
            break

        # Remove the oldest user/assistant pair, preserving the
        # latest user instruction and the complete target response.
        if len(messages) > 2:
            remove_count = 2
            messages = messages[remove_count:]
            stats["trimmed_history_pairs"] += 1
            continue

        # Never silently truncate an assistant response.
        stats["oversized_targets"] += 1
        return None

    supervised = full_ids[len(prefix_ids):]

    if not supervised or IM_END not in supervised:
        raise RuntimeError("Assistant target or message ending missing")

    if full_ids[0] != IM_START:
        raise RuntimeError("Unexpected Qwen3 conversation format")

    input_ids = full_ids + [tokenizer.pad_token_id] * (
        MAX_SEQ_LEN - len(full_ids)
    )
    labels = (
        [IGNORE_INDEX] * len(prefix_ids)
        + supervised
        + [IGNORE_INDEX] * (MAX_SEQ_LEN - len(full_ids))
    )

    if len(input_ids) != MAX_SEQ_LEN or len(labels) != MAX_SEQ_LEN:
        raise RuntimeError("Incorrect stored sequence length")

    if not any(x != IGNORE_INDEX for x in labels):
        raise RuntimeError("Example contains no supervised tokens")

    stats["supervised_tokens"] += len(supervised)
    stats["maximum_length"] = max(
        stats["maximum_length"], len(full_ids)
    )

    return input_ids, labels


def main():
    for path in (MODEL_DIR, SOURCE, LICENSE):
        if not path.exists():
            raise FileNotFoundError(path)

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: {OUTPUT}"
        )

    print("Loading local Qwen3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=False,
    )

    if tokenizer.pad_token_id != 151643:
        raise RuntimeError(
            f"Unexpected Qwen3 PAD ID: {tokenizer.pad_token_id}"
        )

    if tokenizer.convert_tokens_to_ids("<|im_start|>") != IM_START:
        raise RuntimeError("Unexpected Qwen3 im_start token ID")

    if tokenizer.convert_tokens_to_ids("<|im_end|>") != IM_END:
        raise RuntimeError("Unexpected Qwen3 im_end token ID")

    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    tree_counts = Counter()
    target_counts = Counter()
    context_counts = Counter()
    role_counts = Counter()

    stats = Counter()
    total_trees = 0

    def visit(message, ancestors, ancestors_valid, split_name):
        role = message.get("role")
        role_counts[str(role)] += 1

        is_assistant = role == "assistant"

        target_valid = (
            is_assistant
            and message_is_valid(message, assistant_target=True)
        )

        if target_valid:
            target_counts[split_name] += 1

            if ancestors_valid:
                context_counts[split_name] += 1

                example = encode_example(
                    tokenizer,
                    ancestors,
                    message["text"],
                    stats,
                )

                if example is not None:
                    splits[split_name].append(example)

        context_valid = message_is_valid(
            message,
            assistant_target=is_assistant,
        )

        next_valid = ancestors_valid and context_valid
        next_ancestors = ancestors + [message]

        for child in message.get("replies") or []:
            visit(
                child,
                next_ancestors,
                next_valid,
                split_name,
            )

    print("Reading OASST1 and building Qwen3 examples...")

    with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            tree = json.loads(line)
            total_trees += 1

            tree_id = str(tree.get("message_tree_id") or "")
            if not tree_id:
                raise RuntimeError("Missing message_tree_id")

            split_name = split_for_tree(tree_id)
            before = len(splits[split_name])

            prompt = tree.get("prompt")
            if not isinstance(prompt, dict):
                continue

            visit(prompt, [], True, split_name)

            if len(splits[split_name]) > before:
                tree_counts[split_name] += 1

    valid_targets = sum(context_counts.values())

    print("\n=== OASST1 QUALITY AUDIT ===")
    print("Source trees:", total_trees)
    print("Role counts:", dict(role_counts))
    print("Quality-filtered targets:", sum(target_counts.values()))
    print("Valid complete-context targets:", valid_targets)

    for name in splits:
        print(
            f"{name}: trees={tree_counts[name]}, "
            f"valid_targets={context_counts[name]}, "
            f"encoded_examples={len(splits[name])}"
        )

    print("\n=== QWEN3 TOKENIZATION AUDIT ===")
    print("Oversized targets skipped:", stats["oversized_targets"])
    print("Invalid histories:", stats["invalid_history"])
    print("History trimming operations:", stats["trimmed_history_pairs"])
    print("Supervised tokens:", stats["supervised_tokens"])
    print("Maximum sequence length:", stats["maximum_length"])

    if total_trees != EXPECTED_TREES:
        raise RuntimeError(f"Expected {EXPECTED_TREES} trees")

    if valid_targets != EXPECTED_TARGETS:
        raise RuntimeError(
            f"Expected {EXPECTED_TARGETS} valid targets, got {valid_targets}"
        )

    expected_splits = {
        "train": 6660,
        "validation": 372,
        "test": 339,
    }

    for name, expected in expected_splits.items():
        if context_counts[name] != expected:
            raise RuntimeError(
                f"{name} target count mismatch: "
                f"{context_counts[name]} != {expected}"
            )

    if not all(splits[name] for name in splits):
        raise RuntimeError("A dataset split is empty")

    # Save only after all source and filtering audits pass.
    OUTPUT.mkdir(parents=True)

    for name, examples in splits.items():
        input_array = np.asarray(
            [item[0] for item in examples],
            dtype=np.int32,
        )
        label_array = np.asarray(
            [item[1] for item in examples],
            dtype=np.int32,
        )

        np.save(OUTPUT / f"{name}_input_ids.npy", input_array)
        np.save(OUTPUT / f"{name}_labels.npy", label_array)

    manifest = {
        "dataset": "OpenAssistant OASST1",
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "base_model_revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "base_model_license": "Apache-2.0",
        "source_sha256": sha256_file(SOURCE),
        "license_sha256": sha256_file(LICENSE),
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "chat_template": "Qwen3 default thinking-enabled template",
        "seed": SEED,
        "split_method": "SHA256(seed:message_tree_id) modulo 100",
        "max_sequence_length": MAX_SEQ_LEN,
        "padding": "right",
        "padding_label": IGNORE_INDEX,
        "loss_masking": "assistant final message only",
        "assistant_end_token_id": IM_END,
        "filters": {
            "language": "en",
            "deleted": False,
            "review_result": True,
            "synthetic": False,
            "assistant_quality_minimum": 0.75,
            "blocked_positive_flags": list(BLOCKED_FLAGS),
            "complete_valid_ancestor_context_required": True,
        },
        "counts": {
            "source_trees": total_trees,
            "quality_filtered_targets": sum(target_counts.values()),
            "valid_context_targets": valid_targets,
            "splits": {
                name: {
                    "trees": tree_counts[name],
                    "valid_targets": context_counts[name],
                    "encoded_examples": len(splits[name]),
                }
                for name in splits
            },
        },
        "tokenization_stats": dict(stats),
        "array_format": "NumPy int32; ignored labels are -100",
        "source_file": SOURCE.name,
    }

    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n=== DATASET CREATED ===")
    print("Location:", OUTPUT)
    print("Manifest:", manifest_path)
    print("Tokenizer vocabulary:", len(tokenizer))
    print("Maximum sequence length:", MAX_SEQ_LEN)


if __name__ == "__main__":
    main()
