from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from vyrex_ai.tokenizer import VyrexTokenizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/oasst1/oasst_ready.trees.jsonl.gz"
LICENSE = ROOT / "data/raw/oasst1/LICENSE"
TOKENIZER_PATH = ROOT / "artifacts/tokenizer/oanc_production_v1/tokenizer.json"
OUTPUT = ROOT / "data/processed/oasst1_sft_v2"

SEED = 42
MAX_SEQ_LEN = 128
TARGET_CHUNK_SIZE = 48
EXPECTED_VALID_TARGETS = 7371

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

BLOCKED_FLAGS = (
    "spam",
    "lang_mismatch",
    "pii",
    "not_appropriate",
    "hate_speech",
    "sexual_content",
)


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
                continue
            if isinstance(value, (int, float)):
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

    return isinstance(message.get("text"), str) and bool(message["text"].strip())


def split_for_tree(tree_id: str) -> str:
    key = f"{SEED}:{tree_id}".encode("utf-8")
    bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def make_examples(
    tokenizer: VyrexTokenizer,
    history: list[dict],
    response: str,
) -> list[tuple[list[int], list[int]]]:
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if not response_ids:
        return []

    # Find and preserve the latest user instruction in every chunk.
    latest_user_index = next(
        (
            i for i in range(len(history) - 1, -1, -1)
            if history[i].get("role") == "prompter"
            and history[i].get("text", "").strip()
        ),
        None,
    )
    if latest_user_index is None:
        return []

    latest_user = history[latest_user_index]["text"].strip()
    user_ids = tokenizer.encode(
        f"User: {latest_user}\\n",
        add_special_tokens=False,
    )

    # Older conversation context, excluding the latest user message.
    older_parts = []
    for message in history[:latest_user_index]:
        text = message.get("text", "").strip()
        if not text:
            continue
        role = message.get("role")
        if role == "prompter":
            older_parts.append(f"User: {text}\\n")
        elif role == "assistant":
            older_parts.append(f"Assistant: {text}\\n")

    older_ids = tokenizer.encode(
        "".join(older_parts),
        add_special_tokens=False,
    )

    examples = []

    for start in range(0, len(response_ids), TARGET_CHUNK_SIZE):
        target = response_ids[start:start + TARGET_CHUNK_SIZE]
        is_final = start + len(target) == len(response_ids)

        eos_count = 1 if is_final else 0
        context_budget = MAX_SEQ_LEN - len(target) - eos_count - 1

        if context_budget < 0:
            raise RuntimeError("Target chunk exceeds sequence capacity")

        # Reserve up to half the context for the latest user instruction.
        user_budget = min(len(user_ids), max(1, context_budget // 2))
        kept_user = user_ids[:user_budget]

        remaining = context_budget - len(kept_user)

        # Continuations retain the most recent generated assistant tokens.
        previous_response = response_ids[:start]
        kept_previous = previous_response[-remaining:] if remaining else []
        remaining -= len(kept_previous)

        # Use any remaining capacity for older conversation history.
        kept_older = older_ids[-remaining:] if remaining else []

        context = kept_older + kept_user + kept_previous

        input_ids = [BOS_ID] + context + target
        labels = [-100] * len(input_ids)

        target_start = 1 + len(context)
        labels[target_start:target_start + len(target)] = target

        if is_final:
            input_ids.append(EOS_ID)
            labels.append(EOS_ID)

        if len(input_ids) > MAX_SEQ_LEN + 1:
            raise RuntimeError("Sequence exceeds model input capacity")

        padding = MAX_SEQ_LEN + 1 - len(input_ids)
        input_ids.extend([PAD_ID] * padding)
        labels.extend([-100] * padding)

        examples.append((input_ids, labels))

    return examples


def main() -> None:
    for path in (SOURCE, LICENSE, TOKENIZER_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {OUTPUT}"
        )

    tokenizer = VyrexTokenizer.load(TOKENIZER_PATH)

    for token, expected in (
        ("<|pad|>", PAD_ID),
        ("<|bos|>", BOS_ID),
        ("<|eos|>", EOS_ID),
    ):
        actual = tokenizer.encode(token, add_special_tokens=False)
        if actual != [expected]:
            raise RuntimeError(f"Unexpected {token} ID: {actual}")

    splits: dict[str, list[tuple[list[int], list[int]]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    tree_counts = Counter()
    target_counts = Counter()
    context_counts = Counter()
    message_counts = Counter()
    role_counts = Counter()

    def visit(
        message: dict,
        ancestors: list[dict],
        ancestors_valid: bool,
        split_name: str,
    ) -> None:
        role = message.get("role")
        role_counts[role] += 1

        is_assistant = role == "assistant"
        target_valid = is_assistant and message_is_valid(
            message,
            assistant_target=True,
        )

        if target_valid:
            target_counts[split_name] += 1

            if ancestors_valid:
                context_counts[split_name] += 1
                examples = make_examples(
                    tokenizer,
                    ancestors,
                    message["text"],
                )
                splits[split_name].extend(examples)

        context_valid = message_is_valid(
            message,
            assistant_target=is_assistant,
        )
        next_valid = ancestors_valid and context_valid

        next_ancestors = ancestors + [message]

        for child in message.get("replies") or []:
            visit(child, next_ancestors, next_valid, split_name)

    print("Reading OASST1 trees and constructing assistant-only examples...")

    total_trees = 0

    with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            tree = json.loads(line)
            total_trees += 1

            tree_id = str(tree.get("message_tree_id") or "")
            if not tree_id:
                raise RuntimeError("Conversation tree has no message_tree_id")

            split_name = split_for_tree(tree_id)
            before = len(splits[split_name])

            prompt = tree.get("prompt")
            if not isinstance(prompt, dict):
                continue

            visit(
                prompt,
                [],
                True,
                split_name,
            )

            if len(splits[split_name]) > before:
                tree_counts[split_name] += 1

    total_valid_targets = sum(context_counts.values())

    print("\n=== DATASET AUDIT ===")
    print(f"Source trees: {total_trees}")
    print(f"Role counts: {dict(role_counts)}")
    print(f"Quality-filtered assistant targets: {sum(target_counts.values())}")
    print(f"Valid complete-context targets: {total_valid_targets}")

    for name in ("train", "validation", "test"):
        print(
            f"{name}: trees={tree_counts[name]}, "
            f"valid_targets={context_counts[name]}, "
            f"training_chunks={len(splits[name])}"
        )

    if total_trees != 10364:
        raise RuntimeError(f"Unexpected tree count: {total_trees}")

    if total_valid_targets != EXPECTED_VALID_TARGETS:
        raise RuntimeError(
            "Audit mismatch: expected "
            f"{EXPECTED_VALID_TARGETS} valid targets, got {total_valid_targets}. "
            "No dataset files have been written."
        )

    if not all(splits[name] for name in splits):
        raise RuntimeError("At least one split is empty")

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
        "source_file": SOURCE.name,
        "source_sha256": sha256_file(SOURCE),
        "license_file": LICENSE.name,
        "license_sha256": sha256_file(LICENSE),
        "license": "Apache-2.0; verify source distribution terms",
        "tokenizer": str(TOKENIZER_PATH.relative_to(ROOT)),
        "tokenizer_sha256": sha256_file(TOKENIZER_PATH),
        "seed": SEED,
        "split_method": "SHA256(seed:message_tree_id) modulo 100",
        "split_ratios": {
            "train": 0.90,
            "validation": 0.05,
            "test": 0.05,
        },
        "max_seq_len": MAX_SEQ_LEN,
        "stored_sequence_length": MAX_SEQ_LEN + 1,
        "target_chunk_size": TARGET_CHUNK_SIZE,
        "special_token_ids": {
            "pad": PAD_ID,
            "bos": BOS_ID,
            "eos": EOS_ID,
        },
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
            "quality_filtered_assistant_targets": sum(target_counts.values()),
            "valid_context_targets": total_valid_targets,
            "splits": {
                name: {
                    "trees": tree_counts[name],
                    "valid_targets": context_counts[name],
                    "training_chunks": len(splits[name]),
                }
                for name in splits
            },
        },
        "array_format": "NumPy int32; labels use -100 for ignored positions",
    }

    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"\nSUCCESS: dataset written to {OUTPUT}")


if __name__ == "__main__":
    main()
