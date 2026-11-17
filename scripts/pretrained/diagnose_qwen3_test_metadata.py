import sys
import gzip
import json
from pathlib import Path
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path("scripts/pretrained").resolve()))

from prepare_qwen3_oasst import (
    ROOT, MODEL_DIR, SOURCE, OUTPUT, MAX_SEQ_LEN,
    IM_END, message_is_valid, split_for_tree,
    encode_example, sha256_file
)

REPORT = ROOT / "pretrained/qwen3_0.6b/evaluations/oasst1_test_example_metadata.json"
MANIFEST = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))

if sha256_file(SOURCE) != MANIFEST["source_sha256"]:
    raise RuntimeError("Source archive SHA256 does not match manifest.")

if REPORT.exists():
    raise FileExistsError(f"Refusing to overwrite: {REPORT}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR, local_files_only=True, trust_remote_code=False
)

stats = Counter()
metadata = []
context_targets = 0
tree_count = 0

def visit(message, ancestors, ancestors_valid, tree_id):
    global context_targets, tree_count

    is_assistant = message.get("role") == "assistant"
    target_valid = (
        is_assistant
        and message_is_valid(message, assistant_target=True)
    )

    if target_valid and ancestors_valid:
        context_targets += 1

        example = encode_example(
            tokenizer, ancestors, message["text"], stats
        )

        if example is not None:
            input_ids, labels = example

            supervised = [int(x) for x in labels if x != -100]
            history = [
                {
                    "role": "user" if m.get("role") == "prompter" else m.get("role"),
                    "text": m.get("text", "").strip()
                }
                for m in ancestors
            ]

            metadata.append({
                "example": len(metadata),
                "tree_id": tree_id,
                "message_id": message.get("message_id", message.get("id")),
                "response": message["text"].strip(),
                "history": history,
                "history_messages": len(history),
                "response_characters": len(message["text"].strip()),
                "supervised_tokens": len(supervised),
                "final_token_ids": supervised[-20:],
                "final_tokens": tokenizer.convert_ids_to_tokens(supervised[-20:]),
                "final_decoded": tokenizer.decode(
                    supervised[-40:],
                    skip_special_tokens=False
                ),
                "ends_with_im_end": supervised[-1] == IM_END,
            })

    context_valid = message_is_valid(
        message, assistant_target=is_assistant
    )
    next_valid = ancestors_valid and context_valid
    next_ancestors = ancestors + [message]

    for child in message.get("replies") or []:
        visit(child, next_ancestors, next_valid, tree_id)


print("Reconstructing test examples in original traversal order...")

with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue

        tree = json.loads(line)
        tree_id = str(tree.get("message_tree_id") or "")

        if split_for_tree(tree_id) != "test":
            continue

        prompt = tree.get("prompt")
        if isinstance(prompt, dict):
            visit(prompt, [], True, tree_id)

print("Context-valid test targets:", context_targets)
print("Successfully encoded examples:", len(metadata))
print("Oversized targets skipped:", stats["oversized_targets"])

if context_targets != 339 or len(metadata) != 312:
    raise RuntimeError("Reconstructed example counts do not match manifest.")

print("\nLoading saved test arrays...")
saved_x = np.load(OUTPUT / "test_input_ids.npy", mmap_mode="r")
saved_y = np.load(OUTPUT / "test_labels.npy", mmap_mode="r")

if saved_x.shape != (312, MAX_SEQ_LEN) or saved_y.shape != saved_x.shape:
    raise RuntimeError("Unexpected saved test-array shapes.")

print("Verifying exact input IDs...")
for i, item in enumerate(metadata):
    # Re-encode once more is unnecessary: the original traversal's
    # encoded arrays are checked below using a deterministic rebuild.
    pass

# Deterministically rebuild the test arrays for exact comparison.
# Re-run the same traversal while retaining encoded arrays.
rebuilt_x = []
rebuilt_y = []

def rebuild_visit(message, ancestors, ancestors_valid):
    is_assistant = message.get("role") == "assistant"
    target_valid = is_assistant and message_is_valid(
        message, assistant_target=True
    )

    if target_valid and ancestors_valid:
        encoded = encode_example(
            tokenizer, ancestors, message["text"], Counter()
        )
        if encoded is not None:
            rebuilt_x.append(encoded[0])
            rebuilt_y.append(encoded[1])

    valid = message_is_valid(message, assistant_target=is_assistant)
    next_ancestors = ancestors + [message]

    for child in message.get("replies") or []:
        rebuild_visit(child, next_ancestors, ancestors_valid and valid)

with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        tree = json.loads(line)
        if split_for_tree(str(tree.get("message_tree_id") or "")) == "test":
            prompt = tree.get("prompt")
            if isinstance(prompt, dict):
                rebuild_visit(prompt, [], True)

rebuilt_x = np.asarray(rebuilt_x, dtype=np.int32)
rebuilt_y = np.asarray(rebuilt_y, dtype=np.int32)

if not np.array_equal(rebuilt_x, saved_x):
    raise RuntimeError("INPUT IDS MISMATCH: refusing metadata export.")

print("INPUT IDS: EXACT MATCH")

if not np.array_equal(rebuilt_y, saved_y):
    raise RuntimeError("LABELS MISMATCH: refusing metadata export.")

print("LABELS: EXACT MATCH")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(
    json.dumps({
        "source_sha256": MANIFEST["source_sha256"],
        "examples": metadata
    }, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nSUCCESS: Exact mapping verified.")
print("Metadata examples:", len(metadata))
print("Saved:", REPORT)
