from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_1.7b/original"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
REPORT = ROOT / "pretrained/qwen3_1.7b/evaluations/stop_competitors_1_7b_full384.json"

report = json.loads(REPORT.read_text(encoding="utf-8"))
x = np.load(DATA_DIR / "validation_input_ids.npy", mmap_mode="r")
y = np.load(DATA_DIR / "validation_labels.npy", mmap_mode="r")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

groups = defaultdict(list)
for row in report["per_example"]:
    token_id = row["top_competitors"][0]["token_id"]
    groups[token_id].append(row)

selected_ids = [4710, 576, 117975, 1096, 358, 1416, 1084, 116411, 6771, 4354, 271]
shown = set()

for token_id in selected_ids:
    rows = sorted(
        groups[token_id],
        key=lambda r: r["stop_rank"],
        reverse=True
    )[:2]

    for item in rows:
        index = item["example_index"]
        if index in shown:
            continue
        shown.add(index)

        stop_positions = np.flatnonzero(y[index] == 151645)
        if len(stop_positions) != 1:
            continue

        stop_pos = int(stop_positions[0])
        text = tokenizer.decode(
            x[index, :stop_pos + 1].tolist(),
            skip_special_tokens=False
        )

        competitor = item["top_competitors"][0]

        print("\n" + "=" * 90)
        print(
            f"EXAMPLE {index} | response_length={item['response_length']} "
            f"| STOP rank={item['stop_rank']}"
        )
        print(
            f"TOP COMPETITOR: {competitor['decoded']!r} "
            f"| token_id={token_id} "
            f"| probability={competitor['probability']:.6f}"
        )
        print(f"STOP probability: {item['stop_probability']:.8f}")
        print("\n--- PROMPT AND REFERENCE RESPONSE (last 1600 characters) ---")
        print(text[-1600:])

print("\nTotal distinct examples inspected:", len(shown))
