from pathlib import Path
import json
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "pretrained/qwen3_1.7b/original"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
SOURCE = ROOT / "pretrained/qwen3_1.7b/evaluations/stop_competitors_1_7b_full384.json"
OUTPUT = ROOT / "pretrained/qwen3_1.7b/evaluations/newline_competitor_examples.json"

NEWLINE_COMPETITOR_ID = 4710
STOP_ID = 151645
LIMIT = 48

def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")

    with SOURCE.open("r", encoding="utf-8") as f:
        diagnostic = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True
    )

    input_ids = np.load(
        DATA_DIR / "validation_input_ids.npy", mmap_mode="r"
    )
    labels = np.load(
        DATA_DIR / "validation_labels.npy", mmap_mode="r"
    )

    selected = []
    for row in diagnostic["per_example"]:
        if row["top_competitors"][0]["token_id"] == NEWLINE_COMPETITOR_ID:
            selected.append(row)

    print(f"Matching examples: {len(selected)}")
    if len(selected) != LIMIT:
        print(f"WARNING: Expected {LIMIT}, found {len(selected)}")

    results = []

    for item in selected:
        index = item["example_index"]
        ids = np.asarray(input_ids[index], dtype=np.int64)
        target = np.asarray(labels[index], dtype=np.int64)

        supervised = np.flatnonzero(target != -100)
        stop_positions = np.flatnonzero(target == STOP_ID)

        if len(stop_positions) != 1:
            raise ValueError(f"Example {index}: invalid STOP count")

        stop_pos = int(stop_positions[0])
        first_supervised = int(supervised[0])

        prompt_ids = ids[:first_supervised]
        response_ids = ids[first_supervised:stop_pos]
        final_ids = response_ids[-16:]

        prompt_text = tokenizer.decode(
            prompt_ids, skip_special_tokens=False
        )
        response_text = tokenizer.decode(
            response_ids, skip_special_tokens=False
        )

        entry = {
            "example_index": index,
            "response_length": item["response_length"],
            "stop_rank": item["stop_rank"],
            "stop_probability": item["stop_probability"],
            "newline_competitor_probability":
                item["top_competitors"][0]["probability"],
            "prompt": prompt_text[-1200:],
            "response": response_text[-2000:],
            "response_character_count": len(response_text),
            "final_16_token_ids": [int(v) for v in final_ids],
            "final_16_tokens": [
                tokenizer.convert_ids_to_tokens(int(v))
                for v in final_ids
            ],
            "final_16_decoded": tokenizer.decode(
                final_ids, skip_special_tokens=False
            ),
            "ends_with_newline": response_text.endswith("\n"),
            "ends_with_double_newline": response_text.endswith("\n\n"),
            "last_response_character_repr": repr(response_text[-30:]),
        }
        results.append(entry)

    report = {
        "description": "Validation examples where token 4710 (space plus double newline) is the highest-scoring competitor to STOP.",
        "competitor_token_id": NEWLINE_COMPETITOR_ID,
        "competitor_token": tokenizer.convert_ids_to_tokens(
            NEWLINE_COMPETITOR_ID
        ),
        "competitor_decoded": tokenizer.decode(
            [NEWLINE_COMPETITOR_ID], skip_special_tokens=False
        ),
        "stop_token_id": STOP_ID,
        "matching_examples": len(results),
        "examples": results,
    }

    with OUTPUT.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== NEWLINE COMPETITOR INSPECTION ===")
    for item in results:
        print(f"\n--- Example {item['example_index']} ---")
        print(f"Response length: {item['response_length']}")
        print(f"STOP rank: {item['stop_rank']}")
        print(f"STOP probability: {item['stop_probability']:.8g}")
        print(f"Ends with newline: {item['ends_with_newline']}")
        print(f"Ends with double newline: {item['ends_with_double_newline']}")
        print(f"Final text: {item['last_response_character_repr']}")
        print(f"Final tokens: {item['final_16_decoded']!r}")
        print(f"Response ending:\n{item['response'][-500:]}")

    print(f"\nReport saved: {OUTPUT}")

if __name__ == "__main__":
    main()
