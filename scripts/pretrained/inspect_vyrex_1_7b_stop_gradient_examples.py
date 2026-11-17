import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path.cwd()
DATA = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
MODEL = ROOT / "pretrained/qwen3_1.7b/original"
REPORT = ROOT / "pretrained/qwen3_1.7b/evaluations/stop_gradient_analysis_1_7b_full384.json"

SELECTED = [242, 47, 62, 103, 165, 216]

x = np.load(DATA / "validation_input_ids.npy", mmap_mode="r")
y = np.load(DATA / "validation_labels.npy", mmap_mode="r")
report = json.loads(REPORT.read_text(encoding="utf-8"))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

print("=== SELECTED VALIDATION EXAMPLES ===")

for index in SELECTED:
    row = x[index]
    labels = y[index]

    stop_positions = np.where(labels == 151645)[0]

    if len(stop_positions) != 1:
        print(f"\nExample {index}: INVALID STOP COUNT")
        continue

    stop_pos = int(stop_positions[0])
    response_ids = row[labels != -100]

    response_text = tokenizer.decode(
        response_ids[:-1].tolist(),
        skip_special_tokens=False,
    )

    item = next(
        v for v in report["gradient_analysis"]["per_example"]
        if v["example_index"] == index
    )

    print("\n" + "=" * 75)
    print("EXAMPLE:", index)
    print("Ordinary tokens:", item["ordinary_tokens"])
    print("STOP/ordinary gradient ratio:",
          round(item["stop_to_ordinary_gradient_norm_ratio"], 5))
    print("Gradient cosine:",
          round(item["gradient_cosine_similarity"], 5))
    print("STOP gradient fraction:",
          round(item["stop_gradient_fraction_of_component_norms"], 5))
    print("\nRESPONSE ENDING:")
    print(repr(response_text[-1000:]))

