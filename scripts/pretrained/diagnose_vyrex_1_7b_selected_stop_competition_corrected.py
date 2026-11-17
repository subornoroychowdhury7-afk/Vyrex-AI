from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_1.7b/original"
ADAPTER_DIR = ROOT / "pretrained/qwen3_1.7b/adapters/vyrex_1_7b_qlora_full_v1/epoch_01_best"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
OUTPUT = ROOT / "pretrained/qwen3_1.7b/evaluations/selected_stop_competition_corrected_v2_full384.json"

STOP_ID = 151645
SELECTED = [242, 47, 62, 103, 165, 216]

def token_info(tokenizer, token_id):
    return {
        "id": int(token_id),
        "text": repr(tokenizer.decode([int(token_id)])),
    }

def get_rank(logits, token_id):
    return int((logits > logits[token_id]).sum().item() + 1)

def inspect_prediction(tokenizer, logits, target_id):
    probabilities = torch.softmax(logits, dim=-1)
    top_values, top_ids = torch.topk(logits, 5)

    return {
        "target": token_info(tokenizer, target_id),
        "target_probability": float(probabilities[target_id].item()),
        "target_rank": get_rank(logits, target_id),
        "top5": [
            {
                **token_info(tokenizer, token_id),
                "probability": float(probabilities[token_id].item()),
            }
            for token_id in top_ids.tolist()
        ],
    }

def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")

    data_dir = DATA_DIR
    input_ids = np.load(data_dir / "validation_input_ids.npy", mmap_mode="r")
    labels = np.load(data_dir / "validation_labels.npy", mmap_mode="r")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading NF4 base...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )

    print("Loading full384 adapter...", flush=True)

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_DIR,
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()

    results = []

    for index in SELECTED:
        x = torch.tensor(
            np.array(input_ids[index], dtype=np.int64, copy=True),
            device="cuda",
        ).unsqueeze(0)

        y = torch.tensor(
            np.array(labels[index], dtype=np.int64, copy=True),
            device="cuda",
        ).unsqueeze(0)

        stop_positions = torch.where(y[0] == STOP_ID)[0]

        if len(stop_positions) != 1:
            raise RuntimeError(f"Invalid STOP count: {index}")

        stop_position = int(stop_positions.item())

        if stop_position < 2:
            raise RuntimeError(f"Insufficient response context: {index}")

        punctuation_position = stop_position - 2
        stop_prediction_position = stop_position - 1

        final_token_id = int(y[0, stop_position - 1].item())

        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            output = model(
                input_ids=x,
                attention_mask=(x != 151643).long(),
                use_cache=False,
            )

        punctuation_logits = output.logits[
            0, punctuation_position
        ].float()

        stop_logits = output.logits[
            0, stop_prediction_position
        ].float()

        punctuation_result = inspect_prediction(
            tokenizer,
            punctuation_logits,
            final_token_id,
        )

        stop_result = inspect_prediction(
            tokenizer,
            stop_logits,
            STOP_ID,
        )

        result = {
            "example_index": index,
            "final_token_prediction": punctuation_result,
            "stop_prediction": stop_result,
        }

        results.append(result)

        print(f"\n{'=' * 55}")
        print("Example:", index)

        print("\nFINAL TOKEN PREDICTION")
        print("Target:", punctuation_result["target"])
        print("Probability:", punctuation_result["target_probability"])
        print("Rank:", punctuation_result["target_rank"])
        print("Top five:")
        for item in punctuation_result["top5"]:
            print(item["id"], item["text"], f'p={item["probability"]:.6g}')

        print("\nSTOP PREDICTION")
        print("Probability:", stop_result["target_probability"])
        print("Rank:", stop_result["target_rank"])
        print("Top five:")
        for item in stop_result["top5"]:
            print(item["id"], item["text"], f'p={item["probability"]:.6g}')

        del x, y, output, punctuation_logits, stop_logits
        torch.cuda.empty_cache()

    OUTPUT.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nReport:", OUTPUT)

if __name__ == "__main__":
    main()

