from pathlib import Path
import json
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "pretrained/qwen3_1.7b/original"
DATA_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
ADAPTER_DIR = ROOT / "pretrained/qwen3_1.7b/adapters/vyrex_1_7b_qlora_full_v1/epoch_01_best"

OUTPUT = ROOT / "pretrained/qwen3_1.7b/evaluations/selected_stop_competition_full384.json"

SELECTED = [242, 47, 62, 103, 165, 216]
STOP_ID = 151645

def attention_mask(labels):
    positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    last_valid = positions.masked_fill(labels.eq(-100), -1).max(dim=1).values
    return (positions <= last_valid.unsqueeze(1)).long()

def token_info(tokenizer, token_id):
    return {
        "id": int(token_id),
        "text": repr(tokenizer.decode([int(token_id)])),
    }

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    inputs = np.load(DATA_DIR / "validation_input_ids.npy", mmap_mode="r")
    labels = np.load(DATA_DIR / "validation_labels.npy", mmap_mode="r")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True
    )

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading NF4 base...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        quantization_config=quant,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(
        base, ADAPTER_DIR, is_trainable=False
    )
    model.eval()

    results = []

    for index in SELECTED:
        x = torch.as_tensor(
            np.array(inputs[index], dtype=np.int64, copy=True),
            device="cuda"
        ).unsqueeze(0)

        y = torch.as_tensor(
            np.array(labels[index], dtype=np.int64, copy=True),
            device="cuda"
        ).unsqueeze(0)

        stop_positions = torch.where(y[0] == STOP_ID)[0]

        if len(stop_positions) != 1:
            raise RuntimeError(f"Invalid STOP count: {index}")

        stop_position = int(stop_positions.item())
        prediction_position = stop_position - 1

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=x,
                attention_mask=attention_mask(y),
                use_cache=False,
            )

        logits = output.logits[0, prediction_position].float()
        probabilities = torch.softmax(logits, dim=-1)

        stop_probability = probabilities[STOP_ID].item()
        stop_rank = int(
            (logits > logits[STOP_ID]).sum().item() + 1
        )

        final_token_id = int(y[0, stop_position - 1].item())

        top_values, top_ids = torch.topk(logits, 5)

        competitors = []

        for token_id, value in zip(top_ids.tolist(), top_values.tolist()):
            competitors.append({
                **token_info(tokenizer, token_id),
                "probability": float(probabilities[token_id].item()),
                "logit": float(value),
            })

        final_probability = probabilities[final_token_id].item()
        final_rank = int(
            (logits > logits[final_token_id]).sum().item() + 1
        )

        result = {
            "example_index": index,
            "stop_probability": stop_probability,
            "stop_rank": stop_rank,
            "final_token": token_info(tokenizer, final_token_id),
            "final_token_probability": final_probability,
            "final_token_rank": final_rank,
            "stop_to_final_probability_ratio": (
                stop_probability / max(final_probability, 1e-30)
            ),
            "top5_competitors": competitors,
        }

        results.append(result)

        print(f"\nExample {index}")
        print("STOP probability:", stop_probability)
        print("STOP rank:", stop_rank)
        print("Final token:", result["final_token"])
        print("Final-token probability:", final_probability)
        print("Final-token rank:", final_rank)
        print("Top competitors:")

        for item in competitors:
            print(
                item["id"],
                item["text"],
                f"p={item['probability']:.6g}"
            )

        del x, y, output, logits, probabilities
        torch.cuda.empty_cache()

    OUTPUT.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    print("\nReport:", OUTPUT)

if __name__ == "__main__":
    main()
