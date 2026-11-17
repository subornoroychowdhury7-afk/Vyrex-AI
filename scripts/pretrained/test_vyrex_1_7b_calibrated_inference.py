from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
)
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_1.7b/original"
ADAPTER_DIR = ROOT / "pretrained/qwen3_1.7b/adapters/vyrex_1_7b_qlora_full_v1/epoch_01_best"

STOP_ID = 151645
EOS_ID = 151643


class CalibratedStopLogitsProcessor(LogitsProcessor):
    def __init__(self, stop_token_id: int, prompt_length: int, tokenizer, boundary_bias: float = 4.5, min_new_tokens: int = 6):
        self.stop_token_id = stop_token_id
        self.prompt_length = prompt_length
        self.min_new_tokens = min_new_tokens
        self.boundary_bias = boundary_bias
        
        self.boundary_ids = set()
        for s in [".", ".\n", "!", "?", "\n", "\n\n", "```", "```\n"]:
            toks = tokenizer.encode(s, add_special_tokens=False)
            if toks:
                self.boundary_ids.add(toks[-1])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        new_tokens = input_ids.shape[1] - self.prompt_length
        if new_tokens < self.min_new_tokens:
            scores[:, self.stop_token_id] = -float("inf")
            return scores

        last_id = int(input_ids[0, -1].item())
        if last_id in self.boundary_ids:
            scores[:, self.stop_token_id] += self.boundary_bias
        return scores


def main():
    print("=== VYREX AI | PATH A CALIBRATED INFERENCE TEST ===", flush=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=False)
    model.eval()

    prompts = [
        "Say hello in one clear sentence.",
        "What is artificial intelligence? Answer in two short sentences.",
        "Write a python function to add two numbers.",
    ]

    for p in prompts:
        print("\n" + "=" * 65)
        print("PROMPT:", p)
        print("=" * 65)

        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        gen_config_base = GenerationConfig(
            max_new_tokens=96,
            do_sample=False,
            eos_token_id=[STOP_ID, EOS_ID],
            pad_token_id=EOS_ID,
            use_cache=True,
        )

        # 1. Uncalibrated baseline
        with torch.inference_mode():
            out_base = model.generate(**inputs, generation_config=gen_config_base)
        gen_base = out_base[0, prompt_len:]
        stopped_base = STOP_ID in gen_base.tolist() or EOS_ID in gen_base.tolist()
        text_base = tokenizer.decode(gen_base, skip_special_tokens=True).strip()

        # 2. Path A calibrated
        processor = CalibratedStopLogitsProcessor(
            stop_token_id=STOP_ID,
            prompt_length=prompt_len,
            tokenizer=tokenizer,
            boundary_bias=4.5,
            min_new_tokens=6,
        )
        with torch.inference_mode():
            out_cal = model.generate(
                **inputs,
                generation_config=gen_config_base,
                logits_processor=LogitsProcessorList([processor]),
            )
        gen_cal = out_cal[0, prompt_len:]
        stopped_cal = STOP_ID in gen_cal.tolist() or EOS_ID in gen_cal.tolist()
        text_cal = tokenizer.decode(gen_cal, skip_special_tokens=True).strip()

        print("\n--- BASELINE (UNCALIBRATED) ---")
        print(f"Generated tokens: {gen_base.numel()} | Natural STOP: {stopped_base}")
        print(repr(text_base[:120]) + ("..." if len(text_base) > 120 else ""))

        print("\n--- PATH A (CALIBRATED LOGIT BIAS +4.5) ---")
        print(f"Generated tokens: {gen_cal.numel()} | Natural STOP: {stopped_cal}")
        print(repr(text_cal))

    print("\n=== TEST COMPLETED ===", flush=True)


if __name__ == "__main__":
    main()
