import argparse
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
BASE_DIR = ROOT / "pretrained" / "qwen3_1.7b" / "original"
ADAPTER_DIR = ROOT / "pretrained" / "qwen3_1.7b" / "adapters" / "vyrex_1_7b_qlora_v3" / "best_checkpoint"

IM_END_ID = 151645
EOS_ID = 151643


class CalibratedStopLogitsProcessor(LogitsProcessor):
    def __init__(self, stop_token_id: int, prompt_length: int, tokenizer, boundary_bias: float = 4.8, min_new_tokens: int = 5):
        self.stop_token_id = stop_token_id
        self.prompt_length = prompt_length
        self.min_new_tokens = min_new_tokens
        self.boundary_bias = boundary_bias
        
        self.boundary_ids = set()
        for s in [".", ".\n", "!", "?", "\n", "\n\n", "```", "```\n", "`", ")\n", ":\n"]:
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


def load_model():
    if not BASE_DIR.is_dir():
        raise FileNotFoundError(f"Base model not found: {BASE_DIR}")
    if not ADAPTER_DIR.is_dir():
        raise FileNotFoundError(f"Adapter not found: {ADAPTER_DIR}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this inference configuration.")

    print("=== VYREX AI | QWEN3 1.7B ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("Quantization: NF4 + double quantization | Compute: BF16", flush=True)
    print("Inference Mode: Path A Calibrated Boundary STOP (+4.8 bias)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR), local_files_only=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_DIR),
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=192,
    do_sample=False,
    temperature=0.7,
    top_p=0.9,
    boundary_bias=4.8,
):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": 1.1,
        "eos_token_id": [IM_END_ID, EOS_ID],
        "pad_token_id": EOS_ID,
        "use_cache": True,
    }
    if do_sample:
        generation_args["temperature"] = temperature
        generation_args["top_p"] = top_p

    processor = CalibratedStopLogitsProcessor(
        stop_token_id=IM_END_ID,
        prompt_length=prompt_len,
        tokenizer=tokenizer,
        boundary_bias=boundary_bias,
        min_new_tokens=5,
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **generation_args,
            logits_processor=LogitsProcessorList([processor]),
        )

    generated_tokens = output[0, prompt_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response, generated_tokens.numel()


def main():
    parser = argparse.ArgumentParser(description="Run local VYREX AI inference.")
    parser.add_argument("--prompt", type=str, default="Say hello in one clear sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--bias", type=float, default=4.8)
    args = parser.parse_args()

    model, tokenizer = load_model()
    print("\nPROMPT:", args.prompt)
    resp, count = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, do_sample=args.sample, boundary_bias=args.bias)
    print("\nVYREX:", resp)
    print(f"\n[Generated {count} tokens | Natural Termination: {count < args.max_new_tokens}]")


if __name__ == "__main__":
    main()
