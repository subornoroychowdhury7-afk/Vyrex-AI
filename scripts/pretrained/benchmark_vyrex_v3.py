from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_1.7b/original"
V3_ADAPTER = ROOT / "pretrained/qwen3_1.7b/adapters/vyrex_1_7b_qlora_v3/best_checkpoint"

STOP_ID = 151645
EOS_ID = 151643

SYSTEM_PROMPT = (
    "You are VYREX AI, an independent, helpful, intelligent, and concise conversational AI assistant. "
    "You answer questions accurately, write clean code, and maintain a professional and friendly tone. "
    "Always identify yourself as VYREX AI."
)

def main():
    print("=== VYREX AI V3 RAW BENCHMARK (ZERO LOGIT BIAS) ===", flush=True)
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
    model = PeftModel.from_pretrained(base, str(V3_ADAPTER), is_trainable=False)
    model.eval()

    test_cases = [
        {"name": "1. IDENTITY CHECK", "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Who are you, and who developed you?"}
        ]},
        {"name": "2. NEGATIVE CONSTRAINT & TRANSLATION", "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Translate this sentence into French: \"Where is the nearest train station?\" Respond with only the French translation and nothing else."}
        ]},
        {"name": "3. MULTI-TURN ENTITY BINDING", "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "My favorite animal is the red panda and my cat's name is Oliver."},
            {"role": "assistant", "content": "That is wonderful! Oliver is a classic and charming name for a cat, and red pandas are known for their distinctive rust-colored fur and agile climbing skills."},
            {"role": "user", "content": "Based on what I told you, what pet do I have and what is his name?"}
        ]},
        {"name": "4. DIRECT FACTOID TERMINATION", "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is the chemical symbol for gold?"}
        ]},
        {"name": "5. ALGORITHM CODE GENERATION", "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Write a python function to check if a number is prime."}
        ]},
    ]

    for tc in test_cases:
        print("\n" + "=" * 65)
        print(tc["name"])
        print("Prompt:", tc["messages"][-1]["content"])
        print("=" * 65)

        inputs = tokenizer.apply_chat_template(
            tc["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        gen_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=[STOP_ID, EOS_ID],
            pad_token_id=EOS_ID,
            use_cache=True,
        )

        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=gen_config)

        gen = output[0, prompt_len:]
        stopped = STOP_ID in gen.tolist() or EOS_ID in gen.tolist()
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()

        print(f"Generated tokens: {gen.numel()} / 256 | Natural STOP: {stopped}")
        print("Response:\n" + text)

    print("\n=== V3 BENCHMARK EVALUATION FINISHED ===", flush=True)

if __name__ == "__main__":
    main()
