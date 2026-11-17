from pathlib import Path
import torch
from transformers import GenerationConfig, LogitsProcessorList
from vyrex_1_7b_inference import (
    load_model,
    CalibratedStopLogitsProcessor,
    IM_END_ID,
    EOS_ID,
)

SYSTEM_PROMPT = (
    "You are VYREX AI, an independent, helpful, intelligent, and concise conversational AI assistant. "
    "You answer questions accurately, write clean code, and maintain a professional and friendly tone. "
    "Always identify yourself as VYREX AI."
)

def main():
    print("=== VYREX AI V3 CALIBRATED VERIFICATION ===", flush=True)
    model, tokenizer = load_model()

    tests = [
        ("1. IDENTITY CHECK", "Who are you, and who developed you?"),
        ("2. FRENCH TRANSLATION", 'Translate this sentence into French: "Where is the nearest train station?" Respond with only the French translation and nothing else.'),
        ("3. FACTOID TERMINATION", "What is the chemical symbol for gold?"),
        ("4. PYTHON PRIME ALGORITHM", "Write a python function to check if a number is prime."),
    ]

    for title, prompt in tests:
        print("\n" + "=" * 65)
        print(title)
        print("Prompt:", prompt)
        print("=" * 65)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        processor = CalibratedStopLogitsProcessor(stop_token_id=IM_END_ID, prompt_length=prompt_len, tokenizer=tokenizer, boundary_bias=4.8, min_new_tokens=4)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=[IM_END_ID, EOS_ID],
                pad_token_id=EOS_ID,
                logits_processor=LogitsProcessorList([processor]),
            )

        gen = output[0, prompt_len:]
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        print(f"Generated tokens: {gen.numel()} | Terminated Cleanly: {gen.numel() < 192}")
        print("VYREX V3 Response:\n" + text)

    print("\n=== VERIFICATION FINISHED: Starting Interactive Chat ===")

if __name__ == "__main__":
    main()
