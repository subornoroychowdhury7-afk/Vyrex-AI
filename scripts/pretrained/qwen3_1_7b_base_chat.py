import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)

MODEL_DIR = "pretrained/qwen3_1.7b/original"

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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=quant_config,
    dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

print("\nQWEN3-1.7B BASE CHAT")
print("Original base model | No VYREX adapter")
print("Commands: /exit, /reset\n")

history = []

while True:
    try:
        user_text = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if user_text.lower() == "/exit":
        break

    if user_text.lower() == "/reset":
        history.clear()
        print("Conversation reset.\n")
        continue

    if not user_text:
        continue

    history.append({"role": "user", "content": user_text})

    messages = history[-16:]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )["input_ids"].to(model.device)

    print("Base: ", end="", flush=True)

    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=[151645, 151643],
                pad_token_id=151643,
                streamer=streamer,
            )

        generated = output[0, input_ids.shape[1]:]
        response = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        history.append({"role": "assistant", "content": response})
        print("\n")

    except torch.cuda.OutOfMemoryError:
        print("\nCUDA out of memory. Try resetting the conversation.")
        torch.cuda.empty_cache()
        history.pop()

print("\nBase chat closed.")

