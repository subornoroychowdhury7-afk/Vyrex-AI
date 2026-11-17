import torch
from transformers import set_seed
from vyrex_1_7b_inference import load_model

set_seed(42)

model, tokenizer = load_model()

prompt = "Hi! How are you? Please answer in two short sentences."

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

print("\n=== CONTROLLED GENERATION ===", flush=True)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=1.08,
        no_repeat_ngram_size=4,
        eos_token_id=[151645, 151643],
        pad_token_id=151643,
        use_cache=True,
        return_dict_in_generate=True,
    )

new_tokens = output.sequences[0, inputs["input_ids"].shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\nResponse:")
print(response)

print("\n=== GENERATION DIAGNOSTICS ===")
print("Generated tokens:", new_tokens.numel())
print("Last token ID:", new_tokens[-1].item() if new_tokens.numel() else None)
print("Stopped on im_end:", bool((new_tokens == 151645).any()))
print("Stopped on EOS:", bool((new_tokens == 151643).any()))
print("Hit token limit:", new_tokens.numel() == 128)
print("CUDA allocated GB:", round(torch.cuda.memory_allocated() / 1024**3, 2))
