import torch
from diagnose_vyrex_1_7b_generation_logits import load_model
from transformers import GenerationConfig
from vyrex_1_7b_inference import IM_END_ID, EOS_ID

PROMPT = "What is artificial intelligence? Answer in two short sentences."
STEPS = [37, 61, 62]

model, tokenizer = load_model()
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True,
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

config = GenerationConfig(
    max_new_tokens=64,
    do_sample=False,
    eos_token_id=[IM_END_ID, EOS_ID],
    pad_token_id=EOS_ID,
    use_cache=True,
    return_dict_in_generate=True,
    output_logits=True,
)

with torch.inference_mode():
    output = model.generate(**inputs, generation_config=config)

print("\n=== TOP-10 COMPETITORS ===")
for step in STEPS:
    if step > len(output.logits):
        continue
    logits = output.logits[step - 1][0].float()
    probs = torch.softmax(logits, dim=-1)
    values, ids = torch.topk(logits, 10)

    print(f"\nSTEP {step}")
    print("Generated token:", repr(tokenizer.decode(
        [int(output.sequences[0, inputs['input_ids'].shape[1] + step - 1])],
        skip_special_tokens=False
    )))
    for rank, (value, token_id) in enumerate(zip(values.tolist(), ids.tolist()), 1):
        print(
            f"{rank:2d}. ID={token_id:6d} "
            f"prob={probs[token_id].item():.8f} "
            f"token={tokenizer.convert_ids_to_tokens([token_id])[0]!r}"
        )
    print(
        "STOP rank:",
        int((logits > logits[IM_END_ID]).sum().item()) + 1,
        "| STOP probability:",
        f"{probs[IM_END_ID].item():.8f}"
    )
