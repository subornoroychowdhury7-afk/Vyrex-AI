import argparse
from pathlib import Path

import torch
from transformers import GenerationConfig

from vyrex_1_7b_inference import (
    BASE_DIR,
    ADAPTER_DIR,
    IM_END_ID,
    EOS_ID,
    load_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="What is artificial intelligence? Answer in two short sentences.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")

    model, tokenizer = load_model()

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=[IM_END_ID, EOS_ID],
        pad_token_id=EOS_ID,
        use_cache=True,
        return_dict_in_generate=True,
        output_logits=True,
    )

    print("\n=== GENERATION ===", flush=True)
    print("Prompt:", args.prompt, flush=True)

    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=config)

    generated = output.sequences[0, inputs["input_ids"].shape[1]:]
    logits_steps = output.logits

    print("Generated token count:", generated.numel())
    print("Logits steps:", len(logits_steps))
    print("Logits shape:", tuple(logits_steps[0].shape) if logits_steps else None)

    print("\n=== TOKEN-BY-TOKEN STOP ANALYSIS ===")
    print("step | token_id | token_text | stop_prob | stop_rank | top5_contains_stop")

    stop_id = IM_END_ID
    eos_id = EOS_ID

    for i, (token_id, step_logits) in enumerate(zip(generated, logits_steps)):
        row = step_logits[0].float()
        log_probs = torch.log_softmax(row, dim=-1)
        stop_prob = log_probs[stop_id].exp().item()
        stop_rank = int((row > row[stop_id]).sum().item()) + 1
        top5 = torch.topk(row, k=5).indices.tolist()
        token_text = repr(tokenizer.convert_ids_to_tokens([int(token_id.item())])[0])

        print(
            f"{i + 1:4d} | {int(token_id.item()):8d} | "
            f"{token_text[:24]:24s} | {stop_prob:.8f} | "
            f"{stop_rank:9d} | {stop_id in top5}"
        )

    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    generated_ids = generated.tolist()

    print("\n=== RESULT ===")
    print(response if response else "[Empty response]")
    print("Generated token IDs:", generated_ids)
    print("Generated STOP:", IM_END_ID in generated_ids)
    print("Generated EOS:", EOS_ID in generated_ids)
    print("Reached token limit:", generated.numel() >= args.max_new_tokens)
    print("CUDA allocated GB:", round(torch.cuda.memory_allocated() / 1024**3, 3))
    print("CUDA reserved GB:", round(torch.cuda.memory_reserved() / 1024**3, 3))


if __name__ == "__main__":
    main()
