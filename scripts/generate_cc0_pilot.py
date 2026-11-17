from __future__ import annotations

import argparse
import torch

from vyrex_ai.config import load_config
from vyrex_ai.models.transformer import VyrexConfig, VyrexTransformer
from vyrex_ai.tokenizer import VyrexTokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    input_ids = torch.tensor(
        [[tokenizer.bos_token_id, *token_ids]],
        dtype=torch.long,
        device=device,
    )

    model.eval()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = input_ids[:, -model.config.max_seq_len:]

            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(context)
            else:
                output = model(context)

            logits = output[0] if isinstance(output, tuple) else output
            next_logits = logits[:, -1, :].float()

            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature

                if top_k > 0:
                    k = min(top_k, next_logits.size(-1))
                    values, _ = torch.topk(next_logits, k)
                    threshold = values[:, -1, None]
                    next_logits = next_logits.masked_fill(
                        next_logits < threshold, float("-inf")
                    )

                probabilities = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated = input_ids[0].tolist()
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="VYREX AI pilot inference")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if args.top_k < 0:
        parser.error("--top-k cannot be negative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    config = load_config("configs/cc0-pilot.yaml")
    model_config = VyrexConfig(
        vocab_size=config.model.vocab_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        intermediate_size=config.model.intermediate_size,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        tie_embeddings=config.model.tie_embeddings,
    )
    tokenizer = VyrexTokenizer.load(
        "artifacts/tokenizer/tatoeba_cc0_pilot_v1/tokenizer.json"
    )

    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("Tokenizer and model vocabulary sizes do not match")

    checkpoint = torch.load(
        "checkpoints/cc0_pilot_v1/best.pt",
        map_location="cpu",
        weights_only=False,
    )

    model = VyrexTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    print("VYREX AI — PILOT INFERENCE")
    print("Checkpoint step:", checkpoint["step"])
    print("Best validation loss:", checkpoint["best_validation_loss"])
    print("Device:", device)
    print("\nPROMPT:", args.prompt)

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("\nGENERATED TEXT:\n")
    print(result)


if __name__ == "__main__":
    main()
