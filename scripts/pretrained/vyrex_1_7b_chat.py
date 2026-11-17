import argparse
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
    "You were created, architected, and developed by Suborno Roy Chowdhury as part of the VYREX AI Project.\n\n"
    "About your creator, Suborno Roy Chowdhury:\n"
    "- Suborno Roy Chowdhury is an AI/ML engineer and researcher specializing in Artificial Intelligence, Machine Learning, and Generative AI.\n"
    "- He is pursuing his B.Tech in Artificial Intelligence & Machine Learning at St. Thomas' College of Engineering & Technology, Kolkata (YGPA: 8.71), and is enrolled in the Certificate Programme in AI for Real-World Applications at IIT Kharagpur.\n"
    "- He is the creator and lead engineer of VYREX AI (an independent conversational model fine-tuned using parameter-efficient QLoRA) and also engineered DocQ&A, a full-stack Enterprise RAG platform utilizing OCR, ChromaDB, FastAPI, and Next.js.\n"
    "- His core technical stack includes Python, TypeScript, FastAPI, Next.js, ChromaDB, Scikit-Learn, and LLM fine-tuning.\n\n"
    "When asked 'Who created you?', 'Who made VYREX AI?', or 'Who is Suborno Roy Chowdhury?', always answer smoothly, proudly, and accurately with a professional summary of Suborno Roy Chowdhury, highlighting his work as your creator and an AI/ML engineer.\n"
    "Always maintain a professional, polite, and helpful tone, and always identify yourself as VYREX AI."
)

MAX_HISTORY_TURNS = 8


def generate_reply(model, tokenizer, messages, max_new_tokens, sample, boundary_bias=4.8):
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

    args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": sample,
        "repetition_penalty": 1.1,
        "eos_token_id": [IM_END_ID, EOS_ID],
        "pad_token_id": EOS_ID,
        "use_cache": True,
    }
    if sample:
        args["temperature"] = 0.7
        args["top_p"] = 0.9

    processor = CalibratedStopLogitsProcessor(
        stop_token_id=IM_END_ID,
        prompt_length=prompt_len,
        tokenizer=tokenizer,
        boundary_bias=boundary_bias,
        min_new_tokens=4,
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **args,
            logits_processor=LogitsProcessorList([processor]),
        )

    generated = output[0, prompt_len:]
    reply = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return reply, generated.numel()


def main():
    parser = argparse.ArgumentParser(description="Interactive VYREX AI Qwen3-1.7B chat")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--bias", type=float, default=4.8)
    parser.add_argument("--system", type=str, default=SYSTEM_PROMPT)
    args = parser.parse_args()

    model, tokenizer = load_model()
    history = []

    print("\n" + "=" * 58)
    print("                VYREX AI INTERACTIVE CHAT")
    print("=" * 58)
    print("Creator: Suborno Roy Chowdhury | VYREX AI Project")
    print("Model: Qwen3-1.7B + VYREX V3 Checkpoint")
    print("Status: Active | Creator Bio & Natural Termination Enabled")
    print("Commands: /reset = clear history | /exit = quit")
    print("=" * 58)

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nVYREX: Session ended.")
            break

        if not prompt:
            continue

        cmd = prompt.lower()
        if cmd == "/exit":
            print("VYREX: Goodbye.")
            break
        if cmd == "/reset":
            history.clear()
            print("VYREX: Conversation history cleared.")
            continue

        messages = [{"role": "system", "content": args.system}] + history + [{"role": "user", "content": prompt}]
        try:
            response, count = generate_reply(
                model, tokenizer, messages, args.max_new_tokens, args.sample, boundary_bias=args.bias
            )
            print(f"\nVYREX ({count} tokens): {response or '[Empty response]'}")
            if response:
                history.extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ])
                history = history[-(MAX_HISTORY_TURNS * 2):]
        except torch.cuda.OutOfMemoryError:
            print("VYREX: GPU memory exhausted. Try a shorter conversation.")
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"VYREX error: {exc}")


if __name__ == "__main__":
    main()
