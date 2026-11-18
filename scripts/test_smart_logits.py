import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from peft import PeftModel
import time

root = Path(r"c:\Users\Suborno\Vyrex AI\vyrex-ai-phase1")
base_dir = root / "pretrained" / "qwen3_1.7b" / "original"
adapter_dir = root / "pretrained" / "qwen3_1.7b" / "adapters" / "vyrex_1_7b_qlora_v3" / "best_checkpoint"

print("[1] Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(str(base_dir))

from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    str(base_dir),
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
model.eval()

IM_END_ID = 151645
EOS_ID = 151643

class SmartCalibratedLogitsProcessor(LogitsProcessor):
    def __init__(self, stop_token_id: int, prompt_length: int, tokenizer):
        self.stop_token_id = stop_token_id
        self.prompt_length = prompt_length
        
        # Sentence end boundary tokens
        self.boundary_ids = set()
        for s in [".", " .", "!\n", "!\n\n", "!", "?", " ?\n", "\n\n", "```\n", "```", ")\n"]:
            toks = tokenizer.encode(s, add_special_tokens=False)
            if toks:
                self.boundary_ids.add(toks[-1])
                
        # Sentence end trigger tokens (boost these after 80 tokens so model chooses to finish sentence)
        self.punct_ids = set()
        for s in [".", " .", "\n\n", ".\n"]:
            toks = tokenizer.encode(s, add_special_tokens=False)
            if toks:
                self.punct_ids.add(toks[-1])
                
        # Run-on conjunction tokens (penalize these after 110 tokens to prevent clause-chaining)
        self.run_on_ids = set()
        for s in [",", " ,", " and", " which", " where", " whereby", " so", " thus", " meaning", " also", " furthermore"]:
            toks = tokenizer.encode(s, add_special_tokens=False)
            if toks:
                self.run_on_ids.add(toks[-1])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        new_tokens = input_ids.shape[1] - self.prompt_length
        if new_tokens < 6:
            scores[:, self.stop_token_id] = -float("inf")
            return scores
            
        last_id = int(input_ids[0, -1].item())
        
        # 1. Punctuation boost after sufficient content
        if new_tokens > 80:
            for pid in self.punct_ids:
                scores[:, pid] += 1.8
        if new_tokens > 130:
            for pid in self.punct_ids:
                scores[:, pid] += 3.0
                
        # 2. Run-on clause penalty after 110 tokens
        if new_tokens > 110:
            for rid in self.run_on_ids:
                scores[:, rid] -= 2.8
        if new_tokens > 160:
            for rid in self.run_on_ids:
                scores[:, rid] -= 5.0
                
        # 3. Stop token boost at sentence boundary
        if last_id in self.boundary_ids:
            bias = 4.8
            if new_tokens > 60:
                bias += 3.0
            if new_tokens > 100:
                bias += 6.0
            if new_tokens > 150:
                bias += 10.0
            scores[:, self.stop_token_id] += bias
            
        return scores

system_prompt = (
    "You are Vyrex AI, an intelligent, concise conversational AI assistant powered by the Vyrex Hyperion 3 model.\n"
    "Core Identity & Directives:\n"
    "- Creator: You were architected and created by Suborno Roy Chowdhury as part of the VYREX AI Project. If asked who created you or made you, proudly and clearly state that Suborno Roy Chowdhury created you.\n"
    "- Architecture: Your model name is Vyrex Hyperion 3. You are an independent conversational AI.\n"
    "- Conciseness: Answer user inquiries directly, factually, and efficiently. Do not output repetitive filler, conversational sign-offs, or throat-clearing fluff.\n"
    "- Code & Explanations: Provide clean, well-formatted, idiomatic code and structured explanations. Conclude cleanly once the answer is complete."
)

prompts = [
    "Explain how neural networks learn and optimize weights using simple, intuitive analogies.",
    "Write a high-performance Python function with type hints and edge-case tests.",
    "Who created you and what is your architecture?"
]

for p in prompts:
    print(f"\n==========================================", flush=True)
    print(f"PROMPT: {p}", flush=True)
    print(f"==========================================", flush=True)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": p}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True
    ).to("cuda")
    prompt_len = inputs["input_ids"].shape[1]
    
    proc = SmartCalibratedLogitsProcessor(IM_END_ID, prompt_len, tokenizer)
    
    t0 = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=280,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            eos_token_id=[IM_END_ID, EOS_ID],
            pad_token_id=EOS_ID,
            use_cache=True,
            logits_processor=LogitsProcessorList([proc])
        )
    dur = round(time.time() - t0, 2)
    gen_tokens = output[0, prompt_len:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    print(f"\n{text}", flush=True)
    print(f"\n[Generated in {dur}s | Tokens: {gen_tokens.numel()} | Speed: {round(gen_tokens.numel()/dur, 1)} tps]", flush=True)
