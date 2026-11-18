import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import time

root = Path(r"c:\Users\Suborno\Vyrex AI\vyrex-ai-phase1")
base_dir = root / "pretrained" / "qwen3_1.7b" / "original"
adapter_dir = root / "pretrained" / "qwen3_1.7b" / "adapters" / "vyrex_1_7b_qlora_v3" / "best_checkpoint"

tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    str(base_dir),
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
model.eval()

prompt = "Explain in 3 bullet points what machine learning is."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Test 1: with no_repeat_ngram_size=4
t0 = time.time()
with torch.inference_mode():
    out1 = model.generate(**inputs, max_new_tokens=60, repetition_penalty=1.15, no_repeat_ngram_size=4, use_cache=True)
d1 = time.time() - t0
print(f"Test 1 (with no_repeat_ngram_size=4): {d1:.2f}s | Speed: {60/d1:.1f} tps")

# Test 2: without no_repeat_ngram_size
t0 = time.time()
with torch.inference_mode():
    out2 = model.generate(**inputs, max_new_tokens=60, repetition_penalty=1.15, use_cache=True)
d2 = time.time() - t0
print(f"Test 2 (without no_repeat_ngram_size): {d2:.2f}s | Speed: {60/d2:.1f} tps")
