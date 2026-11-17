from pathlib import Path
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

root = Path("pretrained/qwen3_0.6b")
model_dir = root / "original"
out_dir = root / "baselines"
out_dir.mkdir(parents=True, exist_ok=True)

out_file = out_dir / "base_inference_baseline.json"
if out_file.exists():
    raise FileExistsError(f"Refusing to overwrite: {out_file}")

tokenizer = AutoTokenizer.from_pretrained(
    str(model_dir), local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    str(model_dir),
    dtype=torch.bfloat16,
    device_map="cuda",
    local_files_only=True,
    low_cpu_mem_usage=True,
)
model.eval()

prompts = [
    "Explain artificial intelligence in simple English:",
    "What is the difference between machine learning and deep learning?",
    "Write a Python function that checks whether a number is prime:",
]

results = []
torch.cuda.reset_peak_memory_stats()

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated_ids = output[0, inputs["input_ids"].shape[1]:]
    count = generated_ids.numel()

    result = {
        "prompt": prompt,
        "output": tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ),
        "generated_tokens": count,
        "generation_seconds": round(elapsed, 4),
        "tokens_per_second": round(count / max(elapsed, 1e-9), 2),
    }
    results.append(result)

    print(f"\nPrompt: {prompt}")
    print(f"Generated tokens: {count}")
    print(f"Speed: {result['tokens_per_second']} tokens/sec")
    print("Output:", result["output"])

report = {
    "model": "Qwen/Qwen3-0.6B-Base",
    "parameters": sum(p.numel() for p in model.parameters()),
    "device": torch.cuda.get_device_name(0),
    "dtype": str(next(model.parameters()).dtype),
    "peak_gpu_memory_gb": round(
        torch.cuda.max_memory_allocated() / 1024**3, 3
    ),
    "results": results,
}

out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n=== BASELINE COMPLETE ===")
print("Peak GPU memory:", report["peak_gpu_memory_gb"], "GB")
print("Saved:", out_file)
