#!/usr/bin/env python3
"""
Merge VYREX AI QLoRA V3 adapter weights directly into the base Qwen3-1.7B model.
Generates a standalone, unified HuggingFace checkpoint for ultra-fast vLLM serving.

Usage:
    python scripts/merge_lora_weights.py --output-dir pretrained/vyrex_1_7b_merged
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "pretrained" / "qwen3_1.7b" / "original"
DEFAULT_ADAPTER = ROOT / "pretrained" / "qwen3_1.7b" / "adapters" / "vyrex_1_7b_qlora_v3" / "best_checkpoint"
DEFAULT_OUT = ROOT / "pretrained" / "vyrex_1_7b_merged"


def merge_weights(base_path: Path, adapter_path: Path, output_path: Path):
    print("=" * 60)
    print("      VYREX AI: MERGING LORA WEIGHTS INTO BASE MODEL")
    print("=" * 60)
    print(f"Base Model:    {base_path}")
    print(f"Adapter:       {adapter_path}")
    print(f"Output Target: {output_path}\n")

    if not base_path.is_dir():
        raise FileNotFoundError(f"Base model not found at {base_path}")
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")

    print("[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True)

    print("[2/4] Loading base model in BF16/FP16 (unquantized for clean merge)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        local_files_only=True,
    )

    print("[3/4] Attaching PEFT adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("[4/4] Merging LoRA layers into base model weights...")
    merged_model = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[+] Saving merged safetensors weights to {output_path}...")
    merged_model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    print("\n[SUCCESS] Successfully created standalone merged model checkpoint!")
    print(f"You can now serve it directly with vLLM:")
    print(f"vllm serve {output_path} --port 8000 --max-model-len 4096")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into Base Model")
    parser.add_argument("--base-path", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    merge_weights(args.base_path, args.adapter_path, args.output_dir)


if __name__ == "__main__":
    main()
