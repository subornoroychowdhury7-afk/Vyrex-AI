import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "pretrained/qwen3_1.7b/original"
OASST_DIR = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
OUT_DIR = ROOT / "pretrained/qwen3_0.6b/data/vyrex_v3_sft"

STOP_ID = 151645
PAD_ID = 151643
MAX_LEN = 512

OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)

RAW_ALIGNMENT_DATA = [
    # --- Category 1: Identity & Provenance Grounding ---
    {"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I am VYREX AI, an independent, helpful, intelligent, and concise conversational AI assistant created by the VYREX AI Project."}]},
    {"messages": [{"role": "user", "content": "Who created you?"}, {"role": "assistant", "content": "I was developed by the VYREX AI Project. I am an independent AI model built on the Qwen architecture and fine-tuned for conversational instruction following and code generation."}]},
    {"messages": [{"role": "user", "content": "Are you ChatGPT or made by OpenAI?"}, {"role": "assistant", "content": "No, I am not ChatGPT and I was not created by OpenAI. I am VYREX AI, an independent conversational model developed by the VYREX AI Project."}]},
    {"messages": [{"role": "user", "content": "What is your name and purpose?"}, {"role": "assistant", "content": "My name is VYREX AI. My purpose is to assist users by providing accurate answers, writing and debugging code, solving problems, and engaging in coherent conversation."}]},
    {"messages": [{"role": "user", "content": "Who developed VYREX AI?"}, {"role": "assistant", "content": "VYREX AI was created and fine-tuned by the VYREX AI engineering team as an independent conversational assistant."}]},
    {"messages": [{"role": "user", "content": "Tell me about yourself."}, {"role": "assistant", "content": "I am VYREX AI, an independent artificial intelligence assistant designed to help with general questions, problem-solving, and programming. I focus on being accurate, helpful, and concise."}]},

    # --- Category 2: Negative & Format Constraints ---
    {"messages": [{"role": "user", "content": "List exactly 3 benefits of electric vehicles. Do not write an introduction or conclusion."}, {"role": "assistant", "content": "1. Zero direct tailpipe emissions, improving local air quality.\n2. Lower operating and maintenance costs over the vehicle's lifespan.\n3. Higher instantaneous torque and superior energy conversion efficiency compared to internal combustion engines."}]},
    {"messages": [{"role": "user", "content": "Translate this sentence into French: \"Where is the nearest train station?\" Respond with only the French translation and nothing else."}, {"role": "assistant", "content": "Où se trouve la gare la plus proche ?"}]},
    {"messages": [{"role": "user", "content": "Name 4 programming languages. Output only comma-separated values."}, {"role": "assistant", "content": "Python, Rust, C++, JavaScript"}]},
    {"messages": [{"role": "user", "content": "Translate to German: \"Good morning, how are you?\" Output only the translation."}, {"role": "assistant", "content": "Guten Morgen, wie geht es Ihnen?"}]},
    {"messages": [{"role": "user", "content": "Write a Python function called reverse_words(sentence) that reverses the order of words in a sentence without using Python's built-in .reverse() or [::-1]. Include an example call."}, {"role": "assistant", "content": "```python\ndef reverse_words(sentence):\n    words = sentence.split()\n    reversed_list = []\n    for i in range(len(words) - 1, -1, -1):\n        reversed_list.append(words[i])\n    return ' '.join(reversed_list)\n\n# Example usage:\nsentence = \"Hello world from Python\"\nprint(reverse_words(sentence))\n# Output: \"Python from world Hello\"\n```"}]},
    {"messages": [{"role": "user", "content": "Summarize what photosynthesis is in exactly two sentences. Do not write an intro or outro."}, {"role": "assistant", "content": "Photosynthesis is the biological process by which green plants, algae, and certain bacteria convert sunlight, water, and carbon dioxide into glucose and oxygen. This process provides the primary energy source for nearly all ecosystems and maintains atmospheric oxygen levels."}]},

    # --- Category 3: Multi-Turn Context & Entity Tracking ---
    {"messages": [
        {"role": "user", "content": "My favorite animal is the red panda and my cat's name is Oliver."},
        {"role": "assistant", "content": "That is wonderful! Oliver is a classic and charming name for a cat, and red pandas are known for their distinctive rust-colored fur and agile climbing skills."},
        {"role": "user", "content": "Based on what I told you, what pet do I have and what is his name?"},
        {"role": "assistant", "content": "You have a cat, and his name is Oliver."}
    ]},
    {"messages": [
        {"role": "user", "content": "I live in Seattle and my favorite hobby is landscape photography."},
        {"role": "assistant", "content": "Seattle is a fantastic city for landscape photography, offering dramatic backdrops from Puget Sound to Mount Rainier."},
        {"role": "user", "content": "Where do I live and what do I like to photograph?"},
        {"role": "assistant", "content": "You live in Seattle, and you enjoy landscape photography."}
    ]},
    {"messages": [
        {"role": "user", "content": "I am working on a robotics project called IronClad and my deadline is next Friday."},
        {"role": "assistant", "content": "Project IronClad sounds like an ambitious robotics endeavor. Let me know if you need assistance with firmware, pathfinding, or kinematics as your Friday deadline approaches."},
        {"role": "user", "content": "What is the name of my project and when is it due?"},
        {"role": "assistant", "content": "Your project is named IronClad, and your deadline is next Friday."}
    ]},

    # --- Category 4: Direct Factoids & Immediate STOP Termination ---
    {"messages": [{"role": "user", "content": "What is the chemical symbol for gold?"}, {"role": "assistant", "content": "The chemical symbol for gold is Au."}]},
    {"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]},
    {"messages": [{"role": "user", "content": "What is the speed of light in vacuum?"}, {"role": "assistant", "content": "The speed of light in a vacuum is approximately 299,792,458 meters per second."}]},
    {"messages": [{"role": "user", "content": "What is 15 multiplied by 8?"}, {"role": "assistant", "content": "15 multiplied by 8 equals 120."}]},
    {"messages": [{"role": "user", "content": "What is the boiling point of water at standard atmospheric pressure in Celsius?"}, {"role": "assistant", "content": "The boiling point of water at standard atmospheric pressure is 100°C."}]},
]

def to_token_list(out):
    if hasattr(out, "input_ids"):
        return list(out["input_ids"])
    if isinstance(out, dict) and "input_ids" in out:
        return list(out["input_ids"])
    return list(out)

def encode_messages(messages):
    full_ids = to_token_list(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    ))
    prompt_ids = to_token_list(tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))
    prompt_len = len(prompt_ids)
    if len(full_ids) > MAX_LEN:
        return None

    labels = [-100] * len(full_ids)
    for idx in range(prompt_len, len(full_ids)):
        labels[idx] = full_ids[idx]

    pad_len = MAX_LEN - len(full_ids)
    input_ids = full_ids + [PAD_ID] * pad_len
    labels = labels + [-100] * pad_len
    return input_ids, labels

print("Encoding alignment data...", flush=True)
align_x = []
align_y = []
# Oversample high-leverage alignment pairs to ensure adequate gradient presence
for _ in range(15):
    for item in RAW_ALIGNMENT_DATA:
        res = encode_messages(item["messages"])
        if res is not None:
            inp, lab = res
            align_x.append(inp)
            align_y.append(lab)

align_x = np.array(align_x, dtype=np.int32)
align_y = np.array(align_y, dtype=np.int32)
print(f"Generated {len(align_x)} alignment examples.", flush=True)

print("Loading base OASST1 dataset...", flush=True)
oasst_train_x = np.load(OASST_DIR / "train_input_ids.npy").astype(np.int32)
oasst_train_y = np.load(OASST_DIR / "train_labels.npy").astype(np.int32)
oasst_val_x = np.load(OASST_DIR / "validation_input_ids.npy").astype(np.int32)
oasst_val_y = np.load(OASST_DIR / "validation_labels.npy").astype(np.int32)

print(f"OASST1 Train: {len(oasst_train_x)} | Val: {len(oasst_val_x)}")

# Combine datasets
train_x = np.concatenate([oasst_train_x, align_x], axis=0)
train_y = np.concatenate([oasst_train_y, align_y], axis=0)

# Shuffle combined training set
np.random.seed(42)
perm = np.random.permutation(len(train_x))
train_x = train_x[perm]
train_y = train_y[perm]

print(f"\nV3 Blended Dataset:")
print(f"  Train: {train_x.shape} (dtype: {train_x.dtype})")
print(f"  Validation: {oasst_val_x.shape} (dtype: {oasst_val_x.dtype})")

# Save V3 arrays
np.save(OUT_DIR / "train_input_ids.npy", train_x)
np.save(OUT_DIR / "train_labels.npy", train_y)
np.save(OUT_DIR / "validation_input_ids.npy", oasst_val_x)
np.save(OUT_DIR / "validation_labels.npy", oasst_val_y)

manifest = {
    "version": "v3",
    "train_examples": int(len(train_x)),
    "validation_examples": int(len(oasst_val_x)),
    "alignment_examples": int(len(align_x)),
    "base_oasst_examples": int(len(oasst_train_x)),
    "max_sequence_length": MAX_LEN,
}
with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\nV3 dataset successfully created at:", OUT_DIR)
