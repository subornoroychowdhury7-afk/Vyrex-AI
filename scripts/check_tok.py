from transformers import AutoTokenizer
from pathlib import Path

root = Path(r"c:\Users\Suborno\Vyrex AI\vyrex-ai-phase1")
base_dir = root / "pretrained" / "qwen3_1.7b" / "original"
tokenizer = AutoTokenizer.from_pretrained(str(base_dir))

for s in [".", " .", "\n", "\n\n", "!", " !", "?", " ?", ".\n", "!\n", "?\n"]:
    enc = tokenizer.encode(s, add_special_tokens=False)
    print(f"{repr(s):10} -> {enc}")

sentence = "optimal solution. And another sentence! Done?"
tokens = tokenizer.encode(sentence, add_special_tokens=False)
decoded = [tokenizer.decode([t]) for t in tokens]
print("\nTokens for:", repr(sentence))
for t, d in zip(tokens, decoded):
    print(f"  {t:6} -> {repr(d)}")
