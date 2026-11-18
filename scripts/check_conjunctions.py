from transformers import AutoTokenizer
from pathlib import Path

root = Path(r"c:\Users\Suborno\Vyrex AI\vyrex-ai-phase1")
base_dir = root / "pretrained" / "qwen3_1.7b" / "original"
tokenizer = AutoTokenizer.from_pretrained(str(base_dir))

conjunctions = [
    ",", " ,", " and", " which", " where", " whereby", " so", " thus", " meaning",
    " also", " then", " therefore", " furthermore", " additionally", " moreover",
    " but", " because", " hence", " as", " with"
]

for c in conjunctions:
    enc = tokenizer.encode(c, add_special_tokens=False)
    print(f"{repr(c):15} -> {enc}")
