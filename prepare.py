from pathlib import Path
import numpy as np
import tiktoken

DATA_DIR = Path("data/notes")

text = ""

for file in DATA_DIR.glob("*.txt"):
    print("Reading:", file.name)

    with open(file, "r", encoding="utf-8") as f:
        text += f.read()
        text += "\n"

print("\nCharacters:", len(text))

enc = tiktoken.get_encoding("gpt2")

tokens = enc.encode(text)

print("Tokens:", len(tokens))

split = int(len(tokens) * 0.9)

train_ids = tokens[:split]
val_ids = tokens[split:]

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile("train.bin")
val_ids.tofile("val.bin")

print("Train Tokens:", len(train_ids))
print("Val Tokens:", len(val_ids))