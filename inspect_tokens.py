import tiktoken

enc = tiktoken.get_encoding("gpt2")

text = "I love machine learning."

tokens = enc.encode(text)

print("Original Text:")
print(text)

print("\nTokens:")
print(tokens)

decoded = enc.decode(tokens)

print("\nDecoded:")
print(decoded)