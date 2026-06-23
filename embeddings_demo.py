import torch
import torch.nn as nn
import tiktoken

enc = tiktoken.get_encoding("gpt2")

text = "I love machine learning."

tokens = enc.encode(text)

print("Tokens:")
print(tokens)

tokens = torch.tensor(tokens)

# Tiny vocabulary embedding
embedding = nn.Embedding(
    num_embeddings=50257,
    embedding_dim=8
)

vectors = embedding(tokens)

print("\nEmbedding Shape:")
print(vectors.shape)

print("\nEmbeddings:")
print(vectors)