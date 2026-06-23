import torch
import torch.nn as nn
import tiktoken

enc = tiktoken.get_encoding("gpt2")

text = "I love machine learning."

tokens = enc.encode(text)
tokens = torch.tensor(tokens)

print("Tokens:")
print(tokens)

# Token embeddings
token_embedding = nn.Embedding(
    num_embeddings=50257,
    embedding_dim=8
)

token_vectors = token_embedding(tokens)

# Position embeddings
position_embedding = nn.Embedding(
    num_embeddings=1024,
    embedding_dim=8
)

positions = torch.arange(len(tokens))

position_vectors = position_embedding(positions)

# Final embeddings
x = token_vectors + position_vectors

print("\nPositions:")
print(positions)

print("\nFinal Shape:")
print(x.shape)

print("\nFirst Token Vector:")
print(x[0])

print("\nSecond Token Vector:")
print(x[1])