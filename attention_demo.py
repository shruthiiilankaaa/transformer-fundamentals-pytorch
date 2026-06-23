import torch
import torch.nn as nn
import tiktoken

enc = tiktoken.get_encoding("gpt2")

text = "I love machine learning"

tokens = torch.tensor(enc.encode(text))

print("Tokens:")
print(tokens)

# Tiny embeddings
embedding = nn.Embedding(
    num_embeddings=50257,
    embedding_dim=8
)

x = embedding(tokens)

print("\nInput Shape:")
print(x.shape)

# Create Q, K, V matrices
Wq = nn.Linear(8, 8, bias=False)
Wk = nn.Linear(8, 8, bias=False)
Wv = nn.Linear(8, 8, bias=False)

Q = Wq(x)
K = Wk(x)
V = Wv(x)

print("\nQ Shape:", Q.shape)
print("K Shape:", K.shape)
print("V Shape:", V.shape)

# Attention scores
scores = Q @ K.T

print("\nAttention Scores Shape:")
print(scores.shape)

print("\nAttention Scores:")
print(scores)

import torch.nn.functional as F

d = Q.shape[-1]

scaled_scores = scores / (d ** 0.5)

weights = F.softmax(
    scaled_scores,
    dim=-1
)

print("\nAttention Weights Shape:")
print(weights.shape)

print("\nAttention Weights:")
print(weights)

output = weights @ V

print("\nAttention Output Shape:")
print(output.shape)

print("\nAttention Output:")
print(output)