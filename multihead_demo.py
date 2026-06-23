import torch
import torch.nn as nn

torch.manual_seed(42)

# 4 tokens, embedding size 8
x = torch.randn(4, 8)

print("Input Shape:")
print(x.shape)

# Multi-head attention
mha = nn.MultiheadAttention(
    embed_dim=8,
    num_heads=2,
    batch_first=True
)

# Add batch dimension
x = x.unsqueeze(0)

output, weights = mha(
    x,
    x,
    x
)

print("\nOutput Shape:")
print(output.shape)

print("\nAttention Weights Shape:")
print(weights.shape)