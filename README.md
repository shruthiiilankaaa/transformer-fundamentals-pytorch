# Transformer Fundamentals Learning Lab

A hands-on PyTorch project built to understand Transformers and GPTs from first principles. This repository explores the core concepts behind modern Large Language Models (LLMs) by implementing and inspecting each component individually instead of treating transformers as a black box.

**Note:** This is an educational learning project focused on understanding transformer mechanics and GPT architectures. It is not intended to be a production-ready language model.

---

## Objectives

* Understand how text becomes tokens
* Learn token embeddings and positional embeddings
* Build intuition for Query, Key, and Value matrices
* Understand self-attention mathematically and programmatically
* Explore causal masking and next-token prediction
* Understand multi-head attention
* Prepare custom datasets for miniGPT training using nanoGPT

---

## Concepts Explored

### Text Processing

* Text tokenization using `tiktoken`
* GPT-2 vocabulary and token IDs
* Train and validation dataset preparation

### Embeddings

* Token embeddings
* Positional embeddings
* Tensor representations of sequences

### Self-Attention

* Query (Q), Key (K), and Value (V) matrices
* Attention score computation (`QKᵀ`)
* Scaled dot-product attention
* Softmax attention weights
* Contextual representations (`Attention × V`)

### GPT-Specific Concepts

* Causal masking
* Left-to-right generation
* Context windows
* Next-token prediction

### Advanced Concepts

* Multi-head attention
* Transformer blocks
* Dataset preparation for nanoGPT

---

## Project Structure

```text
transformer-fundamentals-pytorch/
│
├── data/
│   └── notes/
│       ├── ai.txt
│       ├── cn.txt
│       ├── dbms.txt
│       ├── journal.txt
│       └── os.txt
│
├── inspect_tokens.py
├── embeddings_demo.py
├── positional_demo.py
├── attention_demo.py
├── causal_mask_demo.py
├── multihead_demo.py
├── prepare.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Learning Pipeline

```text
Text
 ↓
Tokenization
 ↓
Embeddings
 ↓
Positional Embeddings
 ↓
Query, Key, Value
 ↓
Attention Scores
 ↓
Softmax Attention
 ↓
Attention Output
 ↓
Causal Masking
 ↓
Multi-Head Attention
 ↓
Transformer Block
 ↓
Next Token Prediction
```

---

## Files Overview

### `inspect_tokens.py`

Tokenizes text using GPT-2 encoding and converts tokens back to text.

### `embeddings_demo.py`

Demonstrates how token IDs become dense vector representations.

### `positional_demo.py`

Shows why positional information is necessary and how position embeddings are added.

### `attention_demo.py`

Implements self-attention step by step:

* Q, K, V matrices
* Attention scores
* Attention weights
* Attention outputs

### `causal_mask_demo.py`

Demonstrates how GPT prevents tokens from seeing future tokens during training.

### `multihead_demo.py`

Explores multi-head attention using PyTorch's `MultiheadAttention`.

### `prepare.py`

Prepares custom text datasets by:

* Reading `.txt` files
* Tokenizing using GPT-2 encoding
* Splitting into train/validation sets
* Saving binary files for miniGPT training

---

## Key Learnings

Through this project, I learned:

* How text is represented as tokens
* Why embeddings are necessary
* Why transformers need positional information
* How self-attention works mathematically
* How attention weights are computed
* How tokens exchange information through attention
* Why GPT generates text left-to-right
* How causal masking prevents information leakage
* Why multiple attention heads improve representation learning
* How datasets are prepared for training GPT-style models

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run Examples

Tokenization:

```bash
python inspect_tokens.py
```

Embeddings:

```bash
python embeddings_demo.py
```

Positional Embeddings:

```bash
python positional_demo.py
```

Self-Attention:

```bash
python attention_demo.py
```

Causal Masking:

```bash
python causal_mask_demo.py
```

Multi-Head Attention:

```bash
python multihead_demo.py
```

Dataset Preparation:

```bash
python prepare.py
```

---
This repository is an educational learning lab focused on understanding transformer architectures and GPT fundamentals through implementation and experimentation using PyTorch.
