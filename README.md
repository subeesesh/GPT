# GPT From Scratch

A GPT-style Transformer language model built completely from scratch in PyTorch by following Andrej Karpathy's educational GPT tutorials. This project was created to understand the core building blocks behind modern Large Language Models (LLMs), including tokenization, self-attention, transformer blocks, training loops, and autoregressive text generation.

## Overview

This repository demonstrates how a decoder-only Transformer (GPT) works internally without relying on high-level abstractions. The implementation focuses on learning the architecture from first principles and understanding every component involved in training and inference.

## Features

- GPT-style decoder architecture
- Multi-Head Self Attention
- Transformer Blocks
- Feed Forward Networks
- Layer Normalization
- Residual Connections
- Positional Embeddings
- Autoregressive Text Generation
- Model Checkpointing
- CUDA GPU Support

## Tech Stack

- Python
- PyTorch
- NumPy
- CUDA
- Matplotlib

## Installation

```bash
git clone https://github.com/subeesesh/GPT.git
cd GPT

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## GPU Setup

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Training

```bash
python train.py
```

## Text Generation

```bash
python generate.py
```

## Architecture

Input Text
→ Tokenization
→ Token Embeddings
→ Positional Embeddings
→ Transformer Blocks
→ Linear Projection
→ Softmax
→ Next Token Prediction

## Learning Outcomes

- Understanding Transformer Architecture
- Self-Attention Mechanism
- Language Modeling
- Backpropagation
- Training Neural Networks
- GPU Acceleration with PyTorch

## Future Improvements

- GPT-2 Tokenizer (BPE)
- Flash Attention
- Mixed Precision Training
- LoRA Fine-Tuning
- Distributed Training
- Larger Datasets

## Acknowledgements

- Andrej Karpathy's GPT tutorials
- Attention Is All You Need (Transformer Paper)

## Author

Subeesh

GitHub: https://github.com/subeesesh/GPT

## License

MIT License
