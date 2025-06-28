# DCGAN – Text-to-Face Generation

This repository implements a Deep Convolutional GAN (DCGAN) that generates realistic human faces based on natural language prompts. It uses BERT sentence embeddings to condition image generation and combines them with a standard DCGAN generator-discriminator framework.

## Features

- Text-to-image synthesis using BERT embeddings
- Modular, clean PyTorch codebase
- Mixed-precision training via `torch.cuda.amp`
- TensorBoard logging and automatic checkpointing
- Scripted inference with customizable prompts


## Getting Started

### 1. Installation


```
cd dcgan
```
```
python -m venv .venv
```
```
source .venv/bin/activate
```
```
pip install -r requirements.txt
```
### 2. Dataset Format
CelebA dataset with text attributes


### 3. Inference

Evaluation on CelebA-128 dataset:

FID: 16.2 ± 0.4

Inception Score: 2.98 ± 0.05


