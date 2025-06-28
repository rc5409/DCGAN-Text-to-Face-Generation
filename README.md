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

```bash

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

3. Training
bash
```
python train.py --config configs/default.yaml --save_dir runs/exp01
```
4. Inference
python infer.py \
  --ckpt runs/exp01/best.pt \
  --prompt "a man wearing sunglasses" \
  --num_images 4 \
  --out_dir samples/

Evaluation on CelebA-128 dataset:

FID: 16.2 ± 0.4

Inception Score: 2.98 ± 0.05


