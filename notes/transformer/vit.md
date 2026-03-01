---
title: Vision Transformer (ViT)
---

# Vision Transformer (ViT)

Google Research's pioneering work published in 2020, successfully applying Transformer architecture to computer vision tasks for the first time.

## Tags

<div class="tag-cloud">
  <span class="tag-item">Transformer</span>
  <span class="tag-item">Backbone</span>
  <span class="tag-item">Classification</span>
  <span class="tag-item">Self-Attention</span>
  <span class="tag-item">Pre-training</span>
</div>

---

## Overview

ViT demonstrates that pure Transformer architecture can achieve or exceed CNN performance on image classification tasks.

### Core Idea

Divide images into patch sequences as Transformer input tokens.

---

## Model Architecture

```
Input Image (H×W×C)
      ↓
Patch Embedding (N×D)
      ↓
+ [CLS] Token
+ Position Embedding
      ↓
Transformer Encoder × L
      ↓
[CLS] Token → MLP Head → Class Label
```

### Patch Embedding

Divide image into fixed-size patches:

$$
\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}
$$

Where:
- N = HW/P² (number of patches)
- P = patch size (usually 16×16)

### Position Embedding

Using learnable position embeddings:

$$
\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{x}_p^1 \mathbf{E}; \cdots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}
$$

### Transformer Encoder

Each layer contains:

1. **Multi-Head Self-Attention (MSA)**
2. **Layer Normalization (LN)**
3. **MLP (Feed-Forward)**

```python
# Pseudocode
x = x + MSA(LN(x))
x = x + MLP(LN(x))
```

---

## Model Variants

| Model | Layers | Hidden Dim | MLP Dim | Heads | Parameters |
|-------|--------|------------|---------|-------|------------|
| ViT-Ti | 12 | 192 | 768 | 3 | 5.7M |
| ViT-S | 12 | 384 | 1536 | 6 | 22M |
| ViT-B | 12 | 768 | 3072 | 12 | 86M |
| ViT-L | 24 | 1024 | 4096 | 16 | 307M |
| ViT-H | 32 | 1280 | 5120 | 16 | 632M |

---

## Training Strategy

### Pre-training

- **Dataset**: JFT-300M / ImageNet-21K
- **Epochs**: 300
- **Optimizer**: AdamW
- **Learning Rate**: 0.001

### Fine-tuning

- **Dataset**: ImageNet-1K
- **Resolution**: 384×384 (higher resolution)

---

## Performance Comparison

### ImageNet-1K

| Model | Pre-training Data | Top-1 Acc |
|-------|-------------------|-----------|
| ResNet-152 | ImageNet-1K | 82.3% |
| ViT-B/16 | ImageNet-1K | 77.9% |
| ViT-B/16 | ImageNet-21K | 84.2% |
| ViT-B/16 | JFT-300M | 84.5% |
| ViT-L/16 | JFT-300M | 85.2% |

### Key Findings

1. **Large-scale Pre-training is Critical**: ViT underperforms CNNs on small datasets
2. **Higher Computational Efficiency**: Requires less computation for same performance
3. **Global Receptive Field**: Self-attention provides global information interaction

---

## ViT Pros and Cons

### Advantages

- ✅ Global receptive field, captures long-range dependencies
- ✅ Unified architecture, same model for NLP and vision
- ✅ Strong scalability, performance improves with scale

### Limitations

- ❌ Lacks inductive bias, requires large data
- ❌ Sensitive to resolution, computation grows quadratically
- ❌ Training instability, requires strong regularization

---

## Follow-up Work

### DeiT (2021)

- Teacher model distillation
- Training ViT on ImageNet-1K

### Swin Transformer (2021)

- Hierarchical structure
- Shifted window attention
- Linear complexity

### MAE (2022)

- Masked autoencoder pre-training
- Self-supervised learning

---

## Code Example

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        return self.head(x[:, 0])
```

---

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [ViT GitHub (Google Research)](https://github.com/google-research/vision_transformer)
- [timm ViT implementations](https://github.com/huggingface/pytorch-image-models)
