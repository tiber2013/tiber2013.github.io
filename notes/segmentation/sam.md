---
title: SAM - Segment Anything Model
---

# SAM: Segment Anything Model

Meta AI Research's universal image segmentation model released in 2023, achieving powerful zero-shot segmentation capabilities.

## Tags

<div class="tag-cloud">
  <span class="tag-item">Segmentation</span>
  <span class="tag-item">Foundation Model</span>
  <span class="tag-item">Zero-shot</span>
  <span class="tag-item">Prompt Learning</span>
</div>

---

## Overview

SAM (Segment Anything Model) is a promptable image segmentation model that achieves zero-shot generalization through prompt engineering.

### Key Contributions

1. **Promptable Segmentation Task**: Supports points, boxes, text and other prompts
2. **Large-scale Dataset**: SA-1B dataset contains 1 billion masks
3. **Zero-shot Generalization**: Handles new scenarios without fine-tuning

---

## Model Architecture

```
┌─────────────────────────────────────────────┐
│                    SAM                       │
├─────────────────────────────────────────────┤
│  Image Encoder (ViT-H/16)                   │
│       ↓                                      │
│  Prompt Encoder                              │
│  (Points / Boxes / Text → Embeddings)       │
│       ↓                                      │
│  Mask Decoder (Transformer)                  │
│       ↓                                      │
│  Output: Masks + Confidence Scores          │
└─────────────────────────────────────────────┘
```

### Image Encoder

Using ViT-Huge as image encoder:

- Patch Size: 16×16
- Parameters: 636M
- Output: 64×64 feature map

### Prompt Encoder

Supports multiple prompt types:

| Prompt Type | Encoding Method |
|-------------|-----------------|
| Point | Position encoding + foreground/background token |
| Box | Top-left + bottom-right encoding |
| Text | CLIP text encoder |
| Mask | Convolutional encoding |

### Mask Decoder

Lightweight Transformer decoder:

- 2-layer Transformer
- Cross-attention fuses image and prompt features
- Outputs multiple mask candidates

---

## SA-1B Dataset

### Scale

- **Images**: 11 million
- **Masks**: 1 billion
- **Average masks per image**: 91

### Data Collection

Using "data engine" strategy:

1. **Stage 1**: Manual annotation
2. **Stage 2**: Model-assisted + manual correction
3. **Stage 3**: Model auto-generation + manual review

---

## Training Strategy

### Loss Function

$$
\mathcal{L} = \mathcal{L}_{dice} + \mathcal{L}_{bce}
$$

### Multi-mask Prediction

For ambiguous prompts, the model outputs 3 masks:

1. Whole object
2. Part of object
3. Whole object (different interpretation)

---

## Performance Evaluation

### Zero-shot Segmentation

| Dataset | mIoU |
|---------|------|
| COCO | 46.5 |
| LVIS | 49.9 |
| ADE20K | 48.2 |

### Prompt Method Comparison

| Prompt Type | Single Point mIoU | Box mIoU |
|-------------|-------------------|----------|
| SAM-H | 58.1 | 73.3 |

---

## Applications

1. **Image Editing**: Quick cutout, background replacement
2. **Medical Imaging**: Organ segmentation, lesion detection
3. **Video Segmentation**: Object tracking, video editing
4. **Data Annotation**: Accelerating annotation workflow

---

## SAM 2.0

Upgraded version released in 2024:

- Video segmentation support
- Real-time processing capability
- Stronger zero-shot generalization

---

## Code Example

```python
from segment_anything import sam_model_registry, SamPredictor

# Load model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

# Set image
predictor.set_image(image)

# Point prompt segmentation
masks, scores, logits = predictor.predict(
    point_coords=np.array([[x, y]]),
    point_labels=np.array([1])
)

# Box prompt segmentation
masks, scores, logits = predictor.predict(
    box=np.array([x1, y1, x2, y2])
)
```

---

## References

- [Segment Anything Paper](https://arxiv.org/abs/2304.02643)
- [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)
- [SAM 2 Paper](https://arxiv.org/abs/2408.00714)
- [SA-1B Dataset](https://ai.meta.com/datasets/segment-anything/)
