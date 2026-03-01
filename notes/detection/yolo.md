---
title: YOLO Series Notes
---

# YOLO Series Notes

YOLO (You Only Look Once) is the most classic real-time object detection algorithm series, evolving from 2015 to YOLOv8.

## Overview

YOLO's core idea is to transform object detection into a regression problem, directly predicting bounding boxes and class probabilities for real-time detection.

## Tags

<div class="tag-cloud">
  <span class="tag-item">Object Detection</span>
  <span class="tag-item">Real-time</span>
  <span class="tag-item">YOLO</span>
  <span class="tag-item">Anchor-free</span>
</div>

---

## YOLOv1 (2015)

### Key Innovations

1. **Single-stage Detection**: Transform detection into regression
2. **Global Reasoning**: Use entire image for prediction
3. **Real-time Performance**: Achieves 45 FPS

### Network Architecture

```
Input Image (448x448)
    ↓
Conv Layers (GoogLeNet inspired)
    ↓
Fully Connected Layers
    ↓
Output: S×S×(B×5+C)
```

### Loss Function

Using Mean Squared Error loss:

$$
\lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2]
$$

### Limitations

- Poor performance on small objects
- Difficulty with dense objects
- Lower localization accuracy than two-stage methods

---

## YOLOv2 / YOLO9000 (2016)

### Improvements

1. **Batch Normalization**: BN added to all conv layers
2. **Anchor Boxes**: Introduced anchor mechanism
3. **Dimension Clusters**: K-means clustering for anchor sizes
4. **Multi-scale Training**: Multi-scale training strategy

### Darknet-19

Lighter backbone network with 19 conv layers.

---

## YOLOv3 (2018)

### Key Improvements

1. **Multi-scale Prediction**: Three different scale feature maps
2. **Darknet-53**: Deeper backbone network
3. **Residual Connections**: ResNet-style skip connections

### FPN Structure

```
P5 (13x13) ──────────────────→ Output
    ↑
P4 (26x26) ─────────────────→ Output
    ↑
P3 (52x52) ─────────────────→ Output
```

---

## YOLOv4 (2020)

### Bag of Freebies (BoF)

- Data augmentation: Mosaic, MixUp, CutMix
- Class label smoothing
- CIoU loss

### Bag of Specials (BoS)

- SPP module
- PANet feature fusion
- Mish activation function

---

## YOLOv5 (2020)

Ultralytics PyTorch implementation.

### Features

- Easy deployment and usage
- Multiple model sizes: n/s/m/l/x
- Rich data augmentation strategies

---

## YOLOv6 (2022)

Released by Meituan Technical Team.

### Innovations

- RepVGG-style backbone
- Decoupled detection head
- SIoU loss function

---

## YOLOv7 (2022)

### Core Techniques

- **E-ELAN**: Extended Efficient Layer Aggregation Network
- **Model Scaling**: Compound scaling strategy
- **Reparameterization**: Different structures for training and inference

---

## YOLOv8 (2023)

Ultralytics latest version.

### Improvements

- **Anchor-free**: Anchor-free design
- **Decoupled Head**: Separate classification and regression branches
- **New Loss**: DFL + CIoU

---

## Performance Comparison

| Model | mAP (COCO) | FPS (V100) | Parameters |
|-------|------------|------------|------------|
| YOLOv5s | 37.4 | 231 | 7.2M |
| YOLOv6s | 43.1 | 260 | 18.5M |
| YOLOv7 | 51.4 | 161 | 36.9M |
| YOLOv8s | 44.9 | 183 | 11.2M |

---

## Summary

The evolution from YOLOv1 to YOLOv8 demonstrates the progress in object detection:

1. **Speed-Accuracy Trade-off**: Continuously improving accuracy while maintaining real-time performance
2. **Architecture Innovation**: From single-scale to multi-scale, from anchor-based to anchor-free
3. **Training Strategies**: Continuous optimization of data augmentation and loss functions

## References

- [YOLOv1 Paper](https://arxiv.org/abs/1506.02640)
- [YOLOv2 Paper](https://arxiv.org/abs/1612.08242)
- [YOLOv3 Paper](https://arxiv.org/abs/1804.02767)
- [YOLOv4 Paper](https://arxiv.org/abs/2004.10934)
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
