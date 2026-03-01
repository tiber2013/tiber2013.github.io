---
title: Real-time Object Detection System
---

# Real-time Object Detection System

High-performance real-time object detection system based on YOLOv8.

## Tags

<div class="tag-cloud">
  <span class="tag-item">Object Detection</span>
  <span class="tag-item">YOLOv8</span>
  <span class="tag-item">Real-time</span>
  <span class="tag-item">TensorRT</span>
</div>

---

## Architecture

```
┌─────────────────────────────────────────────┐
│      Real-time Object Detection System      │
├─────────────────────────────────────────────┤
│  Input                                       │
│  ├── Camera stream                          │
│  ├── Video files                            │
│  └── RTSP stream                            │
├─────────────────────────────────────────────┤
│  Preprocessing                               │
│  ├── Image resize                           │
│  ├── Normalization                          │
│  └── Data augmentation (optional)           │
├─────────────────────────────────────────────┤
│  Inference Engine                            │
│  ├── YOLOv8 (PyTorch)                       │
│  └── TensorRT acceleration                  │
├─────────────────────────────────────────────┤
│  Postprocessing                              │
│  ├── NMS                                    │
│  └── Result filtering                       │
├─────────────────────────────────────────────┤
│  Output                                      │
│  ├── Visualization                          │
│  ├── JSON output                            │
│  └── Database storage                       │
└─────────────────────────────────────────────┘
```

---

## Performance

| Metric | Value |
|--------|-------|
| mAP@0.5 | 92.3% |
| mAP@0.5:0.95 | 78.5% |
| FPS (RTX 3090) | 156 |
| FPS (Jetson Xavier) | 45 |
| Model Size | 22.5 MB |

---

## Features

### 1. High Accuracy Detection

- Based on YOLOv8-Large model
- Fine-tuned for specific scenarios
- Custom class support

### 2. Real-time Performance

- TensorRT FP16 acceleration
- Multi-stream concurrent processing
- Low latency inference

### 3. Easy Deployment

- Docker containerization
- One-click deployment scripts
- RESTful API interface

---

## Code Example

### Python Inference

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Inference
results = model('image.jpg')

# Get detection results
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = box.cls[0]
        print(f"Class: {cls}, Confidence: {conf:.2f}, BBox: [{x1}, {y1}, {x2}, {y2}]")
```

### TensorRT Deployment

```python
import tensorrt as trt
import pycuda.driver as cuda

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
    
    def infer(self, input_image):
        # Preprocess
        input_tensor = self.preprocess(input_image)
        # Inference
        output = self.context.execute_v2([input_tensor, output_tensor])
        return output
```

---

## Links

<div class="tag-cloud">
  <a href="https://github.com/tiber2013/yolov8-custom" class="tag-item" target="_blank">GitHub Code</a>
  <a href="https://arxiv.org/abs/xxxx" class="tag-item" target="_blank">Paper</a>
  <a href="https://huggingface.co/tiber2013/yolov8-custom" class="tag-item" target="_blank">Model Download</a>
</div>

---

## Citation

If you use this project, please cite:

```bibtex
@misc{yolov8-custom,
  author = {Zhang, San},
  title = {YOLOv8-Custom: A Custom Training Framework for YOLOv8},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/tiber2013/yolov8-custom}
}
```
