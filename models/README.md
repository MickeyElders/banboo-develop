# 智能切竹机模型存储

## 📁 目录结构
```
models/
├── README.md              # 本说明文档
├── best_model.pt          # 训练好的YOLO模型 (PyTorch格式)
├── best_model.onnx        # ONNX格式模型 (可选，用于部署优化)
└── model_info.json       # 模型信息文件
```

## 📝 模型文件说明

### 主要模型文件
- **best_model.pt**: 训练好的PyTorch模型权重文件
- **best_model.onnx**: ONNX格式模型，用于跨平台部署
- **model_info.json**: 模型元数据信息

## 🔧 使用方法

### 1. 加载PyTorch模型
```python
import torch
from ultralytics import YOLO

# 加载模型
model_path = "models/best_model.pt"
model = YOLO(model_path)

# 进行推理
results = model.predict(source="image.jpg")
```

### 2. 加载ONNX模型
```python
import onnxruntime as ort

# 加载ONNX模型
model_path = "models/best_model.onnx"
session = ort.InferenceSession(model_path)

# 进行推理
# inputs = {...}  # 准备输入数据
# outputs = session.run(None, inputs)
```

## 📊 模型信息模板

创建 `model_info.json` 文件包含以下信息：
```json
{
  "model_name": "bamboo_detector",
  "version": "1.0",
  "framework": "YOLOv8",
  "created_date": "2024-02-15",
  "input_size": [640, 640],
  "classes": ["bamboo_node", "bamboo_segment"],
  "performance": {
    "mAP": 0.95,
    "precision": 0.93,
    "recall": 0.97
  },
  "file_size_mb": 50.2,
  "inference_time_ms": 45
}
```

---
**最后更新**: 2024年2月15日 