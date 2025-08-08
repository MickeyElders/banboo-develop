# AI模型目录

存放智能切竹机系统的AI检测模型文件。

## 📁 文件说明

- **best.pt**: 原始PyTorch训练模型
- **bamboo_detection.onnx**: ONNX格式模型（640x640输入）
- **bamboo_detection.trt**: TensorRT优化引擎（推荐）

## 🚀 模型转换

### 使用Python脚本转换
```bash
python convert_model.py --model best.pt --format all --precision int8
```

### 手动转换命令
```bash
# PyTorch -> ONNX (使用ultralytics)
yolo export model=best.pt format=onnx imgsz=640 simplify=True

# ONNX -> TensorRT (INT8量化，减少75%大小)
trtexec --onnx=best.onnx --saveEngine=bamboo_detection.trt --int8 --workspace=1024
```

## 📊 优化效果

- **INT8量化**: 模型大小减少75%，几乎无精度损失
- **TensorRT优化**: 推理速度提升3-5倍

## ⚙️ 配置文件中的路径设置

### 系统配置 (config/system_config.yaml)
```yaml
ai:
  model_path: "models/bamboo_detection.onnx"  # 开发环境
  # model_path: "/opt/bamboo-cut/models/bamboo_detection.onnx"  # 生产环境
  use_tensorrt: true
```

### AI优化配置 (config/ai_optimization.yaml)
```yaml
tensorrt:
  model_path: "models/bamboo_detector.onnx"
  engine_path: "models/bamboo_detector.trt"
```

## 📋 支持的模型格式

| 格式 | 扩展名 | 推荐用途 | 性能 |
|------|-------|----------|------|
| ONNX | .onnx | 跨平台部署 | 标准 |
| TensorRT | .trt | NVIDIA GPU | 最优 |
| OpenVINO | .xml/.bin | Intel CPU/GPU | 良好 |

## 🔧 模型使用示例

### C++代码中的使用
```cpp
// 使用ONNX模型
bamboo_cut::vision::BambooDetector::Config config;
config.model_path = "models/bamboo_detection.onnx";
config.engine_path = "models/bamboo_detection.trt";  // TensorRT加速
config.use_tensorrt = true;

auto detector = std::make_unique<bamboo_cut::vision::BambooDetector>(config);
```

### 命令行测试
```bash
# 测试模型推理
./cpp_backend/build/bamboo_cut_backend --test-model models/bamboo_detection.onnx

# 基准测试
./cpp_backend/build/bamboo_cut_backend --benchmark models/bamboo_detection.onnx
```

## 📝 模型部署注意事项

### 开发环境
- 模型放在项目根目录的 `models/` 文件夹
- 相对路径: `models/bamboo_detection.onnx`

### 生产环境
- 模型部署到: `/opt/bamboo-cut/models/`
- 绝对路径: `/opt/bamboo-cut/models/bamboo_detection.onnx`
- 确保模型文件权限正确: `chmod 644 *.onnx *.trt`

### 文件大小建议
- ONNX模型: < 100MB
- TensorRT引擎: 通常比ONNX小20-50%
- 如果模型过大，考虑使用模型量化技术

## 🐛 常见问题

### 模型加载失败
1. 检查文件路径是否正确
2. 确认模型文件完整性
3. 验证ONNX版本兼容性
4. 检查TensorRT版本匹配

### 推理速度慢
1. 使用TensorRT引擎替代ONNX
2. 启用FP16精度
3. 调整批处理大小
4. 考虑模型剪枝和量化

## 📞 技术支持
如有模型相关问题，请参考：
- [AI优化架构文档](../docs/ai_optimization_architecture.md)
- [竹节检测模型推荐](../docs/bamboo_node_detection_model_recommendation.md)