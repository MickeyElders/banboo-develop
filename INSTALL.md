# 智能切竹机 - 安装指南

本文档将指导您完成智能切竹机AI视觉识别系统的安装和配置。

## 📋 系统要求

### 最低要求
- **操作系统**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8+ (推荐 3.9 或 3.10)
- **内存**: 4GB RAM (推荐 8GB+)
- **存储**: 2GB 可用空间

### 推荐配置
- **GPU**: NVIDIA GTX 1060+ (用于YOLOv8加速)
- **CUDA**: 11.8+ (如果使用GPU)
- **内存**: 16GB+ RAM
- **存储**: 10GB+ 可用空间 (包含训练数据)

## 🚀 快速安装

### 1. 克隆仓库
```bash
git clone https://github.com/MickeyElders/banboo-develop.git
cd banboo-develop
```

### 2. 创建虚拟环境 (推荐)
```bash
# 使用 venv
python -m venv bamboo_env
source bamboo_env/bin/activate  # Linux/macOS
# 或
bamboo_env\Scripts\activate     # Windows

# 使用 conda
conda create -n bamboo_env python=3.9
conda activate bamboo_env
```

### 3. 安装基础依赖
```bash
pip install -r requirements.txt
```

### 4. 运行基础演示
```bash
python demo_ai_vision.py
```

如果基础演示成功运行，您的基础环境就配置好了！

## 🧠 YOLOv8深度学习功能安装

### CPU版本安装 (适合测试)
```bash
# 安装 YOLOv8
pip install ultralytics>=8.3.145

# 安装 PyTorch CPU版本
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu
```

### GPU版本安装 (推荐用于生产)
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应的PyTorch版本 (以CUDA 11.8为例)
pip install torch>=2.0.0+cu118 torchvision>=0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 安装YOLOv8
pip install ultralytics>=8.3.145

# 验证GPU可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 模型部署优化 (可选)
```bash
# 安装ONNX支持
pip install onnx>=1.12.0 onnxruntime>=1.12.0

# GPU版本的ONNX Runtime
pip install onnxruntime-gpu>=1.12.0

# Intel OpenVINO (可选，用于Intel CPU优化)
pip install openvino>=2023.0.0
```

## 🔧 Jetson Nano部署

### 自动安装 (推荐)
```bash
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh
```

### 手动安装
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装系统依赖
sudo apt install -y python3-pip python3-opencv python3-numpy python3-scipy

# 安装Python包
pip3 install ultralytics --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install pyyaml pillow

# 验证安装
python3 -c "from ultralytics import YOLO; print('YOLOv8 安装成功!')"
```

## 📦 依赖说明

### 核心依赖
```
opencv-python>=4.5.0     # 计算机视觉
numpy>=1.21.0           # 数值计算
scipy>=1.7.0            # 科学计算
pillow>=8.0.0           # 图像处理
pyyaml>=5.4.0           # 配置文件
```

### 深度学习依赖
```
ultralytics>=8.3.145    # YOLOv8模型
torch>=2.0.0           # PyTorch深度学习框架
torchvision>=0.15.0    # 计算机视觉工具
```

### 硬件控制依赖
```
pyserial>=3.5          # 串口通信
```

### 开发依赖
```
pytest>=6.2.0         # 单元测试
black>=21.0.0          # 代码格式化
mypy>=0.910            # 类型检查
```

## 🧪 安装验证

### 1. 基础功能测试
```bash
# 运行基础演示
python demo_ai_vision.py

# 运行单元测试
python -m pytest test/ -v
```

### 2. YOLOv8功能测试
```bash
# 检查可用算法
python -c "
from src.vision.bamboo_detector import BambooDetector
detector = BambooDetector()
print('可用算法:', detector.get_available_algorithms())
"

# 测试YOLOv8初始化
python -c "
from src.vision.yolo_detector import YOLODetector
from src.vision.vision_types import create_default_config
detector = YOLODetector(create_default_config())
print('YOLOv8初始化:', detector.initialize())
"
```

### 3. 混合检测测试
```bash
# 测试混合检测器
python -c "
from src.vision.hybrid_detector import HybridDetector
from src.vision.vision_types import create_default_config
detector = HybridDetector(create_default_config())
print('混合检测器初始化:', detector.initialize())
"
```

## 🐛 常见问题解决

### 1. ImportError: No module named 'ultralytics'
```bash
# 解决方案：安装YOLOv8
pip install ultralytics>=8.3.145
```

### 2. CUDA out of memory
```bash
# 解决方案：使用CPU版本或减少batch size
export CUDA_VISIBLE_DEVICES=""  # 强制使用CPU
```

### 3. OpenCV import error
```bash
# 解决方案：重新安装OpenCV
pip uninstall opencv-python opencv-python-headless
pip install opencv-python>=4.5.0
```

### 4. Permission denied on Jetson
```bash
# 解决方案：添加用户权限
sudo usermod -a -G dialout $USER
sudo usermod -a -G video $USER
# 重新登录生效
```

### 5. 模型下载失败
```bash
# 解决方案：手动下载模型
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
```

## 🔄 更新项目

### 更新代码
```bash
git pull origin master
```

### 更新依赖
```bash
pip install -r requirements.txt --upgrade
```

### 清理缓存
```bash
# 清理Python缓存
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# 清理模型缓存
rm -rf ~/.cache/ultralytics/
```

## 🎯 性能优化

### CPU优化
```bash
# 安装优化版本的NumPy
pip install numpy[opt]

# 使用Intel MKL (如果是Intel CPU)
pip install mkl
```

### GPU优化
```bash
# 安装CUDA优化版本
pip install torch>=2.0.0+cu118 torchvision>=0.15.0+cu118

# 验证GPU设置
python -c "
import torch
print(f'GPU数量: {torch.cuda.device_count()}')
print(f'当前GPU: {torch.cuda.get_device_name()}')
print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
"
```

### 内存优化
```bash
# 设置环境变量
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## 📊 基准测试

### 运行性能测试
```bash
# 完整性能测试
python test/test_vision_ai.py

# 算法对比测试
python demo_ai_vision.py
# 选择选项3: 查看算法性能对比
```

### 期望性能指标
| 算法类型 | 处理时间 (CPU) | 处理时间 (GPU) | 内存使用 |
|---------|---------------|---------------|---------|
| 传统CV   | ~20ms         | ~20ms         | ~150MB  |
| YOLOv8   | ~200ms        | ~50ms         | ~500MB  |
| 混合算法 | ~100ms        | ~40ms         | ~300MB  |

## 📞 技术支持

### 问题报告
- GitHub Issues: https://github.com/MickeyElders/banboo-develop/issues
- 请提供：操作系统、Python版本、错误信息、复现步骤

### 社区支持
- 项目Wiki: https://github.com/MickeyElders/banboo-develop/wiki
- 讨论区: https://github.com/MickeyElders/banboo-develop/discussions

### 开发者指南
```bash
# 安装开发环境
pip install -r dev_requirements.txt

# 代码格式化
black src/ test/

# 类型检查
mypy src/

# 运行完整测试
python -m pytest test/ --cov=src/
```

---

**安装遇到问题？** 查看[常见问题](#常见问题解决)部分或在GitHub上提交Issue！ 