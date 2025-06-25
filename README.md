# 智能切竹机开发项目

## 项目概述

智能切竹机是一个基于AI视觉识别技术的自动化工业设备，专门用于竹材的智能化加工。系统通过AI视觉识别竹节与竹筒，精确控制电机运动输送竹子，并控制切片转动实现精准切断。

### 核心目标
- **去除竹节**：准确识别并切除竹节部分
- **定长分割**：按照预设长度精确分离竹筒
- **质量保证**：确保输出的竹筒符合设计长度要求

---

## 系统架构

### 硬件架构

#### 机械结构
- **主体框架**：方钢焊接结构，确保稳定性和精度
- **进料系统**：前段滚筒组进料口，实现平稳上料
- **加工区域**：1米长直线导轨系统，提供精确的线性运动
- **视觉系统**：导轨上方集成摄像头，实现全程监控
- **控制系统**：集成PLC控制器，统一管理所有执行机构

#### 核心硬件组件
```
智能切竹机
├── 机械系统
│   ├── 方钢主框架
│   ├── 滚筒进料装置
│   ├── 直线导轨(1M)
│   ├── 加工滑台
│   └── 切割工具系统
├── 视觉系统
│   ├── Jetson Nano AI计算单元
│   ├── 工业摄像头
│   └── 照明系统
└── 控制系统
    ├── PLC控制器
    ├── 伺服电机驱动器
    └── 传感器系统
```

### 软件架构

#### 系统分层设计
```
应用层：AI视觉识别算法
通信层：Modbus TCP协议栈
控制层：PLC运动控制逻辑
硬件层：电机驱动、传感器采集
```

#### 主要软件模块
- **AI视觉模块**：基于Jetson Nano的竹节识别系统
- **通信模块**：Modbus TCP协议实现
- **PLC控制模块**：设备运动控制与安全管理
- **人机界面**：操作监控界面

---

## 工作流程

### 五个核心节拍

#### 节拍1：上料与待命 (Loading & Standby)
- **位置**：设备后方长条形料架（网格状结构）
- **动作**：操作员将完整竹材放置在料架上
- **状态**：加工滑台处于原点待命位置
- **目的**：准备待加工原材料

#### 节拍2：送料与定位 (Feeding & Positioning)  
- **位置**：加工滑台移动系统
- **动作**：滑台启动，夹爪抓取竹材一端，精确送至加工点
- **控制**：基于视觉识别结果进行精确定位
- **目的**：将竹材精确送至指定切割位置

#### 节拍3：加工执行 (Processing)
- **位置**：加工滑台切割工具
- **动作**：切割电机启动，对竹材执行切断操作
- **监控**：实时监测切割力度和工具状态
- **目的**：完成竹节切除或定长分割

#### 节拍4：单次完成与步进 (Step & Repeat)
- **位置**：整个加工滑台系统
- **动作**：工具复位，滑台移动至下一加工点
- **逻辑**：根据AI识别结果决定下一步操作
- **目的**：在单根竹材上完成所有加工任务

#### 节拍5：卸料与复位 (Unloading & Reset)
- **位置**：整个设备系统
- **动作**：夹爪松开，成品落入收集箱，滑台返回原点
- **状态**：设备回到初始状态
- **目的**：清空工作区，准备下一轮加工

---

## 通信协议

### Modbus TCP通信架构

#### 系统通信拓扑
```
Jetson Nano (AI视觉) ←→ Modbus TCP ←→ PLC控制器
```

#### 发送报文结构 (CuttingCommand)
```protobuf
syntax = "proto3";

message CuttingCommand {
  // 报文头
  uint32 sequence_id = 1;      // 序列号（防重放）
  fixed64 timestamp = 2;       // IEEE 1588精确时间戳（纳秒）
  
  // 切割参数
  float target_position = 3;   // 切割位置（mm，精度0.01）
  float cutting_speed = 4;     // 切割速度（mm/s）
  uint32 tool_id = 5;          // 刀具选择（0:主刀 1:备用刀）
  
  // 安全校验
  bytes safety_signature = 6;  // 数字签名（ECDSA P-256）
  uint32 crc32 = 7;            // 数据校验（多项式0xEDB88320）
}
```

#### 返回报文结构 (DeviceStatus)
```protobuf
message DeviceStatus {
  // 设备基础状态
  enum State {
    IDLE = 0;          // 空闲
    POSITIONING = 1;   // 定位中
    CUTTING = 2;       // 切割中
    FAULT = 3;         // 故障
  }
  State current_state = 1;
  
  // 实时位置反馈
  float actual_position = 2;   // 实际位置（mm）
  float position_error = 3;    // 跟随误差（mm）
  
  // 传感器数据
  float cutting_force = 4;     // 切割阻力（N）
  float motor_temp = 5;        // 电机温度（℃）
  
  // 安全状态
  uint32 emergency_stop = 6;   // 急停状态（0:正常 1:触发）
}
```

---

## 技术规格

### 性能指标
- **加工精度**：±0.01mm
- **切割速度**：可调节，最大100mm/s
- **识别精度**：竹节识别准确率>99%
- **处理能力**：每小时处理竹材100-200根

### 安全特性
- **急停系统**：硬件级紧急停止
- **数字签名**：ECDSA P-256加密通信
- **CRC校验**：数据完整性验证
- **温度监控**：电机过热保护

---

## 开发环境

### 硬件要求
- Jetson Nano开发板
- PLC控制器（支持Modbus TCP）
- 工业级摄像头
- 直线导轨与伺服电机系统

### 软件要求
- Ubuntu 18.04 LTS (Jetson Nano)
- Python 3.6+
- OpenCV 4.x
- TensorFlow/PyTorch
- Modbus TCP库

---

## 项目状态

### 当前阶段
🔄 **开发阶段** - 正在构建核心系统架构

### 下一步计划
1. 搭建硬件测试平台
2. 开发AI视觉识别算法
3. 实现Modbus TCP通信模块
4. 集成PLC控制程序
5. 系统联调与优化

---

## 联系信息

项目负责人：[待填写]
技术支持：[待填写]
更新时间：2025-01-11

---

*本项目致力于推动竹材加工自动化技术发展，提高生产效率和产品质量。*

# 智能切竹机 - AI视觉识别系统

## 项目简介

这是一个基于人工智能视觉识别技术的智能切竹机控制系统。系统能够自动识别竹子的节点位置，分析竹筒段质量，并规划最优的切割策略，实现竹子的自动化高效加工。

## 🚀 核心特性

### 🔍 多算法检测系统
- **传统计算机视觉算法** - 基于边缘检测、轮廓分析的经典方法
- **YOLOv8深度学习模型** - 最新的目标检测神经网络，支持GPU加速
- **混合检测策略** - 智能融合多种算法，提供最佳检测效果

### 🎯 检测功能
- **竹节精确定位** - 亚像素级精度的竹节位置检测
- **节点类型分类** - 自然节、人工节、分支节、损坏节识别
- **竹筒段质量评估** - 基于长度、直径、表面质量的综合评分
- **智能切割规划** - 自动计算最优切割点，最大化竹筒利用率

### ⚡ 性能特性
- **实时处理** - 平均处理时间 < 100ms
- **高精度检测** - 毫米级定位精度
- **多平台支持** - Windows开发 + Jetson Nano部署
- **模块化架构** - 易于扩展和维护

## 🏗️ 系统架构

```
智能切竹机系统
├── 视觉识别模块
│   ├── 传统CV检测器 (Traditional Detector)
│   ├── YOLOv8检测器 (YOLO Detector)
│   └── 混合检测器 (Hybrid Detector)
├── 硬件控制模块
│   ├── 步进电机控制
│   ├── 切割机控制
│   └── 传感器数据采集
└── 通信模块
    ├── Modbus通信
    └── 网络通信
```

## 🔧 算法详解

### 1. 传统计算机视觉算法

**核心原理：**
- 边缘检测：使用Canny算法提取图像边缘
- 水平投影：统计每列边缘像素密度
- 峰值检测：利用scipy.signal.find_peaks寻找竹节位置
- 特征验证：基于几何特征过滤误检

**优势：**
- 处理速度快，资源消耗低
- 参数可调，适应性强
- 不需要训练数据
- 在光照良好的环境下效果稳定

### 2. YOLOv8深度学习算法

**核心特性：**
- 基于YOLOv8.3.145最新版本
- 支持4类竹节类型检测
- GPU加速推理，支持CUDA
- 可导出ONNX、TensorRT等格式

**网络架构：**
```python
# 类别映射
class_mapping = {
    0: "自然节点",      # 天然形成的竹节
    1: "人工节点",      # 人工切割形成
    2: "分支节点",      # 有分支的节点
    3: "损坏节点"       # 有损坏的节点
}
```

**训练数据：**
- 支持合成数据生成
- YOLO格式标注
- 数据增强（旋转、缩放、颜色变换）
- 80/20训练验证分割

### 3. 混合检测策略

**策略类型：**

1. **YOLO优先策略** (`yolo_first`)
   - 优先使用YOLO检测
   - 检测失败时回退到传统算法
   - 适用于有训练好的模型的场景

2. **传统算法优先策略** (`traditional_first`)
   - 优先使用传统CV算法
   - 检测不满意时使用YOLO
   - 适用于计算资源有限的场景

3. **并行融合策略** (`parallel_fusion`)
   - 同时运行两种算法
   - 智能融合检测结果
   - 权重可配置（默认YOLO:0.7, 传统:0.3）

4. **自适应策略** (`adaptive`)
   - 根据图像质量自动选择算法
   - 高质量图像使用YOLO
   - 低质量图像使用传统算法

5. **共识验证策略** (`consensus`)
   - 寻找两种方法的共识节点
   - 高置信度的一致性检测
   - 最高精度但速度较慢

## 📦 安装与环境配置

### 基础依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 主要包含：
# - opencv-python>=4.5.0
# - numpy>=1.21.0
# - scipy>=1.7.0
# - pillow>=8.0.0
# - pyyaml>=5.4.0
```

### YOLOv8深度学习环境

```bash
# 安装YOLOv8和PyTorch
pip install ultralytics>=8.3.145
pip install torch>=2.0.0 torchvision>=0.15.0

# GPU加速（可选，需要CUDA）
pip install torch>=2.0.0+cu118 torchvision>=0.15.0+cu118

# 模型部署优化
pip install onnx>=1.12.0 onnxruntime>=1.12.0
```

### Jetson Nano部署

```bash
# 使用提供的脚本
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh

# 或手动安装
sudo apt update
sudo apt install python3-opencv python3-numpy
pip3 install ultralytics --extra-index-url https://download.pytorch.org/whl/cpu
```

## 🚀 快速开始

### 1. 基础演示

```bash
# 运行演示程序
python demo_ai_vision.py

# 程序将自动：
# - 创建合成竹子图像
# - 选择最佳检测算法
# - 执行竹节检测
# - 显示检测结果
# - 保存可视化图像
```

### 2. 不同算法测试

```python
from src.vision.bamboo_detector import BambooDetector

# 传统算法
detector_cv = BambooDetector("traditional")

# YOLOv8算法（需要模型文件）
detector_yolo = BambooDetector("yolo")

# 混合算法
detector_hybrid = BambooDetector("hybrid")

# 执行检测
result = detector_hybrid.detect_nodes(image)
print(f"检测到 {result.total_nodes} 个竹节")
```

### 3. 自定义混合策略

```python
from src.vision.hybrid_detector import HybridDetector, HybridStrategy

detector = HybridDetector(config, calibration)
detector.initialize()

# 设置策略
detector.set_strategy(
    HybridStrategy.PARALLEL_FUSION,
    yolo_weight=0.8,  # YOLO权重
    thresholds={
        'consensus_threshold': 0.7,
        'min_yolo_detections': 3
    }
)

result = detector.process_image(image)
```

## 🎓 模型训练

### 1. 生成训练数据

```bash
# 生成1000张合成训练图像
python scripts/train_yolo_model.py --action generate --num-images 1000
```

### 2. 训练YOLOv8模型

```bash
# 开始训练
python scripts/train_yolo_model.py --action train

# 从检查点恢复
python scripts/train_yolo_model.py --action train --resume

# 自定义配置
python scripts/train_yolo_model.py --action train --config config/yolo_config.yaml
```

### 3. 模型验证与导出

```bash
# 验证模型
python scripts/train_yolo_model.py --action val --model models/bamboo_yolo_best.pt

# 导出ONNX格式
python scripts/train_yolo_model.py --action export --format onnx
```

## 📊 性能指标

### 检测精度
- **传统算法**：在良好光照下 > 90%
- **YOLOv8模型**：在训练数据上 > 95%
- **混合算法**：综合精度 > 92%

### 处理速度
- **传统算法**：~20ms/图像 (CPU)
- **YOLOv8模型**：~50ms/图像 (GPU), ~200ms/图像 (CPU)
- **混合算法**：根据策略 30-150ms/图像

### 资源消耗
- **内存使用**：150-500MB
- **GPU显存**：1-2GB (使用YOLOv8时)
- **磁盘空间**：基础版本 < 100MB，包含模型 < 500MB

## 🔧 配置说明

### 算法参数配置

```yaml
# config/system_config.yaml
vision:
  algorithm: "hybrid"  # traditional, yolo, hybrid
  
  traditional:
    gaussian_blur_kernel: 5
    canny_low_threshold: 50
    canny_high_threshold: 150
    node_confidence_threshold: 0.3
    min_contour_area: 50.0
    
  yolo:
    model_path: "models/bamboo_yolo_best.pt"
    confidence_threshold: 0.5
    iou_threshold: 0.45
    device: "auto"  # auto, cpu, cuda
    
  hybrid:
    strategy: "adaptive"  # yolo_first, traditional_first, parallel_fusion, adaptive, consensus
    fusion_weights:
      yolo: 0.7
      traditional: 0.3
    thresholds:
      consensus_threshold: 0.6
      image_quality_threshold: 0.7
```

### 标定参数

```python
calibration_data = {
    'pixel_to_mm_ratio': 0.3,  # 像素到毫米转换比例
    'camera_matrix': [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    'dist_coeffs': [k1, k2, p1, p2, k3],  # 畸变系数
    'rotation_angle': 0.0  # 相机旋转角度
}
```

## 📁 项目结构

```
bamboo-cut-develop/
├── src/                          # 源代码
│   ├── vision/                   # 视觉识别模块
│   │   ├── vision_types.py       # 数据类型定义
│   │   ├── vision_processor.py   # 基础处理器
│   │   ├── traditional_detector.py  # 传统CV检测器
│   │   ├── yolo_detector.py      # YOLOv8检测器
│   │   ├── hybrid_detector.py    # 混合检测器
│   │   └── bamboo_detector.py    # 统一检测接口
│   └── communication/            # 通信模块
│       └── modbus_client.py      # Modbus通信
├── scripts/                      # 脚本工具
│   ├── train_yolo_model.py       # YOLO模型训练
│   ├── setup_jetson.sh           # Jetson环境配置
│   └── sync_to_jetson.ps1        # 代码同步脚本
├── config/                       # 配置文件
│   └── system_config.yaml        # 系统配置
├── docs/                         # 文档
│   ├── technical_specs.md        # 技术规格
│   ├── hardware_setup_guide.md   # 硬件配置指南
│   └── communication_protocol.md # 通信协议
├── test/                         # 测试代码
│   ├── test_vision_ai.py         # 视觉算法测试
│   ├── test_hardware.py          # 硬件测试
│   └── test_communication.py     # 通信测试
├── models/                       # 模型文件目录
├── datasets/                     # 数据集目录
├── demo_output/                  # 演示输出
├── requirements.txt              # Python依赖
├── demo_ai_vision.py             # 演示程序
└── main.py                       # 主程序入口
```

## 🔬 算法原理深入解析

### 传统CV算法流程

1. **图像预处理**
   ```python
   # 高斯模糊降噪
   blurred = cv2.GaussianBlur(image, (5, 5), 0)
   
   # 转换为灰度图
   gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
   ```

2. **边缘检测**
   ```python
   # Canny边缘检测
   edges = cv2.Canny(gray, 50, 150)
   ```

3. **特征提取**
   ```python
   # 水平投影
   h_projection = np.sum(edges, axis=0)
   
   # 峰值检测
   peaks, _ = find_peaks(h_projection, height=threshold, distance=min_distance)
   ```

4. **节点验证**
   ```python
   # 基于几何特征验证
   for peak in peaks:
       if validate_node_geometry(peak, edges):
           nodes.append(create_bamboo_node(peak))
   ```

### YOLOv8检测流程

1. **模型加载**
   ```python
   model = YOLO('yolov8n.pt')  # 或自定义训练的模型
   ```

2. **推理**
   ```python
   results = model(image, conf=0.5, iou=0.45)
   ```

3. **结果解析**
   ```python
   for result in results:
       boxes = result.boxes
       for box in boxes:
           x1, y1, x2, y2 = box.xyxy[0]
           confidence = box.conf[0]
           class_id = box.cls[0]
   ```

### 混合算法融合策略

1. **结果对齐**
   ```python
   # 基于距离的节点匹配
   for yolo_node in yolo_nodes:
       for trad_node in traditional_nodes:
           distance = calculate_distance(yolo_node, trad_node)
           if distance < threshold:
               fused_node = merge_nodes(yolo_node, trad_node)
   ```

2. **置信度融合**
   ```python
   # 加权融合置信度
   fused_confidence = (
       yolo_confidence * yolo_weight + 
       traditional_confidence * traditional_weight
   )
   ```

## 🎯 应用场景

### 1. 工业竹材加工
- 大型竹筒处理厂
- 自动化生产线集成
- 质量控制系统

### 2. 手工艺竹器制作
- 竹筷生产
- 竹篮编织原料准备
- 竹乐器制作

### 3. 研究与开发
- 竹材特性研究
- 加工工艺优化
- 算法性能评估

## 🚧 开发路线图

### v3.0 (当前版本)
- ✅ 传统CV算法优化
- ✅ YOLOv8深度学习集成
- ✅ 混合检测策略
- ✅ 完整的演示系统

### v3.1 (计划中)
- 🔄 实时视频流处理
- 🔄 Web界面控制台
- 🔄 云端模型更新
- 🔄 移动端监控应用

### v4.0 (未来版本)
- 📋 3D点云处理
- 📋 多相机立体视觉
- 📋 强化学习优化
- 📋 边缘计算部署

## 🤝 贡献指南

### 代码贡献
1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 问题报告
请使用GitHub Issues报告bug或提出功能请求。

### 开发环境设置
```bash
# 克隆仓库
git clone https://github.com/MickeyElders/banboo-develop.git
cd banboo-develop

# 安装开发依赖
pip install -r dev_requirements.txt

# 运行测试
python -m pytest test/

# 代码格式化
black src/ test/
```

## 📜 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 👥 团队

- **主要开发者**: [Mickey Elders](https://github.com/MickeyElders)
- **算法设计**: AI团队
- **硬件集成**: 硬件团队

## 📞 联系方式

- 项目地址: https://github.com/MickeyElders/banboo-develop
- Issues: https://github.com/MickeyElders/banboo-develop/issues
- 邮箱: [项目邮箱]

## 🙏 致谢

- OpenCV社区提供的计算机视觉算法
- Ultralytics团队的YOLOv8实现
- PyTorch深度学习框架
- 所有贡献者和测试用户

---

**智能切竹机项目** - 让传统竹材加工拥抱人工智能时代！ 