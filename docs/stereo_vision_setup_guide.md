# 双摄像头立体视觉配置指南

## 📋 概述

本指南介绍如何配置和使用双摄像头立体视觉系统解决视差问题，提升竹子切割的3D定位精度。

## 🔧 硬件要求

### 摄像头配置
- **推荐配置**: 两个相同型号的USB摄像头
- **分辨率**: 1280x720 或更高
- **帧率**: 30fps
- **基线距离**: 100-300mm (摄像头间距)
- **安装**: 水平平行安装，避免倾斜

### 基线距离优化
```
基线距离建议：
- 太小 (<50mm): 深度精度不足
- 适中 (100-300mm): 平衡精度和视野重叠 ✅
- 太大 (>500mm): 视野重叠减少，匹配困难
```

## 🚀 系统初始化

### 1. 检查摄像头连接
```bash
# 查看可用摄像头设备
ls /dev/video*

# 测试摄像头
v4l2-ctl --list-devices

# 检查摄像头参数
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video2 --list-formats-ext
```

### 2. 配置摄像头参数
在主程序中调整配置：
```cpp
vision::CameraSyncConfig stereo_config;
stereo_config.left_device = "/dev/video0";   // 左摄像头
stereo_config.right_device = "/dev/video2";  // 右摄像头  
stereo_config.width = 1280;                  // 分辨率
stereo_config.height = 720;
stereo_config.fps = 30;                      // 帧率
stereo_config.sync_tolerance_ms = 10;        // 同步容差
```

## 📐 立体标定流程

### 1. 准备标定板
- **规格**: 9x6棋盘格，方格大小25mm
- **材质**: 平整硬质材料，避免弯曲
- **打印**: 高对比度黑白打印

### 2. 执行标定
```bash
# 编译标定工具
cd cpp_backend
mkdir build && cd build
cmake ..
make stereo_calibration_tool

# 运行标定
./stereo_calibration_tool calibrate
```

### 3. 标定操作步骤
1. 启动标定工具
2. 将标定板放置在不同位置和角度
3. 确保两个摄像头都检测到棋盘格
4. 按 **SPACE** 捕获标定图像
5. 捕获20帧后按 **ENTER** 开始计算
6. 标定文件保存到 `/opt/bamboo-cut/config/stereo_calibration.xml`

### 4. 标定质量检查
```bash
# 测试标定结果
./stereo_calibration_tool test

# 查看实时视差图
./stereo_calibration_tool disparity
```

## 🎯 解决视差问题

### 常见问题和解决方案

#### 1. **基线距离不足**
```
问题: 两个摄像头太近，深度精度差
解决: 增加摄像头间距到150-250mm
验证: 基线距离应显示在标定结果中
```

#### 2. **摄像头同步问题**
```cpp
// 检查同步误差
auto stats = stereo_vision_->get_statistics();
if (stats.avg_sync_error_ms > 10) {
    // 调整同步容差或检查硬件
    stereo_config.sync_tolerance_ms = 15;
}
```

#### 3. **视差计算失败**
```cpp
// 调整SGBM参数
StereoMatchingConfig config;
config.num_disparities = 16 * 6;    // 增加视差范围
config.block_size = 5;               // 调整匹配块大小
config.use_wls_filter = true;        // 启用滤波
stereo_vision_->set_stereo_matching_config(config);
```

#### 4. **深度信息不准确**
```
检查项目:
1. 标定板是否平整
2. 标定图像是否足够多样化
3. 摄像头是否水平对齐
4. 光照是否均匀
```

## 📊 性能优化

### 1. 降低计算延迟
```cpp
// 使用较低分辨率
stereo_config.width = 1280;   // 而不是1920
stereo_config.height = 720;   // 而不是1080

// 优化SGBM参数
config.num_disparities = 16 * 4;  // 减少视差范围
config.block_size = 3;             // 减小匹配块
```

### 2. 提升匹配精度
```cpp
// 启用WLS滤波
config.use_wls_filter = true;
config.lambda = 8000.0;
config.sigma = 1.5;

// 调整置信度阈值
config.min_confidence = 0.7;
```

## 🔍 调试工具

### 1. 实时监控
```cpp
// 查看系统统计
auto stats = stereo_vision_->get_statistics();
std::cout << "成功率: " << stats.successful_captures << "/" << stats.total_frames << std::endl;
std::cout << "同步误差: " << stats.avg_sync_error_ms << "ms" << std::endl;
```

### 2. 视差图可视化
```bash
# 实时查看视差效果
./stereo_calibration_tool disparity
```

### 3. 深度值验证
```cpp
// 在特定像素点检查深度
cv::Point2f test_point(640, 360);  // 图像中心
auto point_3d = stereo_vision_->pixel_to_3d(test_point, disparity);
std::cout << "深度: " << point_3d.z << "mm, 置信度: " << point_3d.confidence << std::endl;
```

## 📈 系统集成

### 1. 主程序集成
立体视觉已集成到主识别程序中：
- 自动加载标定文件
- 实时3D坐标计算
- 深度增强的竹子检测
- 双模式支持（3D/2D切换）

### 2. 通信协议
3D坐标通过Modbus TCP推送给PLC：
- X坐标: 实际世界坐标(mm)
- 深度信息: 用于质量评估
- 置信度: 检测可靠性指标

### 3. 故障恢复
```cpp
// 自动降级到单目模式
if (!stereo_vision_->is_calibrated()) {
    LOG_WARN("立体标定不可用，使用单目模式");
    // 切换到传统2D检测
}
```

## 🎯 最佳实践

### 1. 安装建议
- 摄像头**水平平行**安装
- 基线距离**150-250mm**
- 避免**振动和松动**
- 确保**充足光照**

### 2. 标定技巧
- 在**不同深度**捕获标定图像
- 包含**边缘和中心**位置
- 确保**标定板平整**
- **多角度**覆盖工作区域

### 3. 运行维护
- 定期检查**标定精度**
- 监控**同步误差**
- 验证**深度精度**
- 更新**匹配参数**

## ⚠️ 故障排除

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| 无视差 | 深度图全黑 | 检查基线距离、重新标定 |
| 同步失败 | 时间戳差异大 | 调整同步容差、检查USB带宽 |
| 匹配失败 | 视差图噪声多 | 调整SGBM参数、改善光照 |
| 深度不准 | 3D坐标偏差大 | 重新标定、检查安装精度 |

## 📝 配置示例

完整的配置文件示例：
```yaml
# /opt/bamboo-cut/config/stereo_config.yaml
stereo_vision:
  cameras:
    left_device: "/dev/video0"
    right_device: "/dev/video2"
    width: 1280
    height: 720
    fps: 30
    sync_tolerance_ms: 10
  
  matching:
    num_disparities: 96
    block_size: 5
    use_wls_filter: true
    lambda: 8000.0
    sigma: 1.5
    min_confidence: 0.6
  
  calibration:
    file_path: "/opt/bamboo-cut/config/stereo_calibration.xml"
    board_size: [9, 6]
    square_size: 25.0
```

通过以上配置和步骤，您的双摄像头系统应该能够成功产生视差，并提供准确的3D定位信息。