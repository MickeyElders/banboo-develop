# Flutter嵌入式触摸界面实现指南

## 概述

本文档详细介绍了智能切竹机Flutter嵌入式触摸界面的实现，包括FFI调用C++推理服务、GPU加速主界面结构、以及完整的嵌入式触摸界面架构。

## 架构设计

### 整体架构

```
Flutter UI Layer (Dart)
    ↓ FFI Interface
C++ Backend Layer (C++)
    ↓ Hardware Interface
Hardware Layer (Camera, PLC, etc.)
```

### 核心组件

1. **FFI桥接层**: 实现Flutter与C++后端的通信
2. **GPU加速UI**: 使用CustomPainter和RepaintBoundary实现高性能渲染
3. **状态管理**: 使用Provider模式管理应用状态
4. **触摸界面**: 专为嵌入式设备设计的触摸友好界面

## FFI接口实现

### C++后端接口

#### 主要函数

```cpp
// 初始化推理服务
int32_t initialize_inference_service();

// 检测竹子
int32_t detect_bamboo(uint8_t* image_data, int32_t width, int32_t height, 
                     uint8_t** result_data, int32_t* result_size);

// 获取系统状态
int32_t get_system_status(uint8_t* status_data, int32_t* status_size);

// 摄像头控制
int32_t start_camera_capture();
int32_t stop_camera_capture();
int32_t get_camera_frame(uint8_t** frame_data, int32_t* width, 
                        int32_t* height, int32_t* channels);

// 切割参数设置
int32_t set_cutting_parameters(float x_coordinate, int32_t blade_number, 
                              float quality_threshold);

// 紧急停止
int32_t emergency_stop();

// 关闭服务
int32_t shutdown_inference_service();
```

#### 内存管理

```cpp
// 内存分配
void* allocate_memory(size_t size);

// 内存释放
void free_memory(void* ptr);
```

### Flutter FFI桥接

#### 函数签名定义

```dart
// C++后端函数签名定义
typedef InitializeInferenceServiceNative = ffi.Int32 Function();
typedef InitializeInferenceServiceDart = int Function();

typedef DetectBambooNative = ffi.Int32 Function(
  ffi.Pointer<ffi.Uint8> image_data,
  ffi.Int32 width,
  ffi.Int32 height,
  ffi.Pointer<ffi.Pointer<ffi.Uint8>> result_data,
  ffi.Pointer<ffi.Int32> result_size,
);
```

#### 动态库加载

```dart
static ffi.DynamicLibrary _loadLibrary() {
  if (Platform.isLinux) {
    return ffi.DynamicLibrary.open('libbamboo_cut_backend.so');
  } else {
    throw UnsupportedError('Unsupported platform');
  }
}
```

## GPU加速UI实现

### 视频显示组件

#### GPU加速图像渲染

```dart
class GPUAcceleratedVideoWidget extends StatefulWidget {
  // 使用CustomPainter实现GPU加速渲染
  return CustomPaint(
    painter: _VideoPainter(_currentImage),
    size: Size.infinite,
  );
}
```

#### 图像解码优化

```dart
// 使用GPU加速解码图像数据
final codec = await ui.instantiateImageCodec(
  frameData,
  targetWidth: 1920,
  targetHeight: 1080,
);
```

### 检测结果叠加

#### 实时绘制检测框

```dart
class _DetectionPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    for (final detection in detections) {
      _drawDetectionBox(canvas, size, detection);
      _drawDetectionLabel(canvas, size, detection);
    }
  }
}
```

#### 性能优化

```dart
// 使用RepaintBoundary隔离重绘区域
return RepaintBoundary(
  child: CustomPaint(
    painter: _DetectionPainter(detections),
    size: Size.infinite,
  ),
);
```

## 状态管理

### Provider架构

#### 系统状态管理

```dart
class SystemStateProvider extends ChangeNotifier {
  Map<String, dynamic> _currentStatus = {
    'system_status': 0,
    'fps': 0.0,
    'cpu_usage': 0.0,
    'memory_usage': 0.0,
    'gpu_usage': 0.0,
    'plc_connected': false,
    'camera_connected': false,
    'heartbeat_count': 0,
    'timestamp': DateTime.now(),
    'emergency_stop': false,
  };
}
```

#### 检测结果管理

```dart
class DetectionProvider extends ChangeNotifier {
  Uint8List? _currentVideoFrame;
  List<Map<String, dynamic>> _currentDetections = [];
  Map<String, dynamic> _lastDetectionResult = {};
  bool _isDetecting = false;
  double _detectionConfidence = 0.0;
}
```

### 状态更新机制

```dart
// 定时更新系统状态
_statusTimer = Timer.periodic(const Duration(milliseconds: 100), (timer) {
  _updateStatus();
});

// 定时执行检测
_detectionTimer = Timer.periodic(const Duration(milliseconds: 33), (timer) {
  _performDetection();
});
```

## 触摸界面设计

### 主界面布局

#### 屏幕结构

```
┌─────────────────────────────────────────────────────────────┐
│                    状态面板 (StatusPanel)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                                                             │
│                 GPU加速视频显示区域                          │
│                                                             │
│                                                             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ 控制面板 (ControlPanel) │ 检测结果叠加 (DetectionOverlay)   │
└─────────────────────────────────────────────────────────────┘
```

#### 组件层次

```dart
Stack(
  children: [
    // 主视频显示区域
    Positioned.fill(
      child: RepaintBoundary(
        child: GPUAcceleratedVideoWidget(),
      ),
    ),
    
    // 检测结果叠加层
    Positioned.fill(
      child: RepaintBoundary(
        child: DetectionOverlay(),
      ),
    ),
    
    // 顶部状态栏
    Positioned(
      top: 0,
      left: 0,
      right: 0,
      child: RepaintBoundary(
        child: StatusPanel(),
      ),
    ),
    
    // 右侧控制面板
    Positioned(
      top: 100,
      right: 0,
      bottom: 100,
      child: RepaintBoundary(
        child: ControlPanel(),
      ),
    ),
    
    // 紧急停止按钮
    Positioned(
      bottom: 20,
      right: 20,
      child: RepaintBoundary(
        child: EmergencyButton(),
      ),
    ),
  ],
)
```

### 控制面板设计

#### 功能模块

1. **运行模式切换**: 自动/手动模式
2. **X坐标控制**: 滑块和按钮控制
3. **刀片选择**: 刀片1/刀片2/双刀片
4. **质量阈值**: 检测质量阈值设置
5. **操作按钮**: 开始/停止切割

#### 触摸优化

```dart
// 大尺寸触摸目标
Container(
  padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
  child: Column(
    children: [
      Icon(icon, size: 20),
      Text(label, fontSize: 12),
    ],
  ),
)
```

### 紧急停止按钮

#### 视觉设计

```dart
Container(
  width: 80,
  height: 80,
  decoration: BoxDecoration(
    shape: BoxShape.circle,
    color: Colors.red[700],
    border: Border.all(color: Colors.white, width: 4),
    boxShadow: [
      BoxShadow(
        color: Colors.red.withOpacity(0.5),
        blurRadius: 20,
        spreadRadius: 5,
      ),
    ],
  ),
  child: Icon(Icons.emergency, color: Colors.white, size: 40),
)
```

#### 触觉反馈

```dart
void _onTapDown() {
  HapticFeedback.heavyImpact();
}

void _onTapUp() {
  HapticFeedback.vibrate();
}
```

## 性能优化

### GPU加速策略

1. **RepaintBoundary隔离**: 减少不必要的重绘
2. **CustomPainter优化**: 直接GPU渲染
3. **图像解码加速**: 硬件加速解码
4. **内存管理**: 及时释放图像资源

### 渲染优化

```dart
// 使用FilterQuality.high提升渲染质量
final paint = Paint()
  ..filterQuality = FilterQuality.high
  ..isAntiAlias = true;

// 计算缩放比例保持宽高比
final imageAspectRatio = image!.width / image!.height;
final canvasAspectRatio = size.width / size.height;
```

### 内存优化

```dart
// 及时释放图像资源
@override
void dispose() {
  _currentImage?.dispose();
  super.dispose();
}

// 避免内存泄漏
try {
  // 操作
} finally {
  calloc.free(imagePtr);
  calloc.free(resultPtr);
}
```

## 部署配置

### Linux嵌入式配置

#### 系统要求

- Ubuntu 20.04+ 或 Debian 11+
- Flutter 3.10.0+
- OpenCV 4.5+
- TensorRT 8.0+
- CUDA 11.0+ (可选)

#### 编译配置

```yaml
# pubspec.yaml
flutter_platforms:
  linux:
    desktop: true
    embedded: true
    wayland: true
    x11: true
```

#### 启动配置

```bash
# 设置全屏模式
flutter run -d linux --dart-define=FLUTTER_WEB_USE_SKIA=true

# 生产构建
flutter build linux --release
```

### 系统集成

#### 自启动配置

```ini
# /etc/systemd/system/bamboo-cut.service
[Unit]
Description=Bamboo Cut Flutter App
After=network.target

[Service]
Type=simple
User=bamboo
Environment=DISPLAY=:0
ExecStart=/usr/local/bin/bamboo-cut
Restart=always

[Install]
WantedBy=multi-user.target
```

#### 权限配置

```bash
# 摄像头访问权限
sudo usermod -a -G video bamboo

# 串口访问权限
sudo usermod -a -G dialout bamboo
```

## 故障排除

### 常见问题

1. **FFI库加载失败**
   - 检查库文件路径
   - 确认库文件权限
   - 验证依赖库安装

2. **GPU渲染性能问题**
   - 检查显卡驱动
   - 验证OpenGL支持
   - 调整渲染质量设置

3. **摄像头访问失败**
   - 检查设备权限
   - 验证设备路径
   - 确认V4L2支持

4. **内存泄漏**
   - 检查图像资源释放
   - 验证FFI内存管理
   - 监控内存使用

### 调试工具

```bash
# 性能监控
flutter run --profile

# 内存分析
flutter run --trace-startup

# 调试模式
flutter run --debug
```

## 开发指南

### 代码规范

1. **命名规范**: 使用驼峰命名法
2. **注释规范**: 中文注释，详细说明
3. **错误处理**: 完善的异常处理机制
4. **性能考虑**: 优先考虑性能优化

### 测试策略

1. **单元测试**: 核心逻辑测试
2. **集成测试**: FFI接口测试
3. **性能测试**: GPU渲染性能测试
4. **兼容性测试**: 不同硬件平台测试

### 版本管理

```bash
# 版本标签
git tag -a v1.0.0 -m "Flutter嵌入式界面v1.0.0"

# 发布构建
flutter build linux --release --dart-define=FLUTTER_WEB_USE_SKIA=true
```

## 总结

Flutter嵌入式触摸界面实现了以下核心功能：

1. **高性能GPU加速**: 使用CustomPainter和RepaintBoundary实现流畅的视频显示
2. **FFI无缝集成**: 通过FFI接口调用C++推理服务
3. **触摸友好界面**: 专为嵌入式设备设计的触摸操作界面
4. **实时状态监控**: 实时显示系统状态和性能指标
5. **紧急安全控制**: 一键紧急停止功能

该实现为智能切竹机提供了现代化、高性能的嵌入式触摸界面，满足了工业应用的需求。 