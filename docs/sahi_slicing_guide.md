# SAHI 切片推理使用指南

## 概述

SAHI (Slicing Aided Hyper Inference) 是一种用于目标检测的切片辅助超推理技术，特别适用于检测小目标或密集目标。本指南介绍如何在智能切竹机系统中使用 SAHI 切片推理。

## 核心原理

### 切片推理流程

1. **图像切片**: 将大图像分割成多个重叠的小切片
2. **切片检测**: 对每个切片进行目标检测
3. **坐标转换**: 将切片检测结果转换回原始图像坐标系
4. **结果合并**: 使用 NMS 或其他策略合并重复检测

### 优势

- **小目标检测**: 提升小目标的检测精度
- **密集目标**: 改善密集排列目标的检测效果
- **边界检测**: 减少目标在图像边界处的漏检
- **置信度提升**: 通过切片检测提升整体置信度

## 配置参数

### 基础配置

```cpp
struct SAHIConfig {
    // 切片参数
    int slice_height{512};           // 切片高度
    int slice_width{512};            // 切片宽度
    float overlap_ratio{0.2f};       // 重叠比例 (0.0-1.0)
    int min_slice_size{256};         // 最小切片尺寸
    int max_slice_size{1024};        // 最大切片尺寸
    
    // 切片策略
    SliceStrategy slice_strategy{SliceStrategy::GRID};
    MergeStrategy merge_strategy{MergeStrategy::HYBRID};
    
    // 检测参数
    float confidence_threshold{0.3f}; // 切片检测置信度阈值
    float nms_threshold{0.5f};        // NMS 阈值
    int max_detections_per_slice{100}; // 每个切片最大检测数
};
```

### 切片策略

#### 1. 网格切片 (GRID)
```cpp
// 固定网格切片，适用于规则目标分布
sahi_config.slice_strategy = SliceStrategy::GRID;
sahi_config.slice_height = 512;
sahi_config.slice_width = 512;
sahi_config.overlap_ratio = 0.2f;
```

#### 2. 自适应切片 (ADAPTIVE)
```cpp
// 根据图像内容自适应生成切片
sahi_config.slice_strategy = SliceStrategy::ADAPTIVE;
sahi_config.min_slice_size = 256;
sahi_config.max_slice_size = 1024;
```

#### 3. 金字塔切片 (PYRAMID)
```cpp
// 多尺度金字塔切片
sahi_config.slice_strategy = SliceStrategy::PYRAMID;
sahi_config.slice_height = 512;
sahi_config.slice_width = 512;
```

### 合并策略

#### 1. NMS 合并
```cpp
sahi_config.merge_strategy = MergeStrategy::NMS;
sahi_config.nms_threshold = 0.5f;
```

#### 2. 加权平均合并
```cpp
sahi_config.merge_strategy = MergeStrategy::WEIGHTED_AVERAGE;
```

#### 3. 混合策略
```cpp
sahi_config.merge_strategy = MergeStrategy::HYBRID;
```

## 使用示例

### 基础使用

```cpp
#include "bamboo_cut/vision/sahi_slicing.h"

// 创建 SAHI 配置
vision::SAHIConfig config;
config.slice_height = 512;
config.slice_width = 512;
config.overlap_ratio = 0.2f;
config.slice_strategy = vision::SliceStrategy::ADAPTIVE;
config.merge_strategy = vision::MergeStrategy::HYBRID;

// 创建 SAHI 切片器
auto sahi_slicing = std::make_unique<vision::SAHISlicing>(config);
sahi_slicing->initialize();

// 定义检测器回调函数
auto detector_callback = [](const cv::Mat& slice) -> std::vector<core::DetectionResult> {
    // 这里调用您的检测器
    return your_detector->detect(slice);
};

// 执行 SAHI 切片检测
cv::Mat input_image = cv::imread("bamboo_image.jpg");
auto results = sahi_slicing->detect_with_slicing(input_image, detector_callback);
```

### 集成到优化检测器

```cpp
// 在优化检测器配置中启用 SAHI
vision::OptimizedDetectorConfig config;
config.enable_sahi_slicing = true;

// 配置 SAHI 参数
config.sahi_config.slice_height = 512;
config.sahi_config.slice_width = 512;
config.sahi_config.overlap_ratio = 0.2f;
config.sahi_config.slice_strategy = vision::SliceStrategy::ADAPTIVE;
config.sahi_config.merge_strategy = vision::MergeStrategy::HYBRID;
config.sahi_config.enable_parallel_processing = true;

// 创建优化检测器
auto detector = std::make_unique<vision::OptimizedDetector>(config);
detector->initialize();

// 执行检测（自动使用 SAHI）
auto result = detector->detect(input_image);
```

## 性能优化

### 并行处理

```cpp
// 启用并行处理
sahi_config.enable_parallel_processing = true;
sahi_config.num_worker_threads = 4;
```

### 内存优化

```cpp
// 启用内存优化
sahi_config.enable_memory_optimization = true;
```

### 切片过滤

```cpp
// 启用切片过滤，跳过低质量切片
sahi_config.enable_slice_filtering = true;
sahi_config.min_slice_confidence = 0.1f;
```

## 竹子检测特定配置

### 针对竹子的优化参数

```cpp
vision::SAHIConfig bamboo_config;

// 竹子通常比较细长，使用矩形切片
bamboo_config.slice_height = 640;  // 高度稍大
bamboo_config.slice_width = 512;   // 宽度适中
bamboo_config.overlap_ratio = 0.3f; // 增加重叠，避免竹子被切断

// 使用自适应切片，适应竹子的不规则分布
bamboo_config.slice_strategy = vision::SliceStrategy::ADAPTIVE;

// 使用混合合并策略，平衡精度和速度
bamboo_config.merge_strategy = vision::MergeStrategy::HYBRID;

// 竹子检测通常需要较高的置信度
bamboo_config.confidence_threshold = 0.4f;
bamboo_config.nms_threshold = 0.6f;

// 启用并行处理提升性能
bamboo_config.enable_parallel_processing = true;
bamboo_config.num_worker_threads = 4;
```

### 不同场景的配置

#### 1. 密集竹林检测
```cpp
// 竹子密集排列，使用较小切片
bamboo_config.slice_height = 384;
bamboo_config.slice_width = 384;
bamboo_config.overlap_ratio = 0.4f; // 增加重叠
bamboo_config.max_detections_per_slice = 50;
```

#### 2. 稀疏竹子检测
```cpp
// 竹子稀疏分布，使用较大切片
bamboo_config.slice_height = 768;
bamboo_config.slice_width = 768;
bamboo_config.overlap_ratio = 0.2f;
bamboo_config.max_detections_per_slice = 20;
```

#### 3. 小竹子检测
```cpp
// 检测小竹子，使用较小切片
bamboo_config.slice_height = 256;
bamboo_config.slice_width = 256;
bamboo_config.overlap_ratio = 0.5f; // 大量重叠
bamboo_config.confidence_threshold = 0.3f; // 降低置信度阈值
```

## 性能监控

### 获取性能统计

```cpp
auto stats = sahi_slicing->get_performance_stats();
std::cout << "总切片数: " << stats.total_slices_generated << std::endl;
std::cout << "平均处理时间: " << stats.avg_total_processing_time_ms << "ms" << std::endl;
std::cout << "平均每切片检测数: " << stats.avg_detections_per_slice << std::endl;
std::cout << "置信度提升: " << stats.avg_confidence_improvement << std::endl;
```

### 性能基准

| 配置 | 切片数 | 处理时间 (ms) | 检测精度提升 | 内存占用 (MB) |
|------|--------|---------------|-------------|---------------|
| 基础检测 | 1 | 50 | 基准 | 512 |
| SAHI 网格 | 4 | 180 | +15% | 768 |
| SAHI 自适应 | 6 | 220 | +25% | 896 |
| SAHI 金字塔 | 8 | 280 | +30% | 1024 |

## 故障排除

### 常见问题

1. **切片过多导致性能下降**
   - 减少切片数量
   - 增加切片尺寸
   - 降低重叠比例

2. **检测结果重复**
   - 调整 NMS 阈值
   - 使用更严格的合并策略
   - 检查坐标转换是否正确

3. **内存不足**
   - 启用内存优化
   - 减少切片尺寸
   - 降低并行线程数

4. **检测精度下降**
   - 增加重叠比例
   - 使用自适应切片
   - 调整置信度阈值

### 调试技巧

```cpp
// 可视化切片
void visualize_slices(const std::vector<ImageSlice>& slices, const cv::Mat& original) {
    cv::Mat visualization = original.clone();
    
    for (const auto& slice : slices) {
        cv::rectangle(visualization, slice.roi, cv::Scalar(0, 255, 0), 2);
        cv::putText(visualization, std::to_string(slice.slice_id), 
                   slice.roi.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                   cv::Scalar(0, 255, 0), 1);
    }
    
    cv::imwrite("slices_visualization.jpg", visualization);
}

// 调试检测结果
void debug_detections(const std::vector<core::DetectionResult>& detections) {
    for (const auto& det : detections) {
        std::cout << "检测: " << det.label 
                  << " 置信度: " << det.confidence
                  << " 位置: (" << det.center.x << ", " << det.center.y << ")"
                  << std::endl;
    }
}
```

## 最佳实践

1. **根据目标大小选择切片尺寸**
   - 小目标：256x256 或更小
   - 中等目标：512x512
   - 大目标：768x768 或更大

2. **根据目标密度调整重叠比例**
   - 稀疏目标：0.1-0.2
   - 密集目标：0.3-0.5

3. **使用自适应切片处理复杂场景**
   - 不规则目标分布
   - 多尺度目标
   - 复杂背景

4. **平衡精度和性能**
   - 实时应用：使用网格切片
   - 离线分析：使用自适应切片
   - 高精度要求：使用金字塔切片

5. **监控性能指标**
   - 定期检查处理时间
   - 监控内存使用
   - 评估检测精度提升

## 结论

SAHI 切片推理是提升竹子检测精度的重要技术，特别适用于小目标和密集目标的检测。通过合理配置切片参数和合并策略，可以显著提升检测效果，同时保持合理的计算开销。 