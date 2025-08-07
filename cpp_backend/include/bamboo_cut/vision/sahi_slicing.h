#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <functional>
#include "bamboo_cut/core/types.h"

namespace bamboo_cut {
namespace vision {

/**
 * @brief 切片策略枚举
 */
enum class SliceStrategy {
    GRID,           // 网格切片
    RANDOM,         // 随机切片
    ADAPTIVE,       // 自适应切片
    PYRAMID         // 金字塔切片
};

/**
 * @brief 合并策略枚举
 */
enum class MergeStrategy {
    NMS,            // 非极大值抑制
    WEIGHTED_AVERAGE, // 加权平均
    CONFIDENCE_BASED, // 基于置信度
    HYBRID          // 混合策略
};

/**
 * @brief SAHI 切片配置
 */
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
    
    // 性能优化
    bool enable_parallel_processing{true}; // 启用并行处理
    int num_worker_threads{4};        // 工作线程数
    bool enable_memory_optimization{true}; // 启用内存优化
    
    // 高级参数
    float slice_confidence_boost{0.1f}; // 切片置信度提升
    bool enable_slice_filtering{true};   // 启用切片过滤
    float min_slice_confidence{0.1f};    // 最小切片置信度
    
    SAHIConfig() = default;
    bool validate() const;
};

/**
 * @brief 图像切片结构
 */
struct ImageSlice {
    cv::Mat image;                   // 切片图像
    cv::Rect roi;                    // 原始图像中的区域
    int slice_id;                    // 切片ID
    float confidence_boost;          // 置信度提升值
    
    ImageSlice() : slice_id(0), confidence_boost(1.0f) {}
    ImageSlice(const cv::Mat& img, const cv::Rect& r, int id, float boost = 1.0f)
        : image(img), roi(r), slice_id(id), confidence_boost(boost) {}
};

/**
 * @brief 切片检测结果
 */
struct SliceDetectionResult {
    std::vector<core::DetectionResult> detections; // 检测结果
    cv::Rect slice_roi;              // 切片在原始图像中的位置
    int slice_id;                    // 切片ID
    float processing_time_ms;        // 处理时间
    bool success;                    // 是否成功
    
    SliceDetectionResult() : slice_id(0), processing_time_ms(0.0f), success(false) {}
};

/**
 * @brief SAHI 切片推理器类
 * 
 * 实现切片辅助超推理，提升小目标和密集目标检测精度
 */
class SAHISlicing {
public:
    explicit SAHISlicing(const SAHIConfig& config = SAHIConfig{});
    ~SAHISlicing();

    // 禁用拷贝
    SAHISlicing(const SAHISlicing&) = delete;
    SAHISlicing& operator=(const SAHISlicing&) = delete;

    // 初始化和控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // 主要接口
    std::vector<core::DetectionResult> detect_with_slicing(
        const cv::Mat& input_image,
        std::function<std::vector<core::DetectionResult>(const cv::Mat&)> detector_callback);

    // 切片生成
    std::vector<ImageSlice> generate_slices(const cv::Mat& input_image);
    std::vector<ImageSlice> generate_grid_slices(const cv::Mat& input_image);
    std::vector<ImageSlice> generate_random_slices(const cv::Mat& input_image);
    std::vector<ImageSlice> generate_adaptive_slices(const cv::Mat& input_image);
    std::vector<ImageSlice> generate_pyramid_slices(const cv::Mat& input_image);

    // 结果合并
    std::vector<core::DetectionResult> merge_detections(
        const std::vector<SliceDetectionResult>& slice_results,
        const cv::Size& original_size);

    // 坐标转换
    core::DetectionResult transform_detection_to_original(
        const core::DetectionResult& detection,
        const cv::Rect& slice_roi,
        const cv::Size& original_size);

    // 配置管理
    void set_config(const SAHIConfig& config);
    SAHIConfig get_config() const { return config_; }

    // 性能统计
    struct PerformanceStats {
        uint64_t total_slices_generated{0};
        uint64_t total_detections_merged{0};
        double avg_slice_generation_time_ms{0.0};
        double avg_merge_time_ms{0.0};
        double avg_total_processing_time_ms{0.0};
        double avg_detections_per_slice{0.0};
        double avg_confidence_improvement{0.0};
    };
    PerformanceStats get_performance_stats() const;

private:
    // 内部方法
    bool validate_slice_size(const cv::Size& slice_size, const cv::Size& image_size);
    cv::Rect calculate_slice_roi(int row, int col, const cv::Size& slice_size, 
                                const cv::Size& image_size, float overlap_ratio);
    std::vector<cv::Rect> generate_slice_regions(const cv::Size& image_size);
    
    // 结果处理
    std::vector<core::DetectionResult> apply_nms_merge(
        const std::vector<core::DetectionResult>& detections);
    std::vector<core::DetectionResult> apply_weighted_merge(
        const std::vector<SliceDetectionResult>& slice_results);
    std::vector<core::DetectionResult> apply_confidence_merge(
        const std::vector<SliceDetectionResult>& slice_results);
    std::vector<core::DetectionResult> apply_hybrid_merge(
        const std::vector<SliceDetectionResult>& slice_results,
        const cv::Size& original_size);

    // 辅助计算
    float calculate_overlap_ratio(const cv::Rect& rect1, const cv::Rect& rect2);
    bool is_detection_valid(const core::DetectionResult& detection, 
                           const cv::Size& image_size);
    float boost_confidence(float original_confidence, float boost_factor);

    // 配置和状态
    SAHIConfig config_;
    bool initialized_{false};

    // 性能统计
    mutable std::mutex stats_mutex_;
    PerformanceStats performance_stats_;

    // 错误处理
    std::string last_error_;
    void set_error(const std::string& error);
};

/**
 * @brief SAHI 自适应切片器
 * 
 * 根据图像内容自适应生成切片
 */
class AdaptiveSlicer {
public:
    struct AdaptiveConfig {
        float content_threshold{0.1f};    // 内容密度阈值
        int min_content_area{1000};       // 最小内容区域
        float edge_density_weight{0.3f};  // 边缘密度权重
        float texture_weight{0.4f};       // 纹理权重
        float gradient_weight{0.3f};      // 梯度权重
        int max_slices_per_image{16};     // 每张图像最大切片数
        
        AdaptiveConfig() = default;
        bool validate() const;
    };

    explicit AdaptiveSlicer(const AdaptiveConfig& config = AdaptiveConfig{});
    ~AdaptiveSlicer();

    // 禁用拷贝
    AdaptiveSlicer(const AdaptiveSlicer&) = delete;
    AdaptiveSlicer& operator=(const AdaptiveSlicer&) = delete;

    // 自适应切片生成
    std::vector<ImageSlice> generate_adaptive_slices(const cv::Mat& input_image);
    
    // 内容分析
    cv::Mat analyze_content_density(const cv::Mat& image);
    cv::Mat analyze_edge_density(const cv::Mat& image);
    cv::Mat analyze_texture_density(const cv::Mat& image);
    cv::Mat analyze_gradient_density(const cv::Mat& image);

    // 区域分割
    std::vector<cv::Rect> segment_high_content_regions(const cv::Mat& density_map);
    std::vector<cv::Rect> optimize_slice_regions(const std::vector<cv::Rect>& regions,
                                                const cv::Size& image_size);

    // 配置管理
    void set_config(const AdaptiveConfig& config);
    AdaptiveConfig get_config() const { return config_; }

private:
    // 内部方法
    float calculate_region_density(const cv::Mat& density_map, const cv::Rect& region);
    cv::Rect expand_region(const cv::Rect& region, const cv::Size& image_size, 
                          float expansion_ratio = 0.1f);
    bool should_split_region(const cv::Rect& region, const cv::Mat& density_map);

    // 配置和状态
    AdaptiveConfig config_;
};

} // namespace vision
} // namespace bamboo_cut 