#include "bamboo_cut/vision/sahi_slicing.h"
#include <iostream>
#include <chrono>
#include <mutex>
#include <algorithm>

namespace bamboo_cut {
namespace vision {

// SAHIConfig validation implementation
bool SAHIConfig::validate() const {
    if (slice_height <= 0 || slice_width <= 0) {
        return false;
    }
    if (overlap_ratio < 0.0f || overlap_ratio > 1.0f) {
        return false;
    }
    if (confidence_threshold < 0.0f || confidence_threshold > 1.0f) {
        return false;
    }
    if (nms_threshold < 0.0f || nms_threshold > 1.0f) {
        return false;
    }
    if (min_slice_size <= 0 || max_slice_size <= min_slice_size) {
        return false;
    }
    return true;
}

SAHISlicing::SAHISlicing() : config_(), initialized_(false) {
    std::cout << "创建SAHISlicing实例" << std::endl;
}

SAHISlicing::SAHISlicing(const SAHIConfig& config) : config_(config), initialized_(false) {
    std::cout << "创建SAHISlicing实例，使用自定义配置" << std::endl;
}

SAHISlicing::~SAHISlicing() {
    shutdown();
    std::cout << "销毁SAHISlicing实例" << std::endl;
}

bool SAHISlicing::initialize() {
    if (initialized_) {
        std::cout << "SAHISlicing已初始化" << std::endl;
        return true;
    }
    
    std::cout << "初始化SAHISlicing..." << std::endl;
    
    try {
        // 验证配置
        if (!config_.validate()) {
            std::cerr << "SAHISlicing配置无效" << std::endl;
            return false;
        }
        
        initialized_ = true;
        std::cout << "SAHISlicing初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "SAHISlicing初始化异常: " << e.what() << std::endl;
        return false;
    }
}

void SAHISlicing::shutdown() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "关闭SAHISlicing..." << std::endl;
    initialized_ = false;
    std::cout << "SAHISlicing已关闭" << std::endl;
}

std::vector<core::DetectionResult> SAHISlicing::detect_with_slicing(
    const cv::Mat& input_image,
    std::function<std::vector<core::DetectionResult>(const cv::Mat&)> detector_callback) {
    
    if (!initialized_) {
        std::cerr << "SAHISlicing未初始化" << std::endl;
        return {};
    }
    
    try {
        // 生成切片
        auto slices = generate_slices(input_image);
        
        // 对每个切片进行检测
        std::vector<SliceDetectionResult> slice_results;
        for (const auto& slice : slices) {
            SliceDetectionResult result;
            result.slice_roi = slice.roi;
            result.slice_id = slice.slice_id;
            
            // 调用检测回调
            result.detections = detector_callback(slice.image);
            result.success = true;
            
            slice_results.push_back(result);
        }
        
        // 合并检测结果
        return merge_detections(slice_results, input_image.size());
        
    } catch (const std::exception& e) {
        std::cerr << "SAHI切片检测异常: " << e.what() << std::endl;
        return {};
    }
}

std::vector<ImageSlice> SAHISlicing::generate_slices(const cv::Mat& input_image) {
    switch (config_.slice_strategy) {
        case SliceStrategy::GRID:
            return generate_grid_slices(input_image);
        case SliceStrategy::RANDOM:
            return generate_random_slices(input_image);
        case SliceStrategy::ADAPTIVE:
            return generate_adaptive_slices(input_image);
        case SliceStrategy::PYRAMID:
            return generate_pyramid_slices(input_image);
        default:
            return generate_grid_slices(input_image);
    }
}

std::vector<ImageSlice> SAHISlicing::generate_grid_slices(const cv::Mat& input_image) {
    std::vector<ImageSlice> slices;
    
    if (input_image.empty()) {
        return slices;
    }
    
    int step_x = static_cast<int>(config_.slice_width * (1.0f - config_.overlap_ratio));
    int step_y = static_cast<int>(config_.slice_height * (1.0f - config_.overlap_ratio));
    
    int slice_id = 0;
    for (int y = 0; y < input_image.rows; y += step_y) {
        for (int x = 0; x < input_image.cols; x += step_x) {
            cv::Rect roi(x, y, 
                        std::min(config_.slice_width, input_image.cols - x),
                        std::min(config_.slice_height, input_image.rows - y));
            
            if (roi.width >= config_.min_slice_size && roi.height >= config_.min_slice_size) {
                cv::Mat slice_image = input_image(roi);
                slices.emplace_back(slice_image, roi, slice_id++);
            }
        }
    }
    
    return slices;
}

std::vector<ImageSlice> SAHISlicing::generate_random_slices(const cv::Mat& input_image) {
    // 简化实现：返回网格切片
    return generate_grid_slices(input_image);
}

std::vector<ImageSlice> SAHISlicing::generate_adaptive_slices(const cv::Mat& input_image) {
    // 简化实现：返回网格切片
    return generate_grid_slices(input_image);
}

std::vector<ImageSlice> SAHISlicing::generate_pyramid_slices(const cv::Mat& input_image) {
    // 简化实现：返回网格切片
    return generate_grid_slices(input_image);
}

std::vector<core::DetectionResult> SAHISlicing::merge_detections(
    const std::vector<SliceDetectionResult>& slice_results,
    const cv::Size& original_size) {
    
    if (slice_results.empty()) {
        return {};
    }
    
    // 收集所有检测结果
    std::vector<core::DetectionResult> all_detections;
    for (const auto& slice_result : slice_results) {
        for (const auto& detection : slice_result.detections) {
            // 转换坐标到原图坐标系
            auto transformed = transform_detection_to_original(detection, slice_result.slice_roi, original_size);
            all_detections.push_back(transformed);
        }
    }
    
    // 应用合并策略
    switch (config_.merge_strategy) {
        case MergeStrategy::NMS:
            return apply_nms_merge(all_detections);
        case MergeStrategy::WEIGHTED_AVERAGE:
            return apply_weighted_merge(slice_results);
        case MergeStrategy::CONFIDENCE_BASED:
            return apply_confidence_merge(slice_results);
        case MergeStrategy::HYBRID:
            return apply_hybrid_merge(slice_results, original_size);
        default:
            return apply_nms_merge(all_detections);
    }
}

core::DetectionResult SAHISlicing::transform_detection_to_original(
    const core::DetectionResult& detection,
    const cv::Rect& slice_roi,
    const cv::Size& original_size) {
    
    core::DetectionResult transformed = detection;
    
    // 调整边界框坐标
    transformed.bounding_box.x += slice_roi.x;
    transformed.bounding_box.y += slice_roi.y;
    
    // 确保坐标在原图范围内
    transformed.bounding_box.x = std::max(0, std::min(transformed.bounding_box.x, original_size.width));
    transformed.bounding_box.y = std::max(0, std::min(transformed.bounding_box.y, original_size.height));
    transformed.bounding_box.width = std::min(transformed.bounding_box.width, 
                                            original_size.width - transformed.bounding_box.x);
    transformed.bounding_box.height = std::min(transformed.bounding_box.height, 
                                             original_size.height - transformed.bounding_box.y);
    
    return transformed;
}

void SAHISlicing::set_config(const SAHIConfig& config) {
    config_ = config;
}

SAHISlicing::PerformanceStats SAHISlicing::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return performance_stats_;
}

std::vector<core::DetectionResult> SAHISlicing::apply_nms_merge(
    const std::vector<core::DetectionResult>& detections) {
    // 简化的NMS实现
    std::vector<core::DetectionResult> merged;
    
    for (const auto& detection : detections) {
        if (detection.confidence >= config_.confidence_threshold) {
            merged.push_back(detection);
        }
    }
    
    return merged;
}

std::vector<core::DetectionResult> SAHISlicing::apply_weighted_merge(
    const std::vector<SliceDetectionResult>& slice_results) {
    // 简化实现
    std::vector<core::DetectionResult> merged;
    
    for (const auto& slice_result : slice_results) {
        for (const auto& detection : slice_result.detections) {
            if (detection.confidence >= config_.confidence_threshold) {
                merged.push_back(detection);
            }
        }
    }
    
    return merged;
}

std::vector<core::DetectionResult> SAHISlicing::apply_confidence_merge(
    const std::vector<SliceDetectionResult>& slice_results) {
    // 简化实现
    return apply_weighted_merge(slice_results);
}

std::vector<core::DetectionResult> SAHISlicing::apply_hybrid_merge(
    const std::vector<SliceDetectionResult>& slice_results,
    const cv::Size& original_size) {
    // 简化实现
    return apply_weighted_merge(slice_results);
}

void SAHISlicing::set_error(const std::string& error) {
    last_error_ = error;
    std::cerr << "SAHISlicing错误: " << error << std::endl;
}

// AdaptiveSlicer implementation
bool AdaptiveSlicer::AdaptiveConfig::validate() const {
    if (content_threshold < 0.0f || content_threshold > 1.0f) {
        return false;
    }
    if (min_content_area <= 0) {
        return false;
    }
    if (max_slices_per_image <= 0) {
        return false;
    }
    return true;
}

AdaptiveSlicer::AdaptiveSlicer() : config_() {}

AdaptiveSlicer::AdaptiveSlicer(const AdaptiveConfig& config) : config_(config) {}

AdaptiveSlicer::~AdaptiveSlicer() {}

std::vector<ImageSlice> AdaptiveSlicer::generate_adaptive_slices(const cv::Mat& input_image) {
    // 简化实现：返回一个包含整个图像的切片
    std::vector<ImageSlice> slices;
    if (!input_image.empty()) {
        cv::Rect roi(0, 0, input_image.cols, input_image.rows);
        slices.emplace_back(input_image, roi, 0);
    }
    return slices;
}

cv::Mat AdaptiveSlicer::analyze_content_density(const cv::Mat& image) {
    cv::Mat density;
    cv::cvtColor(image, density, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(density, density, cv::Size(5, 5), 1.0);
    return density;
}

cv::Mat AdaptiveSlicer::analyze_edge_density(const cv::Mat& image) {
    cv::Mat edges;
    cv::Canny(image, edges, 50, 150);
    return edges;
}

cv::Mat AdaptiveSlicer::analyze_texture_density(const cv::Mat& image) {
    return analyze_content_density(image);
}

cv::Mat AdaptiveSlicer::analyze_gradient_density(const cv::Mat& image) {
    cv::Mat grad_x, grad_y, grad;
    cv::Sobel(image, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(image, grad_y, CV_64F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad);
    return grad;
}

std::vector<cv::Rect> AdaptiveSlicer::segment_high_content_regions(const cv::Mat& density_map) {
    // 简化实现：返回整个图像区域
    std::vector<cv::Rect> regions;
    regions.emplace_back(0, 0, density_map.cols, density_map.rows);
    return regions;
}

std::vector<cv::Rect> AdaptiveSlicer::optimize_slice_regions(const std::vector<cv::Rect>& regions,
                                                           const cv::Size& image_size) {
    return regions;
}

void AdaptiveSlicer::set_config(const AdaptiveConfig& config) {
    config_ = config;
}

AdaptiveSlicer::AdaptiveConfig AdaptiveSlicer::get_config() const {
    return config_;
}

float AdaptiveSlicer::calculate_region_density(const cv::Mat& density_map, const cv::Rect& region) {
    if (region.x >= 0 && region.y >= 0 && 
        region.x + region.width <= density_map.cols && 
        region.y + region.height <= density_map.rows) {
        cv::Mat roi = density_map(region);
        return static_cast<float>(cv::sum(roi)[0]) / (roi.rows * roi.cols);
    }
    return 0.0f;
}

cv::Rect AdaptiveSlicer::expand_region(const cv::Rect& region, const cv::Size& image_size, 
                                     float expansion_ratio) {
    int expand_x = static_cast<int>(region.width * expansion_ratio);
    int expand_y = static_cast<int>(region.height * expansion_ratio);
    
    cv::Rect expanded(region.x - expand_x, region.y - expand_y, 
                     region.width + 2 * expand_x, region.height + 2 * expand_y);
    
    // 确保在图像边界内
    expanded.x = std::max(0, expanded.x);
    expanded.y = std::max(0, expanded.y);
    expanded.width = std::min(expanded.width, image_size.width - expanded.x);
    expanded.height = std::min(expanded.height, image_size.height - expanded.y);
    
    return expanded;
}

bool AdaptiveSlicer::should_split_region(const cv::Rect& region, const cv::Mat& density_map) {
    float density = calculate_region_density(density_map, region);
    return density > config_.content_threshold && 
           (region.width > 512 || region.height > 512);
}

} // namespace vision
} // namespace bamboo_cut