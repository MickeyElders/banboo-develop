#include "bamboo_cut/vision/optimized_detector.h"
#include "bamboo_cut/core/logger.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <mutex>
#include <functional>

namespace bamboo_cut {
namespace vision {

OptimizedDetector::OptimizedDetector(const OptimizedDetectorConfig& config) 
    : config_(config), initialized_(false) {
    LOG_INFO("创建OptimizedDetector实例");
}

OptimizedDetector::~OptimizedDetector() {
    LOG_INFO("销毁OptimizedDetector实例");
}

bool OptimizedDetector::initialize() {
    LOG_INFO("初始化OptimizedDetector");
    
    try {
        // 检查配置
        if (config_.model_path.empty()) {
            LOG_ERROR("模型路径为空");
            return false;
        }
        
            // 初始化基础检测器
    if (!initialize_base_detector()) {
        LOG_ERROR("基础检测器初始化失败");
        return false;
    }
    
    // 初始化NAM注意力模块
    nam_attention_ = std::make_unique<NAMAttention>(config_.nam_config);
    if (!nam_attention_->initialize()) {
        LOG_ERROR("NAM注意力模块初始化失败");
        return false;
    }
    
    // 初始化GhostConv模块
    ghost_conv_ = std::make_unique<GhostConv>(config_.ghost_conv_config);
    if (!ghost_conv_->initialize()) {
        LOG_ERROR("GhostConv模块初始化失败");
        return false;
    }
    
    // 初始化VoV-GSCSP模块
    vov_gscsp_ = std::make_unique<VoVGSCSP>(config_.vov_gscsp_config);
    if (!vov_gscsp_->initialize()) {
        LOG_ERROR("VoV-GSCSP模块初始化失败");
        return false;
    }
        
        // 初始化优化组件
        if (!initialize_optimizations()) {
            LOG_ERROR("优化组件初始化失败");
            return false;
        }
        
        initialized_ = true;
        LOG_INFO("OptimizedDetector初始化成功");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("OptimizedDetector初始化异常: {}", e.what());
        return false;
    }
}

core::DetectionResult OptimizedDetector::detect(const cv::Mat& image) {
    core::DetectionResult result;
    result.confidence = 0.0f;
    
    if (!initialized_) {
        LOG_ERROR("检测器未初始化");
        return result;
    }
    
    if (image.empty()) {
        LOG_ERROR("输入图像为空");
        return result;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 应用优化预处理
        cv::Mat optimized_image = apply_optimizations(image);
        
        // 应用NAM注意力增强特征
        cv::Mat enhanced_image = apply_nam_attention(optimized_image);
        
        // 应用GhostConv减少计算量
        cv::Mat ghost_enhanced = apply_ghost_conv(enhanced_image);
        
        // 应用VoV-GSCSP压缩颈部
        cv::Mat compressed_features = apply_vov_gscsp(ghost_enhanced);
        
        // 执行检测
        result = base_detector_->detect(compressed_features);
        
        // 应用后处理优化
        result = apply_post_processing_optimizations(result);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // 更新性能统计
        stats_.total_frames++;
        stats_.processed_frames++;
        stats_.detection_count++;
        stats_.avg_processing_time_ms = duration.count() / 1000.0;
        stats_.fps = 1000.0 / stats_.avg_processing_time_ms;
        stats_.last_update = std::chrono::steady_clock::now();
        
    } catch (const std::exception& e) {
        LOG_ERROR("检测异常: {}", e.what());
        result.confidence = 0.0f;
    }
    
    return result;
}

std::vector<core::DetectionResult> OptimizedDetector::detect_batch(const std::vector<cv::Mat>& images) {
    std::vector<core::DetectionResult> results;
    results.reserve(images.size());
    
    for (const auto& image : images) {
        results.push_back(detect(image));
    }
    
    return results;
}

std::string OptimizedDetector::get_model_info() const {
    std::string info = "OptimizedDetector - ";
    info += "模型: " + config_.model_path + ", ";
    info += "优化级别: " + std::to_string(config_.optimization_level) + ", ";
    info += "SAHI切片: " + (config_.enable_sahi_slicing ? "启用" : "禁用");
    return info;
}

OptimizedDetector::PerformanceStats OptimizedDetector::get_performance_stats() const {
    return stats_;
}

void OptimizedDetector::reset_performance_stats() {
    stats_ = PerformanceStats{};
}

bool OptimizedDetector::initialize_base_detector() {
    try {
        // 创建基础检测器配置
        BambooDetector::Config base_config;
        base_config.model_path = config_.model_path;
        base_config.input_width = config_.input_width;
        base_config.input_height = config_.input_height;
        base_config.confidence_threshold = config_.confidence_threshold;
        base_config.nms_threshold = config_.nms_threshold;
        base_config.max_detections = config_.max_detections;
        base_config.use_tensorrt = config_.use_tensorrt;
        base_config.use_fp16 = config_.use_fp16;
        
        // 创建基础检测器
        base_detector_ = std::make_unique<BambooDetector>(base_config);
        
        if (!base_detector_->initialize()) {
            LOG_ERROR("基础检测器初始化失败");
            return false;
        }
        
        LOG_INFO("基础检测器初始化成功");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("基础检测器初始化异常: {}", e.what());
        return false;
    }
}

bool OptimizedDetector::initialize_optimizations() {
    try {
        // 初始化SAHI切片
        if (config_.enable_sahi_slicing) {
            if (!initialize_sahi_slicing()) {
                LOG_WARN("SAHI切片初始化失败，将禁用此功能");
                config_.enable_sahi_slicing = false;
            }
        }
        
        // 初始化其他优化组件
        if (config_.optimization_level > 0) {
            if (!initialize_advanced_optimizations()) {
                LOG_WARN("高级优化初始化失败，将使用基础优化");
                config_.optimization_level = 0;
            }
        }
        
        LOG_INFO("优化组件初始化完成");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("优化组件初始化异常: {}", e.what());
        return false;
    }
}

bool OptimizedDetector::initialize_sahi_slicing() {
    try {
        // 创建SAHI切片配置
        SAHIConfig sahi_config;
        sahi_config.slice_height = config_.sahi_slice_height;
        sahi_config.slice_width = config_.sahi_slice_width;
        sahi_config.overlap_ratio = config_.sahi_overlap_ratio;
        sahi_config.confidence_threshold = config_.confidence_threshold;
        
        // 创建SAHI切片器
        sahi_slicer_ = std::make_unique<SAHISlicing>(sahi_config);
        
        LOG_INFO("SAHI切片初始化成功");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("SAHI切片初始化异常: {}", e.what());
        return false;
    }
}

bool OptimizedDetector::initialize_advanced_optimizations() {
    try {
        // 这里可以初始化其他高级优化组件
        // 比如NAM注意力、Wise-IoU等
        
        LOG_INFO("高级优化初始化成功");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("高级优化初始化异常: {}", e.what());
        return false;
    }
}

cv::Mat OptimizedDetector::apply_optimizations(const cv::Mat& image) {
    cv::Mat optimized_image = image.clone();
    
    // 应用图像预处理优化
    if (config_.optimization_level > 0) {
        // 图像增强
        optimized_image = apply_image_enhancement(optimized_image);
        
        // 噪声抑制
        if (config_.enable_noise_reduction) {
            optimized_image = apply_noise_reduction(optimized_image);
        }
    }
    
    return optimized_image;
}

cv::Mat OptimizedDetector::apply_image_enhancement(const cv::Mat& image) {
    cv::Mat enhanced = image.clone();
    
    // 直方图均衡化
    if (image.channels() == 3) {
        cv::cvtColor(enhanced, enhanced, cv::COLOR_BGR2YUV);
        std::vector<cv::Mat> channels;
        cv::split(enhanced, channels);
        cv::equalizeHist(channels[0], channels[0]);
        cv::merge(channels, enhanced);
        cv::cvtColor(enhanced, enhanced, cv::COLOR_YUV2BGR);
    } else if (image.channels() == 1) {
        cv::equalizeHist(enhanced, enhanced);
    }
    
    return enhanced;
}

cv::Mat OptimizedDetector::apply_noise_reduction(const cv::Mat& image) {
    cv::Mat denoised;
    cv::bilateralFilter(image, denoised, 9, 75, 75);
    return denoised;
}

core::DetectionResult OptimizedDetector::apply_post_processing_optimizations(const core::DetectionResult& result) {
    core::DetectionResult optimized_result = result;
    
    // 应用后处理优化
    if (config_.optimization_level > 0) {
        // 结果过滤
        optimized_result = apply_result_filtering(optimized_result);
        
        // 结果排序
        optimized_result = apply_result_sorting(optimized_result);
    }
    
    return optimized_result;
}

core::DetectionResult OptimizedDetector::apply_result_filtering(const core::DetectionResult& result) {
    core::DetectionResult filtered_result = result;
    
    // 过滤低置信度的检测结果
    if (result.confidence >= config_.confidence_threshold) {
        return filtered_result;
    }
    
    // 如果置信度低于阈值，返回一个空的检测结果
    filtered_result.confidence = 0.0f;
    filtered_result.center = core::Point2D();
    filtered_result.bounding_box = core::Rectangle();
    filtered_result.label = "";
    filtered_result.class_id = 0;
    
    return filtered_result;
}

core::DetectionResult OptimizedDetector::apply_result_sorting(const core::DetectionResult& result) {
    // 由于单个DetectionResult已经包含了所有信息，不需要排序
    return result;
}

cv::Mat OptimizedDetector::apply_nam_attention(const cv::Mat& features) {
    if (!nam_attention_ || !nam_attention_->is_initialized()) {
        LOG_WARN("NAM注意力模块未初始化，返回原始特征");
        return features;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 应用NAM注意力
        cv::Mat enhanced_features = nam_attention_->forward(features);
        
        // 更新性能统计
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            performance_stats_.nam_stats = nam_attention_->get_performance_stats();
        }
        
        LOG_DEBUG("NAM注意力处理完成，耗时: {}ms", duration.count() / 1000.0);
        return enhanced_features;
        
    } catch (const std::exception& e) {
        LOG_ERROR("NAM注意力处理异常: {}", e.what());
        return features;
    }
}

std::vector<cv::Mat> OptimizedDetector::apply_nam_attention_batch(const std::vector<cv::Mat>& features) {
    if (!nam_attention_ || !nam_attention_->is_initialized()) {
        LOG_WARN("NAM注意力模块未初始化，返回原始特征");
        return features;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 批量应用NAM注意力
        std::vector<cv::Mat> enhanced_features = nam_attention_->forward_batch(features);
        
        // 更新性能统计
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            performance_stats_.nam_stats = nam_attention_->get_performance_stats();
        }
        
        LOG_DEBUG("NAM注意力批量处理完成，耗时: {}ms", duration.count() / 1000.0);
        return enhanced_features;
        
    } catch (const std::exception& e) {
        LOG_ERROR("NAM注意力批量处理异常: {}", e.what());
        return features;
    }
}

cv::Mat OptimizedDetector::apply_ghost_conv(const cv::Mat& features) {
    if (!ghost_conv_ || !ghost_conv_->is_initialized()) {
        LOG_WARN("GhostConv模块未初始化，返回原始特征");
        return features;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 应用GhostConv
        cv::Mat ghost_features = ghost_conv_->forward(features);
        
        // 更新性能统计
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            performance_stats_.ghost_conv_stats = ghost_conv_->get_performance_stats();
        }
        
        LOG_DEBUG("GhostConv处理完成，耗时: {}ms", duration.count() / 1000.0);
        return ghost_features;
        
    } catch (const std::exception& e) {
        LOG_ERROR("GhostConv处理异常: {}", e.what());
        return features;
    }
}

cv::Mat OptimizedDetector::apply_vov_gscsp(const cv::Mat& features) {
    if (!vov_gscsp_ || !vov_gscsp_->is_initialized()) {
        LOG_WARN("VoV-GSCSP模块未初始化，返回原始特征");
        return features;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 应用VoV-GSCSP
        cv::Mat compressed_features = vov_gscsp_->forward(features);
        
        // 更新性能统计
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            performance_stats_.vov_gscsp_stats = vov_gscsp_->get_performance_stats();
        }
        
        LOG_DEBUG("VoV-GSCSP处理完成，耗时: {}ms", duration.count() / 1000.0);
        return compressed_features;
        
    } catch (const std::exception& e) {
        LOG_ERROR("VoV-GSCSP处理异常: {}", e.what());
        return features;
    }
}

} // namespace vision
} // namespace bamboo_cut 