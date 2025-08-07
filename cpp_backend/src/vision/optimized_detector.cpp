#include "bamboo_cut/vision/optimized_detector.h"
#include "bamboo_cut/core/logger.h"
#include <opencv2/dnn.hpp>
#include <chrono>

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
        if (!initializeBaseDetector()) {
            LOG_ERROR("基础检测器初始化失败");
            return false;
        }
        
        // 初始化优化组件
        if (!initializeOptimizations()) {
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

DetectionResult OptimizedDetector::detect(const cv::Mat& image) {
    DetectionResult result;
    
    if (!initialized_) {
        result.error_message = "检测器未初始化";
        return result;
    }
    
    if (image.empty()) {
        result.error_message = "输入图像为空";
        return result;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 应用优化预处理
        cv::Mat optimized_image = applyOptimizations(image);
        
        // 执行检测
        result = base_detector_->detect(optimized_image);
        
        // 应用后处理优化
        result = applyPostProcessingOptimizations(result);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        result.processing_time_ms = duration.count() / 1000.0f;
        
        result.success = true;
        
    } catch (const std::exception& e) {
        result.error_message = std::string("检测异常: ") + e.what();
        LOG_ERROR("检测异常: {}", e.what());
    }
    
    return result;
}

std::vector<DetectionResult> OptimizedDetector::detectBatch(const std::vector<cv::Mat>& images) {
    std::vector<DetectionResult> results;
    results.reserve(images.size());
    
    for (const auto& image : images) {
        results.push_back(detect(image));
    }
    
    return results;
}

std::string OptimizedDetector::getModelInfo() const {
    std::string info = "OptimizedDetector - ";
    info += "模型: " + config_.model_path + ", ";
    info += "优化级别: " + std::to_string(config_.optimization_level) + ", ";
    info += "SAHI切片: " + (config_.enable_sahi_slicing ? "启用" : "禁用");
    return info;
}

OptimizedDetector::PerformanceStats OptimizedDetector::getPerformanceStats() const {
    return stats_;
}

void OptimizedDetector::resetPerformanceStats() {
    stats_ = PerformanceStats{};
}

bool OptimizedDetector::initializeBaseDetector() {
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

bool OptimizedDetector::initializeOptimizations() {
    try {
        // 初始化SAHI切片
        if (config_.enable_sahi_slicing) {
            if (!initializeSAHISlicing()) {
                LOG_WARN("SAHI切片初始化失败，将禁用此功能");
                config_.enable_sahi_slicing = false;
            }
        }
        
        // 初始化其他优化组件
        if (config_.optimization_level > 0) {
            if (!initializeAdvancedOptimizations()) {
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

bool OptimizedDetector::initializeSAHISlicing() {
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

bool OptimizedDetector::initializeAdvancedOptimizations() {
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

cv::Mat OptimizedDetector::applyOptimizations(const cv::Mat& image) {
    cv::Mat optimized_image = image.clone();
    
    // 应用图像预处理优化
    if (config_.optimization_level > 0) {
        // 图像增强
        optimized_image = applyImageEnhancement(optimized_image);
        
        // 噪声抑制
        if (config_.enable_noise_reduction) {
            optimized_image = applyNoiseReduction(optimized_image);
        }
    }
    
    return optimized_image;
}

cv::Mat OptimizedDetector::applyImageEnhancement(const cv::Mat& image) {
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

cv::Mat OptimizedDetector::applyNoiseReduction(const cv::Mat& image) {
    cv::Mat denoised;
    cv::bilateralFilter(image, denoised, 9, 75, 75);
    return denoised;
}

DetectionResult OptimizedDetector::applyPostProcessingOptimizations(const DetectionResult& result) {
    DetectionResult optimized_result = result;
    
    // 应用后处理优化
    if (config_.optimization_level > 0) {
        // 结果过滤
        optimized_result = applyResultFiltering(optimized_result);
        
        // 结果排序
        optimized_result = applyResultSorting(optimized_result);
    }
    
    return optimized_result;
}

DetectionResult OptimizedDetector::applyResultFiltering(const DetectionResult& result) {
    DetectionResult filtered_result = result;
    
    // 过滤低置信度的检测结果
    std::vector<DetectionPoint> filtered_points;
    for (const auto& point : result.points) {
        if (point.confidence >= config_.confidence_threshold) {
            filtered_points.push_back(point);
        }
    }
    
    filtered_result.points = filtered_points;
    return filtered_result;
}

DetectionResult OptimizedDetector::applyResultSorting(const DetectionResult& result) {
    DetectionResult sorted_result = result;
    
    // 按置信度排序
    std::sort(sorted_result.points.begin(), sorted_result.points.end(),
              [](const DetectionPoint& a, const DetectionPoint& b) {
                  return a.confidence > b.confidence;
              });
    
    return sorted_result;
}

} // namespace vision
} // namespace bamboo_cut 