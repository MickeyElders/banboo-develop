#include "bamboo_cut/vision/detector.h"
#include "bamboo_cut/core/logger.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>

namespace bamboo_cut {
namespace vision {

BambooDetector::BambooDetector(const Config& config) 
    : config_(config), initialized_(false) {
    LOG_INFO("创建BambooDetector实例");
}

BambooDetector::~BambooDetector() {
    LOG_INFO("销毁BambooDetector实例");
}

bool BambooDetector::initialize() {
    LOG_INFO("初始化BambooDetector");
    
    try {
        // 检查模型文件是否存在
        if (config_.model_path.empty()) {
            LOG_ERROR("模型路径为空");
            return false;
        }
        
        // 初始化OpenCV DNN
        if (!initializeOpenCV()) {
            LOG_ERROR("OpenCV DNN初始化失败");
            return false;
        }
        
        initialized_ = true;
        LOG_INFO("BambooDetector初始化成功");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("BambooDetector初始化异常: {}", e.what());
        return false;
    }
}

DetectionResult BambooDetector::detect(const cv::Mat& image) {
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
        
        // 预处理图像
        cv::Mat preprocessed = preprocessImage(image);
        
        // 执行推理
        cv::Mat output;
        inferOpenCV(preprocessed, output);
        
        // 后处理
        float scale_x = static_cast<float>(image.cols) / config_.input_width;
        float scale_y = static_cast<float>(image.rows) / config_.input_height;
        result = postprocessOutput(output.ptr<float>(), output.total(), scale_x, scale_y);
        
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

std::vector<DetectionResult> BambooDetector::detectBatch(const std::vector<cv::Mat>& images) {
    std::vector<DetectionResult> results;
    results.reserve(images.size());
    
    for (const auto& image : images) {
        results.push_back(detect(image));
    }
    
    return results;
}

std::string BambooDetector::getModelInfo() const {
    std::string info = "BambooDetector - ";
    info += "模型: " + config_.model_path + ", ";
    info += "输入尺寸: " + std::to_string(config_.input_width) + "x" + std::to_string(config_.input_height) + ", ";
    info += "置信度阈值: " + std::to_string(config_.confidence_threshold);
    return info;
}

BambooDetector::PerformanceStats BambooDetector::getPerformanceStats() const {
    return stats_;
}

void BambooDetector::resetPerformanceStats() {
    stats_ = PerformanceStats{};
    inference_times_.clear();
}

bool BambooDetector::initializeOpenCV() {
    try {
        // 加载模型
        opencv_net_ = cv::dnn::readNet(config_.model_path);
        
        if (opencv_net_.empty()) {
            LOG_ERROR("无法加载模型: {}", config_.model_path);
            return false;
        }
        
        // 设置计算后端和目标
        opencv_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        opencv_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        LOG_INFO("OpenCV DNN模型加载成功");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("OpenCV DNN初始化异常: {}", e.what());
        return false;
    }
}

void BambooDetector::inferOpenCV(const cv::Mat& preprocessed_image, cv::Mat& output) {
    // 创建blob
    cv::Mat blob = cv::dnn::blobFromImage(preprocessed_image, 1.0/255.0, 
                                         cv::Size(config_.input_width, config_.input_height), 
                                         cv::Scalar(), true, false);
    
    // 设置输入
    opencv_net_.setInput(blob);
    
    // 前向传播
    output = opencv_net_.forward();
}

cv::Mat BambooDetector::preprocessImage(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(config_.input_width, config_.input_height));
    return resized;
}

DetectionResult BambooDetector::postprocessOutput(const float* output, int output_size, 
                                                 float scale_x, float scale_y) {
    DetectionResult result;
    
    // 简化的后处理 - 这里应该根据实际的模型输出格式进行解析
    // 目前返回空结果，避免链接错误
    
    result.success = true;
    return result;
}

std::vector<int> BambooDetector::applyNMS(const std::vector<cv::Rect>& boxes, 
                                         const std::vector<float>& scores, 
                                         float nms_threshold) {
    // 简化的NMS实现
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, config_.confidence_threshold, nms_threshold, indices);
    return indices;
}

DetectionPoint BambooDetector::pixelToMM(float pixel_x, float pixel_y, float confidence, int class_id) {
    return DetectionPoint(
        pixel_x * config_.pixel_to_mm_ratio,
        pixel_y * config_.pixel_to_mm_ratio,
        confidence,
        class_id
    );
}

void BambooDetector::updatePerformanceStats(float inference_time) const {
    inference_times_.push_back(inference_time);
    
    // 保持最近100次的记录
    if (inference_times_.size() > 100) {
        inference_times_.erase(inference_times_.begin());
    }
    
    // 计算平均推理时间
    float sum = 0.0f;
    for (float time : inference_times_) {
        sum += time;
    }
    stats_.avg_inference_time_ms = sum / inference_times_.size();
    stats_.total_inferences = inference_times_.size();
    
    if (stats_.avg_inference_time_ms > 0) {
        stats_.fps = 1000.0f / stats_.avg_inference_time_ms;
    }
}

bool BambooDetector::optimizeModel(const std::string& input_model_path,
                                  const std::string& output_model_path,
                                  const OptimizationConfig& config) {
    LOG_INFO("模型优化功能暂未实现");
    return false;
}

float BambooDetector::benchmarkModel(const std::string& model_path, int num_iterations) {
    LOG_INFO("模型基准测试功能暂未实现");
    return 0.0f;
}

} // namespace vision
} // namespace bamboo_cut 