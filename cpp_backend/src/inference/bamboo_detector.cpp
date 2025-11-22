/**
 * @file bamboo_detector.cpp
 * @brief C++ LVGL一体化竹子识别AI推理引擎实现
 * @version 5.0.0
 * @date 2024
 * 
 * YOLOv8推理引擎 + TensorRT加速 + 多线程推理管理
 */

#include "bamboo_cut/inference/bamboo_detector.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <chrono>

#ifdef ENABLE_TENSORRT
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#endif

namespace bamboo_cut {
namespace inference {

BambooDetector::BambooDetector(const DetectorConfig& config) 
    : config_(config), initialized_(false) {
    class_names_ = {"bamboo"};  // 竹子类别
    last_inference_time_ = 0.0f;
    current_fps_ = 0.0f;
    frame_count_ = 0;
    last_frame_time_ = std::chrono::high_resolution_clock::now();
    
#ifdef ENABLE_TENSORRT
    runtime_ = nullptr;
    engine_ = nullptr;
    context_ = nullptr;
    gpu_input_buffer_ = nullptr;
    gpu_output_buffer_ = nullptr;
    stream_ = 0;
    input_size_ = 0;
    output_size_ = 0;
#endif
}

BambooDetector::~BambooDetector() {
    if (initialized_) {
#ifdef ENABLE_TENSORRT
        if (context_) delete context_;
        if (engine_) delete engine_;
        if (runtime_) delete runtime_;
        if (gpu_input_buffer_) cudaFree(gpu_input_buffer_);
        if (gpu_output_buffer_) cudaFree(gpu_output_buffer_);
        if (stream_) cudaStreamDestroy(stream_);
#endif
        initialized_ = false;
    }
}

bool BambooDetector::initialize() {
    std::cout << "[BambooDetector] 初始化AI推理引擎..." << std::endl;
    
    try {
        // 初始化OpenCV DNN网络
        if (!config_.model_path.empty()) {
            dnn_net_ = cv::dnn::readNet(config_.model_path);
            if (dnn_net_.empty()) {
                std::cerr << "[BambooDetector] 无法加载模型: " << config_.model_path << std::endl;
                return false;
            }
            
            // 设置计算后端
            if (config_.use_gpu) {
                dnn_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                dnn_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                std::cout << "[BambooDetector] 启用GPU加速" << std::endl;
            } else {
                dnn_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
                dnn_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
            
            // 获取输出层名称
            output_names_ = dnn_net_.getUnconnectedOutLayersNames();
        }
        
#ifdef ENABLE_TENSORRT
        if (config_.use_tensorrt) {
            if (!initializeTensorRT()) {
                std::cout << "[BambooDetector] TensorRT初始化失败，使用OpenCV DNN" << std::endl;
            } else {
                std::cout << "[BambooDetector] TensorRT加速已启用" << std::endl;
            }
        }
#endif
        
        initialized_ = true;
        std::cout << "[BambooDetector] AI推理引擎初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[BambooDetector] 初始化失败: " << e.what() << std::endl;
        return false;
    }
}

bool BambooDetector::detect(const cv::Mat& frame, core::DetectionResult& result) {
    if (!initialized_ || frame.empty()) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 清空之前的结果
        result.clear();
        
        // 预处理
        cv::Mat processed_frame = preprocess(frame);
        
        // 执行推理
        std::vector<cv::Mat> outputs;
        
#ifdef ENABLE_TENSORRT
        if (config_.use_tensorrt && context_) {
            if (!tensorRTInference(processed_frame, outputs)) {
                // TensorRT失败，回退到OpenCV DNN
                dnn_net_.setInput(processed_frame);
                dnn_net_.forward(outputs, output_names_);
            }
        } else {
#endif
            dnn_net_.setInput(processed_frame);
            dnn_net_.forward(outputs, output_names_);
#ifdef ENABLE_TENSORRT
        }
#endif
        
        // 后处理
        postprocess(outputs, frame.size(), result);
        
        // 更新性能统计
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        {
            std::lock_guard<std::mutex> lock(perf_mutex_);
            last_inference_time_ = duration.count();
            frame_count_++;
            
            auto fps_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - last_frame_time_);
            if (fps_duration.count() > 0) {
                current_fps_ = 1000.0f / fps_duration.count();
            }
            last_frame_time_ = end_time;
        }
        
        result.valid = true;
        result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[BambooDetector] 检测失败: " << e.what() << std::endl;
        return false;
    }
}

void BambooDetector::getPerformanceStats(float& inference_time, float& fps) {
    std::lock_guard<std::mutex> lock(perf_mutex_);
    inference_time = last_inference_time_;
    fps = current_fps_;
}

void BambooDetector::updateConfig(const DetectorConfig& config) {
    config_ = config;
    // 需要重新初始化
    if (initialized_) {
        initialized_ = false;
        initialize();
    }
}

cv::Mat BambooDetector::preprocess(const cv::Mat& frame) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, config_.input_size, cv::Scalar(0,0,0), true, false);
    return blob;
}

void BambooDetector::postprocess(const std::vector<cv::Mat>& outputs, 
                                const cv::Size& original_size,
                                core::DetectionResult& result) {
    if (outputs.empty()) return;
    
    // 简化的后处理实现
    for (const auto& output : outputs) {
        const float* data = (float*)output.data;
        int num_detections = output.size[1];
        int output_dim = output.size[2];  // 通常是85 (4坐标 + 1置信度 + 80类别)
        
        for (int i = 0; i < num_detections; i++) {
            const float* detection = data + i * output_dim;
            
            float confidence = detection[4];
            if (confidence < config_.confidence_threshold) {
                continue;
            }
            
            // 提取坐标 (相对于输入尺寸)
            float center_x = detection[0];
            float center_y = detection[1];
            float width = detection[2];
            float height = detection[3];
            
            // 转换为绝对坐标
            float scale_x = (float)original_size.width / config_.input_size.width;
            float scale_y = (float)original_size.height / config_.input_size.height;
            
            int x = (int)((center_x - width/2) * scale_x);
            int y = (int)((center_y - height/2) * scale_y);
            int w = (int)(width * scale_x);
            int h = (int)(height * scale_y);
            
            // 确保坐标在图像范围内
            x = std::max(0, std::min(x, original_size.width - 1));
            y = std::max(0, std::min(y, original_size.height - 1));
            w = std::max(1, std::min(w, original_size.width - x));
            h = std::max(1, std::min(h, original_size.height - y));
            
            cv::Rect bbox(x, y, w, h);
            result.bboxes.push_back(bbox);
            result.confidences.push_back(confidence);
            
            // 计算切割点 (bbox中心点)
            cv::Point2f cutting_point = calculateCuttingPoint(bbox);
            result.cutting_points.push_back(cutting_point);
        }
    }
}

cv::Point2f BambooDetector::calculateCuttingPoint(const cv::Rect& bbox) {
    // 返回边界框的中心点作为切割点
    return cv::Point2f(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
}

#ifdef ENABLE_TENSORRT
bool BambooDetector::initializeTensorRT() {
    // 创建TensorRT logger
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            // 简单的日志输出
            if (severity != Severity::kINFO) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    };
    
    static Logger gLogger;
    
    // TensorRT初始化实现 (简化版本)
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_) {
        return false;
    }
    
    // TODO: 实现完整的TensorRT初始化
    // 这里需要加载序列化的引擎文件或从ONNX构建
    
    return false; // 暂时返回false，使用OpenCV DNN
}

bool BambooDetector::tensorRTInference(const cv::Mat& input, std::vector<cv::Mat>& outputs) {
    // TensorRT推理实现 (简化版本)
    if (!context_) return false;
    
    // TODO: 实现完整的TensorRT推理
    
    return false; // 暂时返回false，回退到OpenCV DNN
}
#endif

// InferenceThread实现
InferenceThread::InferenceThread(std::shared_ptr<core::DataBridge> data_bridge)
    : data_bridge_(data_bridge)
    , camera_index_(0)
    , total_frames_(0)
    , total_detections_(0) {
    
    DetectorConfig config;
    detector_ = std::make_unique<BambooDetector>(config);
    last_stats_update_ = std::chrono::high_resolution_clock::now();
}

InferenceThread::~InferenceThread() {
    stop();
}

bool InferenceThread::start() {
    if (running_.load()) {
        return false;
    }
    
    if (!detector_->initialize()) {
        std::cerr << "[InferenceThread] 检测器初始化失败" << std::endl;
        return false;
    }
    
    if (!initializeCamera()) {
        std::cerr << "[InferenceThread] 摄像头初始化失败" << std::endl;
        return false;
    }
    
    should_stop_ = false;
    running_ = true;
    inference_thread_ = std::thread(&InferenceThread::inferenceLoop, this);
    
    std::cout << "[InferenceThread] 推理线程已启动" << std::endl;
    return true;
}

void InferenceThread::stop() {
    if (!running_.load()) {
        return;
    }
    
    should_stop_ = true;
    running_ = false;
    
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }
    
    if (camera_.isOpened()) {
        camera_.release();
    }
    
    std::cout << "[InferenceThread] 推理线程已停止" << std::endl;
}

void InferenceThread::inferenceLoop() {
    cv::Mat frame;
    
    while (!should_stop_.load()) {
        if (!camera_.read(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        if (frame.empty()) {
            continue;
        }
        
        // 更新原始帧到数据桥接
        data_bridge_->updateFrame(frame);
        
        // 执行检测
        core::DetectionResult result;
        if (detector_->detect(frame, result)) {
            total_detections_ += result.bboxes.size();
            data_bridge_->updateDetection(result);
        }
        
        total_frames_++;
        
        // 更新系统统计
        updateSystemStats();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

bool InferenceThread::initializeCamera() {
    camera_.open(camera_index_);
    if (!camera_.isOpened()) {
        return false;
    }
    
    // 设置摄像头参数
    camera_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    camera_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    camera_.set(cv::CAP_PROP_FPS, 30);
    
    std::cout << "[InferenceThread] 摄像头已初始化" << std::endl;
    return true;
}

void InferenceThread::updateSystemStats() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_update_);
    
    if (duration.count() >= 1) { // 每秒更新一次
        core::SystemStats stats;
        
        float inference_time, fps;
        detector_->getPerformanceStats(inference_time, fps);
        
        stats.camera_fps = fps;
        stats.inference_fps = fps;
        stats.total_detections = total_detections_;
        stats.system_status = "运行中";

        // 为 UI 填充更详细的 AI/摄像头状态
        stats.ai_model.inference_time_ms = inference_time;
        stats.ai_model.total_detections = total_detections_;

        core::DetectionResult detection;
        if (data_bridge_->getDetectionResult(detection) && !detection.confidences.empty()) {
            float sum_conf = 0.0f;
            for (float c : detection.confidences) {
                sum_conf += c;
            }
            stats.ai_model.average_confidence = sum_conf / detection.confidences.size();
        }

        stats.ai_model.camera_system.system_ready = camera_.isOpened();
        auto& cam1 = stats.ai_model.camera_system.camera1;
        cam1.is_online = camera_.isOpened();
        cam1.fps = (fps > 0.0f) ? fps : static_cast<float>(camera_.get(cv::CAP_PROP_FPS));
        cam1.width = static_cast<int>(camera_.get(cv::CAP_PROP_FRAME_WIDTH));
        cam1.height = static_cast<int>(camera_.get(cv::CAP_PROP_FRAME_HEIGHT));
        
        data_bridge_->updateStats(stats);
        last_stats_update_ = now;
    }
}

} // namespace inference
} // namespace bamboo_cut
