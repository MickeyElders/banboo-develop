/**
 * @file bamboo_detector.h
 * @brief C++ LVGL一体化系统核心推理引擎
 * YOLOv8竹子检测器的C++实现
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include "bamboo_cut/core/data_bridge.h"

#ifdef ENABLE_TENSORRT
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#endif

namespace bamboo_cut {
namespace inference {

/**
 * @brief 检测配置参数
 */
struct DetectorConfig {
    std::string model_path;         // 模型文件路径
    float confidence_threshold;     // 置信度阈值
    float nms_threshold;           // NMS阈值
    cv::Size input_size;           // 输入尺寸
    bool use_gpu;                  // 使用GPU加速
    bool use_tensorrt;             // 使用TensorRT优化
    
    DetectorConfig()
        : model_path("/opt/bamboo-cut/models/bamboo_detection.onnx")
        , confidence_threshold(0.85f)
        , nms_threshold(0.45f)
        , input_size(640, 640)
        , use_gpu(true)
        , use_tensorrt(true) {}
};

/**
 * @brief 竹子检测器类
 * 负责YOLOv8模型的加载、推理和结果处理
 */
class BambooDetector {
public:
    explicit BambooDetector(const DetectorConfig& config);
    ~BambooDetector();

    /**
     * @brief 初始化检测器
     * @return 是否初始化成功
     */
    bool initialize();

    /**
     * @brief 执行检测
     * @param frame 输入图像
     * @param result 检测结果输出
     * @return 是否检测成功
     */
    bool detect(const cv::Mat& frame, core::DetectionResult& result);

    /**
     * @brief 获取推理性能统计
     * @param inference_time 推理时间(ms)
     * @param fps 当前FPS
     */
    void getPerformanceStats(float& inference_time, float& fps);

    /**
     * @brief 检查检测器是否已初始化
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief 更新检测器配置
     */
    void updateConfig(const DetectorConfig& config);

private:
    /**
     * @brief 预处理输入图像
     */
    cv::Mat preprocess(const cv::Mat& frame);

    /**
     * @brief 后处理检测结果
     */
    void postprocess(const std::vector<cv::Mat>& outputs, 
                    const cv::Size& original_size,
                    core::DetectionResult& result);

    /**
     * @brief 计算切割坐标
     */
    cv::Point2f calculateCuttingPoint(const cv::Rect& bbox);

#ifdef ENABLE_TENSORRT
    /**
     * @brief 初始化TensorRT引擎
     */
    bool initializeTensorRT();

    /**
     * @brief TensorRT推理
     */
    bool tensorRTInference(const cv::Mat& input, std::vector<cv::Mat>& outputs);
#endif

private:
    DetectorConfig config_;
    bool initialized_;
    
    // OpenCV DNN
    cv::dnn::Net dnn_net_;
    std::vector<std::string> output_names_;
    
#ifdef ENABLE_TENSORRT
    // TensorRT相关
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    void* gpu_input_buffer_;
    void* gpu_output_buffer_;
    cudaStream_t stream_;
    size_t input_size_;
    size_t output_size_;
#endif

    // 性能统计
    mutable std::mutex perf_mutex_;
    float last_inference_time_;
    float current_fps_;
    std::chrono::high_resolution_clock::time_point last_frame_time_;
    int frame_count_;
    
    // 类别名称
    std::vector<std::string> class_names_;
};

/**
 * @brief 推理线程管理器
 * 管理独立的推理线程，与LVGL界面线程分离
 */
class InferenceThread {
public:
    InferenceThread(std::shared_ptr<core::DataBridge> data_bridge);
    ~InferenceThread();

    /**
     * @brief 启动推理线程
     */
    bool start();

    /**
     * @brief 停止推理线程
     */
    void stop();

    /**
     * @brief 检查线程是否在运行
     */
    bool isRunning() const { return running_.load(); }

private:
    /**
     * @brief 推理线程主循环
     */
    void inferenceLoop();

    /**
     * @brief 初始化摄像头
     */
    bool initializeCamera();

    /**
     * @brief 更新系统性能统计
     */
    void updateSystemStats();

private:
    std::shared_ptr<core::DataBridge> data_bridge_;
    std::unique_ptr<BambooDetector> detector_;
    
    std::thread inference_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    
    // 摄像头相关
    cv::VideoCapture camera_;
    int camera_index_;
    
    // 性能统计
    std::chrono::high_resolution_clock::time_point last_stats_update_;
    int total_frames_;
    int total_detections_;
};

} // namespace inference
} // namespace bamboo_cut