/**
 * C++推理核心模块头文件
 * 高性能TensorRT推理引擎接口定义
 */

#ifndef INFERENCE_CORE_H
#define INFERENCE_CORE_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <filesystem>

// 前向声明
namespace nvinfer1 {
    class ICudaEngine;
    class IExecutionContext;
}

/**
 * 检测器配置结构
 */
struct DetectorConfig {
    std::string model_path = "/opt/bamboo-cut/models/bamboo_detection.onnx";           // 模型文件路径
    std::string engine_path = "/opt/bamboo-cut/models/bamboo_detection.onnx_b1_gpu0_fp16.engine";     // 引擎文件路径
    float confidence_threshold = 0.5f;                    // 置信度阈值
    float nms_threshold = 0.4f;                          // NMS阈值
    int max_detections = 100;                            // 最大检测数量
    bool use_tensorrt = true;                            // 使用TensorRT
    bool use_fp16 = true;                               // 使用FP16精度
    int batch_size = 1;                                 // 批处理大小
    int input_width = 640;                              // 输入图像宽度
    int input_height = 640;                             // 输入图像高度
    
    // 构造函数
    DetectorConfig() = default;
    
    // 验证配置
    bool validate() const {
        return !model_path.empty() && 
               !engine_path.empty() && 
               confidence_threshold >= 0.0f && confidence_threshold <= 1.0f &&
               nms_threshold >= 0.0f && nms_threshold <= 1.0f &&
               max_detections > 0 &&
               input_width > 0 && input_height > 0;
    }
};

/**
 * 检测点结构
 */
struct DetectionPoint {
    float x = 0.0f;              // X坐标 (像素)
    float y = 0.0f;              // Y坐标 (像素)
    float confidence = 0.0f;     // 置信度 [0.0, 1.0]
    int class_id = 0;            // 类别ID (0=切点, 1=节点)
    
    DetectionPoint() = default;
    DetectionPoint(float x_, float y_, float conf_, int cls_) 
        : x(x_), y(y_), confidence(conf_), class_id(cls_) {}
    
    // 转换为字符串
    std::string toString() const {
        return "Point(" + std::to_string(x) + ", " + std::to_string(y) + 
               ", conf=" + std::to_string(confidence) + 
               ", cls=" + std::to_string(class_id) + ")";
    }
};

/**
 * 检测结果结构
 */
struct DetectionResult {
    std::vector<DetectionPoint> points;     // 检测到的点
    float processing_time_ms = 0.0f;        // 处理时间(毫秒)
    bool success = false;                   // 是否成功
    std::string error_message;              // 错误信息
    
    DetectionResult() = default;
    
    // 添加检测点
    void addPoint(const DetectionPoint& point) {
        points.push_back(point);
    }
    
    // 获取检测点数量
    size_t getPointCount() const {
        return points.size();
    }
    
    // 清空结果
    void clear() {
        points.clear();
        processing_time_ms = 0.0f;
        success = false;
        error_message.clear();
    }
    
    // 转换为字符串
    std::string toString() const {
        std::string result = "DetectionResult(";
        result += "points=" + std::to_string(points.size());
        result += ", time=" + std::to_string(processing_time_ms) + "ms";
        result += std::string(", success=") + (success ? "true" : "false");
        if (!error_message.empty()) {
            result += ", error=\"" + error_message + "\"";
        }
        result += ")";
        return result;
    }
};

/**
 * 竹子检测器主类
 */
class BambooDetector {
public:
    /**
     * 构造函数
     */
    BambooDetector();
    
    /**
     * 析构函数
     */
    ~BambooDetector();
    
    /**
     * 初始化检测器
     * @param config 检测器配置
     * @return 是否成功
     */
    bool initialize(const DetectorConfig& config);
    
    /**
     * 执行检测
     * @param image 输入图像
     * @return 检测结果
     */
    DetectionResult detect(const cv::Mat& image);
    
    /**
     * 批量检测
     * @param images 输入图像列表
     * @return 检测结果列表
     */
    std::vector<DetectionResult> detectBatch(const std::vector<cv::Mat>& images);
    
    /**
     * 检查是否已初始化
     * @return 是否已初始化
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * 获取配置信息
     * @return 配置结构
     */
    const DetectorConfig& getConfig() const { return config_; }
    
    /**
     * 获取性能统计
     * @return 性能统计信息
     */
    struct PerformanceStats {
        int total_detections = 0;           // 总检测次数
        float total_time_ms = 0.0f;         // 总处理时间
        float avg_time_ms = 0.0f;           // 平均处理时间
        float min_time_ms = 0.0f;           // 最小处理时间
        float max_time_ms = 0.0f;           // 最大处理时间
        float fps = 0.0f;                   // 平均FPS
    };
    
    PerformanceStats getPerformanceStats() const { return perf_stats_; }
    
    /**
     * 重置性能统计
     */
    void resetPerformanceStats();
    
    /**
     * 清理资源
     */
    void cleanup();
    
    // 禁用拷贝构造和赋值
    BambooDetector(const BambooDetector&) = delete;
    BambooDetector& operator=(const BambooDetector&) = delete;
    
    // 启用移动构造和赋值
    BambooDetector(BambooDetector&&) noexcept = default;
    BambooDetector& operator=(BambooDetector&&) noexcept = default;

private:
    /**
     * 从ONNX文件构建TensorRT引擎
     * @param onnx_path ONNX文件路径
     * @param engine_path 引擎保存路径
     * @return 是否成功
     */
    bool buildEngineFromOnnx(const std::string& onnx_path, const std::string& engine_path);
    
    /**
     * 加载TensorRT引擎
     * @param engine_path 引擎文件路径
     * @return 是否成功
     */
    bool loadEngine(const std::string& engine_path);
    
    /**
     * 分配GPU内存缓冲区
     * @return 是否成功
     */
    bool allocateBuffers();
    
    /**
     * 预热模型
     * @return 是否成功
     */
    bool warmupModel();
    
    /**
     * 预处理图像
     * @param image 输入图像
     * @param processed 预处理后的图像
     * @return 是否成功
     */
    bool preprocessImage(const cv::Mat& image, cv::Mat& processed);
    
    /**
     * 执行推理
     * @param preprocessed_image 预处理后的图像
     * @return 是否成功
     */
    bool runInference(const cv::Mat& preprocessed_image);
    
    /**
     * 后处理结果
     * @param original_image 原始图像
     * @param result 检测结果
     * @return 是否成功
     */
    bool postprocessResults(const cv::Mat& original_image, DetectionResult& result);
    
    /**
     * 更新性能统计
     * @param processing_time_ms 处理时间
     */
    void updatePerformanceStats(float processing_time_ms);

private:
    // TensorRT组件
    nvinfer1::ICudaEngine* engine_;          // TensorRT引擎
    nvinfer1::IExecutionContext* context_;   // 执行上下文
    
    // CUDA资源
    cudaStream_t stream_;                    // CUDA流
    void* input_device_buffer_;              // GPU输入缓冲区
    void* output_device_buffer_;             // GPU输出缓冲区
    
    // CPU缓冲区
    std::vector<float> input_host_buffer_;   // CPU输入缓冲区
    std::vector<float> output_host_buffer_;  // CPU输出缓冲区
    
    // 缓冲区大小
    size_t input_size_;                      // 输入缓冲区大小
    size_t output_size_;                     // 输出缓冲区大小
    
    // 绑定索引
    int input_binding_index_;                // 输入绑定索引
    int output_binding_index_;               // 输出绑定索引
    
    // 配置和状态
    DetectorConfig config_;                  // 检测器配置
    bool initialized_;                       // 是否已初始化
    
    // 性能统计
    mutable PerformanceStats perf_stats_;    // 性能统计
};

/**
 * 工具函数
 */
namespace BambooUtils {
    /**
     * 检查CUDA设备
     * @return CUDA设备信息
     */
    struct CudaDeviceInfo {
        int device_count = 0;
        int current_device = 0;
        std::string device_name;
        size_t total_memory = 0;
        size_t free_memory = 0;
        int compute_capability_major = 0;
        int compute_capability_minor = 0;
    };
    
    CudaDeviceInfo getCudaDeviceInfo();
    
    /**
     * 验证TensorRT版本
     * @return TensorRT版本信息
     */
    std::string getTensorRTVersion();
    
    /**
     * 验证OpenCV版本
     * @return OpenCV版本信息
     */
    std::string getOpenCVVersion();
    
    /**
     * 创建日志记录器
     * @param log_level 日志级别
     * @return 是否成功
     */
    bool setupLogging(const std::string& log_level = "INFO");
}

#endif // INFERENCE_CORE_H