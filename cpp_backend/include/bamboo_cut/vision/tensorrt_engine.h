#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#ifdef ENABLE_TENSORRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#endif

#include "bamboo_cut/core/types.h"

namespace bamboo_cut {
namespace vision {

#ifdef ENABLE_TENSORRT
// TensorRT 日志记录器
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// TensorRT 内存管理
class TensorRTAllocator {
public:
    void* allocate(size_t size);
    void free(void* ptr);
    ~TensorRTAllocator();
private:
    std::vector<void*> allocated_memory_;
};
#endif

/**
 * @brief TensorRT 推理引擎配置
 */
struct TensorRTConfig {
    std::string model_path;           // ONNX 模型路径
    std::string engine_path;          // 序列化引擎路径
    int max_batch_size{1};            // 最大批处理大小
    bool enable_fp16{true};           // 启用 FP16 半精度
    bool enable_int8{false};          // 启用 INT8 量化
    float fp16_threshold{0.1f};       // FP16 精度阈值
    int max_workspace_size{1 << 30};  // 最大工作空间大小 (1GB)
    int device_id{0};                 // GPU 设备 ID
    bool enable_profiling{false};     // 启用性能分析
    int num_inference_threads{1};     // 推理线程数
    
    TensorRTConfig() = default;
    bool validate() const;
};

/**
 * @brief 推理结果结构
 */
struct InferenceResult {
    std::vector<core::DetectionResult> detections;
    float inference_time_ms{0.0f};
    bool success{false};
    std::string error_message;
    
    InferenceResult() = default;
};

/**
 * @brief TensorRT 推理引擎类
 * 
 * 支持 FP16 半精度推理的高性能 TensorRT 引擎
 */
class TensorRTEngine {
public:
    explicit TensorRTEngine(const TensorRTConfig& config = TensorRTConfig{});
    ~TensorRTEngine();

    // 禁用拷贝
    TensorRTEngine(const TensorRTEngine&) = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;

    // 初始化和控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // 模型管理
    bool load_model(const std::string& model_path);
    bool save_engine(const std::string& engine_path);
    bool load_engine(const std::string& engine_path);

    // 推理接口
    InferenceResult infer(const cv::Mat& input_image);
    InferenceResult infer_batch(const std::vector<cv::Mat>& input_images);

    // 性能优化
    void enable_profiling(bool enable);
    void set_fp16_mode(bool enable);
    void set_int8_mode(bool enable);
    
    // 统计信息
    struct PerformanceStats {
        uint64_t total_inferences{0};
        double avg_inference_time_ms{0.0};
        double min_inference_time_ms{0.0};
        double max_inference_time_ms{0.0};
        double fps{0.0};
        core::Timestamp last_update;
    };
    PerformanceStats get_performance_stats() const;

private:
#ifdef ENABLE_TENSORRT
    // TensorRT 组件
    std::unique_ptr<TensorRTLogger> logger_;
    std::unique_ptr<TensorRTAllocator> allocator_;
    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
    
    // 内存管理
    void* input_buffer_{nullptr};
    void* output_buffer_{nullptr};
    size_t input_size_{0};
    size_t output_size_{0};
    
    // 模型信息
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;
    std::string input_name_;
    std::string output_name_;
    
    // 内部方法
    bool build_engine_from_onnx(const std::string& onnx_path);
    bool create_execution_context();
    void* allocate_gpu_memory(size_t size);
    void free_gpu_memory(void* ptr);
    bool preprocess_image(const cv::Mat& input, void* gpu_buffer);
    bool postprocess_output(void* gpu_buffer, std::vector<core::DetectionResult>& detections);
#endif

    // 配置和状态
    TensorRTConfig config_;
    bool initialized_{false};
    bool profiling_enabled_{false};
    
    // 性能统计
    mutable std::mutex stats_mutex_;
    PerformanceStats performance_stats_;
    
    // 错误处理
    std::string last_error_;
    void set_error(const std::string& error);
};

} // namespace vision
} // namespace bamboo_cut 