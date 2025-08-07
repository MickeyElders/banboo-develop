#ifndef BAMBOO_CUT_DETECTOR_H
#define BAMBOO_CUT_DETECTOR_H

#include <vector>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

#ifdef ENABLE_TENSORRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#endif

namespace bamboo_cut {
namespace vision {

struct DetectionPoint {
    float x;            // X坐标 (mm)
    float y;            // Y坐标 (mm) 
    float confidence;   // 置信度 [0.0, 1.0]
    int class_id;       // 类别ID (0=切点, 1=节点)
    
    DetectionPoint(float x = 0.0f, float y = 0.0f, float conf = 0.0f, int cls = 0)
        : x(x), y(y), confidence(conf), class_id(cls) {}
};

struct DetectionResult {
    std::vector<DetectionPoint> points;
    float processing_time_ms;
    bool success;
    std::string error_message;
    
    DetectionResult() : processing_time_ms(0.0f), success(false) {}
};

class BambooDetector {
public:
    struct Config {
        std::string model_path;
        std::string engine_path;        // TensorRT引擎文件路径
        int input_width = 640;
        int input_height = 640;
        float confidence_threshold = 0.5f;
        float nms_threshold = 0.4f;
        int max_detections = 10;
        bool use_tensorrt = true;
        bool use_fp16 = true;          // 是否使用FP16精度
        int max_batch_size = 1;
        float pixel_to_mm_ratio = 1.0f; // 像素到毫米的转换比例
    };

    explicit BambooDetector(const Config& config);
    virtual ~BambooDetector();

    // 初始化检测器
    bool initialize();
    
    // 检测竹材切点
    DetectionResult detect(const cv::Mat& image);
    
    // 批量检测
    std::vector<DetectionResult> detectBatch(const std::vector<cv::Mat>& images);
    
    // 获取模型信息
    std::string getModelInfo() const;
    
    // 性能统计
    struct PerformanceStats {
        float avg_inference_time_ms;
        float avg_preprocessing_time_ms;
        float avg_postprocessing_time_ms;
        int total_inferences;
        float fps;
    };
    
    PerformanceStats getPerformanceStats() const;
    void resetPerformanceStats();

private:
    Config config_;
    bool initialized_;
    
    // 性能统计
    mutable PerformanceStats stats_;
    mutable std::vector<float> inference_times_;
    
#ifdef ENABLE_TENSORRT
    // TensorRT相关
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    void* gpu_input_buffer_;
    void* gpu_output_buffer_;
    void* cpu_output_buffer_;
    
    int input_binding_index_;
    int output_binding_index_;
    size_t input_size_;
    size_t output_size_;
    
    bool initializeTensorRT();
    bool buildEngine();
    bool loadEngine();
    void inferTensorRT(const cv::Mat& preprocessed_image, float* output);
#endif

    // OpenCV DNN后备方案
    cv::dnn::Net opencv_net_;
    bool initializeOpenCV();
    void inferOpenCV(const cv::Mat& preprocessed_image, cv::Mat& output);
    
    // 图像预处理
    cv::Mat preprocessImage(const cv::Mat& image);
    
    // 后处理
    DetectionResult postprocessOutput(const float* output, int output_size, 
                                    float scale_x, float scale_y);
    
    // NMS (非极大值抑制)
    std::vector<int> applyNMS(const std::vector<cv::Rect>& boxes, 
                             const std::vector<float>& scores, 
                             float nms_threshold);
    
    // 坐标转换
    DetectionPoint pixelToMM(float pixel_x, float pixel_y, float confidence, int class_id);
    
    // 性能监控
    void updatePerformanceStats(float inference_time) const;
};

// 模型优化技术接口
class ModelOptimizer {
public:
    struct OptimizationConfig {
        bool enable_ghostconv = true;      // GhostConv优化
        bool enable_gsconv = true;         // GSConv优化  
        bool enable_vov_gscsp = true;      // VoV-GSCSP优化
        bool enable_nam = true;            // NAM注意力机制
        float pruning_ratio = 0.1f;        // 剪枝比例
        bool quantize_int8 = false;        // INT8量化
    };
    
    static bool optimizeModel(const std::string& input_model_path,
                            const std::string& output_model_path,
                            const OptimizationConfig& config);
    
    static float benchmarkModel(const std::string& model_path, 
                              int num_iterations = 100);
};

} // namespace vision
} // namespace bamboo_cut

#endif // BAMBOO_CUT_DETECTOR_H 