#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "bamboo_cut/core/types.h"
#include "bamboo_cut/vision/tensorrt_engine.h"
#include "bamboo_cut/vision/nam_attention.h"
#include "bamboo_cut/vision/ghost_conv.h"
#include "bamboo_cut/vision/vov_gscsp.h"
#include "bamboo_cut/vision/wise_iou.h"
#include "bamboo_cut/vision/deepstream_pipeline.h"
#include "bamboo_cut/vision/stereo_vision.h"
#include "bamboo_cut/vision/sahi_slicing.h"
#include "bamboo_cut/vision/hardware_accelerated_camera.h"
#include "bamboo_cut/vision/bamboo_detector.h"

namespace bamboo_cut {
namespace vision {

/**
 * @brief 优化检测器配置
 */
struct OptimizedDetectorConfig {
    // TensorRT 配置
    TensorRTConfig tensorrt_config;
    
    // NAM 注意力配置
    NAMAttention::NAMConfig nam_config;
    
    // GhostConv 配置
    GhostConvConfig ghost_conv_config;
    
    // VoV-GSCSP 配置
    VoVGSCSPConfig vov_gscsp_config;
    
    // Wise-IoU 配置
    WiseIoUConfig wise_iou_config;
    
    // DeepStream 配置
    DeepStreamConfig deepstream_config;
    
    // 立体视觉配置
    CameraSyncConfig stereo_config;
    
    // SAHI 切片配置
    SAHIConfig sahi_config;
    
    // 硬件加速摄像头配置
    HardwareAccelerationConfig camera_config;
    
    // 检测参数
    float confidence_threshold{0.5f};
    float nms_threshold{0.4f};
    int max_detections{100};
    bool enable_stereo_depth{true};
    bool enable_fp16{true};
    bool enable_tracking{true};
    bool enable_sahi_slicing{true};  // 启用 SAHI 切片推理
    bool enable_hardware_acceleration{true}; // 启用硬件加速
    
    // 图像优化参数
    int optimization_level{1};  // 优化级别 (0-3)
    bool enable_noise_reduction{true}; // 启用噪声抑制
    
    // 性能优化
    bool enable_batch_processing{true};
    int batch_size{4};
    bool enable_async_processing{true};
    int num_worker_threads{4};
    
    // 模型参数
    std::string model_path;
    int input_width{640};
    int input_height{640};
    bool use_tensorrt{true};
    bool use_fp16{true};
    
    // SAHI切片参数
    int sahi_slice_height{512};
    int sahi_slice_width{512};
    float sahi_overlap_ratio{0.2f};
    
    OptimizedDetectorConfig() = default;
    bool validate() const;
};

/**
 * @brief 优化检测结果
 */
struct OptimizedDetectionResult {
    std::vector<core::DetectionResult> detections_2d;      // 2D 检测结果
    std::vector<core::Point3D> detections_3d;              // 3D 检测结果
    std::vector<core::DetectionResult> tracked_objects;    // 跟踪对象
    float processing_time_ms{0.0f};                        // 处理时间
    bool success{false};                                   // 是否成功
    std::string error_message;                             // 错误信息
    
    OptimizedDetectionResult() = default;
};

/**
 * @brief 优化的视觉检测器类
 * 
 * 集成 TensorRT 推理、NAM 注意力、Wise-IoU 损失、DeepStream 流水线的完整检测系统
 */
class OptimizedDetector {
public:
    OptimizedDetector();
    explicit OptimizedDetector(const OptimizedDetectorConfig& config);
    ~OptimizedDetector();

    // 禁用拷贝
    OptimizedDetector(const OptimizedDetector&) = delete;
    OptimizedDetector& operator=(const OptimizedDetector&) = delete;

    // 初始化和控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // 单帧检测
    core::DetectionResult detect(const cv::Mat& image);
    
    // 立体检测
    core::DetectionResult detect_stereo(const StereoFrame& stereo_frame);
    
    // 批处理检测
    std::vector<core::DetectionResult> detect_batch(const std::vector<cv::Mat>& images);
    std::vector<core::DetectionResult> detect_stereo_batch(const std::vector<StereoFrame>& stereo_frames);
    
    // 获取模型信息
    std::string get_model_info() const;

    // 模型管理
    bool load_model(const std::string& model_path);
    bool save_model(const std::string& model_path);
    bool export_to_onnx(const std::string& onnx_path);

    // 配置管理
    void set_config(const OptimizedDetectorConfig& config);
    OptimizedDetectorConfig get_config() const { return config_; }

    // 性能优化
    void enable_fp16(bool enable);
    void enable_batch_processing(bool enable);
    void set_batch_size(int batch_size);
    void enable_async_processing(bool enable);

    // 训练接口
    bool train(const std::vector<cv::Mat>& images,
              const std::vector<std::vector<core::DetectionResult>>& ground_truth);
    
    bool validate(const std::vector<cv::Mat>& images,
                 const std::vector<std::vector<core::DetectionResult>>& ground_truth);

    // 性能统计
    struct PerformanceStats {
        uint64_t total_detections{0};
        uint64_t total_frames_processed{0};
        double avg_detection_time_ms{0.0};
        double min_detection_time_ms{0.0};
        double max_detection_time_ms{0.0};
        double fps{0.0};
        double avg_confidence{0.0};
        double avg_precision{0.0};
        double avg_recall{0.0};
        
        // 组件性能
        TensorRTEngine::PerformanceStats tensorrt_stats;
        NAMAttention::PerformanceStats nam_stats;
        GhostConv::PerformanceStats ghost_conv_stats;
        VoVGSCSP::PerformanceStats vov_gscsp_stats;
        WiseIoULoss::PerformanceStats wise_iou_stats;
        DeepStreamPipeline::PerformanceStats deepstream_stats;
        StereoVision::Statistics stereo_stats;
        
        core::Timestamp last_update;
    };
    PerformanceStats get_performance_stats() const;

    // 回调函数类型
    using DetectionCallback = std::function<void(const OptimizedDetectionResult&)>;
    using ErrorCallback = std::function<void(const std::string&)>;
    using TrainingCallback = std::function<void(int epoch, float loss)>;

    // 设置回调
    void set_detection_callback(DetectionCallback callback);
    void set_error_callback(ErrorCallback callback);
    void set_training_callback(TrainingCallback callback);

private:
    // 内部检测方法
    OptimizedDetectionResult detect_with_tensorrt(const cv::Mat& image);
    OptimizedDetectionResult detect_with_deepstream(const cv::Mat& image);
    OptimizedDetectionResult detect_with_stereo(const StereoFrame& stereo_frame);
    OptimizedDetectionResult detect_with_sahi_slicing(const cv::Mat& image);
    OptimizedDetectionResult detect_with_hardware_acceleration(const HardwareAcceleratedFrame& frame);
    
    // 后处理
    std::vector<core::DetectionResult> apply_nms(const std::vector<core::DetectionResult>& detections);
    std::vector<core::DetectionResult> apply_confidence_filter(const std::vector<core::DetectionResult>& detections);
    std::vector<core::Point3D> convert_to_3d(const std::vector<core::DetectionResult>& detections_2d, 
                                            const cv::Mat& disparity);
    
    // 特征增强
    cv::Mat apply_nam_attention(const cv::Mat& features);
    std::vector<cv::Mat> apply_nam_attention_batch(const std::vector<cv::Mat>& features);
    cv::Mat apply_ghost_conv(const cv::Mat& features);
    cv::Mat apply_vov_gscsp(const cv::Mat& features);
    
    // 损失计算
    float compute_wise_iou_loss(const std::vector<core::DetectionResult>& predictions,
                               const std::vector<core::DetectionResult>& ground_truth);

    // 图像优化处理
    cv::Mat apply_optimizations(const cv::Mat& image);
    cv::Mat apply_image_enhancement(const cv::Mat& image);
    cv::Mat apply_noise_reduction(const cv::Mat& image);
    
    // 后处理优化
    core::DetectionResult apply_post_processing_optimizations(const core::DetectionResult& result);
    core::DetectionResult apply_result_filtering(const core::DetectionResult& result);
    core::DetectionResult apply_result_sorting(const core::DetectionResult& result);

    // 初始化辅助函数
    bool initialize_base_detector();
    bool initialize_optimizations();
    bool initialize_sahi_slicing();
    bool initialize_advanced_optimizations();

    // 配置和状态
    OptimizedDetectorConfig config_;
    bool initialized_{false};

    // 组件
    std::unique_ptr<BambooDetector> base_detector_;
    std::unique_ptr<TensorRTEngine> tensorrt_engine_;
    std::unique_ptr<NAMAttention> nam_attention_;
    std::unique_ptr<GhostConv> ghost_conv_;
    std::unique_ptr<VoVGSCSP> vov_gscsp_;
    std::unique_ptr<WiseIoULoss> wise_iou_loss_;
    std::unique_ptr<DeepStreamPipeline> deepstream_pipeline_;
    std::unique_ptr<StereoVision> stereo_vision_;
    std::unique_ptr<SAHISlicing> sahi_slicing_;
    std::unique_ptr<HardwareAcceleratedCamera> hardware_camera_;
    std::unique_ptr<MultiCameraHardwareManager> multi_camera_manager_;

    // 回调函数
    DetectionCallback detection_callback_;
    ErrorCallback error_callback_;
    TrainingCallback training_callback_;

    // 性能统计
    mutable std::mutex stats_mutex_;
    PerformanceStats performance_stats_;

    // 错误处理
    std::string last_error_;
    void set_error(const std::string& error);
};

} // namespace vision
} // namespace bamboo_cut 