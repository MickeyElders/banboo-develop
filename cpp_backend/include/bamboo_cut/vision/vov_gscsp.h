#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <mutex>
#include <chrono>
#include <map>
#include "bamboo_cut/vision/ghost_conv.h"

namespace bamboo_cut {
namespace vision {

/**
 * @brief VoV-GSCSP配置
 * 
 * VoV-GSCSP (VoV-Ghost Spatial Pyramid) 是一种高效的颈部压缩模块，
 * 结合了VoVNet的密集连接和GhostConv的高效性
 */
struct VoVGSCSPConfig {
    // 基础参数
    int in_channels{256};          // 输入通道数
    int out_channels{256};         // 输出通道数
    int num_blocks{3};             // 块数量
    bool use_residual{true};       // 是否使用残差连接
    
    // GhostConv参数
    GhostConvConfig ghost_config;  // GhostConv配置
    
    // 空间金字塔参数
    std::vector<int> pyramid_scales{1, 2, 4};  // 金字塔尺度
    bool use_attention{true};      // 是否使用注意力机制
    float attention_ratio{0.5f};   // 注意力比例
    
    // 性能优化
    bool enable_fusion{true};      // 启用特征融合
    bool enable_compression{true}; // 启用压缩
    float compression_ratio{0.5f}; // 压缩比例
    
    VoVGSCSPConfig() = default;
    bool validate() const;
};

/**
 * @brief VoV-GSCSP模块
 * 
 * 实现VoV-Ghost Spatial Pyramid模块，包括：
 * 1. 多尺度特征提取
 * 2. GhostConv高效卷积
 * 3. 密集连接和特征融合
 * 4. 空间金字塔池化
 */
class VoVGSCSP {
public:
    explicit VoVGSCSP(const VoVGSCSPConfig& config = VoVGSCSPConfig());
    ~VoVGSCSP();
    
    // 禁用拷贝
    VoVGSCSP(const VoVGSCSP&) = delete;
    VoVGSCSP& operator=(const VoVGSCSP&) = delete;
    
    // 初始化和控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }
    
    // 前向传播
    cv::Mat forward(const cv::Mat& input);
    std::vector<cv::Mat> forward_batch(const std::vector<cv::Mat>& inputs);
    
    // 多尺度处理
    cv::Mat process_multi_scale(const cv::Mat& input);
    cv::Mat apply_spatial_pyramid(const cv::Mat& input);
    cv::Mat apply_attention_fusion(const std::vector<cv::Mat>& features);
    
    // 特征融合
    cv::Mat fuse_features(const std::vector<cv::Mat>& features);
    cv::Mat apply_dense_connection(const std::vector<cv::Mat>& features);
    
    // 配置管理
    void set_config(const VoVGSCSPConfig& config);
    VoVGSCSPConfig get_config() const { return config_; }
    
    // 性能统计
    struct PerformanceStats {
        uint64_t total_forward_passes{0};
        double avg_processing_time_ms{0.0};
        double min_processing_time_ms{0.0};
        double max_processing_time_ms{0.0};
        double avg_memory_usage_mb{0.0};
        double fps{0.0};
        
        // 压缩统计
        double compression_ratio{0.0};
        double feature_reduction_ratio{0.0};
        double computation_savings{0.0};
        
        // 时间戳
        std::chrono::steady_clock::time_point last_update;
    };
    PerformanceStats get_performance_stats() const;
    void reset_performance_stats();
    
    // 模型管理
    bool load_model(const std::string& model_path);
    bool save_model(const std::string& model_path);
    bool export_to_onnx(const std::string& onnx_path);
    
    // 训练接口
    bool train(const std::vector<cv::Mat>& inputs,
              const std::vector<cv::Mat>& targets);
    bool validate(const std::vector<cv::Mat>& inputs,
                 const std::vector<cv::Mat>& targets);
    
    // 回调函数类型
    using FeatureCallback = std::function<void(const std::vector<cv::Mat>& multi_scale_features)>;
    using TrainingCallback = std::function<void(int epoch, float loss)>;
    
    // 设置回调
    void set_feature_callback(FeatureCallback callback);
    void set_training_callback(TrainingCallback callback);

private:
    // 多尺度处理
    cv::Mat apply_scale_convolution(const cv::Mat& input, int scale);
    cv::Mat apply_pyramid_pooling(const cv::Mat& input, int scale);
    
    // 注意力机制
    cv::Mat compute_channel_attention(const cv::Mat& input);
    cv::Mat compute_spatial_attention(const cv::Mat& input);
    cv::Mat apply_attention(const cv::Mat& input, const cv::Mat& attention_weights);
    
    // 特征压缩
    cv::Mat compress_features(const cv::Mat& input);
    cv::Mat decompress_features(const cv::Mat& compressed);
    
    // 归一化
    cv::Mat batch_normalization(const cv::Mat& input, const std::string& layer_name);
    cv::Mat layer_normalization(const cv::Mat& input, const std::string& layer_name);
    
    // 激活函数
    cv::Mat relu_activation(const cv::Mat& input);
    cv::Mat sigmoid_activation(const cv::Mat& input);
    cv::Mat tanh_activation(const cv::Mat& input);
    
    // 损失计算
    float compute_loss(const cv::Mat& prediction, const cv::Mat& target);
    cv::Mat compute_gradient(const cv::Mat& prediction, const cv::Mat& target);
    
    // 优化器
    void update_weights(const std::vector<cv::Mat>& gradients);
    void apply_weight_decay();
    
    // 配置和状态
    VoVGSCSPConfig config_;
    bool initialized_{false};
    
    // GhostConv模块
    std::vector<std::unique_ptr<GhostConv>> ghost_convs_;
    
    // 多尺度卷积核
    std::vector<cv::Mat> scale_kernels_;
    std::vector<cv::Mat> scale_biases_;
    
    // 注意力权重
    cv::Mat channel_attention_weights_;
    cv::Mat spatial_attention_weights_;
    
    // 压缩参数
    cv::Mat compression_matrix_;
    cv::Mat decompression_matrix_;
    
    // 批归一化参数
    std::map<std::string, cv::Mat> bn_weights_;
    std::map<std::string, cv::Mat> bn_biases_;
    std::map<std::string, cv::Mat> bn_running_mean_;
    std::map<std::string, cv::Mat> bn_running_var_;
    
    // 优化器状态
    std::vector<cv::Mat> momentum_;
    std::vector<cv::Mat> velocity_;
    float learning_rate_{1e-4f};
    float weight_decay_{1e-4f};
    
    // 回调函数
    FeatureCallback feature_callback_;
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