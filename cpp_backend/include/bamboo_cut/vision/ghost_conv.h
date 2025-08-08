#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>

namespace bamboo_cut {
namespace vision {

/**
 * @brief GhostConv配置
 * 
 * GhostConv是一种高效的卷积操作，通过生成"幽灵"特征图来减少计算量
 */
struct GhostConvConfig {
    // 卷积参数
    int in_channels{64};           // 输入通道数
    int out_channels{64};          // 输出通道数
    int kernel_size{3};            // 卷积核大小
    int stride{1};                 // 步长
    int padding{1};                // 填充
    bool use_bias{true};           // 是否使用偏置
    
    // Ghost参数
    int ghost_channels{32};        // 幽灵通道数（通常为out_channels的一半）
    float reduction_ratio{2.0f};   // 降维比例
    bool use_relu{true};           // 是否使用ReLU激活
    bool use_batch_norm{true};     // 是否使用批归一化
    
    // 性能优化
    bool enable_fusion{true};      // 启用特征融合
    bool enable_residual{true};    // 启用残差连接
    float dropout_rate{0.1f};      // Dropout率
    
    GhostConvConfig() = default;
    bool validate() const;
};

/**
 * @brief GhostConv模块
 * 
 * 实现Ghost卷积操作，通过以下步骤减少计算量：
 * 1. 使用少量卷积核生成主要特征
 * 2. 通过线性变换生成"幽灵"特征
 * 3. 将主要特征和幽灵特征拼接
 */
class GhostConv {
public:
    explicit GhostConv(const GhostConvConfig& config = GhostConvConfig());
    ~GhostConv();
    
    // 禁用拷贝
    GhostConv(const GhostConv&) = delete;
    GhostConv& operator=(const GhostConv&) = delete;
    
    // 初始化和控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }
    
    // 前向传播
    cv::Mat forward(const cv::Mat& input);
    std::vector<cv::Mat> forward_batch(const std::vector<cv::Mat>& inputs);
    
    // Ghost特征生成
    cv::Mat generate_primary_features(const cv::Mat& input);
    cv::Mat generate_ghost_features(const cv::Mat& primary_features);
    cv::Mat concatenate_features(const cv::Mat& primary, const cv::Mat& ghost);
    
    // 配置管理
    void set_config(const GhostConvConfig& config);
    GhostConvConfig get_config() const { return config_; }
    
    // 性能统计
    struct PerformanceStats {
        uint64_t total_forward_passes{0};
        double avg_processing_time_ms{0.0};
        double min_processing_time_ms{0.0};
        double max_processing_time_ms{0.0};
        double avg_memory_usage_mb{0.0};
        double fps{0.0};
        
        // Ghost特征统计
        double avg_primary_feature_weight{0.0};
        double avg_ghost_feature_weight{0.0};
        double computation_reduction_ratio{0.0};
        
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
    using FeatureCallback = std::function<void(const cv::Mat& primary_features, const cv::Mat& ghost_features)>;
    using TrainingCallback = std::function<void(int epoch, float loss)>;
    
    // 设置回调
    void set_feature_callback(FeatureCallback callback);
    void set_training_callback(TrainingCallback callback);

private:
    // 卷积操作
    cv::Mat apply_convolution(const cv::Mat& input, const cv::Mat& kernel, const cv::Mat& bias);
    cv::Mat apply_depthwise_convolution(const cv::Mat& input, const cv::Mat& kernel);
    cv::Mat apply_pointwise_convolution(const cv::Mat& input, const cv::Mat& kernel);
    
    // 线性变换
    cv::Mat apply_linear_transform(const cv::Mat& input, const cv::Mat& weight, const cv::Mat& bias);
    
    // 归一化
    cv::Mat batch_normalization(const cv::Mat& input, const std::string& layer_name);
    cv::Mat layer_normalization(const cv::Mat& input, const std::string& layer_name);
    
    // 激活函数
    cv::Mat relu_activation(const cv::Mat& input);
    cv::Mat sigmoid_activation(const cv::Mat& input);
    cv::Mat tanh_activation(const cv::Mat& input);
    
    // 特征处理
    cv::Mat apply_dropout(const cv::Mat& input);
    cv::Mat apply_residual_connection(const cv::Mat& input, const cv::Mat& residual);
    
    // 损失计算
    float compute_loss(const cv::Mat& prediction, const cv::Mat& target);
    cv::Mat compute_gradient(const cv::Mat& prediction, const cv::Mat& target);
    
    // 优化器
    void update_weights(const std::vector<cv::Mat>& gradients);
    void apply_weight_decay();
    
    // 配置和状态
    GhostConvConfig config_;
    bool initialized_{false};
    
    // 模型参数
    cv::Mat primary_conv_kernel_;      // 主要卷积核
    cv::Mat primary_conv_bias_;        // 主要卷积偏置
    cv::Mat ghost_linear_weight_;      // 幽灵线性变换权重
    cv::Mat ghost_linear_bias_;        // 幽灵线性变换偏置
    
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