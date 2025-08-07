#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>
#include <functional> // Added for std::function
#include <map>        // Added for std::map
#include <chrono>     // Added for std::chrono

namespace bamboo_cut {
namespace vision {

/**
 * @brief NAM注意力机制配置
 * 
 * NAM (Normalization-based Attention Module) 是一种基于归一化的注意力机制，
 * 通过通道和空间维度的归一化来增强特征表示，替代传统的EMA机制。
 */
struct NAMConfig {
    // 通道注意力参数
    float channel_reduction_ratio{16.0f};  // 通道降维比例
    float channel_gamma{2.0f};             // 通道注意力gamma参数
    float channel_beta{4.0f};              // 通道注意力beta参数
    
    // 空间注意力参数
    float spatial_kernel_size{7.0f};       // 空间注意力卷积核大小
    float spatial_gamma{2.0f};             // 空间注意力gamma参数
    float spatial_beta{4.0f};              // 空间注意力beta参数
    
    // 归一化参数
    float normalization_epsilon{1e-5f};    // 归一化epsilon值
    bool use_batch_norm{true};             // 是否使用批归一化
    bool use_layer_norm{false};            // 是否使用层归一化
    
    // 训练参数
    float learning_rate{1e-4f};            // 学习率
    float weight_decay{1e-4f};             // 权重衰减
    bool enable_dropout{true};             // 是否启用dropout
    float dropout_rate{0.1f};              // dropout率
    
    NAMConfig() = default;
    bool validate() const;
};

/**
 * @brief NAM注意力模块
 * 
 * 实现基于归一化的注意力机制，包括：
 * 1. 通道注意力：通过通道维度的归一化增强重要通道
 * 2. 空间注意力：通过空间维度的归一化增强重要区域
 * 3. 特征融合：将通道和空间注意力结果融合
 */
class NAMAttention {
public:
    explicit NAMAttention(const NAMConfig& config = NAMConfig());
    ~NAMAttention();
    
    // 禁用拷贝
    NAMAttention(const NAMAttention&) = delete;
    NAMAttention& operator=(const NAMAttention&) = delete;
    
    // 初始化和控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }
    
    // 前向传播
    cv::Mat forward(const cv::Mat& input);
    std::vector<cv::Mat> forward_batch(const std::vector<cv::Mat>& inputs);
    
    // 通道注意力
    cv::Mat apply_channel_attention(const cv::Mat& input);
    
    // 空间注意力
    cv::Mat apply_spatial_attention(const cv::Mat& input);
    
    // 特征融合
    cv::Mat fuse_attention(const cv::Mat& channel_attn, const cv::Mat& spatial_attn);
    
    // 配置管理
    void set_config(const NAMConfig& config);
    NAMConfig get_config() const { return config_; }
    
    // 性能统计
    struct PerformanceStats {
        uint64_t total_forward_passes{0};
        double avg_processing_time_ms{0.0};
        double min_processing_time_ms{0.0};
        double max_processing_time_ms{0.0};
        double avg_memory_usage_mb{0.0};
        double fps{0.0};
        
        // 注意力统计
        double avg_channel_attention_weight{0.0};
        double avg_spatial_attention_weight{0.0};
        double avg_feature_enhancement_ratio{0.0};
        
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
    using AttentionCallback = std::function<void(const cv::Mat& attention_map)>;
    using TrainingCallback = std::function<void(int epoch, float loss)>;
    
    // 设置回调
    void set_attention_callback(AttentionCallback callback);
    void set_training_callback(TrainingCallback callback);

private:
    // 通道注意力实现
    cv::Mat compute_channel_attention(const cv::Mat& input);
    cv::Mat channel_global_pooling(const cv::Mat& input);
    cv::Mat channel_fc_layers(const cv::Mat& pooled);
    cv::Mat channel_sigmoid_activation(const cv::Mat& fc_output);
    
    // 空间注意力实现
    cv::Mat compute_spatial_attention(const cv::Mat& input);
    cv::Mat spatial_conv_layer(const cv::Mat& input);
    cv::Mat spatial_sigmoid_activation(const cv::Mat& conv_output);
    
    // 归一化实现
    cv::Mat batch_normalization(const cv::Mat& input, const std::string& layer_name);
    cv::Mat layer_normalization(const cv::Mat& input, const std::string& layer_name);
    cv::Mat instance_normalization(const cv::Mat& input, const std::string& layer_name);
    
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
    NAMConfig config_;
    bool initialized_{false};
    
    // 模型参数
    std::vector<cv::Mat> channel_fc_weights_;
    std::vector<cv::Mat> channel_fc_biases_;
    cv::Mat spatial_conv_kernel_;
    cv::Mat spatial_conv_bias_;
    
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
    AttentionCallback attention_callback_;
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