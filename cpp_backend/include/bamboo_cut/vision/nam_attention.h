#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace bamboo_cut {
namespace vision {

/**
 * @brief NAM (Normalization-based Attention Module) 注意力模块
 * 
 * 替代传统的 EMA 注意力机制，提供更好的特征增强效果
 */
class NAMAttention {
public:
    struct NAMConfig {
        int channels{64};              // 通道数
        float reduction_ratio{16.0f};  // 降维比例
        float eps{1e-5f};              // 数值稳定性参数
        bool use_bias{true};           // 是否使用偏置
        bool use_scale{true};          // 是否使用缩放
        std::string activation{"relu"}; // 激活函数类型
        
        NAMConfig() = default;
        bool validate() const;
    };

    explicit NAMAttention(const NAMConfig& config = NAMConfig{});
    ~NAMAttention();

    // 禁用拷贝
    NAMAttention(const NAMAttention&) = delete;
    NAMAttention& operator=(const NAMAttention&) = delete;

    // 初始化和控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // 注意力计算
    cv::Mat apply_attention(const cv::Mat& input);
    std::vector<cv::Mat> apply_attention_batch(const std::vector<cv::Mat>& inputs);

    // 通道注意力
    cv::Mat channel_attention(const cv::Mat& input);
    
    // 空间注意力
    cv::Mat spatial_attention(const cv::Mat& input);

    // 配置管理
    void set_config(const NAMConfig& config);
    NAMConfig get_config() const { return config_; }

    // 性能统计
    struct PerformanceStats {
        uint64_t total_operations{0};
        double avg_processing_time_ms{0.0};
        double min_processing_time_ms{0.0};
        double max_processing_time_ms{0.0};
    };
    PerformanceStats get_performance_stats() const;

private:
    // 内部方法
    cv::Mat global_average_pooling(const cv::Mat& input);
    cv::Mat global_max_pooling(const cv::Mat& input);
    cv::Mat sigmoid_activation(const cv::Mat& input);
    cv::Mat relu_activation(const cv::Mat& input);
    cv::Mat batch_normalization(const cv::Mat& input, const cv::Mat& mean, 
                               const cv::Mat& variance, const cv::Mat& gamma, 
                               const cv::Mat& beta);
    
    // 通道注意力计算
    cv::Mat compute_channel_weights(const cv::Mat& input);
    
    // 空间注意力计算
    cv::Mat compute_spatial_weights(const cv::Mat& input);

    // 配置和状态
    NAMConfig config_;
    bool initialized_{false};

    // 性能统计
    mutable std::mutex stats_mutex_;
    PerformanceStats performance_stats_;

    // 错误处理
    std::string last_error_;
    void set_error(const std::string& error);
};

/**
 * @brief NAM 注意力网络
 * 
 * 完整的 NAM 注意力网络，包含多个 NAM 模块
 */
class NAMNetwork {
public:
    struct NetworkConfig {
        int num_layers{3};             // NAM 层数
        std::vector<int> channels;     // 每层通道数
        std::vector<float> reduction_ratios; // 每层降维比例
        bool use_residual{true};       // 是否使用残差连接
        bool use_bottleneck{true};     // 是否使用瓶颈结构
        
        NetworkConfig() = default;
        bool validate() const;
    };

    explicit NAMNetwork(const NetworkConfig& config = NetworkConfig{});
    ~NAMNetwork();

    // 禁用拷贝
    NAMNetwork(const NAMNetwork&) = delete;
    NAMNetwork& operator=(const NAMNetwork&) = delete;

    // 初始化和控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // 前向传播
    cv::Mat forward(const cv::Mat& input);
    std::vector<cv::Mat> forward_batch(const std::vector<cv::Mat>& inputs);

    // 特征提取
    std::vector<cv::Mat> extract_features(const cv::Mat& input);

    // 配置管理
    void set_config(const NetworkConfig& config);
    NetworkConfig get_config() const { return config_; }

    // 性能统计
    struct PerformanceStats {
        uint64_t total_forward_passes{0};
        double avg_forward_time_ms{0.0};
        double min_forward_time_ms{0.0};
        double max_forward_time_ms{0.0};
        std::vector<double> layer_times_ms;
    };
    PerformanceStats get_performance_stats() const;

private:
    // 内部方法
    bool create_layers();
    cv::Mat residual_connection(const cv::Mat& input, const cv::Mat& output);
    cv::Mat bottleneck_connection(const cv::Mat& input, const cv::Mat& output);

    // 配置和状态
    NetworkConfig config_;
    bool initialized_{false};

    // NAM 层
    std::vector<std::unique_ptr<NAMAttention>> nam_layers_;

    // 性能统计
    mutable std::mutex stats_mutex_;
    PerformanceStats performance_stats_;

    // 错误处理
    std::string last_error_;
    void set_error(const std::string& error);
};

} // namespace vision
} // namespace bamboo_cut 