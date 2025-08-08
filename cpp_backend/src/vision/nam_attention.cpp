#include "bamboo_cut/vision/nam_attention.h"
#include "bamboo_cut/core/logger.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <algorithm>

namespace bamboo_cut {
namespace vision {

NAMAttention::NAMAttention(const NAMConfig& config) 
    : config_(config), initialized_(false) {
    LOG_INFO("创建NAM注意力模块实例");
}

NAMAttention::~NAMAttention() {
    shutdown();
    LOG_INFO("销毁NAM注意力模块实例");
}

bool NAMAttention::initialize() {
    LOG_INFO("初始化NAM注意力模块");
    
    try {
        // 验证配置
        if (!config_.validate()) {
            LOG_ERROR("NAM配置验证失败");
            return false;
        }
        
        // 初始化模型参数
        if (!initialize_parameters()) {
            LOG_ERROR("模型参数初始化失败");
            return false;
        }
        
        // 初始化批归一化参数
        if (!initialize_batch_norm()) {
            LOG_ERROR("批归一化参数初始化失败");
            return false;
        }
        
        // 初始化优化器状态
        initialize_optimizer();
        
        initialized_ = true;
        LOG_INFO("NAM注意力模块初始化成功");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("NAM注意力模块初始化异常: {}", e.what());
        return false;
    }
}

void NAMAttention::shutdown() {
    if (!initialized_) {
        return;
    }
    
    LOG_INFO("关闭NAM注意力模块");
    initialized_ = false;
}

cv::Mat NAMAttention::forward(const cv::Mat& input) {
    if (!initialized_) {
        LOG_ERROR("NAM注意力模块未初始化");
        return cv::Mat();
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 计算通道注意力
        cv::Mat channel_attn = apply_channel_attention(input);
        
        // 计算空间注意力
        cv::Mat spatial_attn = apply_spatial_attention(input);
        
        // 融合注意力结果
        cv::Mat fused_attn = fuse_attention(channel_attn, spatial_attn);
        
        // 应用注意力到输入特征
        cv::Mat output = input.mul(fused_attn);
        
        // 更新性能统计
        update_performance_stats(start_time, channel_attn, spatial_attn, output);
        
        // 调用注意力回调
        if (attention_callback_) {
            attention_callback_(fused_attn);
        }
        
        return output;
        
    } catch (const std::exception& e) {
        LOG_ERROR("NAM前向传播异常: {}", e.what());
        return cv::Mat();
    }
}

std::vector<cv::Mat> NAMAttention::forward_batch(const std::vector<cv::Mat>& inputs) {
    std::vector<cv::Mat> outputs;
    outputs.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        outputs.push_back(forward(input));
    }
    
    return outputs;
}

cv::Mat NAMAttention::apply_channel_attention(const cv::Mat& input) {
    // 计算通道注意力权重
    cv::Mat channel_weights = compute_channel_attention(input);
    
    // 应用通道注意力
    cv::Mat output = input.clone();
    for (int c = 0; c < input.channels(); ++c) {
        cv::Mat channel = input.channel(c);
        float weight = channel_weights.at<float>(c);
        output.channel(c) = channel * weight;
    }
    
    return output;
}

cv::Mat NAMAttention::apply_spatial_attention(const cv::Mat& input) {
    // 计算空间注意力权重
    cv::Mat spatial_weights = compute_spatial_attention(input);
    
    // 应用空间注意力
    cv::Mat output = input.clone();
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            float weight = spatial_weights.at<float>(y, x);
            for (int c = 0; c < input.channels(); ++c) {
                output.at<cv::Vec3f>(y, x)[c] *= weight;
            }
        }
    }
    
    return output;
}

cv::Mat NAMAttention::fuse_attention(const cv::Mat& channel_attn, const cv::Mat& spatial_attn) {
    // 融合通道和空间注意力
    cv::Mat fused = channel_attn.mul(spatial_attn);
    
    // 应用归一化
    if (config_.use_batch_norm) {
        fused = batch_normalization(fused, "fused_attention");
    } else if (config_.use_layer_norm) {
        fused = layer_normalization(fused, "fused_attention");
    }
    
    // 应用激活函数
    fused = sigmoid_activation(fused);
    
    return fused;
}

cv::Mat NAMAttention::compute_channel_attention(const cv::Mat& input) {
    // 全局池化
    cv::Mat pooled = channel_global_pooling(input);
    
    // 全连接层
    cv::Mat fc_output = channel_fc_layers(pooled);
    
    // Sigmoid激活
    cv::Mat weights = channel_sigmoid_activation(fc_output);
    
    return weights;
}

cv::Mat NAMAttention::channel_global_pooling(const cv::Mat& input) {
    cv::Mat pooled;
    
    if (input.channels() == 3) {
        // 对于彩色图像，计算每个通道的平均值
        std::vector<cv::Mat> channels;
        cv::split(input, channels);
        
        cv::Mat avg_pool, max_pool;
        cv::reduce(channels[0], avg_pool, 0, cv::REDUCE_AVG);
        cv::reduce(channels[0], max_pool, 0, cv::REDUCE_MAX);
        
        // 结合平均池化和最大池化
        pooled = (avg_pool + max_pool) * 0.5f;
    } else {
        // 对于灰度图像
        cv::reduce(input, pooled, 0, cv::REDUCE_AVG);
    }
    
    return pooled;
}

cv::Mat NAMAttention::channel_fc_layers(const cv::Mat& pooled) {
    // 第一个全连接层（降维）
    int reduced_channels = static_cast<int>(pooled.total() / config_.channel_reduction_ratio);
    cv::Mat fc1 = pooled.reshape(1, 1) * channel_fc_weights_[0] + channel_fc_biases_[0];
    fc1 = relu_activation(fc1);
    
    // 第二个全连接层（升维）
    cv::Mat fc2 = fc1 * channel_fc_weights_[1] + channel_fc_biases_[1];
    
    return fc2;
}

cv::Mat NAMAttention::channel_sigmoid_activation(const cv::Mat& fc_output) {
    cv::Mat sigmoid;
    cv::exp(-fc_output, sigmoid);
    sigmoid = 1.0f / (1.0f + sigmoid);
    return sigmoid;
}

cv::Mat NAMAttention::compute_spatial_attention(const cv::Mat& input) {
    // 空间卷积层
    cv::Mat conv_output = spatial_conv_layer(input);
    
    // Sigmoid激活
    cv::Mat weights = spatial_sigmoid_activation(conv_output);
    
    return weights;
}

cv::Mat NAMAttention::spatial_conv_layer(const cv::Mat& input) {
    // 创建卷积核
    int kernel_size = static_cast<int>(config_.spatial_kernel_size);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    
    // 应用卷积
    cv::Mat conv_output;
    cv::filter2D(input, conv_output, -1, kernel);
    
    return conv_output;
}

cv::Mat NAMAttention::spatial_sigmoid_activation(const cv::Mat& conv_output) {
    cv::Mat sigmoid;
    cv::exp(-conv_output, sigmoid);
    sigmoid = 1.0f / (1.0f + sigmoid);
    return sigmoid;
}

cv::Mat NAMAttention::batch_normalization(const cv::Mat& input, const std::string& layer_name) {
    auto& running_mean = bn_running_mean_[layer_name];
    auto& running_var = bn_running_var_[layer_name];
    auto& gamma = bn_weights_[layer_name];
    auto& beta = bn_biases_[layer_name];
    
    cv::Mat normalized = (input - running_mean) / cv::sqrt(running_var + config_.normalization_epsilon);
    return gamma.mul(normalized) + beta;
}

cv::Mat NAMAttention::layer_normalization(const cv::Mat& input, const std::string& layer_name) {
    // 计算均值和方差
    cv::Scalar mean, stddev;
    cv::meanStdDev(input, mean, stddev);
    
    // 归一化
    cv::Mat normalized = (input - mean[0]) / (stddev[0] + config_.normalization_epsilon);
    
    // 应用缩放和偏移
    auto& gamma = bn_weights_[layer_name];
    auto& beta = bn_biases_[layer_name];
    
    return gamma.mul(normalized) + beta;
}

cv::Mat NAMAttention::instance_normalization(const cv::Mat& input, const std::string& layer_name) {
    // 对每个通道分别进行归一化
    cv::Mat normalized = input.clone();
    
    for (int c = 0; c < input.channels(); ++c) {
        cv::Mat channel = input.channel(c);
        cv::Scalar mean, stddev;
        cv::meanStdDev(channel, mean, stddev);
        
        cv::Mat norm_channel = (channel - mean[0]) / (stddev[0] + config_.normalization_epsilon);
        normalized.channel(c) = norm_channel;
    }
    
    // 应用缩放和偏移
    auto& gamma = bn_weights_[layer_name];
    auto& beta = bn_biases_[layer_name];
    
    return gamma.mul(normalized) + beta;
}

cv::Mat NAMAttention::relu_activation(const cv::Mat& input) {
    cv::Mat output = input.clone();
    cv::threshold(output, output, 0, 0, cv::THRESH_TOZERO);
    return output;
}

cv::Mat NAMAttention::sigmoid_activation(const cv::Mat& input) {
    cv::Mat sigmoid;
    cv::exp(-input, sigmoid);
    sigmoid = 1.0f / (1.0f + sigmoid);
    return sigmoid;
}

cv::Mat NAMAttention::tanh_activation(const cv::Mat& input) {
    cv::Mat output;
    cv::tanh(input, output);
    return output;
}

void NAMAttention::update_performance_stats(const std::chrono::steady_clock::time_point& start_time,
                                          const cv::Mat& channel_attn, const cv::Mat& spatial_attn,
                                          const cv::Mat& output) {
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    performance_stats_.total_forward_passes++;
    performance_stats_.avg_processing_time_ms = duration.count() / 1000.0;
    performance_stats_.fps = 1000.0 / performance_stats_.avg_processing_time_ms;
    performance_stats_.last_update = end_time;
    
    // 计算注意力统计
    cv::Scalar channel_mean = cv::mean(channel_attn);
    cv::Scalar spatial_mean = cv::mean(spatial_attn);
    
    performance_stats_.avg_channel_attention_weight = channel_mean[0];
    performance_stats_.avg_spatial_attention_weight = spatial_mean[0];
    
    // 计算特征增强比例
    cv::Scalar output_mean = cv::mean(output);
    cv::Scalar input_mean = cv::mean(output);
    performance_stats_.avg_feature_enhancement_ratio = output_mean[0] / (input_mean[0] + 1e-6f);
}

NAMAttention::PerformanceStats NAMAttention::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return performance_stats_;
}

void NAMAttention::reset_performance_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    performance_stats_ = PerformanceStats{};
}

bool NAMAttention::initialize_parameters() {
    // 初始化通道注意力全连接层权重
    int input_channels = 3; // 假设输入是RGB图像
    int reduced_channels = static_cast<int>(input_channels / config_.channel_reduction_ratio);
    
    // 第一个全连接层（降维）
    channel_fc_weights_.push_back(cv::Mat::randn(reduced_channels, input_channels) * 0.01f);
    channel_fc_biases_.push_back(cv::Mat::zeros(reduced_channels, 1, CV_32F));
    
    // 第二个全连接层（升维）
    channel_fc_weights_.push_back(cv::Mat::randn(input_channels, reduced_channels) * 0.01f);
    channel_fc_biases_.push_back(cv::Mat::zeros(input_channels, 1, CV_32F));
    
    // 初始化空间注意力卷积核
    int kernel_size = static_cast<int>(config_.spatial_kernel_size);
    spatial_conv_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    spatial_conv_bias_ = cv::Mat::zeros(1, 1, CV_32F);
    
    return true;
}

bool NAMAttention::initialize_batch_norm() {
    // 初始化批归一化参数
    std::vector<std::string> layer_names = {"channel_attention", "spatial_attention", "fused_attention"};
    
    for (const auto& name : layer_names) {
        bn_weights_[name] = cv::Mat::ones(1, 1, CV_32F);
        bn_biases_[name] = cv::Mat::zeros(1, 1, CV_32F);
        bn_running_mean_[name] = cv::Mat::zeros(1, 1, CV_32F);
        bn_running_var_[name] = cv::Mat::ones(1, 1, CV_32F);
    }
    
    return true;
}

void NAMAttention::initialize_optimizer() {
    // 初始化优化器状态
    learning_rate_ = config_.learning_rate;
    weight_decay_ = config_.weight_decay;
    
    // 为每个权重矩阵创建动量和速度
    for (size_t i = 0; i < channel_fc_weights_.size(); ++i) {
        momentum_.push_back(cv::Mat::zeros(channel_fc_weights_[i].size(), CV_32F));
        velocity_.push_back(cv::Mat::zeros(channel_fc_weights_[i].size(), CV_32F));
    }
}

void NAMAttention::set_attention_callback(AttentionCallback callback) {
    attention_callback_ = callback;
}

void NAMAttention::set_training_callback(TrainingCallback callback) {
    training_callback_ = callback;
}

void NAMAttention::set_config(const NAMConfig& config) {
    config_ = config;
}

bool NAMConfig::validate() const {
    if (channel_reduction_ratio <= 0) {
        return false;
    }
    if (spatial_kernel_size <= 0) {
        return false;
    }
    if (normalization_epsilon <= 0) {
        return false;
    }
    if (learning_rate <= 0) {
        return false;
    }
    if (dropout_rate < 0 || dropout_rate > 1) {
        return false;
    }
    return true;
}

} // namespace vision
} // namespace bamboo_cut 