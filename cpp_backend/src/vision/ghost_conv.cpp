#include "bamboo_cut/vision/ghost_conv.h"
#include "bamboo_cut/core/logger.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <algorithm>

namespace bamboo_cut {
namespace vision {

GhostConv::GhostConv(const GhostConvConfig& config) 
    : config_(config), initialized_(false) {
    LOG_INFO("创建GhostConv实例");
}

GhostConv::~GhostConv() {
    shutdown();
    LOG_INFO("销毁GhostConv实例");
}

bool GhostConv::initialize() {
    LOG_INFO("初始化GhostConv");
    
    try {
        // 验证配置
        if (!config_.validate()) {
            LOG_ERROR("GhostConv配置验证失败");
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
        LOG_INFO("GhostConv初始化成功");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("GhostConv初始化异常: {}", e.what());
        return false;
    }
}

void GhostConv::shutdown() {
    if (!initialized_) {
        return;
    }
    
    LOG_INFO("关闭GhostConv");
    initialized_ = false;
}

cv::Mat GhostConv::forward(const cv::Mat& input) {
    if (!initialized_) {
        LOG_ERROR("GhostConv未初始化");
        return cv::Mat();
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 生成主要特征
        cv::Mat primary_features = generate_primary_features(input);
        
        // 生成幽灵特征
        cv::Mat ghost_features = generate_ghost_features(primary_features);
        
        // 拼接特征
        cv::Mat output = concatenate_features(primary_features, ghost_features);
        
        // 应用批归一化
        if (config_.use_batch_norm) {
            output = batch_normalization(output, "ghost_conv");
        }
        
        // 应用激活函数
        if (config_.use_relu) {
            output = relu_activation(output);
        }
        
        // 应用Dropout
        if (config_.dropout_rate > 0) {
            output = apply_dropout(output);
        }
        
        // 应用残差连接
        if (config_.enable_residual && input.size() == output.size()) {
            output = apply_residual_connection(output, input);
        }
        
        // 更新性能统计
        update_performance_stats(start_time, primary_features, ghost_features, output);
        
        // 调用特征回调
        if (feature_callback_) {
            feature_callback_(primary_features, ghost_features);
        }
        
        return output;
        
    } catch (const std::exception& e) {
        LOG_ERROR("GhostConv前向传播异常: {}", e.what());
        return cv::Mat();
    }
}

std::vector<cv::Mat> GhostConv::forward_batch(const std::vector<cv::Mat>& inputs) {
    std::vector<cv::Mat> outputs;
    outputs.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        outputs.push_back(forward(input));
    }
    
    return outputs;
}

cv::Mat GhostConv::generate_primary_features(const cv::Mat& input) {
    // 使用少量卷积核生成主要特征
    cv::Mat primary_features = apply_convolution(input, primary_conv_kernel_, primary_conv_bias_);
    
    // 应用批归一化
    if (config_.use_batch_norm) {
        primary_features = batch_normalization(primary_features, "primary_features");
    }
    
    // 应用激活函数
    if (config_.use_relu) {
        primary_features = relu_activation(primary_features);
    }
    
    return primary_features;
}

cv::Mat GhostConv::generate_ghost_features(const cv::Mat& primary_features) {
    // 通过线性变换生成幽灵特征
    cv::Mat ghost_features = apply_linear_transform(primary_features, ghost_linear_weight_, ghost_linear_bias_);
    
    // 应用批归一化
    if (config_.use_batch_norm) {
        ghost_features = batch_normalization(ghost_features, "ghost_features");
    }
    
    // 应用激活函数
    if (config_.use_relu) {
        ghost_features = relu_activation(ghost_features);
    }
    
    return ghost_features;
}

cv::Mat GhostConv::concatenate_features(const cv::Mat& primary, const cv::Mat& ghost) {
    // 在通道维度上拼接主要特征和幽灵特征
    std::vector<cv::Mat> channels;
    
    // 分离主要特征的通道
    for (int c = 0; c < primary.channels(); ++c) {
        channels.push_back(primary.channel(c));
    }
    
    // 分离幽灵特征的通道
    for (int c = 0; c < ghost.channels(); ++c) {
        channels.push_back(ghost.channel(c));
    }
    
    // 合并所有通道
    cv::Mat concatenated;
    cv::merge(channels, concatenated);
    
    return concatenated;
}

cv::Mat GhostConv::apply_convolution(const cv::Mat& input, const cv::Mat& kernel, const cv::Mat& bias) {
    cv::Mat output;
    cv::filter2D(input, output, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    
    // 添加偏置
    if (config_.use_bias && !bias.empty()) {
        for (int c = 0; c < output.channels(); ++c) {
            cv::Mat channel = output.channel(c);
            channel += bias.at<float>(c);
        }
    }
    
    return output;
}

cv::Mat GhostConv::apply_depthwise_convolution(const cv::Mat& input, const cv::Mat& kernel) {
    cv::Mat output;
    cv::filter2D(input, output, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    return output;
}

cv::Mat GhostConv::apply_pointwise_convolution(const cv::Mat& input, const cv::Mat& kernel) {
    // 1x1卷积实现
    cv::Mat output;
    cv::filter2D(input, output, -1, kernel, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
    return output;
}

cv::Mat GhostConv::apply_linear_transform(const cv::Mat& input, const cv::Mat& weight, const cv::Mat& bias) {
    // 重塑输入为2D矩阵
    cv::Mat input_2d = input.reshape(1, input.total());
    
    // 应用线性变换
    cv::Mat output_2d = input_2d * weight.t();
    
    // 添加偏置
    if (config_.use_bias && !bias.empty()) {
        output_2d += bias.t();
    }
    
    // 重塑回原始形状
    cv::Mat output = output_2d.reshape(input.channels(), input.rows);
    
    return output;
}

cv::Mat GhostConv::batch_normalization(const cv::Mat& input, const std::string& layer_name) {
    auto& running_mean = bn_running_mean_[layer_name];
    auto& running_var = bn_running_var_[layer_name];
    auto& gamma = bn_weights_[layer_name];
    auto& beta = bn_biases_[layer_name];
    
    cv::Mat normalized = (input - running_mean) / cv::sqrt(running_var + 1e-5f);
    return gamma.mul(normalized) + beta;
}

cv::Mat GhostConv::layer_normalization(const cv::Mat& input, const std::string& layer_name) {
    // 计算均值和方差
    cv::Scalar mean, stddev;
    cv::meanStdDev(input, mean, stddev);
    
    // 归一化
    cv::Mat normalized = (input - mean[0]) / (stddev[0] + 1e-5f);
    
    // 应用缩放和偏移
    auto& gamma = bn_weights_[layer_name];
    auto& beta = bn_biases_[layer_name];
    
    return gamma.mul(normalized) + beta;
}

cv::Mat GhostConv::relu_activation(const cv::Mat& input) {
    cv::Mat output = input.clone();
    cv::threshold(output, output, 0, 0, cv::THRESH_TOZERO);
    return output;
}

cv::Mat GhostConv::sigmoid_activation(const cv::Mat& input) {
    cv::Mat sigmoid;
    cv::exp(-input, sigmoid);
    sigmoid = 1.0f / (1.0f + sigmoid);
    return sigmoid;
}

cv::Mat GhostConv::tanh_activation(const cv::Mat& input) {
    cv::Mat output;
    cv::tanh(input, output);
    return output;
}

cv::Mat GhostConv::apply_dropout(const cv::Mat& input) {
    if (config_.dropout_rate <= 0) {
        return input;
    }
    
    cv::Mat output = input.clone();
    cv::RNG rng;
    
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            for (int c = 0; c < output.channels(); ++c) {
                if (rng.uniform(0.0f, 1.0f) < config_.dropout_rate) {
                    output.at<cv::Vec3f>(y, x)[c] = 0.0f;
                }
            }
        }
    }
    
    return output;
}

cv::Mat GhostConv::apply_residual_connection(const cv::Mat& input, const cv::Mat& residual) {
    return input + residual;
}

void GhostConv::update_performance_stats(const std::chrono::steady_clock::time_point& start_time,
                                       const cv::Mat& primary_features, const cv::Mat& ghost_features,
                                       const cv::Mat& output) {
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    performance_stats_.total_forward_passes++;
    performance_stats_.avg_processing_time_ms = duration.count() / 1000.0;
    performance_stats_.fps = 1000.0 / performance_stats_.avg_processing_time_ms;
    performance_stats_.last_update = end_time;
    
    // 计算特征统计
    cv::Scalar primary_mean = cv::mean(primary_features);
    cv::Scalar ghost_mean = cv::mean(ghost_features);
    
    performance_stats_.avg_primary_feature_weight = primary_mean[0];
    performance_stats_.avg_ghost_feature_weight = ghost_mean[0];
    
    // 计算计算量减少比例
    int total_channels = primary_features.channels() + ghost_features.channels();
    int primary_channels = primary_features.channels();
    performance_stats_.computation_reduction_ratio = static_cast<double>(primary_channels) / total_channels;
}

GhostConv::PerformanceStats GhostConv::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return performance_stats_;
}

void GhostConv::reset_performance_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    performance_stats_ = PerformanceStats{};
}

bool GhostConv::initialize_parameters() {
    // 初始化主要卷积核
    int kernel_size = config_.kernel_size;
    primary_conv_kernel_ = cv::Mat::randn(kernel_size, kernel_size, CV_32F) * 0.01f;
    
    // 初始化主要卷积偏置
    primary_conv_bias_ = cv::Mat::zeros(config_.ghost_channels, 1, CV_32F);
    
    // 初始化幽灵线性变换权重
    ghost_linear_weight_ = cv::Mat::randn(config_.ghost_channels, config_.ghost_channels, CV_32F) * 0.01f;
    
    // 初始化幽灵线性变换偏置
    ghost_linear_bias_ = cv::Mat::zeros(config_.ghost_channels, 1, CV_32F);
    
    return true;
}

bool GhostConv::initialize_batch_norm() {
    // 初始化批归一化参数
    std::vector<std::string> layer_names = {"primary_features", "ghost_features", "ghost_conv"};
    
    for (const auto& name : layer_names) {
        bn_weights_[name] = cv::Mat::ones(1, 1, CV_32F);
        bn_biases_[name] = cv::Mat::zeros(1, 1, CV_32F);
        bn_running_mean_[name] = cv::Mat::zeros(1, 1, CV_32F);
        bn_running_var_[name] = cv::Mat::ones(1, 1, CV_32F);
    }
    
    return true;
}

void GhostConv::initialize_optimizer() {
    // 初始化优化器状态
    learning_rate_ = 1e-4f;
    weight_decay_ = 1e-4f;
    
    // 为每个权重矩阵创建动量和速度
    std::vector<cv::Mat*> weights = {&primary_conv_kernel_, &ghost_linear_weight_};
    
    for (auto* weight : weights) {
        momentum_.push_back(cv::Mat::zeros(weight->size(), CV_32F));
        velocity_.push_back(cv::Mat::zeros(weight->size(), CV_32F));
    }
}

void GhostConv::set_feature_callback(FeatureCallback callback) {
    feature_callback_ = callback;
}

void GhostConv::set_training_callback(TrainingCallback callback) {
    training_callback_ = callback;
}

void GhostConv::set_config(const GhostConvConfig& config) {
    config_ = config;
}

bool GhostConvConfig::validate() const {
    if (in_channels <= 0 || out_channels <= 0) {
        return false;
    }
    if (kernel_size <= 0) {
        return false;
    }
    if (ghost_channels <= 0 || ghost_channels >= out_channels) {
        return false;
    }
    if (reduction_ratio <= 0) {
        return false;
    }
    if (dropout_rate < 0 || dropout_rate > 1) {
        return false;
    }
    return true;
}

} // namespace vision
} // namespace bamboo_cut 