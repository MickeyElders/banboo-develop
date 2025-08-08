#include "bamboo_cut/vision/vov_gscsp.h"
#include "bamboo_cut/core/logger.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <algorithm>

namespace bamboo_cut {
namespace vision {

bool VoVGSCSPConfig::validate() const {
    if (in_channels <= 0 || out_channels <= 0) {
        return false;
    }
    if (num_blocks <= 0) {
        return false;
    }
    if (attention_ratio < 0.0f || attention_ratio > 1.0f) {
        return false;
    }
    if (compression_ratio < 0.0f || compression_ratio > 1.0f) {
        return false;
    }
    return ghost_config.validate();
}

VoVGSCSP::VoVGSCSP(const VoVGSCSPConfig& config) 
    : config_(config), initialized_(false) {
    LOG_INFO("创建VoVGSCSP实例");
}

VoVGSCSP::~VoVGSCSP() {
    shutdown();
    LOG_INFO("销毁VoVGSCSP实例");
}

bool VoVGSCSP::initialize() {
    LOG_INFO("初始化VoVGSCSP");
    
    try {
        // 验证配置
        if (!config_.validate()) {
            LOG_ERROR("VoVGSCSP配置验证失败");
            return false;
        }
        
        // 初始化GhostConv模块
        for (int i = 0; i < config_.num_blocks; ++i) {
            auto ghost_conv = std::make_unique<GhostConv>(config_.ghost_config);
            if (!ghost_conv->initialize()) {
                LOG_ERROR("GhostConv {} 初始化失败", i);
                return false;
            }
            ghost_convs_.push_back(std::move(ghost_conv));
        }
        
        // 初始化多尺度卷积核
        for (int scale : config_.pyramid_scales) {
            cv::Mat kernel = cv::Mat::randn(scale, scale, CV_32F) * 0.01f;
            cv::Mat bias = cv::Mat::zeros(1, 1, CV_32F);
            scale_kernels_.push_back(kernel);
            scale_biases_.push_back(bias);
        }
        
        // 初始化注意力权重
        channel_attention_weights_ = cv::Mat::ones(1, 1, CV_32F);
        spatial_attention_weights_ = cv::Mat::ones(1, 1, CV_32F);
        
        // 初始化压缩矩阵
        compression_matrix_ = cv::Mat::eye(config_.in_channels, config_.in_channels, CV_32F);
        decompression_matrix_ = compression_matrix_.clone();
        
        // 初始化批归一化参数
        initialize_batch_norm();
        
        // 初始化优化器状态
        initialize_optimizer();
        
        initialized_ = true;
        LOG_INFO("VoVGSCSP初始化成功");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("VoVGSCSP初始化异常: {}", e.what());
        return false;
    }
}

void VoVGSCSP::shutdown() {
    if (!initialized_) {
        return;
    }
    
    LOG_INFO("关闭VoVGSCSP");
    initialized_ = false;
    
    ghost_convs_.clear();
    scale_kernels_.clear();
    scale_biases_.clear();
}

cv::Mat VoVGSCSP::forward(const cv::Mat& input) {
    if (!initialized_) {
        LOG_ERROR("VoVGSCSP未初始化");
        return cv::Mat();
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 多尺度处理
        cv::Mat multi_scale_features = process_multi_scale(input);
        
        // 应用空间金字塔
        cv::Mat pyramid_features = apply_spatial_pyramid(multi_scale_features);
        
        // 密集连接处理
        std::vector<cv::Mat> dense_features;
        cv::Mat current_features = pyramid_features;
        
        for (auto& ghost_conv : ghost_convs_) {
            cv::Mat ghost_output = ghost_conv->forward(current_features);
            dense_features.push_back(ghost_output);
            
            if (config_.use_residual) {
                current_features = current_features + ghost_output;
            } else {
                current_features = ghost_output;
            }
        }
        
        // 特征融合
        cv::Mat fused_features = fuse_features(dense_features);
        
        // 应用注意力机制
        if (config_.use_attention) {
            fused_features = apply_attention_fusion({fused_features});
        }
        
        // 特征压缩
        if (config_.enable_compression) {
            fused_features = compress_features(fused_features);
        }
        
        // 更新性能统计
        update_performance_stats(start_time, fused_features);
        
        return fused_features;
        
    } catch (const std::exception& e) {
        LOG_ERROR("VoVGSCSP前向传播异常: {}", e.what());
        return cv::Mat();
    }
}

std::vector<cv::Mat> VoVGSCSP::forward_batch(const std::vector<cv::Mat>& inputs) {
    std::vector<cv::Mat> outputs;
    outputs.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        outputs.push_back(forward(input));
    }
    
    return outputs;
}

cv::Mat VoVGSCSP::process_multi_scale(const cv::Mat& input) {
    std::vector<cv::Mat> multi_scale_features;
    
    for (size_t i = 0; i < config_.pyramid_scales.size(); ++i) {
        cv::Mat scale_features = apply_scale_convolution(input, config_.pyramid_scales[i]);
        multi_scale_features.push_back(scale_features);
    }
    
    return fuse_features(multi_scale_features);
}

cv::Mat VoVGSCSP::apply_spatial_pyramid(const cv::Mat& input) {
    std::vector<cv::Mat> pyramid_features;
    
    for (int scale : config_.pyramid_scales) {
        cv::Mat pyramid_feature = apply_pyramid_pooling(input, scale);
        pyramid_features.push_back(pyramid_feature);
    }
    
    return fuse_features(pyramid_features);
}

cv::Mat VoVGSCSP::apply_attention_fusion(const std::vector<cv::Mat>& features) {
    if (features.empty()) {
        return cv::Mat();
    }
    
    cv::Mat fused = features[0].clone();
    
    for (size_t i = 1; i < features.size(); ++i) {
        cv::Mat attention_weights = compute_channel_attention(features[i]);
        cv::Mat attended = apply_attention(features[i], attention_weights);
        fused = fused + attended * config_.attention_ratio;
    }
    
    return fused;
}

cv::Mat VoVGSCSP::fuse_features(const std::vector<cv::Mat>& features) {
    if (features.empty()) {
        return cv::Mat();
    }
    
    if (features.size() == 1) {
        return features[0];
    }
    
    // 简单的特征拼接
    std::vector<cv::Mat> channels;
    for (const auto& feature : features) {
        std::vector<cv::Mat> feature_channels;
        cv::split(feature, feature_channels);
        channels.insert(channels.end(), feature_channels.begin(), feature_channels.end());
    }
    
    cv::Mat fused;
    cv::merge(channels, fused);
    
    return fused;
}

cv::Mat VoVGSCSP::apply_dense_connection(const std::vector<cv::Mat>& features) {
    return fuse_features(features);
}

void VoVGSCSP::set_config(const VoVGSCSPConfig& config) {
    config_ = config;
}

VoVGSCSP::PerformanceStats VoVGSCSP::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void VoVGSCSP::reset_performance_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = PerformanceStats{};
}

bool VoVGSCSP::load_model(const std::string& model_path) {
    LOG_INFO("加载VoVGSCSP模型: {}", model_path);
    // TODO: 实现模型加载
    return true;
}

bool VoVGSCSP::save_model(const std::string& model_path) {
    LOG_INFO("保存VoVGSCSP模型: {}", model_path);
    // TODO: 实现模型保存
    return true;
}

bool VoVGSCSP::export_to_onnx(const std::string& onnx_path) {
    LOG_INFO("导出VoVGSCSP模型到ONNX: {}", onnx_path);
    // TODO: 实现ONNX导出
    return true;
}

bool VoVGSCSP::train(const std::vector<cv::Mat>& inputs, const std::vector<cv::Mat>& targets) {
    LOG_INFO("训练VoVGSCSP模型");
    // TODO: 实现训练逻辑
    return true;
}

bool VoVGSCSP::validate(const std::vector<cv::Mat>& inputs, const std::vector<cv::Mat>& targets) {
    LOG_INFO("验证VoVGSCSP模型");
    // TODO: 实现验证逻辑
    return true;
}

void VoVGSCSP::set_feature_callback(FeatureCallback callback) {
    feature_callback_ = callback;
}

void VoVGSCSP::set_training_callback(TrainingCallback callback) {
    training_callback_ = callback;
}

// 私有方法实现
cv::Mat VoVGSCSP::apply_scale_convolution(const cv::Mat& input, int scale) {
    cv::Mat output;
    cv::filter2D(input, output, -1, scale_kernels_[scale], cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    return output;
}

cv::Mat VoVGSCSP::apply_pyramid_pooling(const cv::Mat& input, int scale) {
    cv::Mat output;
    cv::resize(input, output, cv::Size(), 1.0/scale, 1.0/scale, cv::INTER_AREA);
    cv::resize(output, output, input.size(), 0, 0, cv::INTER_LINEAR);
    return output;
}

cv::Mat VoVGSCSP::compute_channel_attention(const cv::Mat& input) {
    cv::Mat attention = channel_attention_weights_.clone();
    return attention;
}

cv::Mat VoVGSCSP::compute_spatial_attention(const cv::Mat& input) {
    cv::Mat attention = spatial_attention_weights_.clone();
    return attention;
}

cv::Mat VoVGSCSP::apply_attention(const cv::Mat& input, const cv::Mat& attention_weights) {
    return input.mul(attention_weights);
}

cv::Mat VoVGSCSP::compress_features(const cv::Mat& input) {
    cv::Mat compressed;
    cv::gemm(input, compression_matrix_, 1.0, cv::Mat(), 0.0, compressed);
    return compressed;
}

cv::Mat VoVGSCSP::decompress_features(const cv::Mat& compressed) {
    cv::Mat decompressed;
    cv::gemm(compressed, decompression_matrix_, 1.0, cv::Mat(), 0.0, decompressed);
    return decompressed;
}

cv::Mat VoVGSCSP::batch_normalization(const cv::Mat& input, const std::string& layer_name) {
    auto it = bn_running_mean_.find(layer_name);
    if (it == bn_running_mean_.end()) {
        return input;
    }
    
    cv::Mat running_mean = it->second;
    cv::Mat running_var = bn_running_var_[layer_name];
    
    cv::Mat normalized = (input - running_mean) / cv::sqrt(running_var + 1e-5f);
    return normalized;
}

cv::Mat VoVGSCSP::layer_normalization(const cv::Mat& input, const std::string& layer_name) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(input, mean, stddev);
    
    cv::Mat normalized = (input - mean[0]) / (stddev[0] + 1e-5f);
    return normalized;
}

cv::Mat VoVGSCSP::relu_activation(const cv::Mat& input) {
    cv::Mat output = input.clone();
    cv::threshold(output, output, 0, 0, cv::THRESH_TOZERO);
    return output;
}

cv::Mat VoVGSCSP::sigmoid_activation(const cv::Mat& input) {
    cv::Mat sigmoid;
    cv::exp(-input, sigmoid);
    return 1.0 / (1.0 + sigmoid);
}

cv::Mat VoVGSCSP::tanh_activation(const cv::Mat& input) {
    cv::Mat output;
    cv::tanh(input, output);
    return output;
}

float VoVGSCSP::compute_loss(const cv::Mat& prediction, const cv::Mat& target) {
    cv::Mat diff = prediction - target;
    cv::Mat squared_diff;
    cv::multiply(diff, diff, squared_diff);
    
    cv::Scalar sum = cv::sum(squared_diff);
    return static_cast<float>(sum[0]) / (prediction.rows * prediction.cols * prediction.channels());
}

cv::Mat VoVGSCSP::compute_gradient(const cv::Mat& prediction, const cv::Mat& target) {
    return prediction - target;
}

void VoVGSCSP::update_weights(const std::vector<cv::Mat>& gradients) {
    // TODO: 实现权重更新
}

void VoVGSCSP::apply_weight_decay() {
    // TODO: 实现权重衰减
}

void VoVGSCSP::initialize_batch_norm() {
    std::vector<std::string> layer_names = {"vov_gscsp"};
    
    for (const auto& name : layer_names) {
        bn_weights_[name] = cv::Mat::ones(1, 1, CV_32F);
        bn_biases_[name] = cv::Mat::zeros(1, 1, CV_32F);
        bn_running_mean_[name] = cv::Mat::zeros(1, 1, CV_32F);
        bn_running_var_[name] = cv::Mat::ones(1, 1, CV_32F);
    }
}

void VoVGSCSP::initialize_optimizer() {
    std::vector<cv::Mat*> weights = {&channel_attention_weights_, &spatial_attention_weights_};
    
    for (auto* weight : weights) {
        momentum_.push_back(cv::Mat::zeros(weight->size(), CV_32F));
        velocity_.push_back(cv::Mat::zeros(weight->size(), CV_32F));
    }
}

void VoVGSCSP::update_performance_stats(const std::chrono::high_resolution_clock::time_point& start_time, 
                                       const cv::Mat& output) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double processing_time_ms = duration.count() / 1000.0;
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.total_forward_passes++;
    stats_.avg_processing_time_ms = 
        (stats_.avg_processing_time_ms * (stats_.total_forward_passes - 1) + processing_time_ms) / 
        stats_.total_forward_passes;
    
    if (processing_time_ms < stats_.min_processing_time_ms || stats_.min_processing_time_ms == 0.0) {
        stats_.min_processing_time_ms = processing_time_ms;
    }
    
    if (processing_time_ms > stats_.max_processing_time_ms) {
        stats_.max_processing_time_ms = processing_time_ms;
    }
    
    stats_.fps = 1000.0 / stats_.avg_processing_time_ms;
    stats_.last_update = end_time;
}

void VoVGSCSP::set_error(const std::string& error) {
    last_error_ = error;
    LOG_ERROR("VoVGSCSP错误: {}", error);
}

} // namespace vision
} // namespace bamboo_cut 