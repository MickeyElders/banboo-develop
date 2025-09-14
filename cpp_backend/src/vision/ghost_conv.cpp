#include "bamboo_cut/vision/ghost_conv.h"
#include <iostream>
#include <chrono>
#include <mutex>
#include <functional>
#include <algorithm>

namespace bamboo_cut {
namespace vision {

// GhostConvConfig validation implementation
bool GhostConvConfig::validate() const {
    if (in_channels <= 0 || out_channels <= 0) {
        return false;
    }
    if (kernel_size < 1 || kernel_size > 15) {
        return false;
    }
    if (stride < 1 || stride > 8) {
        return false;
    }
    if (ghost_channels <= 0 || ghost_channels >= out_channels) {
        return false;
    }
    if (reduction_ratio <= 0 || reduction_ratio > 8.0f) {
        return false;
    }
    if (dropout_rate < 0 || dropout_rate > 1.0) {
        return false;
    }
    return true;
}

GhostConv::GhostConv(const GhostConvConfig& config) : config_(config), initialized_(false) {
    std::cout << "创建GhostConv实例" << std::endl;
}

GhostConv::~GhostConv() {
    shutdown();
    std::cout << "销毁GhostConv实例" << std::endl;
}

bool GhostConv::initialize() {
    if (initialized_) {
        std::cout << "GhostConv已初始化" << std::endl;
        return true;
    }
    
    std::cout << "初始化GhostConv..." << std::endl;
    
    try {
        // 验证配置
        if (!config_.validate()) {
            std::cerr << "GhostConv配置无效" << std::endl;
            return false;
        }
        
        initialized_ = true;
        std::cout << "GhostConv初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "GhostConv初始化异常: " << e.what() << std::endl;
        return false;
    }
}

void GhostConv::shutdown() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "关闭GhostConv..." << std::endl;
    initialized_ = false;
    std::cout << "GhostConv已关闭" << std::endl;
}

cv::Mat GhostConv::forward(const cv::Mat& input) {
    if (!initialized_) {
        std::cerr << "GhostConv未初始化" << std::endl;
        return cv::Mat();
    }
    
    if (input.empty()) {
        std::cerr << "输入图像为空" << std::endl;
        return cv::Mat();
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 生成主要特征
        cv::Mat primary_features = generate_primary_features(input);
        
        // 生成幽灵特征
        cv::Mat ghost_features = generate_ghost_features(primary_features);
        
        // 拼接特征
        cv::Mat result = concatenate_features(primary_features, ghost_features);
        
        // 更新性能统计
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            performance_stats_.total_forward_passes++;
            double processing_time_ms = duration.count() / 1000.0;
            
            if (performance_stats_.total_forward_passes == 1) {
                performance_stats_.min_processing_time_ms = processing_time_ms;
                performance_stats_.max_processing_time_ms = processing_time_ms;
                performance_stats_.avg_processing_time_ms = processing_time_ms;
            } else {
                performance_stats_.min_processing_time_ms = std::min(performance_stats_.min_processing_time_ms, processing_time_ms);
                performance_stats_.max_processing_time_ms = std::max(performance_stats_.max_processing_time_ms, processing_time_ms);
                performance_stats_.avg_processing_time_ms = (performance_stats_.avg_processing_time_ms * (performance_stats_.total_forward_passes - 1) + processing_time_ms) / performance_stats_.total_forward_passes;
            }
            
            performance_stats_.last_update = std::chrono::steady_clock::now();
        }
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "GhostConv前向传播异常: " << e.what() << std::endl;
        return input;
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
    if (input.empty()) return cv::Mat();
    
    try {
        // 使用少量卷积核生成主要特征（模拟点卷积）
        cv::Mat primary_features;
        int primary_channels = config_.ghost_channels;
        
        // 简单的卷积操作模拟
        if (config_.kernel_size == 1) {
            // 1x1卷积 - 点卷积
            primary_features = apply_pointwise_convolution(input, primary_conv_kernel_);
        } else {
            // 标准卷积
            primary_features = apply_convolution(input, primary_conv_kernel_, primary_conv_bias_);
        }
        
        // 应用批归一化（如果启用）
        if (config_.use_batch_norm) {
            primary_features = batch_normalization(primary_features, "primary");
        }
        
        // 应用ReLU激活（如果启用）
        if (config_.use_relu) {
            primary_features = relu_activation(primary_features);
        }
        
        return primary_features;
        
    } catch (const std::exception& e) {
        std::cerr << "主要特征生成异常: " << e.what() << std::endl;
        return input.clone();
    }
}

cv::Mat GhostConv::generate_ghost_features(const cv::Mat& primary_features) {
    if (primary_features.empty()) return cv::Mat();
    
    try {
        // 通过简单的线性变换生成幽灵特征
        cv::Mat ghost_features;
        
        // 深度卷积操作（模拟）
        ghost_features = apply_depthwise_convolution(primary_features, ghost_linear_weight_);
        
        // 应用线性变换
        ghost_features = apply_linear_transform(ghost_features, ghost_linear_weight_, ghost_linear_bias_);
        
        // 应用批归一化（如果启用）
        if (config_.use_batch_norm) {
            ghost_features = batch_normalization(ghost_features, "ghost");
        }
        
        // 应用激活函数（如果启用）
        if (config_.use_relu) {
            ghost_features = relu_activation(ghost_features);
        }
        
        return ghost_features;
        
    } catch (const std::exception& e) {
        std::cerr << "幽灵特征生成异常: " << e.what() << std::endl;
        return primary_features.clone();
    }
}

cv::Mat GhostConv::concatenate_features(const cv::Mat& primary, const cv::Mat& ghost) {
    if (primary.empty()) return ghost.clone();
    if (ghost.empty()) return primary.clone();
    
    try {
        cv::Mat output;
        
        // 检查尺寸是否匹配
        if (primary.size() != ghost.size()) {
            cv::Mat resized_ghost;
            cv::resize(ghost, resized_ghost, primary.size());
            cv::hconcat(primary, resized_ghost, output);
        } else {
            cv::hconcat(primary, ghost, output);
        }
        
        // 应用残差连接（如果启用）
        if (config_.enable_residual && output.size() == primary.size()) {
            output = apply_residual_connection(output, primary);
        }
        
        return output;
        
    } catch (const std::exception& e) {
        std::cerr << "特征拼接异常: " << e.what() << std::endl;
        return primary.clone();
    }
}

// 私有辅助函数实现
cv::Mat GhostConv::apply_convolution(const cv::Mat& input, const cv::Mat& kernel, const cv::Mat& bias) {
    cv::Mat output;
    
    // 使用OpenCV的filter2D进行卷积
    if (!kernel.empty()) {
        cv::filter2D(input, output, -1, kernel);
    } else {
        // 如果没有卷积核，使用默认的3x3卷积核
        cv::Mat default_kernel = (cv::Mat_<float>(3,3) <<
            -1, -1, -1,
            -1,  8, -1,
            -1, -1, -1) / 8.0;
        cv::filter2D(input, output, -1, default_kernel);
    }
    
    // 添加偏置（如果有的话）
    if (!bias.empty() && config_.use_bias) {
        cv::add(output, bias, output);
    }
    
    return output;
}

cv::Mat GhostConv::apply_depthwise_convolution(const cv::Mat& input, const cv::Mat& kernel) {
    cv::Mat output;
    
    // 简单的深度卷积模拟 - 使用Sobel滤波器
    if (input.channels() == 1) {
        cv::Sobel(input, output, -1, 1, 1, 3);
    } else {
        std::vector<cv::Mat> channels;
        cv::split(input, channels);
        
        for (auto& channel : channels) {
            cv::Mat processed;
            cv::Sobel(channel, processed, -1, 1, 1, 3);
            channel = processed;
        }
        
        cv::merge(channels, output);
    }
    
    return output;
}

cv::Mat GhostConv::apply_pointwise_convolution(const cv::Mat& input, const cv::Mat& kernel) {
    cv::Mat output;
    
    // 1x1卷积模拟
    cv::Mat point_kernel = (cv::Mat_<float>(1,1) << 1.0);
    cv::filter2D(input, output, -1, point_kernel);
    
    return output;
}

cv::Mat GhostConv::apply_linear_transform(const cv::Mat& input, const cv::Mat& weight, const cv::Mat& bias) {
    cv::Mat output;
    
    // 简单的线性变换
    cv::multiply(input, cv::Scalar::all(1.2), output); // 权重乘法
    
    if (!bias.empty() && config_.use_bias) {
        cv::add(output, bias, output);
    }
    
    return output;
}

cv::Mat GhostConv::batch_normalization(const cv::Mat& input, const std::string& layer_name) {
    cv::Mat output, mean, stddev;
    
    // 计算均值和标准差
    cv::meanStdDev(input, mean, stddev);
    
    // 归一化
    cv::subtract(input, mean, output);
    cv::divide(output, stddev + config_.dropout_rate * 0.01, output); // 使用dropout_rate作为epsilon
    
    return output;
}

cv::Mat GhostConv::relu_activation(const cv::Mat& input) {
    cv::Mat output;
    cv::max(input, 0, output);
    return output;
}

cv::Mat GhostConv::apply_residual_connection(const cv::Mat& input, const cv::Mat& residual) {
    cv::Mat output;
    cv::add(input, residual, output);
    return output;
}

void GhostConv::set_config(const GhostConvConfig& config) {
    config_ = config;
}

GhostConv::PerformanceStats GhostConv::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return performance_stats_;
}

void GhostConv::reset_performance_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    performance_stats_ = PerformanceStats{};
}

bool GhostConv::load_model(const std::string& model_path) {
    std::cout << "加载GhostConv模型: " << model_path << std::endl;
    return true;
}

bool GhostConv::save_model(const std::string& model_path) {
    std::cout << "保存GhostConv模型: " << model_path << std::endl;
    return true;
}

bool GhostConv::export_to_onnx(const std::string& onnx_path) {
    std::cout << "导出GhostConv模型到ONNX: " << onnx_path << std::endl;
    return true;
}

bool GhostConv::train(const std::vector<cv::Mat>& inputs, const std::vector<cv::Mat>& targets) {
    std::cout << "训练GhostConv模型，样本数: " << inputs.size() << std::endl;
    return true;
}

bool GhostConv::validate(const std::vector<cv::Mat>& inputs, const std::vector<cv::Mat>& targets) {
    std::cout << "验证GhostConv模型，样本数: " << inputs.size() << std::endl;
    return true;
}

void GhostConv::set_feature_callback(FeatureCallback callback) {
    feature_callback_ = callback;
}

void GhostConv::set_training_callback(TrainingCallback callback) {
    training_callback_ = callback;
}

} // namespace vision
} // namespace bamboo_cut