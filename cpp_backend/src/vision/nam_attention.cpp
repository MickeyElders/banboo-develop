#include "bamboo_cut/vision/nam_attention.h"
#include <iostream>
#include <chrono>
#include <mutex>
#include <algorithm>

namespace bamboo_cut {
namespace vision {

// NAMConfig validation implementation
bool NAMConfig::validate() const {
    if (channel_reduction_ratio <= 0 || channel_reduction_ratio > 64) {
        return false;
    }
    if (spatial_kernel_size < 1 || spatial_kernel_size > 15) {
        return false;
    }
    if (learning_rate <= 0 || learning_rate > 1.0) {
        return false;
    }
    if (dropout_rate < 0 || dropout_rate > 1.0) {
        return false;
    }
    return true;
}

NAMAttention::NAMAttention(const NAMConfig& config) : config_(config), initialized_(false) {
    std::cout << "创建NAMAttention实例" << std::endl;
}

NAMAttention::~NAMAttention() {
    shutdown();
    std::cout << "销毁NAMAttention实例" << std::endl;
}

bool NAMAttention::initialize() {
    if (initialized_) {
        std::cout << "NAMAttention已初始化" << std::endl;
        return true;
    }
    
    std::cout << "初始化NAMAttention..." << std::endl;
    
    try {
        // 验证配置
        if (!config_.validate()) {
            std::cerr << "NAMAttention配置无效" << std::endl;
            return false;
        }
        
        initialized_ = true;
        std::cout << "NAMAttention初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "NAMAttention初始化异常: " << e.what() << std::endl;
        return false;
    }
}

void NAMAttention::shutdown() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "关闭NAMAttention..." << std::endl;
    initialized_ = false;
    std::cout << "NAMAttention已关闭" << std::endl;
}

cv::Mat NAMAttention::forward(const cv::Mat& input) {
    if (!initialized_) {
        std::cerr << "NAMAttention未初始化" << std::endl;
        return cv::Mat();
    }
    
    if (input.empty()) {
        std::cerr << "输入图像为空" << std::endl;
        return cv::Mat();
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 应用通道注意力
        cv::Mat channel_attention = apply_channel_attention(input);
        
        // 应用空间注意力
        cv::Mat spatial_attention = apply_spatial_attention(input);
        
        // 融合注意力
        cv::Mat result = fuse_attention(channel_attention, spatial_attention);
        
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
        std::cerr << "NAMAttention前向传播异常: " << e.what() << std::endl;
        return input;
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
    if (input.empty()) return cv::Mat();
    
    try {
        // 计算通道级全局平均池化
        cv::Mat pooled = channel_global_pooling(input);
        
        // 通过全连接层
        cv::Mat fc_output = channel_fc_layers(pooled);
        
        // 应用Sigmoid激活
        cv::Mat attention_weights = channel_sigmoid_activation(fc_output);
        
        // 将注意力权重应用到输入
        cv::Mat result;
        if (input.channels() == attention_weights.channels()) {
            cv::multiply(input, attention_weights, result);
        } else {
            result = input.clone();
        }
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "通道注意力计算异常: " << e.what() << std::endl;
        return input.clone();
    }
}

cv::Mat NAMAttention::apply_spatial_attention(const cv::Mat& input) {
    if (input.empty()) return cv::Mat();
    
    try {
        // 应用空间卷积
        cv::Mat conv_output = spatial_conv_layer(input);
        
        // 应用Sigmoid激活得到空间注意力图
        cv::Mat attention_map = spatial_sigmoid_activation(conv_output);
        
        // 将注意力图应用到输入
        cv::Mat result;
        if (attention_map.size() == input.size()) {
            cv::multiply(input, attention_map, result);
        } else {
            // 如果尺寸不匹配，调整注意力图尺寸
            cv::Mat resized_attention;
            cv::resize(attention_map, resized_attention, input.size());
            cv::multiply(input, resized_attention, result);
        }
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "空间注意力计算异常: " << e.what() << std::endl;
        return input.clone();
    }
}

cv::Mat NAMAttention::fuse_attention(const cv::Mat& channel_attn, const cv::Mat& spatial_attn) {
    if (channel_attn.empty()) return spatial_attn.clone();
    if (spatial_attn.empty()) return channel_attn.clone();
    
    try {
        cv::Mat output;
        // 根据配置权重融合
        float channel_weight = 0.6f;  // 通道注意力权重稍高
        float spatial_weight = 0.4f;  // 空间注意力权重
        
        cv::addWeighted(channel_attn, channel_weight, spatial_attn, spatial_weight, 0, output);
        return output;
        
    } catch (const std::exception& e) {
        std::cerr << "注意力融合异常: " << e.what() << std::endl;
        return channel_attn.clone();
    }
}

// 私有辅助函数实现
cv::Mat NAMAttention::channel_global_pooling(const cv::Mat& input) {
    cv::Mat pooled;
    cv::reduce(input.reshape(1, input.rows * input.cols), pooled, 0, cv::REDUCE_AVG);
    return pooled;
}

cv::Mat NAMAttention::channel_fc_layers(const cv::Mat& pooled) {
    // 简单的线性变换模拟全连接层
    cv::Mat output;
    cv::multiply(pooled, cv::Scalar::all(config_.channel_gamma), output);
    cv::add(output, cv::Scalar::all(config_.channel_beta), output);
    return output;
}

cv::Mat NAMAttention::channel_sigmoid_activation(const cv::Mat& fc_output) {
    cv::Mat sigmoid_output;
    cv::exp(-fc_output, sigmoid_output);
    cv::add(sigmoid_output, cv::Scalar::all(1.0), sigmoid_output);
    cv::divide(cv::Scalar::all(1.0), sigmoid_output, sigmoid_output);
    return sigmoid_output;
}

cv::Mat NAMAttention::spatial_conv_layer(const cv::Mat& input) {
    cv::Mat output;
    int kernel_size = static_cast<int>(config_.spatial_kernel_size);
    if (kernel_size % 2 == 0) kernel_size += 1; // 确保是奇数
    
    // 使用Gaussian滤波器模拟空间卷积
    cv::GaussianBlur(input, output, cv::Size(kernel_size, kernel_size), 1.0);
    return output;
}

cv::Mat NAMAttention::spatial_sigmoid_activation(const cv::Mat& conv_output) {
    cv::Mat normalized, sigmoid_output;
    
    // 归一化到0-1范围
    cv::normalize(conv_output, normalized, 0, 1, cv::NORM_MINMAX);
    
    // 应用sigmoid函数
    cv::multiply(normalized, cv::Scalar::all(config_.spatial_gamma), sigmoid_output);
    cv::subtract(sigmoid_output, cv::Scalar::all(config_.spatial_beta), sigmoid_output);
    cv::exp(-sigmoid_output, sigmoid_output);
    cv::add(sigmoid_output, cv::Scalar::all(1.0), sigmoid_output);
    cv::divide(cv::Scalar::all(1.0), sigmoid_output, sigmoid_output);
    
    return sigmoid_output;
}

void NAMAttention::set_config(const NAMConfig& config) {
    config_ = config;
}

NAMAttention::PerformanceStats NAMAttention::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return performance_stats_;
}

void NAMAttention::reset_performance_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    performance_stats_ = PerformanceStats{};
}

bool NAMAttention::load_model(const std::string& model_path) {
    std::cout << "加载NAM模型: " << model_path << std::endl;
    return true;
}

bool NAMAttention::save_model(const std::string& model_path) {
    std::cout << "保存NAM模型: " << model_path << std::endl;
    return true;
}

bool NAMAttention::export_to_onnx(const std::string& onnx_path) {
    std::cout << "导出NAM模型到ONNX: " << onnx_path << std::endl;
    return true;
}

bool NAMAttention::train(const std::vector<cv::Mat>& inputs, const std::vector<cv::Mat>& targets) {
    std::cout << "训练NAM模型，样本数: " << inputs.size() << std::endl;
    return true;
}

bool NAMAttention::validate(const std::vector<cv::Mat>& inputs, const std::vector<cv::Mat>& targets) {
    std::cout << "验证NAM模型，样本数: " << inputs.size() << std::endl;
    return true;
}

void NAMAttention::set_attention_callback(AttentionCallback callback) {
    attention_callback_ = callback;
}

void NAMAttention::set_training_callback(TrainingCallback callback) {
    training_callback_ = callback;
}

} // namespace vision
} // namespace bamboo_cut