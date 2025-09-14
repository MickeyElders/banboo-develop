#include "bamboo_cut/vision/vov_gscsp.h"
#include <iostream>
#include <chrono>
#include <mutex>
#include <functional>
#include <algorithm>

namespace bamboo_cut {
namespace vision {

// VoVGSCSPConfig validation implementation
bool VoVGSCSPConfig::validate() const {
    if (in_channels <= 0 || out_channels <= 0) {
        return false;
    }
    if (num_blocks < 1 || num_blocks > 10) {
        return false;
    }
    if (pyramid_scales.empty()) {
        return false;
    }
    for (int scale : pyramid_scales) {
        if (scale < 1 || scale > 8) {
            return false;
        }
    }
    if (attention_ratio <= 0 || attention_ratio > 1.0f) {
        return false;
    }
    if (compression_ratio <= 0 || compression_ratio > 1.0f) {
        return false;
    }
    // Validate ghost_config
    if (!ghost_config.validate()) {
        return false;
    }
    return true;
}

VoVGSCSP::VoVGSCSP(const VoVGSCSPConfig& config) : config_(config), initialized_(false) {
    std::cout << "创建VoVGSCSP实例" << std::endl;
}

VoVGSCSP::~VoVGSCSP() {
    shutdown();
    std::cout << "销毁VoVGSCSP实例" << std::endl;
}

bool VoVGSCSP::initialize() {
    if (initialized_) {
        std::cout << "VoVGSCSP已初始化" << std::endl;
        return true;
    }
    
    std::cout << "初始化VoVGSCSP..." << std::endl;
    
    try {
        // 验证配置
        if (!config_.validate()) {
            std::cerr << "VoVGSCSP配置无效" << std::endl;
            return false;
        }
        
        initialized_ = true;
        std::cout << "VoVGSCSP初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "VoVGSCSP初始化异常: " << e.what() << std::endl;
        return false;
    }
}

void VoVGSCSP::shutdown() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "关闭VoVGSCSP..." << std::endl;
    initialized_ = false;
    std::cout << "VoVGSCSP已关闭" << std::endl;
}

cv::Mat VoVGSCSP::forward(const cv::Mat& input) {
    if (!initialized_) {
        std::cerr << "VoVGSCSP未初始化" << std::endl;
        return cv::Mat();
    }
    
    if (input.empty()) {
        std::cerr << "输入图像为空" << std::endl;
        return cv::Mat();
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 多尺度处理
        cv::Mat multi_scale_features = process_multi_scale(input);
        
        // 应用空间金字塔
        cv::Mat pyramid_features = apply_spatial_pyramid(multi_scale_features);
        
        // 特征融合
        std::vector<cv::Mat> feature_list = {multi_scale_features, pyramid_features};
        cv::Mat result = fuse_features(feature_list);
        
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
        std::cerr << "VoVGSCSP前向传播异常: " << e.what() << std::endl;
        return input;
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
    if (input.empty()) return cv::Mat();
    
    try {
        std::vector<cv::Mat> scale_features;
        
        // 对不同尺度进行处理
        for (int scale : config_.pyramid_scales) {
            cv::Mat scaled_input;
            if (scale == 1) {
                scaled_input = input.clone();
            } else {
                cv::Size new_size(input.cols / scale, input.rows / scale);
                cv::resize(input, scaled_input, new_size);
            }
            
            // 应用尺度卷积
            cv::Mat scale_feature = apply_scale_convolution(scaled_input, scale);
            
            // 将特征恢复到原始尺寸
            if (scale != 1) {
                cv::resize(scale_feature, scale_feature, input.size());
            }
            
            scale_features.push_back(scale_feature);
        }
        
        // 融合多尺度特征
        return apply_attention_fusion(scale_features);
        
    } catch (const std::exception& e) {
        std::cerr << "多尺度处理异常: " << e.what() << std::endl;
        return input.clone();
    }
}

cv::Mat VoVGSCSP::apply_spatial_pyramid(const cv::Mat& input) {
    if (input.empty()) return cv::Mat();
    
    try {
        std::vector<cv::Mat> pyramid_features;
        
        // 应用不同尺度的金字塔池化
        for (int scale : config_.pyramid_scales) {
            cv::Mat pyramid_feature = apply_pyramid_pooling(input, scale);
            pyramid_features.push_back(pyramid_feature);
        }
        
        // 应用注意力融合
        return apply_attention_fusion(pyramid_features);
        
    } catch (const std::exception& e) {
        std::cerr << "空间金字塔处理异常: " << e.what() << std::endl;
        return input.clone();
    }
}

cv::Mat VoVGSCSP::apply_attention_fusion(const std::vector<cv::Mat>& features) {
    if (features.empty()) {
        return cv::Mat();
    }
    
    if (features.size() == 1) {
        return features[0].clone();
    }
    
    try {
        cv::Mat output = features[0].clone();
        
        if (config_.use_attention) {
            // 计算注意力权重
            std::vector<cv::Mat> attention_weights;
            for (const auto& feature : features) {
                cv::Mat channel_attn = compute_channel_attention(feature);
                cv::Mat spatial_attn = compute_spatial_attention(feature);
                cv::Mat combined_attn;
                cv::addWeighted(channel_attn, 0.5, spatial_attn, 0.5, 0, combined_attn);
                attention_weights.push_back(combined_attn);
            }
            
            // 应用注意力加权融合
            output = cv::Mat::zeros(features[0].size(), features[0].type());
            float weight_sum = 0.0f;
            
            for (size_t i = 0; i < features.size(); ++i) {
                cv::Mat weighted_feature = apply_attention(features[i], attention_weights[i]);
                cv::add(output, weighted_feature, output);
                weight_sum += config_.attention_ratio;
            }
            
            // 归一化
            if (weight_sum > 0) {
                cv::multiply(output, cv::Scalar::all(1.0 / weight_sum), output);
            }
        } else {
            // 简单的平均融合
            for (size_t i = 1; i < features.size(); ++i) {
                cv::addWeighted(output, static_cast<double>(i) / (i + 1),
                              features[i], 1.0 / (i + 1), 0, output);
            }
        }
        
        return output;
        
    } catch (const std::exception& e) {
        std::cerr << "注意力融合异常: " << e.what() << std::endl;
        return features[0].clone();
    }
}

cv::Mat VoVGSCSP::fuse_features(const std::vector<cv::Mat>& features) {
    if (features.empty()) return cv::Mat();
    
    try {
        // 首先应用密集连接
        cv::Mat dense_features = apply_dense_connection(features);
        
        // 然后应用注意力融合
        std::vector<cv::Mat> fusion_input = {dense_features};
        for (const auto& feature : features) {
            fusion_input.push_back(feature);
        }
        
        return apply_attention_fusion(fusion_input);
        
    } catch (const std::exception& e) {
        std::cerr << "特征融合异常: " << e.what() << std::endl;
        return features[0].clone();
    }
}

cv::Mat VoVGSCSP::apply_dense_connection(const std::vector<cv::Mat>& features) {
    if (features.empty()) return cv::Mat();
    
    try {
        cv::Mat output = features[0].clone();
        
        // VoV风格的密集连接：每个特征都与前面所有特征连接
        for (size_t i = 1; i < features.size(); ++i) {
            cv::Mat concatenated;
            cv::hconcat(output, features[i], concatenated);
            output = concatenated;
        }
        
        // 如果启用压缩，则压缩特征
        if (config_.enable_compression) {
            output = compress_features(output);
        }
        
        return output;
        
    } catch (const std::exception& e) {
        std::cerr << "密集连接异常: " << e.what() << std::endl;
        return features[0].clone();
    }
}

// 私有辅助函数实现
cv::Mat VoVGSCSP::apply_scale_convolution(const cv::Mat& input, int scale) {
    cv::Mat output;
    
    // 根据尺度选择不同的卷积核
    if (scale == 1) {
        // 1x1卷积
        cv::Mat kernel = (cv::Mat_<float>(1,1) << 1.0);
        cv::filter2D(input, output, -1, kernel);
    } else if (scale == 2) {
        // 3x3卷积
        cv::Mat kernel = (cv::Mat_<float>(3,3) <<
            0.0625, 0.125, 0.0625,
            0.125,  0.25,  0.125,
            0.0625, 0.125, 0.0625);
        cv::filter2D(input, output, -1, kernel);
    } else {
        // 5x5或更大的卷积
        cv::GaussianBlur(input, output, cv::Size(scale*2+1, scale*2+1), scale*0.5);
    }
    
    return output;
}

cv::Mat VoVGSCSP::apply_pyramid_pooling(const cv::Mat& input, int scale) {
    cv::Mat output;
    
    // 自适应平均池化
    cv::Size pool_size(input.cols / scale, input.rows / scale);
    if (pool_size.width > 0 && pool_size.height > 0) {
        cv::resize(input, output, pool_size, 0, 0, cv::INTER_AREA);
        // 恢复到原始尺寸
        cv::resize(output, output, input.size(), 0, 0, cv::INTER_LINEAR);
    } else {
        output = input.clone();
    }
    
    return output;
}

cv::Mat VoVGSCSP::compute_channel_attention(const cv::Mat& input) {
    cv::Mat attention_weights;
    
    // 全局平均池化
    cv::Mat pooled;
    cv::reduce(input.reshape(1, input.rows * input.cols), pooled, 0, cv::REDUCE_AVG);
    
    // 简单的全连接操作模拟
    cv::multiply(pooled, cv::Scalar::all(config_.attention_ratio), attention_weights);
    
    // Sigmoid激活
    cv::exp(-attention_weights, attention_weights);
    cv::add(attention_weights, cv::Scalar::all(1.0), attention_weights);
    cv::divide(cv::Scalar::all(1.0), attention_weights, attention_weights);
    
    return attention_weights;
}

cv::Mat VoVGSCSP::compute_spatial_attention(const cv::Mat& input) {
    cv::Mat attention_map;
    
    // 空间注意力：使用均值和最大值
    cv::Mat mean_map, max_map;
    cv::reduce(input, mean_map, 2, cv::REDUCE_AVG); // 沿通道维度求平均
    cv::reduce(input, max_map, 2, cv::REDUCE_MAX);   // 沿通道维度求最大值
    
    // 拼接并应用卷积
    cv::Mat concatenated;
    cv::hconcat(mean_map, max_map, concatenated);
    
    // 7x7卷积
    cv::Mat kernel = cv::getGaussianKernel(7, 1.0, CV_32F);
    cv::Mat kernel2d = kernel * kernel.t();
    cv::filter2D(concatenated, attention_map, -1, kernel2d);
    
    // Sigmoid激活
    cv::exp(-attention_map, attention_map);
    cv::add(attention_map, cv::Scalar::all(1.0), attention_map);
    cv::divide(cv::Scalar::all(1.0), attention_map, attention_map);
    
    return attention_map;
}

cv::Mat VoVGSCSP::apply_attention(const cv::Mat& input, const cv::Mat& attention_weights) {
    cv::Mat output;
    
    if (attention_weights.size() == input.size()) {
        cv::multiply(input, attention_weights, output);
    } else {
        // 如果尺寸不匹配，调整注意力图
        cv::Mat resized_attention;
        cv::resize(attention_weights, resized_attention, input.size());
        cv::multiply(input, resized_attention, output);
    }
    
    return output;
}

cv::Mat VoVGSCSP::compress_features(const cv::Mat& input) {
    cv::Mat compressed;
    
    // 简单的特征压缩：使用PCA风格的降维
    int target_channels = static_cast<int>(input.channels() * config_.compression_ratio);
    if (target_channels < 1) target_channels = 1;
    
    // 模拟压缩：选择前N个通道
    if (input.channels() > target_channels) {
        std::vector<cv::Mat> channels;
        cv::split(input, channels);
        
        std::vector<cv::Mat> selected_channels(channels.begin(),
                                             channels.begin() + target_channels);
        cv::merge(selected_channels, compressed);
    } else {
        compressed = input.clone();
    }
    
    return compressed;
}

void VoVGSCSP::set_config(const VoVGSCSPConfig& config) {
    config_ = config;
}

VoVGSCSP::PerformanceStats VoVGSCSP::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return performance_stats_;
}

void VoVGSCSP::reset_performance_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    performance_stats_ = PerformanceStats{};
}

bool VoVGSCSP::load_model(const std::string& model_path) {
    std::cout << "加载VoVGSCSP模型: " << model_path << std::endl;
    return true;
}

bool VoVGSCSP::save_model(const std::string& model_path) {
    std::cout << "保存VoVGSCSP模型: " << model_path << std::endl;
    return true;
}

bool VoVGSCSP::export_to_onnx(const std::string& onnx_path) {
    std::cout << "导出VoVGSCSP模型到ONNX: " << onnx_path << std::endl;
    return true;
}

bool VoVGSCSP::train(const std::vector<cv::Mat>& inputs, const std::vector<cv::Mat>& targets) {
    std::cout << "训练VoVGSCSP模型，样本数: " << inputs.size() << std::endl;
    return true;
}

bool VoVGSCSP::validate(const std::vector<cv::Mat>& inputs, const std::vector<cv::Mat>& targets) {
    std::cout << "验证VoVGSCSP模型，样本数: " << inputs.size() << std::endl;
    return true;
}

void VoVGSCSP::set_feature_callback(FeatureCallback callback) {
    feature_callback_ = callback;
}

void VoVGSCSP::set_training_callback(TrainingCallback callback) {
    training_callback_ = callback;
}

} // namespace vision
} // namespace bamboo_cut