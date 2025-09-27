/**
 * @file nam_attention.h
 * @brief NAM (Normalization-based Attention Module) 注意力机制
 * 轻量化注意力模块，基于归一化统计的高效注意力计算
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <memory>

namespace bamboo_cut {
namespace vision {

/**
 * @brief NAM注意力模块配置参数
 */
struct NAMConfig {
    int num_channels;                // 输入通道数
    int spatial_kernel_size;         // 空间注意力核大小
    float channel_reduction_ratio;   // 通道缩减比例
    float channel_weight;            // 通道注意力权重
    float spatial_weight;            // 空间注意力权重
    bool enable_channel_attention;   // 启用通道注意力
    bool enable_spatial_attention;   // 启用空间注意力
    
    NAMConfig() 
        : num_channels(3)
        , spatial_kernel_size(7)
        , channel_reduction_ratio(16.0f)
        , channel_weight(0.5f)
        , spatial_weight(0.5f)
        , enable_channel_attention(true)
        , enable_spatial_attention(true) {}
};

/**
 * @brief NAM (Normalization-based Attention Module) 注意力机制类
 * 
 * NAM是一种轻量化的注意力机制，基于归一化统计信息计算注意力权重，
 * 相比传统的注意力机制具有更低的计算复杂度和内存占用。
 * 
 * 主要特点：
 * - 基于局部统计的注意力计算
 * - 低计算复杂度和内存占用
 * - 适合边缘设备和实时应用
 * - 有效提升模型性能
 */
class NAMAttention {
public:
    /**
     * @brief 构造函数
     * @param config NAM配置参数
     */
    explicit NAMAttention(const NAMConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~NAMAttention();
    
    /**
     * @brief 初始化NAM模块
     * @return 是否初始化成功
     */
    bool initialize();
    
    /**
     * @brief 前向推理
     * @param input 输入特征图
     * @return 经过注意力加权的特征图
     */
    cv::Mat forward(const cv::Mat& input);
    
    /**
     * @brief 更新配置
     * @param new_config 新的配置参数
     */
    void updateConfig(const NAMConfig& new_config);
    
    /**
     * @brief 获取当前配置
     * @return 当前配置参数
     */
    NAMConfig getConfig() const;
    
    /**
     * @brief 检查是否已初始化
     * @return 初始化状态
     */
    bool isInitialized() const;

private:
    /**
     * @brief 计算通道注意力
     * @param input 输入特征图
     * @return 通道注意力图
     */
    cv::Mat computeChannelAttention(const cv::Mat& input);
    
    /**
     * @brief 计算空间注意力
     * @param input 输入特征图
     * @return 空间注意力图
     */
    cv::Mat computeSpatialAttention(const cv::Mat& input);
    
    /**
     * @brief 计算空间特征
     * @param input 输入图像
     * @param features 输出空间特征
     */
    void computeSpatialFeatures(const cv::Mat& input, cv::Mat& features);
    
    /**
     * @brief 融合通道和空间注意力
     * @param channel_attention 通道注意力
     * @param spatial_attention 空间注意力
     * @return 融合后的注意力图
     */
    cv::Mat combineAttentions(const cv::Mat& channel_attention, 
                             const cv::Mat& spatial_attention);
    
    /**
     * @brief 初始化权重
     */
    void initializeWeights();
    
    /**
     * @brief 分配内存缓冲区
     */
    void allocateBuffers();
    
    /**
     * @brief Sigmoid激活函数
     * @param x 输入值
     * @return 激活后的值
     */
    static float sigmoid(float x);
    
    /**
     * @brief 应用Softmax归一化
     * @param input 输入矩阵（就地修改）
     */
    void applySoftmax(cv::Mat& input);
    
    /**
     * @brief 应用Sigmoid激活
     * @param input 输入矩阵
     * @param output 输出矩阵
     */
    void applySigmoid(const cv::Mat& input, cv::Mat& output);

private:
    NAMConfig config_;          // 配置参数
    bool initialized_;          // 初始化状态
    
    // 权重和缓冲区
    cv::Mat channel_scale_;     // 通道缩放因子
    cv::Mat spatial_scale_;     // 空间缩放因子
    cv::Mat buffer_channel_;    // 通道计算缓冲区
    cv::Mat buffer_spatial_;    // 空间计算缓冲区
};

} // namespace vision
} // namespace bamboo_cut