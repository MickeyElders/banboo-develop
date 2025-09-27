/**
 * @file ghost_conv.h
 * @brief GhostConv轻量化卷积模块
 * 通过线性变换生成ghost特征图，大幅减少计算量和参数量
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

namespace bamboo_cut {
namespace vision {

/**
 * @brief 激活函数类型
 */
enum class ActivationType {
    NONE = 0,
    RELU = 1,
    RELU6 = 2,
    SWISH = 3
};

/**
 * @brief GhostConv配置参数
 */
struct GhostConvConfig {
    int input_channels;         // 输入通道数
    int output_channels;        // 输出通道数
    int kernel_size;           // 卷积核大小
    int stride;                // 步长
    int padding;               // 填充
    int ghost_ratio;           // Ghost比例 (通常为2)
    bool use_batch_norm;       // 是否使用批归一化
    bool use_activation;       // 是否使用激活函数
    ActivationType activation_type; // 激活函数类型
    int input_height;          // 输入高度（用于缓冲区分配）
    int input_width;           // 输入宽度（用于缓冲区分配）
    
    GhostConvConfig()
        : input_channels(3)
        , output_channels(16)
        , kernel_size(3)
        , stride(1)
        , padding(1)
        , ghost_ratio(2)
        , use_batch_norm(true)
        , use_activation(true)
        , activation_type(ActivationType::RELU)
        , input_height(0)
        , input_width(0) {}
};

/**
 * @brief GhostConv轻量化卷积类
 * 
 * GhostConv通过以下步骤减少计算量：
 * 1. 使用常规卷积生成一部分intrinsic特征图
 * 2. 通过cheap线性变换生成ghost特征图
 * 3. 连接intrinsic和ghost特征图作为最终输出
 * 
 * 相比传统卷积，GhostConv可以减少约50%的计算量和参数量
 */
class GhostConv {
public:
    /**
     * @brief 构造函数
     * @param config GhostConv配置参数
     */
    explicit GhostConv(const GhostConvConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~GhostConv();
    
    /**
     * @brief 初始化GhostConv模块
     * @return 是否初始化成功
     */
    bool initialize();
    
    /**
     * @brief 前向推理
     * @param input 输入特征图
     * @return 输出特征图
     */
    cv::Mat forward(const cv::Mat& input);
    
    /**
     * @brief 更新配置
     * @param new_config 新的配置参数
     */
    void updateConfig(const GhostConvConfig& new_config);
    
    /**
     * @brief 获取当前配置
     * @return 当前配置参数
     */
    GhostConvConfig getConfig() const;
    
    /**
     * @brief 检查是否已初始化
     * @return 初始化状态
     */
    bool isInitialized() const;

private:
    /**
     * @brief 生成本征特征图
     * @param input 输入图像
     * @return 本征特征图
     */
    cv::Mat generateIntrinsicFeatures(const cv::Mat& input);
    
    /**
     * @brief 生成Ghost特征图
     * @param intrinsic_features 本征特征图
     * @return Ghost特征图
     */
    cv::Mat generateGhostFeatures(const cv::Mat& intrinsic_features);
    
    /**
     * @brief 应用Ghost变换
     * @param feature_map 输入特征图
     * @param transform_id 变换ID
     * @return 变换后的特征图
     */
    cv::Mat applyGhostTransform(const cv::Mat& feature_map, int transform_id);
    
    /**
     * @brief 连接本征特征和Ghost特征
     * @param intrinsic 本征特征
     * @param ghost Ghost特征
     * @return 连接后的特征图
     */
    cv::Mat concatenateFeatures(const cv::Mat& intrinsic, const cv::Mat& ghost);
    
    /**
     * @brief 应用点卷积(1x1卷积)
     * @param input 输入图像
     * @param output_channels 输出通道数
     * @return 卷积结果
     */
    cv::Mat applyPointwiseConv(const cv::Mat& input, int output_channels);
    
    /**
     * @brief 应用标准卷积
     * @param input 输入图像
     * @param output_channels 输出通道数
     * @return 卷积结果
     */
    cv::Mat applyStandardConv(const cv::Mat& input, int output_channels);
    
    /**
     * @brief 获取卷积核
     * @param kernel_size 核大小
     * @return 卷积核
     */
    cv::Mat getConvKernel(int kernel_size);
    
    /**
     * @brief 应用批归一化
     * @param input 输入特征图
     * @return 归一化后的特征图
     */
    cv::Mat applyBatchNorm(const cv::Mat& input);
    
    /**
     * @brief 应用激活函数
     * @param input 输入特征图
     * @return 激活后的特征图
     */
    cv::Mat applyActivation(const cv::Mat& input);
    
    /**
     * @brief 应用Sigmoid激活函数
     * @param input 输入矩阵
     * @param output 输出矩阵
     */
    void applySigmoid(const cv::Mat& input, cv::Mat& output);
    
    /**
     * @brief 初始化卷积核
     */
    void initializeKernels();
    
    /**
     * @brief 初始化Ghost变换核
     */
    void initializeGhostKernels();
    
    /**
     * @brief 分配内存缓冲区
     */
    void allocateBuffers();

private:
    GhostConvConfig config_;     // 配置参数
    bool initialized_;           // 初始化状态
    
    // 卷积核存储
    std::map<int, cv::Mat> conv_kernels_;        // 标准卷积核
    std::vector<cv::Mat> ghost_kernels_;         // Ghost变换核
    
    // 内存缓冲区
    cv::Mat buffer_intrinsic_;   // 本征特征缓冲区
    cv::Mat buffer_ghost_;       // Ghost特征缓冲区
};

} // namespace vision
} // namespace bamboo_cut