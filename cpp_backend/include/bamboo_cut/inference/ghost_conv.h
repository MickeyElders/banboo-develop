/**
 * @file ghost_conv.h
 * @brief GhostConv轻量化卷积模块实现
 * @version 1.0
 * @date 2024
 * 
 * GhostConv: 通过生成更多特征图来减少计算量的轻量化卷积
 * 基于论文: "GhostNet: More Features from Cheap Operations"
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <memory>

namespace bamboo_cut {
namespace inference {

/**
 * @brief GhostConv配置参数
 */
struct GhostConvConfig {
    int input_channels;      // 输入通道数
    int output_channels;     // 输出通道数
    int kernel_size;         // 卷积核大小
    int stride;              // 步长
    int padding;             // 填充
    float ratio;             // Ghost比率，通常为2.0
    bool use_bias;           // 是否使用偏置
    
    GhostConvConfig(int in_ch = 64, int out_ch = 128, int ks = 3, int s = 1, int p = 1, float r = 2.0)
        : input_channels(in_ch), output_channels(out_ch), kernel_size(ks), 
          stride(s), padding(p), ratio(r), use_bias(false) {}
};

/**
 * @brief GhostConv轻量化卷积模块
 * 
 * 原理：
 * 1. 首先用传统卷积生成intrinsic特征图
 * 2. 然后用cheap operations(如depthwise conv)生成ghost特征图
 * 3. 将两部分特征图拼接得到最终输出
 * 
 * 优势：
 * - 减少约50%的计算量
 * - 保持相似的准确率
 * - 特别适合移动端和嵌入式设备
 */
class GhostConv {
public:
    explicit GhostConv(const GhostConvConfig& config);
    ~GhostConv() = default;

    /**
     * @brief 初始化GhostConv模块
     */
    bool initialize();

    /**
     * @brief 执行GhostConv前向推理
     * @param input 输入特征图 [N, C, H, W]
     * @param output 输出特征图 [N, out_channels, H', W']
     * @return 是否成功
     */
    bool forward(const cv::Mat& input, cv::Mat& output);

    /**
     * @brief 设置权重参数
     * @param primary_weights 主卷积权重
     * @param cheap_weights 轻量卷积权重
     */
    void setWeights(const cv::Mat& primary_weights, const cv::Mat& cheap_weights);

    /**
     * @brief 获取计算量统计
     * @return FLOPs数量
     */
    long long getComputationalCost() const;

    /**
     * @brief 获取参数数量
     */
    int getParameterCount() const;

private:
    /**
     * @brief 主要卷积操作（生成intrinsic特征）
     */
    bool primaryConvolution(const cv::Mat& input, cv::Mat& intrinsic_features);

    /**
     * @brief 轻量操作（生成ghost特征）
     */
    bool cheapOperations(const cv::Mat& intrinsic_features, cv::Mat& ghost_features);

    /**
     * @brief 特征图拼接
     */
    bool concatenateFeatures(const cv::Mat& intrinsic, const cv::Mat& ghost, cv::Mat& output);

    /**
     * @brief 初始化权重
     */
    void initializeWeights();

private:
    GhostConvConfig config_;
    bool initialized_;
    
    // 网络层
    cv::dnn::Net primary_conv_;     // 主卷积层
    cv::dnn::Net cheap_conv_;       // 轻量卷积层（通常是depthwise conv）
    
    // 权重矩阵
    cv::Mat primary_weights_;
    cv::Mat cheap_weights_;
    cv::Mat primary_bias_;
    cv::Mat cheap_bias_;
    
    // 中间特征
    cv::Mat intrinsic_features_;
    cv::Mat ghost_features_;
    
    // 计算统计
    int intrinsic_channels_;        // intrinsic特征通道数
    int ghost_channels_;           // ghost特征通道数
    long long computational_cost_; // 计算量（FLOPs）
    int parameter_count_;          // 参数数量
};

/**
 * @brief Ghost瓶颈模块
 * 结合GhostConv和残差连接的完整模块
 */
class GhostBottleneck {
public:
    struct Config {
        int input_channels;
        int hidden_channels;
        int output_channels;
        int kernel_size;
        int stride;
        bool use_se;           // 是否使用SE注意力
        float se_ratio;        // SE压缩比
        std::string activation; // 激活函数类型
        
        Config(int in_ch = 64, int hid_ch = 128, int out_ch = 64, int ks = 3, int s = 1)
            : input_channels(in_ch), hidden_channels(hid_ch), output_channels(out_ch),
              kernel_size(ks), stride(s), use_se(false), se_ratio(0.25f), activation("relu") {}
    };

    explicit GhostBottleneck(const Config& config);
    ~GhostBottleneck() = default;

    bool initialize();
    bool forward(const cv::Mat& input, cv::Mat& output);

private:
    Config config_;
    bool initialized_;
    
    std::unique_ptr<GhostConv> ghost_conv1_;
    std::unique_ptr<GhostConv> ghost_conv2_;
    cv::dnn::Net dw_conv_;         // Depthwise卷积
    cv::dnn::Net se_module_;       // SE注意力模块
    cv::dnn::Net shortcut_;        // 残差连接
    
    bool use_residual_;            // 是否使用残差连接
};

/**
 * @brief GhostNet网络构建器
 * 用于构建完整的GhostNet架构
 */
class GhostNetBuilder {
public:
    struct NetworkConfig {
        int input_size;            // 输入图像尺寸
        int num_classes;           // 分类数量
        float width_multiplier;    // 宽度倍数
        std::vector<std::vector<int>> cfg; // 网络配置
        
        NetworkConfig() : input_size(224), num_classes(1000), width_multiplier(1.0f) {
            // 默认GhostNet配置
            cfg = {
                // [k, exp, c, se, s]
                {3, 16, 16, 0, 1},
                {3, 48, 24, 0, 2},
                {3, 72, 24, 0, 1},
                {5, 72, 40, 1, 2},
                {5, 120, 40, 1, 1},
                {3, 240, 80, 0, 2},
                {3, 200, 80, 0, 1},
                {3, 184, 80, 0, 1},
                {3, 184, 80, 0, 1},
                {3, 480, 112, 1, 1},
                {3, 672, 112, 1, 1},
                {5, 672, 160, 1, 2},
                {5, 960, 160, 0, 1},
                {5, 960, 160, 1, 1},
                {5, 960, 160, 0, 1},
                {5, 960, 160, 1, 1}
            };
        }
    };

    explicit GhostNetBuilder(const NetworkConfig& config);
    
    /**
     * @brief 构建GhostNet网络
     */
    cv::dnn::Net buildNetwork();

    /**
     * @brief 创建轻量化的竹子检测模型
     */
    cv::dnn::Net buildBambooDetector(int num_classes = 1);

private:
    NetworkConfig config_;
    
    /**
     * @brief 创建Ghost瓶颈层
     */
    cv::dnn::Net createGhostBottleneck(int input_ch, int hidden_ch, int output_ch, 
                                      int kernel_size, int stride, bool use_se);
};

} // namespace inference
} // namespace bamboo_cut