/**
 * @file vov_gscsp.h
 * @brief VoV-GSCSP (Versatile Object detection and Visual-Gradient SegmePath Channel Spatial Pyramid) 模块实现
 * @version 1.0
 * @date 2024
 * 
 * VoV-GSCSP: 结合VoVNet和GSCSP的高效特征提取模块
 * 基于论文: "An Energy and GPU-Computation Efficient Backbone Network"
 * 和 "Scaled-YOLOv4: Scaling Cross Stage Partial Network"
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <memory>
#include <string>

namespace bamboo_cut {
namespace inference {

/**
 * @brief VoV模块配置参数
 */
struct VoVConfig {
    int input_channels;          // 输入通道数
    int output_channels;         // 输出通道数
    int num_layers;              // VoV层数
    int growth_rate;             // 增长率
    bool use_residual;           // 是否使用残差连接
    std::string activation;      // 激活函数类型
    
    VoVConfig(int in_ch = 64, int out_ch = 128, int layers = 4, int growth = 32)
        : input_channels(in_ch), output_channels(out_ch), num_layers(layers), 
          growth_rate(growth), use_residual(true), activation("swish") {}
};

/**
 * @brief GSCSP配置参数
 */
struct GSCSPConfig {
    int input_channels;          // 输入通道数
    int output_channels;         // 输出通道数
    int hidden_channels;         // 隐藏通道数
    int num_blocks;              // 块数量
    float expansion_ratio;       // 扩展比率
    bool use_depthwise;          // 是否使用深度卷积
    bool use_attention;          // 是否使用注意力机制
    
    GSCSPConfig(int in_ch = 128, int out_ch = 256, int hid_ch = 64, int blocks = 3)
        : input_channels(in_ch), output_channels(out_ch), hidden_channels(hid_ch),
          num_blocks(blocks), expansion_ratio(2.0f), use_depthwise(true), use_attention(true) {}
};

/**
 * @brief VoV (Versatile Object detection) 模块
 * 
 * 原理：
 * 1. 使用多个3x3卷积层进行特征提取
 * 2. 每层的输出都连接到最终的聚合层
 * 3. 避免了DenseNet中的密集连接，降低了内存使用
 * 4. 保持了有效的特征重用
 */
class VoVModule {
public:
    explicit VoVModule(const VoVConfig& config);
    ~VoVModule() = default;

    /**
     * @brief 初始化VoV模块
     */
    bool initialize();

    /**
     * @brief 执行前向推理
     * @param input 输入特征图
     * @param output 输出特征图
     * @return 是否成功
     */
    bool forward(const cv::Mat& input, cv::Mat& output);

    /**
     * @brief 设置权重参数
     */
    void setWeights(const std::vector<cv::Mat>& conv_weights, const cv::Mat& aggregation_weights);

    /**
     * @brief 获取计算量统计
     */
    long long getComputationalCost() const;

    /**
     * @brief 获取参数数量
     */
    int getParameterCount() const;

private:
    /**
     * @brief 卷积层前向传播
     */
    bool convolutionLayer(const cv::Mat& input, cv::Mat& output, int layer_idx);

    /**
     * @brief 特征聚合
     */
    bool aggregateFeatures(const std::vector<cv::Mat>& features, cv::Mat& output);

    /**
     * @brief 初始化权重
     */
    void initializeWeights();

private:
    VoVConfig config_;
    bool initialized_;
    
    // 卷积层权重
    std::vector<cv::Mat> conv_weights_;
    std::vector<cv::Mat> conv_bias_;
    
    // 聚合层权重
    cv::Mat aggregation_weights_;
    cv::Mat aggregation_bias_;
    
    // 中间特征存储
    std::vector<cv::Mat> intermediate_features_;
    
    // 计算统计
    long long computational_cost_;
    int parameter_count_;
};

/**
 * @brief GSCSP (Gradient SegmePath Channel Spatial Pyramid) 模块
 * 
 * 原理：
 * 1. 结合了CSP (Cross Stage Partial) 网络的思想
 * 2. 使用梯度分割路径提高特征表示能力
 * 3. 通道空间金字塔提取多尺度特征
 * 4. 减少计算量同时保持精度
 */
class GSCSPModule {
public:
    explicit GSCSPModule(const GSCSPConfig& config);
    ~GSCSPModule() = default;

    /**
     * @brief 初始化GSCSP模块
     */
    bool initialize();

    /**
     * @brief 执行前向推理
     */
    bool forward(const cv::Mat& input, cv::Mat& output);

    /**
     * @brief 设置权重参数
     */
    void setWeights(const cv::Mat& stem_weights, const std::vector<cv::Mat>& block_weights, 
                   const cv::Mat& transition_weights);

    /**
     * @brief 获取计算量统计
     */
    long long getComputationalCost() const;

private:
    /**
     * @brief Stem卷积
     */
    bool stemConvolution(const cv::Mat& input, cv::Mat& output);

    /**
     * @brief CSP块处理
     */
    bool cspBlock(const cv::Mat& input, cv::Mat& output, int block_idx);

    /**
     * @brief 梯度分割路径
     */
    bool gradientSegmentPath(const cv::Mat& input, cv::Mat& path1, cv::Mat& path2);

    /**
     * @brief 通道空间金字塔
     */
    bool channelSpatialPyramid(const cv::Mat& input, cv::Mat& output);

    /**
     * @brief 特征融合
     */
    bool featureFusion(const cv::Mat& path1, const cv::Mat& path2, cv::Mat& output);

    /**
     * @brief 转换层
     */
    bool transitionLayer(const cv::Mat& input, cv::Mat& output);

private:
    GSCSPConfig config_;
    bool initialized_;
    
    // 网络权重
    cv::Mat stem_weights_;
    std::vector<cv::Mat> block_weights_;
    cv::Mat transition_weights_;
    cv::Mat fusion_weights_;
    
    // 中间特征
    cv::Mat stem_features_;
    std::vector<cv::Mat> block_features_;
    cv::Mat path1_features_;
    cv::Mat path2_features_;
    
    // 计算统计
    long long computational_cost_;
    int parameter_count_;
};

/**
 * @brief VoV-GSCSP融合模块
 * 结合VoV和GSCSP的优势，提供高效的特征提取
 */
class VoVGSCSPFusion {
public:
    struct Config {
        VoVConfig vov_config;
        GSCSPConfig gscsp_config;
        bool parallel_processing;    // 是否并行处理
        float fusion_weight;         // 融合权重
        std::string fusion_method;   // 融合方法: "add", "concat", "attention"
        
        Config() : parallel_processing(true), fusion_weight(0.5f), fusion_method("attention") {}
    };

    explicit VoVGSCSPFusion(const Config& config);
    ~VoVGSCSPFusion() = default;

    /**
     * @brief 初始化融合模块
     */
    bool initialize();

    /**
     * @brief 执行前向推理
     */
    bool forward(const cv::Mat& input, cv::Mat& output);

    /**
     * @brief 获取总计算量
     */
    long long getTotalComputationalCost() const;

    /**
     * @brief 获取总参数量
     */
    int getTotalParameterCount() const;

private:
    /**
     * @brief 注意力融合
     */
    bool attentionFusion(const cv::Mat& vov_features, const cv::Mat& gscsp_features, cv::Mat& output);

    /**
     * @brief 特征对齐
     */
    bool alignFeatures(const cv::Mat& features1, const cv::Mat& features2, 
                      cv::Mat& aligned1, cv::Mat& aligned2);

private:
    Config config_;
    bool initialized_;
    
    std::unique_ptr<VoVModule> vov_module_;
    std::unique_ptr<GSCSPModule> gscsp_module_;
    
    // 融合权重
    cv::Mat attention_weights_;
    cv::Mat fusion_weights_;
    
    // 中间特征
    cv::Mat vov_features_;
    cv::Mat gscsp_features_;
    cv::Mat aligned_vov_;
    cv::Mat aligned_gscsp_;
};

/**
 * @brief VoV-GSCSP网络构建器
 */
class VoVGSCSPNetworkBuilder {
public:
    struct NetworkConfig {
        int input_size;              // 输入尺寸
        int num_classes;             // 类别数
        std::vector<int> stages;     // 各阶段通道数
        bool use_spp;                // 是否使用SPP
        bool use_pan;                // 是否使用PAN
        std::string detection_head;  // 检测头类型
        
        NetworkConfig() : input_size(640), num_classes(80), 
                         stages({64, 128, 256, 512, 1024}),
                         use_spp(true), use_pan(true), detection_head("yolo") {}
    };

    explicit VoVGSCSPNetworkBuilder(const NetworkConfig& config);

    /**
     * @brief 构建完整网络
     */
    cv::dnn::Net buildNetwork();

    /**
     * @brief 构建竹子检测器
     */
    cv::dnn::Net buildBambooDetector();

    /**
     * @brief 创建骨干网络
     */
    cv::dnn::Net createBackbone();

    /**
     * @brief 创建颈部网络（FPN+PAN）
     */
    cv::dnn::Net createNeck();

    /**
     * @brief 创建检测头
     */
    cv::dnn::Net createDetectionHead();

private:
    /**
     * @brief 创建VoV-GSCSP阶段
     */
    cv::dnn::Net createStage(int stage_idx, int input_channels, int output_channels);

    /**
     * @brief 创建SPP模块
     */
    cv::dnn::Net createSPP(int channels);

    /**
     * @brief 创建PAN模块
     */
    cv::dnn::Net createPAN(const std::vector<int>& channels);

private:
    NetworkConfig config_;
    std::vector<std::unique_ptr<VoVGSCSPFusion>> stages_;
};

/**
 * @brief 性能优化工具
 */
class VoVGSCSPOptimizer {
public:
    struct OptimizationConfig {
        bool enable_pruning;         // 启用剪枝
        float pruning_ratio;         // 剪枝比率
        bool enable_quantization;    // 启用量化
        std::string quantization_method; // 量化方法
        bool enable_knowledge_distillation; // 启用知识蒸馏
        
        OptimizationConfig() : enable_pruning(false), pruning_ratio(0.1f),
                              enable_quantization(false), quantization_method("int8"),
                              enable_knowledge_distillation(false) {}
    };

    explicit VoVGSCSPOptimizer(const OptimizationConfig& config);

    /**
     * @brief 优化网络
     */
    cv::dnn::Net optimizeNetwork(const cv::dnn::Net& original_network);

    /**
     * @brief 结构化剪枝
     */
    cv::dnn::Net structuredPruning(const cv::dnn::Net& network, float ratio);

    /**
     * @brief 量化网络
     */
    cv::dnn::Net quantizeNetwork(const cv::dnn::Net& network);

    /**
     * @brief 知识蒸馏
     */
    cv::dnn::Net knowledgeDistillation(const cv::dnn::Net& teacher, const cv::dnn::Net& student);

private:
    OptimizationConfig config_;
};

} // namespace inference
} // namespace bamboo_cut