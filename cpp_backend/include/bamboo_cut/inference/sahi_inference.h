/**
 * @file sahi_inference.h
 * @brief SAHI (Slicing Aided Hyper Inference) 切片推理模块实现
 * @version 1.0
 * @date 2024
 * 
 * SAHI: 通过图像切片来提高大尺寸图像检测精度的推理框架
 * 基于论文: "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection"
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include "bamboo_cut/core/data_bridge.h"

namespace bamboo_cut {
namespace inference {

/**
 * @brief SAHI配置参数
 */
struct SAHIConfig {
    int slice_width;             // 切片宽度
    int slice_height;            // 切片高度
    int overlap_width_ratio;     // 宽度重叠比率 (0-1)
    int overlap_height_ratio;    // 高度重叠比率 (0-1)
    float conf_threshold;        // 置信度阈值
    float nms_threshold;         // NMS阈值
    int max_detections;          // 最大检测数量
    bool auto_slice_resolution;  // 自动切片分辨率
    std::string postprocess_type; // 后处理类型: "NMS", "GREEDYNMM", "NMM"
    bool use_multiscale;         // 是否使用多尺度推理
    std::vector<float> scale_factors; // 尺度因子
    
    SAHIConfig() 
        : slice_width(640), slice_height(640)
        , overlap_width_ratio(20), overlap_height_ratio(20)
        , conf_threshold(0.5f), nms_threshold(0.5f)
        , max_detections(1000), auto_slice_resolution(true)
        , postprocess_type("NMS"), use_multiscale(false)
        , scale_factors({1.0f}) {}
};

/**
 * @brief 切片信息结构
 */
struct SliceInfo {
    cv::Rect slice_bbox;         // 切片在原图中的位置
    cv::Mat slice_image;         // 切片图像
    int slice_index;             // 切片索引
    float scale_factor;          // 缩放因子
    cv::Point2f offset;          // 偏移量
    
    SliceInfo() : slice_index(-1), scale_factor(1.0f), offset(0, 0) {}
};

/**
 * @brief 检测结果结构
 */
struct SAHIDetection {
    cv::Rect bbox;               // 检测框
    float confidence;            // 置信度
    int class_id;               // 类别ID
    cv::Point2f center;         // 中心点
    int slice_index;            // 来源切片索引
    
    SAHIDetection() : confidence(0.0f), class_id(-1), slice_index(-1) {}
    
    SAHIDetection(const cv::Rect& box, float conf, int cls_id, int slice_idx = -1)
        : bbox(box), confidence(conf), class_id(cls_id), slice_index(slice_idx) {
        center = cv::Point2f(box.x + box.width * 0.5f, box.y + box.height * 0.5f);
    }
};

/**
 * @brief SAHI推理引擎
 * 
 * 主要功能：
 * 1. 图像智能切片
 * 2. 多尺度推理
 * 3. 结果融合和NMS
 * 4. 小目标检测优化
 */
class SAHIInferenceEngine {
public:
    explicit SAHIInferenceEngine(const SAHIConfig& config);
    ~SAHIInferenceEngine() = default;

    /**
     * @brief 初始化SAHI推理引擎
     * @param detector 基础检测器
     */
    bool initialize(cv::dnn::Net detector);

    /**
     * @brief 执行SAHI推理
     * @param image 输入图像
     * @param detections 输出检测结果
     * @return 是否成功
     */
    bool inference(const cv::Mat& image, std::vector<SAHIDetection>& detections);

    /**
     * @brief 设置检测器
     */
    void setDetector(cv::dnn::Net detector);

    /**
     * @brief 更新配置
     */
    void updateConfig(const SAHIConfig& config);

    /**
     * @brief 获取性能统计
     */
    void getPerformanceStats(float& total_time, int& num_slices, float& avg_slice_time);

private:
    /**
     * @brief 图像切片
     */
    std::vector<SliceInfo> sliceImage(const cv::Mat& image);

    /**
     * @brief 自动确定切片参数
     */
    void autoSliceResolution(const cv::Mat& image);

    /**
     * @brief 多尺度切片
     */
    std::vector<SliceInfo> multiscaleSlicing(const cv::Mat& image);

    /**
     * @brief 单个切片推理
     */
    std::vector<SAHIDetection> inferenceSlice(const SliceInfo& slice_info);

    /**
     * @brief 结果融合
     */
    std::vector<SAHIDetection> mergeDetections(const std::vector<std::vector<SAHIDetection>>& all_detections);

    /**
     * @brief 坐标变换（切片坐标 -> 原图坐标）
     */
    SAHIDetection transformDetection(const SAHIDetection& detection, const SliceInfo& slice_info);

    /**
     * @brief 高级NMS
     */
    std::vector<SAHIDetection> advancedNMS(const std::vector<SAHIDetection>& detections);

    /**
     * @brief Greedy NMM (Non-Maximum Merging)
     */
    std::vector<SAHIDetection> greedyNMM(const std::vector<SAHIDetection>& detections);

    /**
     * @brief 计算IoU
     */
    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);

    /**
     * @brief 计算重叠率
     */
    float calculateOverlapRatio(const cv::Rect& box1, const cv::Rect& box2);

private:
    SAHIConfig config_;
    bool initialized_;
    cv::dnn::Net detector_;
    
    // 性能统计
    float total_inference_time_;
    int total_slices_;
    std::vector<float> slice_times_;
    
    // 缓存
    std::vector<SliceInfo> cached_slices_;
    cv::Size last_image_size_;
};

/**
 * @brief 自适应切片策略
 */
class AdaptiveSlicingStrategy {
public:
    struct StrategyConfig {
        float target_object_size;   // 目标物体尺寸
        float min_slice_size;       // 最小切片尺寸
        float max_slice_size;       // 最大切片尺寸
        float complexity_threshold; // 复杂度阈值
        bool use_attention_map;     // 使用注意力图
        
        StrategyConfig() : target_object_size(32.0f), min_slice_size(320.0f),
                          max_slice_size(1024.0f), complexity_threshold(0.5f),
                          use_attention_map(false) {}
    };

    explicit AdaptiveSlicingStrategy(const StrategyConfig& config);

    /**
     * @brief 生成自适应切片
     */
    std::vector<SliceInfo> generateAdaptiveSlices(const cv::Mat& image);

    /**
     * @brief 分析图像复杂度
     */
    float analyzeImageComplexity(const cv::Mat& image);

    /**
     * @brief 生成注意力图
     */
    cv::Mat generateAttentionMap(const cv::Mat& image);

private:
    /**
     * @brief 基于梯度的切片
     */
    std::vector<SliceInfo> gradientBasedSlicing(const cv::Mat& image);

    /**
     * @brief 基于内容的切片
     */
    std::vector<SliceInfo> contentBasedSlicing(const cv::Mat& image);

private:
    StrategyConfig config_;
};

/**
 * @brief SAHI后处理器
 */
class SAHIPostProcessor {
public:
    enum class MergeStrategy {
        UNION,              // 并集
        INTERSECTION,       // 交集
        WEIGHTED_AVERAGE,   // 加权平均
        CONFIDENCE_BASED    // 基于置信度
    };

    struct PostProcessConfig {
        MergeStrategy merge_strategy;
        float merge_threshold;       // 合并阈值
        bool use_class_agnostic_nms; // 类别无关NMS
        float confidence_boost;      // 置信度提升
        int max_merge_candidates;    // 最大合并候选数
        
        PostProcessConfig() : merge_strategy(MergeStrategy::WEIGHTED_AVERAGE),
                             merge_threshold(0.5f), use_class_agnostic_nms(false),
                             confidence_boost(0.1f), max_merge_candidates(5) {}
    };

    explicit SAHIPostProcessor(const PostProcessConfig& config);

    /**
     * @brief 后处理检测结果
     */
    std::vector<SAHIDetection> postProcess(const std::vector<SAHIDetection>& detections,
                                          const cv::Size& image_size);

    /**
     * @brief 智能合并重叠检测
     */
    std::vector<SAHIDetection> smartMergeOverlaps(const std::vector<SAHIDetection>& detections);

    /**
     * @brief 边界框细化
     */
    SAHIDetection refineBoundingBox(const SAHIDetection& detection, const cv::Mat& image);

    /**
     * @brief 置信度校准
     */
    float calibrateConfidence(float original_confidence, const std::vector<SAHIDetection>& neighbors);

private:
    /**
     * @brief 合并两个检测结果
     */
    SAHIDetection mergeTwoDetections(const SAHIDetection& det1, const SAHIDetection& det2);

    /**
     * @brief 计算合并权重
     */
    std::vector<float> calculateMergeWeights(const std::vector<SAHIDetection>& candidates);

private:
    PostProcessConfig config_;
};

/**
 * @brief SAHI优化器
 */
class SAHIOptimizer {
public:
    struct OptimizationConfig {
        bool enable_gpu_acceleration;   // GPU加速
        bool enable_batch_processing;   // 批处理
        int batch_size;                 // 批大小
        bool enable_caching;            // 启用缓存
        bool enable_parallel_slicing;   // 并行切片
        int num_threads;               // 线程数
        
        OptimizationConfig() : enable_gpu_acceleration(true), enable_batch_processing(false),
                              batch_size(4), enable_caching(true), enable_parallel_slicing(true),
                              num_threads(4) {}
    };

    explicit SAHIOptimizer(const OptimizationConfig& config);

    /**
     * @brief 优化推理管道
     */
    void optimizeInferencePipeline(SAHIInferenceEngine& engine);

    /**
     * @brief 并行切片处理
     */
    std::vector<std::vector<SAHIDetection>> parallelSliceInference(
        const std::vector<SliceInfo>& slices, cv::dnn::Net& detector);

    /**
     * @brief 批处理推理
     */
    std::vector<std::vector<SAHIDetection>> batchInference(
        const std::vector<SliceInfo>& slices, cv::dnn::Net& detector);

    /**
     * @brief 内存优化
     */
    void optimizeMemoryUsage();

private:
    OptimizationConfig config_;
    std::vector<std::thread> worker_threads_;
};

/**
 * @brief SAHI指标评估器
 */
class SAHIMetricsEvaluator {
public:
    struct Metrics {
        float precision;            // 精确率
        float recall;              // 召回率
        float f1_score;            // F1分数
        float average_precision;   // 平均精确率
        float inference_time;      // 推理时间
        int total_detections;      // 总检测数
        int false_positives;       // 假正例
        int false_negatives;       // 假负例
        float small_object_ap;     // 小目标AP
        
        Metrics() : precision(0.0f), recall(0.0f), f1_score(0.0f),
                   average_precision(0.0f), inference_time(0.0f),
                   total_detections(0), false_positives(0), false_negatives(0),
                   small_object_ap(0.0f) {}
    };

    /**
     * @brief 评估检测结果
     */
    Metrics evaluate(const std::vector<SAHIDetection>& predictions,
                    const std::vector<SAHIDetection>& ground_truth,
                    float iou_threshold = 0.5f);

    /**
     * @brief 计算AP（平均精确率）
     */
    float calculateAveragePrecision(const std::vector<SAHIDetection>& predictions,
                                   const std::vector<SAHIDetection>& ground_truth);

    /**
     * @brief 分析小目标检测性能
     */
    float analyzeSmallObjectPerformance(const std::vector<SAHIDetection>& predictions,
                                       const std::vector<SAHIDetection>& ground_truth,
                                       float small_object_threshold = 32.0f);

    /**
     * @brief 生成详细报告
     */
    std::string generateDetailedReport(const Metrics& metrics);

private:
    /**
     * @brief 匹配预测和真值
     */
    std::vector<std::pair<int, int>> matchPredictionsToGroundTruth(
        const std::vector<SAHIDetection>& predictions,
        const std::vector<SAHIDetection>& ground_truth,
        float iou_threshold);
};

} // namespace inference
} // namespace bamboo_cut