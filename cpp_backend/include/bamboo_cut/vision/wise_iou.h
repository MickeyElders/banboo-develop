#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "bamboo_cut/core/types.h"

namespace bamboo_cut {
namespace vision {

/**
 * @brief Wise-IoU 损失函数配置
 */
struct WiseIoUConfig {
    float alpha{1.0f};                 // 权重参数
    float beta{1.0f};                  // 形状参数
    float gamma{1.0f};                 // 尺度参数
    float delta{1.0f};                 // 偏移参数
    float eps{1e-7f};                  // 数值稳定性参数
    bool use_focal{true};              // 是否使用 Focal 损失
    float focal_alpha{0.25f};          // Focal 损失 alpha 参数
    float focal_gamma{2.0f};           // Focal 损失 gamma 参数
    bool use_giou{false};              // 是否使用 GIoU
    bool use_diou{false};              // 是否使用 DIoU
    bool use_ciou{false};              // 是否使用 CIoU
    
    WiseIoUConfig() = default;
    bool validate() const;
};

/**
 * @brief Wise-IoU 损失函数类
 * 
 * 替代传统 IoU 的智能损失函数，提供更好的训练效果
 */
class WiseIoULoss {
public:
    WiseIoULoss();
    explicit WiseIoULoss(const WiseIoUConfig& config);
    ~WiseIoULoss();

    // 禁用拷贝
    WiseIoULoss(const WiseIoULoss&) = delete;
    WiseIoULoss& operator=(const WiseIoULoss&) = delete;

    // 初始化和控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // 损失计算
    float compute_loss(const core::Rectangle& pred, const core::Rectangle& target);
    float compute_loss_batch(const std::vector<core::Rectangle>& preds, 
                           const std::vector<core::Rectangle>& targets);
    
    // IoU 计算
    float compute_iou(const core::Rectangle& box1, const core::Rectangle& box2);
    float compute_giou(const core::Rectangle& box1, const core::Rectangle& box2);
    float compute_diou(const core::Rectangle& box1, const core::Rectangle& box2);
    float compute_ciou(const core::Rectangle& box1, const core::Rectangle& box2);
    float compute_wise_iou(const core::Rectangle& box1, const core::Rectangle& box2);

    // 检测结果评估
    struct DetectionMetrics {
        float precision{0.0f};
        float recall{0.0f};
        float f1_score{0.0f};
        float mAP{0.0f};
        float avg_iou{0.0f};
        float avg_wise_iou{0.0f};
        uint32_t true_positives{0};
        uint32_t false_positives{0};
        uint32_t false_negatives{0};
    };
    
    DetectionMetrics evaluate_detections(const std::vector<core::DetectionResult>& predictions,
                                       const std::vector<core::DetectionResult>& ground_truth,
                                       float iou_threshold = 0.5f);

    // 配置管理
    void set_config(const WiseIoUConfig& config);
    WiseIoUConfig get_config() const { return config_; }

    // 性能统计
    struct PerformanceStats {
        uint64_t total_loss_computations{0};
        double avg_computation_time_ms{0.0};
        double min_computation_time_ms{0.0};
        double max_computation_time_ms{0.0};
        double total_loss_value{0.0};
        double avg_loss_value{0.0};
    };
    PerformanceStats get_performance_stats() const;

private:
    // 内部方法
    float compute_area(const core::Rectangle& box);
    float compute_intersection_area(const core::Rectangle& box1, const core::Rectangle& box2);
    float compute_union_area(const core::Rectangle& box1, const core::Rectangle& box2);
    float compute_center_distance(const core::Rectangle& box1, const core::Rectangle& box2);
    float compute_diagonal_distance(const core::Rectangle& box1, const core::Rectangle& box2);
    float compute_aspect_ratio_similarity(const core::Rectangle& box1, const core::Rectangle& box2);
    
    // 辅助计算
    float focal_loss(float confidence, float target);
    float smooth_l1_loss(float pred, float target, float beta = 1.0f);
    float huber_loss(float pred, float target, float delta = 1.0f);

    // 配置和状态
    WiseIoUConfig config_;
    bool initialized_{false};

    // 性能统计
    mutable std::mutex stats_mutex_;
    PerformanceStats performance_stats_;

    // 错误处理
    std::string last_error_;
    void set_error(const std::string& error);
};

/**
 * @brief Wise-IoU 损失函数网络
 * 
 * 集成 Wise-IoU 损失函数的训练网络
 */
class WiseIoUNetwork {
public:
    struct NetworkConfig {
        WiseIoUConfig loss_config;     // 损失函数配置
        float learning_rate{0.001f};   // 学习率
        float weight_decay{0.0001f};   // 权重衰减
        int batch_size{32};            // 批处理大小
        int num_epochs{100};           // 训练轮数
        bool use_mixed_precision{true}; // 是否使用混合精度
        std::string optimizer{"adam"}; // 优化器类型
        
        NetworkConfig() = default;
        bool validate() const;
    };

    WiseIoUNetwork();
    explicit WiseIoUNetwork(const NetworkConfig& config);
    ~WiseIoUNetwork();

    // 禁用拷贝
    WiseIoUNetwork(const WiseIoUNetwork&) = delete;
    WiseIoUNetwork& operator=(const WiseIoUNetwork&) = delete;

    // 初始化和控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // 训练接口
    bool train(const std::vector<cv::Mat>& images,
              const std::vector<std::vector<core::DetectionResult>>& ground_truth);
    
    // 验证接口
    WiseIoULoss::DetectionMetrics validate(const std::vector<cv::Mat>& images,
                                          const std::vector<std::vector<core::DetectionResult>>& ground_truth);

    // 模型保存和加载
    bool save_model(const std::string& model_path);
    bool load_model(const std::string& model_path);

    // 配置管理
    void set_config(const NetworkConfig& config);
    NetworkConfig get_config() const { return config_; }

    // 性能统计
    struct TrainingStats {
        uint64_t total_training_steps{0};
        double avg_training_loss{0.0};
        double best_validation_loss{0.0};
        double current_learning_rate{0.0};
        std::vector<double> training_losses;
        std::vector<double> validation_losses;
        std::vector<WiseIoULoss::DetectionMetrics> validation_metrics;
    };
    TrainingStats get_training_stats() const;

private:
    // 内部方法
    bool setup_optimizer();
    bool update_learning_rate(int epoch);
    float compute_batch_loss(const std::vector<cv::Mat>& batch_images,
                           const std::vector<std::vector<core::DetectionResult>>& batch_targets);

    // 配置和状态
    NetworkConfig config_;
    bool initialized_{false};

    // 组件
    std::unique_ptr<WiseIoULoss> loss_function_;

    // 训练状态
    mutable std::mutex training_mutex_;
    TrainingStats training_stats_;

    // 错误处理
    std::string last_error_;
    void set_error(const std::string& error);
};

} // namespace vision
} // namespace bamboo_cut 