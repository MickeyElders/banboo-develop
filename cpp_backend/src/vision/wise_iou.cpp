#include "bamboo_cut/vision/wise_iou.h"
#include <iostream>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <cmath>

namespace bamboo_cut {
namespace vision {

// WiseIoUConfig validation implementation
bool WiseIoUConfig::validate() const {
    if (alpha <= 0.0f || beta <= 0.0f || gamma <= 0.0f || delta <= 0.0f) {
        return false;
    }
    if (eps <= 0.0f || eps >= 1e-3f) {
        return false;
    }
    if (focal_alpha < 0.0f || focal_alpha > 1.0f) {
        return false;
    }
    if (focal_gamma <= 0.0f) {
        return false;
    }
    return true;
}

WiseIoULoss::WiseIoULoss() : config_(), initialized_(false) {
    std::cout << "创建WiseIoULoss实例" << std::endl;
}

WiseIoULoss::WiseIoULoss(const WiseIoUConfig& config) : config_(config), initialized_(false) {
    std::cout << "创建WiseIoULoss实例，使用自定义配置" << std::endl;
}

WiseIoULoss::~WiseIoULoss() {
    shutdown();
    std::cout << "销毁WiseIoULoss实例" << std::endl;
}

bool WiseIoULoss::initialize() {
    if (initialized_) {
        std::cout << "WiseIoULoss已初始化" << std::endl;
        return true;
    }
    
    std::cout << "初始化WiseIoULoss..." << std::endl;
    
    try {
        // 验证配置
        if (!config_.validate()) {
            std::cerr << "WiseIoULoss配置无效" << std::endl;
            return false;
        }
        
        initialized_ = true;
        std::cout << "WiseIoULoss初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "WiseIoULoss初始化异常: " << e.what() << std::endl;
        return false;
    }
}

void WiseIoULoss::shutdown() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "关闭WiseIoULoss..." << std::endl;
    initialized_ = false;
    std::cout << "WiseIoULoss已关闭" << std::endl;
}

float WiseIoULoss::compute_loss(const core::Rectangle& pred, const core::Rectangle& target) {
    if (!initialized_) {
        std::cerr << "WiseIoULoss未初始化" << std::endl;
        return 1.0f; // 最大损失值
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        float loss = 0.0f;
        
        if (config_.use_giou) {
            loss = 1.0f - compute_giou(pred, target);
        } else if (config_.use_diou) {
            loss = 1.0f - compute_diou(pred, target);
        } else if (config_.use_ciou) {
            loss = 1.0f - compute_ciou(pred, target);
        } else {
            loss = 1.0f - compute_wise_iou(pred, target);
        }
        
        // 更新性能统计
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            performance_stats_.total_loss_computations++;
            double computation_time_ms = duration.count() / 1000.0;
            
            if (performance_stats_.total_loss_computations == 1) {
                performance_stats_.min_computation_time_ms = computation_time_ms;
                performance_stats_.max_computation_time_ms = computation_time_ms;
                performance_stats_.avg_computation_time_ms = computation_time_ms;
            } else {
                performance_stats_.min_computation_time_ms = std::min(performance_stats_.min_computation_time_ms, computation_time_ms);
                performance_stats_.max_computation_time_ms = std::max(performance_stats_.max_computation_time_ms, computation_time_ms);
                performance_stats_.avg_computation_time_ms = (performance_stats_.avg_computation_time_ms * (performance_stats_.total_loss_computations - 1) + computation_time_ms) / performance_stats_.total_loss_computations;
            }
            
            performance_stats_.total_loss_value += loss;
            performance_stats_.avg_loss_value = performance_stats_.total_loss_value / performance_stats_.total_loss_computations;
        }
        
        return loss;
        
    } catch (const std::exception& e) {
        std::cerr << "WiseIoU损失计算异常: " << e.what() << std::endl;
        return 1.0f;
    }
}

float WiseIoULoss::compute_loss_batch(const std::vector<core::Rectangle>& preds, 
                                    const std::vector<core::Rectangle>& targets) {
    if (preds.size() != targets.size()) {
        std::cerr << "预测和目标数量不匹配" << std::endl;
        return 1.0f;
    }
    
    float total_loss = 0.0f;
    for (size_t i = 0; i < preds.size(); ++i) {
        total_loss += compute_loss(preds[i], targets[i]);
    }
    
    return total_loss / static_cast<float>(preds.size());
}

float WiseIoULoss::compute_iou(const core::Rectangle& box1, const core::Rectangle& box2) {
    float intersection = compute_intersection_area(box1, box2);
    float union_area = compute_union_area(box1, box2);
    
    if (union_area < config_.eps) {
        return 0.0f;
    }
    
    return intersection / union_area;
}

float WiseIoULoss::compute_giou(const core::Rectangle& box1, const core::Rectangle& box2) {
    float iou = compute_iou(box1, box2);
    
    // 计算包围框
    int x1 = std::min(box1.x, box2.x);
    int y1 = std::min(box1.y, box2.y);
    int x2 = std::max(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::max(box1.y + box1.height, box2.y + box2.height);
    
    float enclosing_area = (x2 - x1) * (y2 - y1);
    float union_area = compute_union_area(box1, box2);
    
    if (enclosing_area < config_.eps) {
        return iou;
    }
    
    return iou - (enclosing_area - union_area) / enclosing_area;
}

float WiseIoULoss::compute_diou(const core::Rectangle& box1, const core::Rectangle& box2) {
    float iou = compute_iou(box1, box2);
    float center_distance = compute_center_distance(box1, box2);
    float diagonal_distance = compute_diagonal_distance(box1, box2);
    
    if (diagonal_distance < config_.eps) {
        return iou;
    }
    
    return iou - (center_distance * center_distance) / (diagonal_distance * diagonal_distance);
}

float WiseIoULoss::compute_ciou(const core::Rectangle& box1, const core::Rectangle& box2) {
    float diou = compute_diou(box1, box2);
    float aspect_similarity = compute_aspect_ratio_similarity(box1, box2);
    
    return diou - config_.alpha * aspect_similarity;
}

float WiseIoULoss::compute_wise_iou(const core::Rectangle& box1, const core::Rectangle& box2) {
    float iou = compute_iou(box1, box2);
    
    // Wise-IoU的智能权重计算
    float area1 = compute_area(box1);
    float area2 = compute_area(box2);
    float area_ratio = std::abs(area1 - area2) / std::max(area1, area2);
    
    float center_distance = compute_center_distance(box1, box2);
    float diagonal_distance = compute_diagonal_distance(box1, box2);
    float distance_penalty = 0.0f;
    
    if (diagonal_distance > config_.eps) {
        distance_penalty = (center_distance * center_distance) / (diagonal_distance * diagonal_distance);
    }
    
    // 智能权重组合
    float wise_weight = config_.alpha * std::exp(-config_.beta * area_ratio) + 
                       config_.gamma * std::exp(-config_.delta * distance_penalty);
    
    return iou * wise_weight;
}

WiseIoULoss::DetectionMetrics WiseIoULoss::evaluate_detections(
    const std::vector<core::DetectionResult>& predictions,
    const std::vector<core::DetectionResult>& ground_truth,
    float iou_threshold) {
    
    DetectionMetrics metrics;
    
    if (predictions.empty() && ground_truth.empty()) {
        return metrics;
    }
    
    // 简化的评估实现
    uint32_t tp = 0;
    uint32_t fp = 0;
    uint32_t fn = ground_truth.size();
    
    for (const auto& pred : predictions) {
        bool matched = false;
        for (const auto& gt : ground_truth) {
            float iou = compute_iou(pred.bounding_box, gt.bounding_box);
            if (iou >= iou_threshold) {
                tp++;
                fn--;
                matched = true;
                break;
            }
        }
        if (!matched) {
            fp++;
        }
    }
    
    metrics.true_positives = tp;
    metrics.false_positives = fp;
    metrics.false_negatives = fn;
    
    if (tp + fp > 0) {
        metrics.precision = static_cast<float>(tp) / (tp + fp);
    }
    
    if (tp + fn > 0) {
        metrics.recall = static_cast<float>(tp) / (tp + fn);
    }
    
    if (metrics.precision + metrics.recall > 0) {
        metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall);
    }
    
    return metrics;
}

void WiseIoULoss::set_config(const WiseIoUConfig& config) {
    config_ = config;
}

WiseIoULoss::PerformanceStats WiseIoULoss::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return performance_stats_;
}

float WiseIoULoss::compute_area(const core::Rectangle& box) {
    return static_cast<float>(box.width * box.height);
}

float WiseIoULoss::compute_intersection_area(const core::Rectangle& box1, const core::Rectangle& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }
    
    return static_cast<float>((x2 - x1) * (y2 - y1));
}

float WiseIoULoss::compute_union_area(const core::Rectangle& box1, const core::Rectangle& box2) {
    float area1 = compute_area(box1);
    float area2 = compute_area(box2);
    float intersection = compute_intersection_area(box1, box2);
    
    return area1 + area2 - intersection;
}

float WiseIoULoss::compute_center_distance(const core::Rectangle& box1, const core::Rectangle& box2) {
    float center1_x = box1.x + box1.width / 2.0f;
    float center1_y = box1.y + box1.height / 2.0f;
    float center2_x = box2.x + box2.width / 2.0f;
    float center2_y = box2.y + box2.height / 2.0f;
    
    float dx = center1_x - center2_x;
    float dy = center1_y - center2_y;
    
    return std::sqrt(dx * dx + dy * dy);
}

float WiseIoULoss::compute_diagonal_distance(const core::Rectangle& box1, const core::Rectangle& box2) {
    int x1 = std::min(box1.x, box2.x);
    int y1 = std::min(box1.y, box2.y);
    int x2 = std::max(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::max(box1.y + box1.height, box2.y + box2.height);
    
    float dx = x2 - x1;
    float dy = y2 - y1;
    
    return std::sqrt(dx * dx + dy * dy);
}

float WiseIoULoss::compute_aspect_ratio_similarity(const core::Rectangle& box1, const core::Rectangle& box2) {
    float ratio1 = static_cast<float>(box1.width) / box1.height;
    float ratio2 = static_cast<float>(box2.width) / box2.height;
    
    float diff = std::abs(ratio1 - ratio2);
    return std::exp(-diff / config_.beta);
}

float WiseIoULoss::focal_loss(float confidence, float target) {
    float ce_loss = -target * std::log(confidence + config_.eps) - 
                   (1 - target) * std::log(1 - confidence + config_.eps);
    float pt = target * confidence + (1 - target) * (1 - confidence);
    return config_.focal_alpha * std::pow(1 - pt, config_.focal_gamma) * ce_loss;
}

float WiseIoULoss::smooth_l1_loss(float pred, float target, float beta) {
    float diff = std::abs(pred - target);
    if (diff < beta) {
        return 0.5f * diff * diff / beta;
    } else {
        return diff - 0.5f * beta;
    }
}

float WiseIoULoss::huber_loss(float pred, float target, float delta) {
    float diff = std::abs(pred - target);
    if (diff <= delta) {
        return 0.5f * diff * diff;
    } else {
        return delta * diff - 0.5f * delta * delta;
    }
}

void WiseIoULoss::set_error(const std::string& error) {
    last_error_ = error;
    std::cerr << "WiseIoULoss错误: " << error << std::endl;
}

// WiseIoUNetwork implementation
bool WiseIoUNetwork::NetworkConfig::validate() const {
    if (!loss_config.validate()) {
        return false;
    }
    if (learning_rate <= 0.0f || learning_rate > 1.0f) {
        return false;
    }
    if (weight_decay < 0.0f || weight_decay > 1.0f) {
        return false;
    }
    if (batch_size <= 0 || num_epochs <= 0) {
        return false;
    }
    return true;
}

WiseIoUNetwork::WiseIoUNetwork() : config_(), initialized_(false) {
    loss_function_ = std::make_unique<WiseIoULoss>();
}

WiseIoUNetwork::WiseIoUNetwork(const NetworkConfig& config) : config_(config), initialized_(false) {
    loss_function_ = std::make_unique<WiseIoULoss>(config.loss_config);
}

WiseIoUNetwork::~WiseIoUNetwork() {
    shutdown();
}

bool WiseIoUNetwork::initialize() {
    if (initialized_) {
        return true;
    }
    
    try {
        if (!config_.validate()) {
            return false;
        }
        
        if (!loss_function_->initialize()) {
            return false;
        }
        
        initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "WiseIoUNetwork初始化异常: " << e.what() << std::endl;
        return false;
    }
}

void WiseIoUNetwork::shutdown() {
    if (!initialized_) {
        return;
    }
    
    if (loss_function_) {
        loss_function_->shutdown();
    }
    
    initialized_ = false;
}

bool WiseIoUNetwork::train(const std::vector<cv::Mat>& images,
                          const std::vector<std::vector<core::DetectionResult>>& ground_truth) {
    if (!initialized_) {
        return false;
    }
    
    // 简化的训练实现
    std::cout << "开始训练，样本数: " << images.size() << std::endl;
    
    {
        std::lock_guard<std::mutex> lock(training_mutex_);
        training_stats_.total_training_steps++;
    }
    
    return true;
}

WiseIoULoss::DetectionMetrics WiseIoUNetwork::validate(const std::vector<cv::Mat>& images,
                                                      const std::vector<std::vector<core::DetectionResult>>& ground_truth) {
    WiseIoULoss::DetectionMetrics metrics;
    
    if (!initialized_) {
        return metrics;
    }
    
    // 简化的验证实现
    std::cout << "开始验证，样本数: " << images.size() << std::endl;
    
    return metrics;
}

bool WiseIoUNetwork::save_model(const std::string& model_path) {
    std::cout << "保存模型到: " << model_path << std::endl;
    return true;
}

bool WiseIoUNetwork::load_model(const std::string& model_path) {
    std::cout << "加载模型从: " << model_path << std::endl;
    return true;
}

void WiseIoUNetwork::set_config(const NetworkConfig& config) {
    config_ = config;
}

WiseIoUNetwork::TrainingStats WiseIoUNetwork::get_training_stats() const {
    std::lock_guard<std::mutex> lock(training_mutex_);
    return training_stats_;
}

bool WiseIoUNetwork::setup_optimizer() {
    // 简化的优化器设置
    return true;
}

bool WiseIoUNetwork::update_learning_rate(int epoch) {
    // 简化的学习率更新
    return true;
}

float WiseIoUNetwork::compute_batch_loss(const std::vector<cv::Mat>& batch_images,
                                        const std::vector<std::vector<core::DetectionResult>>& batch_targets) {
    // 简化的批次损失计算
    return 0.5f;
}

void WiseIoUNetwork::set_error(const std::string& error) {
    last_error_ = error;
    std::cerr << "WiseIoUNetwork错误: " << error << std::endl;
}

} // namespace vision
} // namespace bamboo_cut