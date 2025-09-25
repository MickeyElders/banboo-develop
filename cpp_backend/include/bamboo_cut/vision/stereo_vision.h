/**
 * @file stereo_vision.h
 * @brief 双摄立体视觉系统
 * 支持立体标定、深度计算和竹子检测
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <chrono>

namespace bamboo_cut {
namespace vision {

/**
 * @brief 立体视觉配置参数
 */
struct StereoConfig {
    std::string calibration_file;      // 立体标定文件路径
    int left_camera_id;               // 左摄像头ID
    int right_camera_id;              // 右摄像头ID
    cv::Size frame_size;              // 帧尺寸
    int fps;                          // 帧率
    
    StereoConfig() 
        : calibration_file("config/stereo_calibration.xml")
        , left_camera_id(0)
        , right_camera_id(1) 
        , frame_size(640, 480)
        , fps(30) {}
};

/**
 * @brief 立体帧数据
 */
struct StereoFrame {
    cv::Mat left_image;               // 左图像
    cv::Mat right_image;              // 右图像
    cv::Mat rectified_left;           // 校正后左图像
    cv::Mat rectified_right;          // 校正后右图像
    cv::Mat disparity;                // 视差图
    cv::Mat depth_map;                // 深度图
    uint64_t timestamp;               // 时间戳
    bool valid;                       // 数据有效性
    
    StereoFrame() : timestamp(0), valid(false) {}
};

/**
 * @brief 双摄立体视觉系统类
 */
class StereoVision {
public:
    explicit StereoVision(const StereoConfig& config);
    ~StereoVision();
    
    /**
     * @brief 初始化立体视觉系统
     * @return 是否初始化成功
     */
    bool initialize();
    
    /**
     * @brief 捕获立体帧
     * @param frame 输出的立体帧数据
     * @return 是否成功捕获
     */
    bool capture_stereo_frame(StereoFrame& frame);
    
    /**
     * @brief 计算深度信息
     * @param frame 立体帧数据，会更新disparity和depth_map
     */
    void compute_depth(StereoFrame& frame);
    
    /**
     * @brief 检查系统是否已初始化
     */
    bool is_initialized() const { return initialized_; }
    
    /**
     * @brief 获取立体标定参数
     */
    const StereoConfig& get_config() const { return config_; }
    
    /**
     * @brief 获取性能统计
     */
    void get_performance_stats(float& capture_fps, float& process_time_ms) const;

private:
    /**
     * @brief 加载立体标定参数
     */
    bool load_calibration();
    
    /**
     * @brief 初始化双摄像头
     */
    bool initialize_cameras();
    
    /**
     * @brief 图像校正
     */
    void rectify_images(const cv::Mat& left, const cv::Mat& right, 
                       cv::Mat& rect_left, cv::Mat& rect_right);

private:
    StereoConfig config_;
    bool initialized_;
    
    // 双摄像头
    cv::VideoCapture left_camera_;
    cv::VideoCapture right_camera_;
    
    // 立体标定参数
    cv::Mat left_camera_matrix_, right_camera_matrix_;
    cv::Mat left_dist_coeffs_, right_dist_coeffs_;
    cv::Mat R_, T_, E_, F_;
    cv::Mat R1_, R2_, P1_, P2_, Q_;
    cv::Mat map1_left_, map2_left_;
    cv::Mat map1_right_, map2_right_;
    
    // 视差计算
    cv::Ptr<cv::StereoSGBM> stereo_matcher_;
    
    // 性能统计
    mutable std::chrono::high_resolution_clock::time_point last_capture_time_;
    mutable float current_fps_;
    mutable float last_process_time_ms_;
    mutable int frame_count_;
};

} // namespace vision
} // namespace bamboo_cut