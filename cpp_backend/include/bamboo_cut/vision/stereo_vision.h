#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>  // 添加WLS滤波器支持
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>    // 添加时间支持
#include <atomic>    // 添加原子操作支持

namespace bamboo_cut {
namespace vision {

// 3D点结构
struct Point3D {
    double x{0.0};      // X坐标 (mm)
    double y{0.0};      // Y坐标 (mm) 
    double z{0.0};      // Z坐标/深度 (mm)
    double confidence{0.0}; // 置信度 (0-1)
    
    Point3D() = default;
    Point3D(double x_, double y_, double z_, double conf = 1.0)
        : x(x_), y(y_), z(z_), confidence(conf) {}
};

// 立体视觉标定参数
struct StereoCalibrationParams {
    // 左相机内参
    cv::Mat left_camera_matrix;
    cv::Mat left_dist_coeffs;
    
    // 右相机内参
    cv::Mat right_camera_matrix;  
    cv::Mat right_dist_coeffs;
    
    // 立体视觉参数
    cv::Mat R;          // 旋转矩阵
    cv::Mat T;          // 平移向量
    cv::Mat E;          // 本质矩阵
    cv::Mat F;          // 基础矩阵
    
    // 校正参数
    cv::Mat R1, R2;     // 校正旋转矩阵
    cv::Mat P1, P2;     // 校正投影矩阵
    cv::Mat Q;          // 重投影矩阵
    cv::Mat map1_left, map2_left;   // 左相机去畸变映射
    cv::Mat map1_right, map2_right; // 右相机去畸变映射
    
    // 图像尺寸
    cv::Size image_size;
    
    // 基线距离 (mm)
    double baseline{0.0};
    
    bool is_calibrated{false};
};

// 立体匹配配置
struct StereoMatchingConfig {
    // SGBM参数
    int min_disparity{0};           // 最小视差
    int num_disparities{16*5};      // 视差范围 (必须是16的倍数)
    int block_size{5};              // 匹配块大小 (奇数)
    int P1{8*3*3*3};               // 控制视差变化平滑性的参数
    int P2{32*3*3*3};              // 控制视差变化平滑性的参数
    int disp12_max_diff{1};         // 左右一致性检查
    int pre_filter_cap{63};         // 预滤波器截止频率
    int uniqueness_ratio{10};       // 唯一性比率百分比
    int speckle_window_size{100};   // 斑点窗口大小
    int speckle_range{32};          // 斑点范围
    int mode{cv::StereoSGBM::MODE_SGBM_3WAY}; // SGBM模式
    
    // 后处理参数
    bool use_wls_filter{true};      // 使用加权最小二乘滤波
    double lambda{8000.0};          // WLS滤波lambda参数
    double sigma{1.5};              // WLS滤波sigma参数
    
    // 置信度阈值
    double min_confidence{0.5};     // 最小置信度
};

// 相机同步配置
struct CameraSyncConfig {
    std::string left_device{"/dev/video0"};   // 左相机设备
    std::string right_device{"/dev/video2"};  // 右相机设备
    int width{1920};                          // 图像宽度
    int height{1080};                         // 图像高度
    int fps{30};                             // 帧率
    bool hardware_sync{false};               // 硬件同步
    int sync_tolerance_ms{5};                // 同步容差(毫秒)
};

// 立体帧数据
struct StereoFrame {
    cv::Mat left_image;         // 左图像
    cv::Mat right_image;        // 右图像
    cv::Mat disparity;          // 视差图
    cv::Mat depth;              // 深度图
    std::chrono::steady_clock::time_point timestamp;
    bool valid{false};
    double sync_error_ms{0.0};  // 同步误差
};

// 标定板检测结果
struct CalibrationDetection {
    std::vector<cv::Point2f> left_corners;   // 左图像角点
    std::vector<cv::Point2f> right_corners;  // 右图像角点
    bool left_found{false};                  // 左图像检测成功
    bool right_found{false};                 // 右图像检测成功
    double reprojection_error{0.0};         // 重投影误差
};

/**
 * @brief 立体视觉系统类
 * 
 * 实现双摄像头立体视觉功能：
 * - 双相机标定和校正
 * - 立体匹配和视差计算
 * - 3D坐标重建
 * - 相机同步管理
 */
class StereoVision {
public:
    explicit StereoVision(const CameraSyncConfig& sync_config = CameraSyncConfig{});
    ~StereoVision();

    // 禁用拷贝构造和赋值
    StereoVision(const StereoVision&) = delete;
    StereoVision& operator=(const StereoVision&) = delete;

    // 系统控制
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // 相机标定
    bool start_calibration(const cv::Size& board_size, float square_size);
    CalibrationDetection detect_calibration_pattern(const cv::Mat& left_img, const cv::Mat& right_img);
    bool add_calibration_frame(const cv::Mat& left_img, const cv::Mat& right_img);
    bool calibrate_cameras();
    bool load_calibration(const std::string& filepath);
    bool save_calibration(const std::string& filepath) const;
    
    // 图像获取和处理
    bool capture_stereo_frame(StereoFrame& frame);
    bool rectify_images(const cv::Mat& left_raw, const cv::Mat& right_raw, 
                       cv::Mat& left_rect, cv::Mat& right_rect);
    
    // 立体匹配和深度估计
    bool compute_disparity(const cv::Mat& left_rect, const cv::Mat& right_rect, cv::Mat& disparity);
    bool disparity_to_depth(const cv::Mat& disparity, cv::Mat& depth);
    
    // 3D坐标转换
    Point3D pixel_to_3d(const cv::Point2f& pixel, const cv::Mat& disparity);
    std::vector<Point3D> pixels_to_3d(const std::vector<cv::Point2f>& pixels, const cv::Mat& disparity);
    
    // 竹子检测增强 (结合深度信息)
    std::vector<cv::Point2f> detect_bamboo_with_depth(const cv::Mat& left_image, 
                                                      const cv::Mat& disparity,
                                                      double min_depth_mm = 100.0,
                                                      double max_depth_mm = 2000.0);
    
    // 配置管理
    void set_stereo_matching_config(const StereoMatchingConfig& config);
    StereoMatchingConfig get_stereo_matching_config() const;
    StereoCalibrationParams get_calibration_params() const;
    
    // 状态和统计
    bool is_calibrated() const { return calibration_params_.is_calibrated; }
    double get_baseline_mm() const { return calibration_params_.baseline; }
    cv::Size get_image_size() const { return calibration_params_.image_size; }
    
    // 性能统计
    struct Statistics {
        uint64_t total_frames{0};
        uint64_t successful_captures{0};
        uint64_t sync_failures{0};
        double avg_sync_error_ms{0.0};
        double avg_processing_time_ms{0.0};
        std::chrono::steady_clock::time_point last_capture_time;
    };
    Statistics get_statistics() const;

private:
    // 相机管理
    bool open_cameras();
    void close_cameras();
    bool capture_synchronized_frames(cv::Mat& left, cv::Mat& right);
    
    // 标定辅助函数
    void generate_calibration_pattern_points(const cv::Size& board_size, float square_size);
    double compute_reprojection_error(const std::vector<std::vector<cv::Point3f>>& object_points,
                                    const std::vector<std::vector<cv::Point2f>>& image_points,
                                    const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
                                    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs);
    
    // 立体匹配优化
    void optimize_sgbm_parameters();
    cv::Mat apply_wls_filter(const cv::Mat& disparity, const cv::Mat& left_image);
    
    // 质量评估
    double evaluate_disparity_quality(const cv::Mat& disparity);
    bool validate_calibration_quality();
    
    // 配置参数
    CameraSyncConfig sync_config_;
    StereoMatchingConfig matching_config_;
    
    // 标定数据
    StereoCalibrationParams calibration_params_;
    std::vector<std::vector<cv::Point3f>> calibration_object_points_;
    std::vector<std::vector<cv::Point2f>> calibration_left_points_;
    std::vector<std::vector<cv::Point2f>> calibration_right_points_;
    cv::Size calibration_board_size_;
    float calibration_square_size_{0.0f};
    
    // 相机对象
    cv::VideoCapture left_camera_;
    cv::VideoCapture right_camera_;
    
    // 立体匹配器
    cv::Ptr<cv::StereoSGBM> stereo_matcher_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter_;
    
    // 状态管理
    std::atomic<bool> initialized_{false};
    std::atomic<bool> calibrating_{false};
    mutable std::mutex calibration_mutex_;
    mutable std::mutex capture_mutex_;
    
    // 统计信息
    mutable std::mutex stats_mutex_;
    Statistics statistics_;
    
    // 线程同步
    std::chrono::steady_clock::time_point last_left_timestamp_;
    std::chrono::steady_clock::time_point last_right_timestamp_;
};

// 工具函数
namespace stereo_utils {
    // 计算两个相机的基线距离
    double calculate_baseline(const cv::Mat& T);
    
    // 视差图可视化
    cv::Mat visualize_disparity(const cv::Mat& disparity, bool color = true);
    
    // 深度图可视化  
    cv::Mat visualize_depth(const cv::Mat& depth, double max_depth = 2000.0);
    
    // 3D点云可视化 (简单的投影视图)
    cv::Mat visualize_3d_points(const std::vector<Point3D>& points, 
                                const cv::Size& image_size,
                                double scale = 1.0);
    
    // 标定质量评估
    std::string evaluate_calibration_quality(const StereoCalibrationParams& params);
    
    // 推荐的相机配置
    CameraSyncConfig get_recommended_camera_config(const std::string& camera_type = "usb");
    
    // 推荐的匹配参数
    StereoMatchingConfig get_recommended_matching_config(const cv::Size& image_size);
}

} // namespace vision  
} // namespace bamboo_cut