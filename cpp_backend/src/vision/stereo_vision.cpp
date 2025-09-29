/**
 * @file stereo_vision.cpp
 * @brief 双摄立体视觉系统实现
 */

#include "bamboo_cut/vision/stereo_vision.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdlib>

namespace bamboo_cut {
namespace vision {

StereoVision::StereoVision(const StereoConfig& config)
    : config_(config), initialized_(false), current_fps_(0.0f), 
      last_process_time_ms_(0.0f), frame_count_(0) {
}

StereoVision::~StereoVision() {
    if (left_camera_.isOpened()) {
        left_camera_.release();
    }
    if (right_camera_.isOpened()) {
        right_camera_.release();
    }
}

bool StereoVision::initialize() {
    std::cout << "初始化双摄立体视觉系统..." << std::endl;
    
    // 加载立体标定参数
    if (!load_calibration()) {
        std::cout << "立体标定参数加载失败" << std::endl;
        return false;
    }
    
    // 初始化双摄像头
    if (!initialize_cameras()) {
        std::cout << "双摄像头初始化失败" << std::endl;
        return false;
    }
    
    // 创建立体匹配器
    stereo_matcher_ = cv::StereoSGBM::create(
        0,          // minDisparity
        160,        // numDisparities
        21,         // blockSize
        8 * 21 * 21, // P1
        32 * 21 * 21, // P2
        1,          // disp12MaxDiff
        63,         // preFilterCap
        10,         // uniquenessRatio
        100,        // speckleWindowSize
        32          // speckleRange
    );
    
    last_capture_time_ = std::chrono::high_resolution_clock::now();
    initialized_ = true;
    
    std::cout << "双摄立体视觉系统初始化完成" << std::endl;
    return true;
}

bool StereoVision::load_calibration() {
    cv::FileStorage fs(config_.calibration_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cout << "无法打开立体标定文件: " << config_.calibration_file << std::endl;
        return false;
    }
    
    // 读取相机内参
    fs["left_camera_matrix"] >> left_camera_matrix_;
    fs["right_camera_matrix"] >> right_camera_matrix_;
    fs["left_dist_coeffs"] >> left_dist_coeffs_;
    fs["right_dist_coeffs"] >> right_dist_coeffs_;
    
    // 读取立体标定参数
    fs["R"] >> R_;
    fs["T"] >> T_;
    fs["E"] >> E_;
    fs["F"] >> F_;
    
    // 读取校正矩阵
    fs["R1"] >> R1_;
    fs["R2"] >> R2_;
    fs["P1"] >> P1_;
    fs["P2"] >> P2_;
    fs["Q"] >> Q_;
    
    // 读取重映射矩阵
    fs["map1_left"] >> map1_left_;
    fs["map2_left"] >> map2_left_;
    fs["map1_right"] >> map1_right_;
    fs["map2_right"] >> map2_right_;
    
    fs.release();
    
    std::cout << "立体标定参数加载成功" << std::endl;
    std::cout << "基线距离: " << cv::norm(T_) << "mm" << std::endl;
    
    return true;
}

bool StereoVision::initialize_cameras() {
    std::cout << "初始化 Jetson CSI 双摄像头系统..." << std::endl;
    
    // 确保任何已打开的摄像头被释放
    if (left_camera_.isOpened()) {
        left_camera_.release();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if (right_camera_.isOpened()) {
        right_camera_.release();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    // 优先使用 GStreamer 管道（nvarguscamerasrc）
    if (config_.use_gstreamer && !config_.left_camera_pipeline.empty() && !config_.right_camera_pipeline.empty()) {
        std::cout << "使用 nvarguscamerasrc 访问 CSI 摄像头..." << std::endl;
        
        // 初始化左摄像头 (nvarguscamerasrc)
        std::cout << "初始化左摄像头 (nvarguscamerasrc sensor-id=0)..." << std::endl;
        left_camera_.open(config_.left_camera_pipeline, cv::CAP_GSTREAMER);
        if (!left_camera_.isOpened()) {
            std::cout << "无法打开左摄像头 nvarguscamerasrc 管道" << std::endl;
            return false;
        }
        
        // 初始化右摄像头 (nvarguscamerasrc)
        std::cout << "初始化右摄像头 (nvarguscamerasrc sensor-id=1)..." << std::endl;
        right_camera_.open(config_.right_camera_pipeline, cv::CAP_GSTREAMER);
        if (!right_camera_.isOpened()) {
            std::cout << "无法打开右摄像头 nvarguscamerasrc 管道" << std::endl;
            left_camera_.release();
            return false;
        }
        
        std::cout << "nvarguscamerasrc 双摄像头初始化成功" << std::endl;
    } else {
        // 回退到传统 V4L2 方式，但只释放 video0（避免影响 nvargus-daemon 的 video1）
        std::cout << "回退到 V4L2 方式访问摄像头..." << std::endl;
        
        // 只释放 video0 设备（video1 被 nvargus-daemon 正常占用）
        if (config_.left_camera_id == 0) {
            std::string cmd = "fuser -k /dev/video0 2>/dev/null || true";
            system(cmd.c_str());
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        
        std::cout << "初始化左摄像头 (ID: " << config_.left_camera_id << ")..." << std::endl;
        left_camera_.open(config_.left_camera_id);
        if (!left_camera_.isOpened()) {
            std::cout << "无法打开左摄像头，建议使用 nvarguscamerasrc" << std::endl;
            return false;
        }
        
        std::cout << "初始化右摄像头 (ID: " << config_.right_camera_id << ")..." << std::endl;
        right_camera_.open(config_.right_camera_id);
        if (!right_camera_.isOpened()) {
            std::cout << "无法打开右摄像头，nvargus-daemon 可能正在使用该设备" << std::endl;
            left_camera_.release();
            return false;
        }
        
        std::cout << "V4L2 双摄像头初始化成功" << std::endl;
    }
    
    // 设置摄像头参数
    left_camera_.set(cv::CAP_PROP_FRAME_WIDTH, config_.frame_size.width);
    left_camera_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.frame_size.height);
    left_camera_.set(cv::CAP_PROP_FPS, config_.fps);
    
    right_camera_.set(cv::CAP_PROP_FRAME_WIDTH, config_.frame_size.width);
    right_camera_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.frame_size.height);
    right_camera_.set(cv::CAP_PROP_FPS, config_.fps);
    
    // 测试读取帧
    cv::Mat test_left, test_right;
    if (!left_camera_.read(test_left) || test_left.empty()) {
        std::cout << "左摄像头无法读取帧" << std::endl;
        return false;
    }
    
    if (!right_camera_.read(test_right) || test_right.empty()) {
        std::cout << "右摄像头无法读取帧" << std::endl;
        return false;
    }
    
    std::cout << "双摄像头初始化成功" << std::endl;
    std::cout << "左摄像头分辨率: " << test_left.size() << std::endl;
    std::cout << "右摄像头分辨率: " << test_right.size() << std::endl;
    
    return true;
}

bool StereoVision::capture_stereo_frame(StereoFrame& frame) {
    if (!initialized_) {
        return false;
    }
    
    auto capture_start = std::chrono::high_resolution_clock::now();
    
    // 同时读取左右摄像头
    bool left_ok = left_camera_.read(frame.left_image);
    bool right_ok = right_camera_.read(frame.right_image);
    
    if (!left_ok || !right_ok || frame.left_image.empty() || frame.right_image.empty()) {
        frame.valid = false;
        return false;
    }
    
    // 图像校正
    rectify_images(frame.left_image, frame.right_image, 
                   frame.rectified_left, frame.rectified_right);
    
    // 设置时间戳和有效性
    frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    frame.valid = true;
    
    // 更新性能统计
    auto capture_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(capture_end - capture_start);
    last_process_time_ms_ = duration.count();
    
    frame_count_++;
    auto time_elapsed = std::chrono::duration_cast<std::chrono::seconds>(capture_end - last_capture_time_);
    if (time_elapsed.count() >= 1) {
        current_fps_ = static_cast<float>(frame_count_) / time_elapsed.count();
        frame_count_ = 0;
        last_capture_time_ = capture_end;
    }
    
    return true;
}

void StereoVision::rectify_images(const cv::Mat& left, const cv::Mat& right, 
                                 cv::Mat& rect_left, cv::Mat& rect_right) {
    // 使用预计算的重映射矩阵进行图像校正
    cv::remap(left, rect_left, map1_left_, map2_left_, cv::INTER_LINEAR);
    cv::remap(right, rect_right, map1_right_, map2_right_, cv::INTER_LINEAR);
}

void StereoVision::compute_depth(StereoFrame& frame) {
    if (!frame.valid || frame.rectified_left.empty() || frame.rectified_right.empty()) {
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 转换为灰度图
    cv::Mat gray_left, gray_right;
    cv::cvtColor(frame.rectified_left, gray_left, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame.rectified_right, gray_right, cv::COLOR_BGR2GRAY);
    
    // 计算视差图
    stereo_matcher_->compute(gray_left, gray_right, frame.disparity);
    
    // 转换为浮点型视差图
    frame.disparity.convertTo(frame.disparity, CV_32F, 1.0/16.0);
    
    // 计算深度图
    cv::reprojectImageTo3D(frame.disparity, frame.depth_map, Q_, true);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    last_process_time_ms_ = duration.count();
}

void StereoVision::get_performance_stats(float& capture_fps, float& process_time_ms) const {
    capture_fps = current_fps_;
    process_time_ms = last_process_time_ms_;
}

} // namespace vision
} // namespace bamboo_cut