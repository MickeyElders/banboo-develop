#include "bamboo_cut/vision/stereo_vision.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <atomic>
#include <opencv2/opencv.hpp>

#ifdef ENABLE_OPENCV_CONTRIB
    #include <opencv2/ximgproc.hpp>
#endif

namespace bamboo_cut {
namespace vision {

StereoVision::StereoVision(const CameraSyncConfig& sync_config)
    : sync_config_(sync_config)
    , stream_enabled_(false)
    , display_mode_(DisplayMode::SIDE_BY_SIDE)
    , frame_counter_(0)
    , gst_pipeline_(nullptr)
    , gst_appsrc_(nullptr)
{
    // 初始化统计信息
    statistics_.last_capture_time = std::chrono::steady_clock::now();
    
    // 设置默认匹配配置（不使用函数调用，直接设置默认值）
    matching_config_.min_disparity = 0;
    matching_config_.num_disparities = 16 * 6;  // 96
    matching_config_.block_size = 5;
    matching_config_.P1 = 8 * 3 * 5 * 5;  // 600
    matching_config_.P2 = 32 * 3 * 5 * 5; // 2400
    matching_config_.disp12_max_diff = 1;
    matching_config_.pre_filter_cap = 63;
    matching_config_.uniqueness_ratio = 10;
    matching_config_.speckle_window_size = 100;
    matching_config_.speckle_range = 32;
    matching_config_.mode = cv::StereoSGBM::MODE_SGBM_3WAY;
    matching_config_.use_wls_filter = true;
    matching_config_.lambda = 8000.0;
    matching_config_.sigma = 1.5;
    matching_config_.min_confidence = 0.5;
}

StereoVision::~StereoVision() {
    shutdown();
}

bool StereoVision::initialize() {
    if (initialized_.load()) {
        std::cerr << "立体视觉系统已初始化" << std::endl;
        return true;
    }
    
    std::cout << "初始化立体视觉系统..." << std::endl;
    
    // 打开相机
    if (!open_cameras()) {
        std::cerr << "无法打开双摄像头" << std::endl;
        return false;
    }
    
    // 创建立体匹配器
    stereo_matcher_ = cv::StereoSGBM::create(
        matching_config_.min_disparity,
        matching_config_.num_disparities,
        matching_config_.block_size,
        matching_config_.P1,
        matching_config_.P2,
        matching_config_.disp12_max_diff,
        matching_config_.pre_filter_cap,
        matching_config_.uniqueness_ratio,
        matching_config_.speckle_window_size,
        matching_config_.speckle_range,
        matching_config_.mode
    );
    
    // 创建WLS滤波器
    if (matching_config_.use_wls_filter) {
#if HAS_WLS_FILTER
        wls_filter_ = cv::ximgproc::createDisparityWLSFilter(stereo_matcher_);
        wls_filter_->setLambda(matching_config_.lambda);
        wls_filter_->setSigmaColor(matching_config_.sigma);
#else
        std::cerr << "警告: OpenCV扩展模块未安装，WLS滤波器不可用" << std::endl;
        matching_config_.use_wls_filter = false;
#endif
    }
    
    initialized_.store(true);
    std::cout << "立体视觉系统初始化完成" << std::endl;
    
    return true;
}

void StereoVision::shutdown() {
    if (!initialized_.load()) {
        return;
    }
    
    std::cout << "关闭立体视觉系统..." << std::endl;
    
    // 首先禁用流输出
    stream_enabled_ = false;
    
    // 设置初始化标志为false，停止所有处理循环
    initialized_.store(false);
    
    // 强制停止GStreamer管道并等待
    if (gst_pipeline_) {
        std::cout << "停止GStreamer管道..." << std::endl;
        
        // 发送EOS事件
        gst_element_send_event(gst_pipeline_, gst_event_new_eos());
        
        // 等待EOS处理完成
        GstBus* bus = gst_element_get_bus(gst_pipeline_);
        if (bus) {
            GstMessage* msg = gst_bus_timed_pop_filtered(bus, 2 * GST_SECOND, GST_MESSAGE_EOS);
            if (msg) {
                gst_message_unref(msg);
            }
            gst_object_unref(bus);
        }
        
        // 强制设置为NULL状态
        GstStateChangeReturn ret = gst_element_set_state(gst_pipeline_, GST_STATE_NULL);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cout << "警告：GStreamer管道停止失败" << std::endl;
        }
        
        // 等待状态变化完成
        gst_element_get_state(gst_pipeline_, nullptr, nullptr, GST_SECOND);
        
        // 清理对象引用
        gst_object_unref(gst_pipeline_);
        gst_pipeline_ = nullptr;
        gst_appsrc_ = nullptr;
        
        std::cout << "GStreamer管道已停止" << std::endl;
    }
    
    // 关闭摄像头
    close_cameras();
    
    // 清理OpenCV对象
    if (stereo_matcher_) {
        stereo_matcher_.release();
    }
#if HAS_WLS_FILTER
    if (wls_filter_) {
        wls_filter_.release();
    }
#endif
    
    // 等待一小段时间确保所有资源释放
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::cout << "立体视觉系统已关闭" << std::endl;
}

bool StereoVision::open_cameras() {
    std::cout << "🔍 尝试打开双摄像头..." << std::endl;
    
    bool left_opened = false;
    bool right_opened = false;
    
    // 尝试打开左相机（带超时检测）
    std::cout << "📷 尝试打开左相机: " << sync_config_.left_device << " (5秒超时)" << std::endl;
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // 在独立线程中打开相机，避免主线程阻塞
        std::atomic<bool> camera_opened{false};
        std::atomic<bool> timeout_occurred{false};
        
        std::thread camera_thread([&]() {
            left_camera_.open(sync_config_.left_device);
            camera_opened.store(true);
        });
        
        // 等待5秒或相机打开成功
        auto timeout_time = start_time + std::chrono::seconds(5);
        while (!camera_opened.load() && std::chrono::steady_clock::now() < timeout_time) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        if (!camera_opened.load()) {
            std::cout << "⏰ 左相机打开超时，跳过" << std::endl;
            timeout_occurred.store(true);
            camera_thread.detach(); // 超时后分离线程，让它在后台自然结束
        } else {
            camera_thread.join();
            
            if (left_camera_.isOpened()) {
                // 配置左相机参数
                left_camera_.set(cv::CAP_PROP_FRAME_WIDTH, sync_config_.width);
                left_camera_.set(cv::CAP_PROP_FRAME_HEIGHT, sync_config_.height);
                left_camera_.set(cv::CAP_PROP_FPS, sync_config_.fps);
                left_camera_.set(cv::CAP_PROP_BUFFERSIZE, 1);
                
                std::cout << "✅ 左相机打开成功: " << sync_config_.left_device << std::endl;
                left_opened = true;
            } else {
                std::cout << "❌ 左相机打开失败: " << sync_config_.left_device << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cout << "❌ 左相机打开异常: " << e.what() << std::endl;
    }
    
    // 尝试打开右相机（带超时检测）
    std::cout << "📷 尝试打开右相机: " << sync_config_.right_device << " (5秒超时)" << std::endl;
    start_time = std::chrono::steady_clock::now();
    
    try {
        std::atomic<bool> camera_opened{false};
        std::atomic<bool> timeout_occurred{false};
        
        std::thread camera_thread([&]() {
            right_camera_.open(sync_config_.right_device);
            camera_opened.store(true);
        });
        
        // 等待5秒或相机打开成功
        auto timeout_time = start_time + std::chrono::seconds(5);
        while (!camera_opened.load() && std::chrono::steady_clock::now() < timeout_time) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        if (!camera_opened.load()) {
            std::cout << "⏰ 右相机打开超时，跳过" << std::endl;
            timeout_occurred.store(true);
            camera_thread.detach(); // 超时后分离线程
        } else {
            camera_thread.join();
            
            if (right_camera_.isOpened()) {
                // 配置右相机参数
                right_camera_.set(cv::CAP_PROP_FRAME_WIDTH, sync_config_.width);
                right_camera_.set(cv::CAP_PROP_FRAME_HEIGHT, sync_config_.height);
                right_camera_.set(cv::CAP_PROP_FPS, sync_config_.fps);
                right_camera_.set(cv::CAP_PROP_BUFFERSIZE, 1);
                
                std::cout << "✅ 右相机打开成功: " << sync_config_.right_device << std::endl;
                right_opened = true;
            } else {
                std::cout << "❌ 右相机打开失败: " << sync_config_.right_device << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cout << "❌ 右相机打开异常: " << e.what() << std::endl;
    }
    
    // 硬件调试模式：即使摄像头不可用也继续运行
    if (!left_opened && !right_opened) {
        std::cout << "⚠️ 硬件调试模式：双摄像头都不可用，将生成测试画面" << std::endl;
        std::cout << "💡 前端将显示彩色测试图案用于调试GStreamer流" << std::endl;
    } else if (!left_opened) {
        std::cout << "⚠️ 硬件调试模式：左相机不可用，将复制右相机画面" << std::endl;
    } else if (!right_opened) {
        std::cout << "⚠️ 硬件调试模式：右相机不可用，将复制左相机画面" << std::endl;
    } else {
        std::cout << "🎉 双摄像头都打开成功！" << std::endl;
    }
    
    std::cout << "🚀 立体视觉系统将以硬件调试模式运行" << std::endl;
    std::cout << "📺 预期视频流: UDP://127.0.0.1:5000 (H.264, 640x480@30fps)" << std::endl;
    
    return true; // 硬件调试模式：总是返回成功
}

void StereoVision::close_cameras() {
    if (left_camera_.isOpened()) {
        left_camera_.release();
    }
    if (right_camera_.isOpened()) {
        right_camera_.release();
    }
}

bool StereoVision::capture_synchronized_frames(cv::Mat& left, cv::Mat& right) {
    // 硬件调试模式：尽力获取画面，忽略同步误差
    static int debug_frame_count = 0;
    debug_frame_count++;
    
    // 优先尝试左摄像头
    bool left_success = false;
    bool right_success = false;
    cv::Mat left_temp, right_temp;
    
    if (left_camera_.isOpened()) {
        left_success = left_camera_.read(left_temp);
    }
    
    if (right_camera_.isOpened()) {
        right_success = right_camera_.read(right_temp);
    }
    
    // 如果两个都失败，生成测试画面
    if (!left_success && !right_success) {
        if (debug_frame_count % 100 == 0) {
            std::cout << "📷 双摄像头都不可用，生成测试画面 (帧 #" << debug_frame_count << ")" << std::endl;
        }
        
        // 生成彩色测试画面
        left = cv::Mat(sync_config_.height, sync_config_.width, CV_8UC3);
        right = cv::Mat(sync_config_.height, sync_config_.width, CV_8UC3);
        
        // 左摄像头: 蓝色渐变
        for (int y = 0; y < left.rows; y++) {
            for (int x = 0; x < left.cols; x++) {
                cv::Vec3b& pixel = left.at<cv::Vec3b>(y, x);
                pixel[0] = 255;  // B
                pixel[1] = (x * 255) / left.cols;  // G
                pixel[2] = (y * 255) / left.rows;  // R
            }
        }
        
        // 右摄像头: 红色渐变
        for (int y = 0; y < right.rows; y++) {
            for (int x = 0; x < right.cols; x++) {
                cv::Vec3b& pixel = right.at<cv::Vec3b>(y, x);
                pixel[0] = (y * 255) / right.rows;  // B
                pixel[1] = (x * 255) / right.cols;  // G
                pixel[2] = 255;  // R
            }
        }
        
        // 添加文字标识
        cv::putText(left, "LEFT CAM (TEST)", cv::Point(20, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        cv::putText(right, "RIGHT CAM (TEST)", cv::Point(20, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        
        std::lock_guard<std::mutex> lock(stats_mutex_);
        statistics_.total_frames++;
        statistics_.sync_failures++;
        return true;
    }
    
    // 如果只有一个摄像头可用，复制到另一个
    if (left_success && !right_success) {
        if (debug_frame_count % 100 == 0) {
            std::cout << "📷 只有左摄像头可用 (帧 #" << debug_frame_count << ")" << std::endl;
        }
        left = left_temp.clone();
        right = left_temp.clone();
        cv::putText(right, "COPIED FROM LEFT", cv::Point(20, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    } else if (right_success && !left_success) {
        if (debug_frame_count % 100 == 0) {
            std::cout << "📷 只有右摄像头可用 (帧 #" << debug_frame_count << ")" << std::endl;
        }
        right = right_temp.clone();
        left = right_temp.clone();
        cv::putText(left, "COPIED FROM RIGHT", cv::Point(20, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    } else {
        // 两个都成功，忽略同步误差直接使用
        if (debug_frame_count % 300 == 0) {
            std::cout << "📷 双摄像头都可用 (帧 #" << debug_frame_count << ")" << std::endl;
        }
        left = left_temp.clone();
        right = right_temp.clone();
    }
    
    // 更新统计信息
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        statistics_.total_frames++;
        statistics_.successful_captures++;
        statistics_.last_capture_time = std::chrono::steady_clock::now();
    }
    
    return true;
}

bool StereoVision::capture_stereo_frame(StereoFrame& frame) {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    
    if (!initialized_.load()) {
        return false;
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // 捕获原始图像
    if (!capture_synchronized_frames(frame.left_image, frame.right_image)) {
        frame.valid = false;
        return false;
    }
    
    frame.timestamp = std::chrono::steady_clock::now();
    frame.valid = true;
    
    // 如果已标定，进行校正和视差计算
    if (calibration_params_.is_calibrated) {
        cv::Mat left_rect, right_rect;
        if (rectify_images(frame.left_image, frame.right_image, left_rect, right_rect)) {
            if (compute_disparity(left_rect, right_rect, frame.disparity)) {
                disparity_to_depth(frame.disparity, frame.depth);
            }
        }
    }
    
    // 更新性能统计
    auto end_time = std::chrono::steady_clock::now();
    double processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count() / 1000.0;
    
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        statistics_.avg_processing_time_ms = 
            (statistics_.avg_processing_time_ms * (statistics_.total_frames - 1) + processing_time) /
            statistics_.total_frames;
    }
    
    return true;
}

bool StereoVision::rectify_images(const cv::Mat& left_raw, const cv::Mat& right_raw,
                                 cv::Mat& left_rect, cv::Mat& right_rect) {
    if (!calibration_params_.is_calibrated) {
        std::cerr << "相机未标定，无法进行图像校正" << std::endl;
        return false;
    }
    
    // 使用预计算的映射进行校正
    cv::remap(left_raw, left_rect, calibration_params_.map1_left, 
              calibration_params_.map2_left, cv::INTER_LINEAR);
    cv::remap(right_raw, right_rect, calibration_params_.map1_right, 
              calibration_params_.map2_right, cv::INTER_LINEAR);
    
    return true;
}

bool StereoVision::compute_disparity(const cv::Mat& left_rect, const cv::Mat& right_rect,
                                    cv::Mat& disparity) {
    if (!stereo_matcher_) {
        std::cerr << "立体匹配器未初始化" << std::endl;
        return false;
    }
    
    // 转换为灰度图
    cv::Mat left_gray, right_gray;
    if (left_rect.channels() == 3) {
        cv::cvtColor(left_rect, left_gray, cv::COLOR_BGR2GRAY);
    } else {
        left_gray = left_rect;
    }
    
    if (right_rect.channels() == 3) {
        cv::cvtColor(right_rect, right_gray, cv::COLOR_BGR2GRAY);
    } else {
        right_gray = right_rect;
    }
    
    // 计算视差
    cv::Mat raw_disparity;
    stereo_matcher_->compute(left_gray, right_gray, raw_disparity);
    
    // 应用WLS滤波
    if (matching_config_.use_wls_filter) {
#if HAS_WLS_FILTER
        if (wls_filter_) {
            wls_filter_->filter(raw_disparity, left_gray, disparity);
        } else {
            disparity = raw_disparity;
        }
#else
        disparity = raw_disparity;
#endif
    } else {
        disparity = raw_disparity;
    }
    
    return true;
}

bool StereoVision::disparity_to_depth(const cv::Mat& disparity, cv::Mat& depth) {
    if (!calibration_params_.is_calibrated) {
        return false;
    }
    
    // 使用重投影矩阵计算深度
    cv::reprojectImageTo3D(disparity, depth, calibration_params_.Q, true);
    
    return true;
}

Point3D StereoVision::pixel_to_3d(const cv::Point2f& pixel, const cv::Mat& disparity) {
    if (!calibration_params_.is_calibrated || disparity.empty()) {
        return Point3D{};
    }
    
    // 获取像素处的视差值
    int x = static_cast<int>(pixel.x);
    int y = static_cast<int>(pixel.y);
    
    if (x < 0 || x >= disparity.cols || y < 0 || y >= disparity.rows) {
        return Point3D{};
    }
    
    float disp_value = disparity.at<float>(y, x);
    if (disp_value <= 0) {
        return Point3D{};
    }
    
    // 使用重投影矩阵计算3D坐标
    cv::Mat point_4d = (cv::Mat_<float>(4, 1) << pixel.x, pixel.y, disp_value, 1.0);
    cv::Mat point_3d_h = calibration_params_.Q * point_4d;
    
    if (point_3d_h.at<float>(3, 0) == 0) {
        return Point3D{};
    }
    
    Point3D result;
    result.x = point_3d_h.at<float>(0, 0) / point_3d_h.at<float>(3, 0);
    result.y = point_3d_h.at<float>(1, 0) / point_3d_h.at<float>(3, 0);
    result.z = point_3d_h.at<float>(2, 0) / point_3d_h.at<float>(3, 0);
    
    // 计算置信度 (基于视差值的强度)
    result.confidence = std::min(1.0, disp_value / static_cast<double>(matching_config_.num_disparities));
    
    return result;
}

std::vector<Point3D> StereoVision::pixels_to_3d(const std::vector<cv::Point2f>& pixels,
                                                const cv::Mat& disparity) {
    std::vector<Point3D> result;
    result.reserve(pixels.size());
    
    for (const auto& pixel : pixels) {
        Point3D point_3d = pixel_to_3d(pixel, disparity);
        if (point_3d.confidence > matching_config_.min_confidence) {
            result.push_back(point_3d);
        }
    }
    
    return result;
}

std::vector<cv::Point2f> StereoVision::detect_bamboo_with_depth(const cv::Mat& left_image,
                                                               const cv::Mat& disparity,
                                                               double min_depth_mm,
                                                               double max_depth_mm) {
    std::vector<cv::Point2f> bamboo_points;
    
    if (left_image.empty() || disparity.empty()) {
        return bamboo_points;
    }
    
    // 基本的边缘检测来寻找竹子轮廓
    cv::Mat gray, edges;
    if (left_image.channels() == 3) {
        cv::cvtColor(left_image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = left_image;
    }
    
    cv::Canny(gray, edges, 50, 150);
    
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    for (const auto& contour : contours) {
        // 过滤小轮廓
        if (cv::contourArea(contour) < 100) {
            continue;
        }
        
        // 计算轮廓中心点
        cv::Moments moments = cv::moments(contour);
        if (moments.m00 == 0) continue;
        
        cv::Point2f center(moments.m10 / moments.m00, moments.m01 / moments.m00);
        
        // 检查深度范围
        Point3D point_3d = pixel_to_3d(center, disparity);
        if (point_3d.z >= min_depth_mm && point_3d.z <= max_depth_mm && 
            point_3d.confidence > matching_config_.min_confidence) {
            bamboo_points.push_back(center);
        }
    }
    
    return bamboo_points;
}

// 标定相关功能
bool StereoVision::start_calibration(const cv::Size& board_size, float square_size) {
    std::lock_guard<std::mutex> lock(calibration_mutex_);
    
    calibrating_.store(true);
    calibration_board_size_ = board_size;
    calibration_square_size_ = square_size;
    
    // 清除之前的标定数据
    calibration_object_points_.clear();
    calibration_left_points_.clear();
    calibration_right_points_.clear();
    
    // 生成标定板的3D点
    generate_calibration_pattern_points(board_size, square_size);
    
    std::cout << "开始标定，标定板尺寸: " << board_size.width << "x" << board_size.height 
              << "，方格大小: " << square_size << "mm" << std::endl;
    
    return true;
}

void StereoVision::generate_calibration_pattern_points(const cv::Size& board_size, float square_size) {
    std::vector<cv::Point3f> pattern_points;
    
    for (int i = 0; i < board_size.height; i++) {
        for (int j = 0; j < board_size.width; j++) {
            pattern_points.push_back(cv::Point3f(j * square_size, i * square_size, 0));
        }
    }
    
    // 这个模式会被重复用于每一帧
    calibration_object_points_.clear();
    calibration_object_points_.push_back(pattern_points);
}

CalibrationDetection StereoVision::detect_calibration_pattern(const cv::Mat& left_img, 
                                                             const cv::Mat& right_img) {
    CalibrationDetection result;
    
    // 转换为灰度图
    cv::Mat left_gray, right_gray;
    if (left_img.channels() == 3) {
        cv::cvtColor(left_img, left_gray, cv::COLOR_BGR2GRAY);
    } else {
        left_gray = left_img;
    }
    
    if (right_img.channels() == 3) {
        cv::cvtColor(right_img, right_gray, cv::COLOR_BGR2GRAY);
    } else {
        right_gray = right_img;
    }
    
    // 检测棋盘格角点
    result.left_found = cv::findChessboardCorners(left_gray, calibration_board_size_, 
                                                 result.left_corners);
    result.right_found = cv::findChessboardCorners(right_gray, calibration_board_size_, 
                                                  result.right_corners);
    
    // 如果检测成功，进行亚像素精度优化
    if (result.left_found) {
        cv::cornerSubPix(left_gray, result.left_corners, cv::Size(11, 11), cv::Size(-1, -1),
                        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    }
    
    if (result.right_found) {
        cv::cornerSubPix(right_gray, result.right_corners, cv::Size(11, 11), cv::Size(-1, -1),
                        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    }
    
    return result;
}

bool StereoVision::add_calibration_frame(const cv::Mat& left_img, const cv::Mat& right_img) {
    if (!calibrating_.load()) {
        return false;
    }
    
    auto detection = detect_calibration_pattern(left_img, right_img);
    
    if (detection.left_found && detection.right_found) {
        std::lock_guard<std::mutex> lock(calibration_mutex_);
        
        calibration_left_points_.push_back(detection.left_corners);
        calibration_right_points_.push_back(detection.right_corners);
        calibration_object_points_.push_back(calibration_object_points_[0]); // 重复使用模式
        
        std::cout << "添加标定帧 #" << calibration_left_points_.size() << std::endl;
        return true;
    }
    
    return false;
}

bool StereoVision::calibrate_cameras() {
    std::lock_guard<std::mutex> lock(calibration_mutex_);
    
    if (calibration_left_points_.size() < 10) {
        std::cerr << "标定帧数不足，至少需要10帧，当前: " << calibration_left_points_.size() << std::endl;
        return false;
    }
    
    std::cout << "开始标定，使用 " << calibration_left_points_.size() << " 帧图像..." << std::endl;
    
    cv::Size image_size(sync_config_.width, sync_config_.height);
    
    // 初始化相机矩阵
    calibration_params_.left_camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    calibration_params_.right_camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    calibration_params_.left_dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    calibration_params_.right_dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    
    // 立体标定
    double rms_error = cv::stereoCalibrate(
        calibration_object_points_,
        calibration_left_points_,
        calibration_right_points_,
        calibration_params_.left_camera_matrix,
        calibration_params_.left_dist_coeffs,
        calibration_params_.right_camera_matrix,
        calibration_params_.right_dist_coeffs,
        image_size,
        calibration_params_.R,
        calibration_params_.T,
        calibration_params_.E,
        calibration_params_.F,
        cv::CALIB_SAME_FOCAL_LENGTH | cv::CALIB_ZERO_TANGENT_DIST,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5)
    );
    
    std::cout << "立体标定完成，RMS误差: " << rms_error << std::endl;
    
    // 立体校正
    cv::stereoRectify(
        calibration_params_.left_camera_matrix,
        calibration_params_.left_dist_coeffs,
        calibration_params_.right_camera_matrix,
        calibration_params_.right_dist_coeffs,
        image_size,
        calibration_params_.R,
        calibration_params_.T,
        calibration_params_.R1,
        calibration_params_.R2,
        calibration_params_.P1,
        calibration_params_.P2,
        calibration_params_.Q,
        cv::CALIB_ZERO_DISPARITY,
        -1,
        image_size
    );
    
    // 计算校正映射
    cv::initUndistortRectifyMap(
        calibration_params_.left_camera_matrix,
        calibration_params_.left_dist_coeffs,
        calibration_params_.R1,
        calibration_params_.P1,
        image_size,
        CV_16SC2,
        calibration_params_.map1_left,
        calibration_params_.map2_left
    );
    
    cv::initUndistortRectifyMap(
        calibration_params_.right_camera_matrix,
        calibration_params_.right_dist_coeffs,
        calibration_params_.R2,
        calibration_params_.P2,
        image_size,
        CV_16SC2,
        calibration_params_.map1_right,
        calibration_params_.map2_right
    );
    
    // 计算基线距离
    calibration_params_.baseline = stereo_utils::calculate_baseline(calibration_params_.T);
    calibration_params_.image_size = image_size;
    calibration_params_.is_calibrated = true;
    
    calibrating_.store(false);
    
    std::cout << "立体校正完成" << std::endl;
    std::cout << "基线距离: " << calibration_params_.baseline << "mm" << std::endl;
    
    return true;
}

StereoVision::Statistics StereoVision::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return statistics_;
}

bool StereoVision::load_calibration(const std::string& calibration_file) {
    try {
        cv::FileStorage fs(calibration_file, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "无法打开标定文件: " << calibration_file << std::endl;
            return false;
        }
        
        // 读取标定参数
        fs["left_camera_matrix"] >> calibration_params_.left_camera_matrix;
        fs["right_camera_matrix"] >> calibration_params_.right_camera_matrix;
        fs["left_dist_coeffs"] >> calibration_params_.left_dist_coeffs;
        fs["right_dist_coeffs"] >> calibration_params_.right_dist_coeffs;
        fs["R"] >> calibration_params_.R;
        fs["T"] >> calibration_params_.T;
        fs["E"] >> calibration_params_.E;
        fs["F"] >> calibration_params_.F;
        fs["Q"] >> calibration_params_.Q;
        fs["R1"] >> calibration_params_.R1;
        fs["R2"] >> calibration_params_.R2;
        fs["P1"] >> calibration_params_.P1;
        fs["P2"] >> calibration_params_.P2;
        fs["map1_left"] >> calibration_params_.map1_left;
        fs["map2_left"] >> calibration_params_.map2_left;
        fs["map1_right"] >> calibration_params_.map1_right;
        fs["map2_right"] >> calibration_params_.map2_right;
        
        cv::Size temp_size;
        fs["image_size"] >> temp_size;
        calibration_params_.image_size = temp_size;
        
        double baseline;
        fs["baseline"] >> baseline;
        calibration_params_.baseline = baseline;
        
        fs.release();
        
        calibration_params_.is_calibrated = true;
        std::cout << "标定文件加载成功: " << calibration_file << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "加载标定文件异常: " << e.what() << std::endl;
        return false;
    }
}

StereoCalibrationParams StereoVision::get_calibration_params() const {
    return calibration_params_;
}

// 工具函数实现
namespace stereo_utils {

double calculate_baseline(const cv::Mat& T) {
    // 基线距离是平移向量的模长
    return cv::norm(T);
}

cv::Mat visualize_disparity(const cv::Mat& disparity, bool color) {
    cv::Mat vis;
    cv::normalize(disparity, vis, 0, 255, cv::NORM_MINMAX, CV_8U);
    
    if (color) {
        cv::applyColorMap(vis, vis, cv::COLORMAP_JET);
    }
    
    return vis;
}

cv::Mat visualize_depth(const cv::Mat& depth, double max_depth) {
    cv::Mat vis;
    cv::Mat depth_norm;
    depth.copyTo(depth_norm);
    
    // 限制深度范围
    depth_norm.setTo(max_depth, depth_norm > max_depth);
    depth_norm.setTo(0, depth_norm < 0);
    
    // 归一化并转换为8位
    depth_norm /= max_depth;
    depth_norm *= 255;
    depth_norm.convertTo(vis, CV_8U);
    
    // 应用颜色映射
    cv::applyColorMap(vis, vis, cv::COLORMAP_INFERNO);
    
    return vis;
}

CameraSyncConfig get_recommended_camera_config(const std::string& camera_type) {
    CameraSyncConfig config;
    
    if (camera_type == "usb") {
        config.left_device = "/dev/video0";
        config.right_device = "/dev/video2";
        config.width = 1280;
        config.height = 720;
        config.fps = 30;
        config.hardware_sync = false;
        config.sync_tolerance_ms = 10;
    } else if (camera_type == "csi") {
        // CSI相机配置（如Jetson平台）
        config.left_device = "0";
        config.right_device = "1";
        config.width = 1920;
        config.height = 1080;
        config.fps = 30;
        config.hardware_sync = true;
        config.sync_tolerance_ms = 5;
    }
    
    return config;
}

StereoMatchingConfig get_recommended_matching_config(const cv::Size& image_size) {
    StereoMatchingConfig config;
    
    // 根据图像尺寸调整参数
    int area = image_size.width * image_size.height;
    
    if (area > 1920*1080) {  // 高分辨率
        config.num_disparities = 16 * 8;
        config.block_size = 7;
        config.P1 = 8 * 3 * 7 * 7;
        config.P2 = 32 * 3 * 7 * 7;
    } else if (area > 1280*720) {  // 中等分辨率
        config.num_disparities = 16 * 6;
        config.block_size = 5;
        config.P1 = 8 * 3 * 5 * 5;
        config.P2 = 32 * 3 * 5 * 5;
    } else {  // 低分辨率
        config.num_disparities = 16 * 4;
        config.block_size = 3;
        config.P1 = 8 * 3 * 3 * 3;
        config.P2 = 32 * 3 * 3 * 3;
    }
    
    return config;
}

std::string evaluate_calibration_quality(const StereoCalibrationParams& params) {
    std::stringstream ss;
    
    if (!params.is_calibrated) {
        ss << "未标定";
        return ss.str();
    }
    
    ss << "基线距离: " << params.baseline << "mm\n";
    
    if (params.baseline < 50) {
        ss << "⚠️ 基线距离过小，可能影响深度精度\n";
    } else if (params.baseline > 500) {
        ss << "⚠️ 基线距离过大，可能影响视野重叠\n";
    } else {
        ss << "✅ 基线距离适中\n";
    }
    
    return ss.str();
}

} // namespace stereo_utils

// GStreamer流输出功能实现

bool StereoVision::initialize_video_stream() {
    std::cout << "初始化立体视觉GStreamer流输出..." << std::endl;
    
    // 初始化GStreamer
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
        std::cout << "GStreamer已初始化" << std::endl;
    }
    
    std::string pipeline_desc = build_stream_pipeline();
    if (pipeline_desc.empty()) {
        std::cerr << "构建GStreamer管道失败" << std::endl;
        return false;
    }
    
    std::cout << "GStreamer管道构建成功: " << pipeline_desc << std::endl;
    
    // 启动管道
    GstStateChangeReturn ret = gst_element_set_state(gst_pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "启动GStreamer管道失败" << std::endl;
        gst_object_unref(gst_pipeline_);
        gst_pipeline_ = nullptr;
        gst_appsrc_ = nullptr;
        return false;
    }
    
    // 等待管道状态切换完成
    GstState state;
    ret = gst_element_get_state(gst_pipeline_, &state, nullptr, GST_SECOND * 2);
    if (ret == GST_STATE_CHANGE_FAILURE || state != GST_STATE_PLAYING) {
        std::cerr << "GStreamer管道启动超时或失败，当前状态: " << state << std::endl;
        return false;
    }
    
    std::cout << "立体视觉GStreamer流输出初始化完成，管道状态: PLAYING" << std::endl;
    return true;
}

std::string StereoVision::build_stream_pipeline() {
    // 创建管道元素
    gst_pipeline_ = gst_pipeline_new("stereo-video-pipeline");
    if (!gst_pipeline_) {
        std::cerr << "创建GStreamer管道失败" << std::endl;
        return ""; // 返回空字符串表示失败
    }
    
    // 创建appsrc
    gst_appsrc_ = gst_element_factory_make("appsrc", "stereo-source");
    if (!gst_appsrc_) {
        std::cerr << "创建appsrc失败" << std::endl;
        return "";
    }
    
    // 创建视频处理元素
    GstElement* videoconvert = gst_element_factory_make("videoconvert", "convert");
    GstElement* videoscale = gst_element_factory_make("videoscale", "scale");
    GstElement* capsfilter = gst_element_factory_make("capsfilter", "caps");
    
    // Jetson Orin NX专用硬件编码器优先级列表
    GstElement* encoder = nullptr;
    const char* encoder_names[] = {
        "nvv4l2h264enc",    // NVIDIA V4L2 H.264编码器 (推荐)
        "omxh264enc",       // OpenMAX H.264编码器 (备用)
        "nvh264enc",        // NVIDIA H.264编码器 (备用)
        "x264enc",          // 软件编码器 (最后备用)
        NULL
    };
    const char* used_encoder = nullptr;
    
    std::cout << "🔍 检测Jetson Orin NX可用编码器..." << std::endl;
    for (int i = 0; encoder_names[i] != NULL; i++) {
        std::cout << "   尝试: " << encoder_names[i];
        encoder = gst_element_factory_make(encoder_names[i], "encoder");
        if (encoder) {
            used_encoder = encoder_names[i];
            std::cout << " ✅" << std::endl;
            break;
        } else {
            std::cout << " ❌" << std::endl;
        }
    }
    
    GstElement* parser = nullptr;
    GstElement* payloader = nullptr;
    
    // 如果编码器检测失败，使用基础的x264软件编码器（总是可用的）
    if (!encoder) {
        std::cout << "⚠️ 硬件编码器不可用，回退到软件编码器" << std::endl;
        encoder = gst_element_factory_make("x264enc", "encoder");
        used_encoder = "x264enc (fallback)";
    }
    
    // 创建解析器和负载器
    parser = gst_element_factory_make("h264parse", "parser");
    payloader = gst_element_factory_make("rtph264pay", "payload");
    
    if (encoder) {
        std::cout << "✅ 使用编码器: " << used_encoder << std::endl;
    } else {
        std::cerr << "❌ 连软件编码器也不可用，这不应该发生" << std::endl;
        if (gst_pipeline_) { gst_object_unref(gst_pipeline_); gst_pipeline_ = nullptr; }
        return "";
    }
    
    GstElement* udpsink = gst_element_factory_make("udpsink", "sink");
    
    // 检查关键元素是否创建成功
    if (!videoconvert || !videoscale || !capsfilter || !encoder || !parser || !payloader || !udpsink) {
        std::cerr << "❌ 创建GStreamer元素失败:" << std::endl;
        std::cerr << "   videoconvert: " << (videoconvert ? "✅" : "❌") << std::endl;
        std::cerr << "   videoscale: " << (videoscale ? "✅" : "❌") << std::endl;
        std::cerr << "   capsfilter: " << (capsfilter ? "✅" : "❌") << std::endl;
        std::cerr << "   encoder (" << (used_encoder ? used_encoder : "unknown") << "): " << (encoder ? "✅" : "❌") << std::endl;
        std::cerr << "   h264parse: " << (parser ? "✅" : "❌") << std::endl;
        std::cerr << "   rtph264pay: " << (payloader ? "✅" : "❌") << std::endl;
        std::cerr << "   udpsink: " << (udpsink ? "✅" : "❌") << std::endl;
        
        // 清理已创建的元素
        if (gst_pipeline_) { gst_object_unref(gst_pipeline_); gst_pipeline_ = nullptr; }
        return "";
    }
    
    // 配置appsrc
    g_object_set(G_OBJECT(gst_appsrc_),
        "caps", gst_caps_from_string("video/x-raw,format=BGR,width=640,height=480,framerate=30/1"),
        "format", GST_FORMAT_TIME,
        "is-live", TRUE,
        "do-timestamp", TRUE,
        "max-buffers", 2,      // 限制缓冲区防止延迟
        "drop", TRUE,          // 允许丢帧
        NULL);
    
    // 配置缩放和格式转换
    GstCaps* scale_caps = gst_caps_from_string("video/x-raw,width=640,height=480,framerate=30/1");
    g_object_set(G_OBJECT(capsfilter), "caps", scale_caps, NULL);
    gst_caps_unref(scale_caps);
    
    // 配置编码器 (针对不同编码器优化)
    if (strstr(used_encoder, "nvv4l2h264enc")) {
        // NVIDIA V4L2编码器配置 (推荐)
        g_object_set(G_OBJECT(encoder),
            "bitrate", 2000000,          // 2Mbps
            "preset-level", 1,           // UltraFastPreset
            "profile", 0,                // Baseline
            "iframeinterval", 30,        // I帧间隔
            "control-rate", 1,           // CBR
            NULL);
        std::cout << "🚀 使用NVIDIA V4L2硬件编码器 (最佳性能)" << std::endl;
    } else if (strstr(used_encoder, "omxh264enc")) {
        // OpenMAX编码器配置
        g_object_set(G_OBJECT(encoder),
            "bitrate", 2000000,          // 2Mbps
            "preset-level", 0,           // UltraFastPreset
            "profile", 0,                // Baseline
            "iframeinterval", 30,        // I帧间隔
            NULL);
        std::cout << "⚡ 使用OpenMAX硬件编码器" << std::endl;
    } else if (strstr(used_encoder, "nvh264enc")) {
        // NVIDIA编码器配置
        g_object_set(G_OBJECT(encoder),
            "bitrate", 2000000,          // 2Mbps
            "preset", 1,                 // low-latency-default
            NULL);
        std::cout << "🔧 使用NVIDIA编码器" << std::endl;
    } else if (strstr(used_encoder, "x264enc")) {
        // 软件编码器配置
        g_object_set(G_OBJECT(encoder),
            "tune", 4,                   // zerolatency
            "bitrate", 2000,             // 2Mbps
            "speed-preset", 6,           // ultrafast
            "key-int-max", 30,           // GOP size
            NULL);
        std::cout << "💻 使用软件编码器 (性能较低)" << std::endl;
    }
    
    // 配置RTP负载器
    g_object_set(G_OBJECT(payloader),
        "pt", 96,
        "config-interval", 1,
        NULL);
    
    // 配置UDP输出
    g_object_set(G_OBJECT(udpsink),
        "host", "127.0.0.1",
        "port", 5000,
        "sync", FALSE,               // 异步发送，减少延迟
        NULL);
    
    // 添加所有元素到管道
    gst_bin_add_many(GST_BIN(gst_pipeline_),
        gst_appsrc_, videoconvert, videoscale, capsfilter,
        encoder, parser, payloader, udpsink, NULL);
    
    // 连接元素
    if (!gst_element_link_many(gst_appsrc_, videoconvert, videoscale, capsfilter,
                               encoder, parser, payloader, udpsink, NULL)) {
        std::cerr << "❌ 连接GStreamer元素失败" << std::endl;
        gst_object_unref(gst_pipeline_);
        gst_pipeline_ = nullptr;
        gst_appsrc_ = nullptr;
        return "";
    }
    
    std::cout << "✅ Jetson Orin NX GStreamer管道构建成功: " << used_encoder << " -> RTP -> UDP:5000" << std::endl;
    return "stereo-video-pipeline";
}

void StereoVision::push_frame_to_stream(const cv::Mat& frame) {
    static int push_failures = 0;
    static int push_successes = 0;
    static auto last_debug_time = std::chrono::steady_clock::now();
    
    if (!gst_appsrc_) {
        if (push_failures % 100 == 0) {
            std::cerr << "GStreamer appsrc 未初始化" << std::endl;
        }
        push_failures++;
        return;
    }
    
    if (!stream_enabled_) {
        if (push_failures % 100 == 0) {
            std::cerr << "视频流未启用" << std::endl;
        }
        push_failures++;
        return;
    }
    
    if (frame.empty()) {
        if (push_failures % 100 == 0) {
            std::cerr << "输入帧为空" << std::endl;
        }
        push_failures++;
        return;
    }
    
    // 确保帧格式正确 (640x480 BGR)
    cv::Mat output_frame;
    if (frame.size() != cv::Size(640, 480)) {
        cv::resize(frame, output_frame, cv::Size(640, 480));
        if (frame_counter_ % 300 == 0) {
            std::cout << "缩放帧: " << frame.cols << "x" << frame.rows << " -> 640x480" << std::endl;
        }
    } else {
        output_frame = frame;
    }
    
    // 确保是BGR格式
    if (output_frame.channels() != 3) {
        cv::cvtColor(output_frame, output_frame, cv::COLOR_GRAY2BGR);
        if (frame_counter_ % 300 == 0) {
            std::cout << "转换为BGR格式" << std::endl;
        }
    }
    
    // 创建GStreamer缓冲区
    gsize size = output_frame.total() * output_frame.elemSize();
    GstBuffer* buffer = gst_buffer_new_allocate(NULL, size, NULL);
    
    if (!buffer) {
        std::cerr << "创建GStreamer缓冲区失败，大小: " << size << " 字节" << std::endl;
        push_failures++;
        return;
    }
    
    // 复制数据到缓冲区
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        std::cerr << "映射GStreamer缓冲区失败" << std::endl;
        gst_buffer_unref(buffer);
        push_failures++;
        return;
    }
    
    memcpy(map.data, output_frame.data, size);
    gst_buffer_unmap(buffer, &map);
    
    // 设置时间戳
    GST_BUFFER_PTS(buffer) = frame_counter_ * GST_SECOND / 30;  // 30fps
    GST_BUFFER_DURATION(buffer) = GST_SECOND / 30;
    
    // 推送到appsrc
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(gst_appsrc_), buffer);
    if (ret != GST_FLOW_OK) {
        push_failures++;
        if (push_failures % 10 == 0) {
            std::cerr << "推送视频帧失败: " << ret << " (失败次数: " << push_failures << ")" << std::endl;
        }
    } else {
        push_successes++;
        frame_counter_++;
        
        // 完全移除所有调试输出和统计信息，避免前端界面干扰
        // 移除了所有FPS统计、帧计数和数据大小显示
    }
}

cv::Mat StereoVision::create_display_frame(const cv::Mat& left, const cv::Mat& right) {
    cv::Mat display_frame;
    
    if (left.empty()) {
        return display_frame;
    }
    
    switch (display_mode_) {
        case DisplayMode::SIDE_BY_SIDE:
            // 并排显示模式
            if (!right.empty() && left.size() == right.size()) {
                cv::hconcat(left, right, display_frame);
                // 缩放到640x480 (左右各320x480)
                cv::resize(display_frame, display_frame, cv::Size(640, 480));
            } else {
                // 只有左摄像头，缩放到640x480
                cv::resize(left, display_frame, cv::Size(640, 480));
            }
            break;
            
        case DisplayMode::FUSED:
            // 融合显示模式，只显示左摄像头
            cv::resize(left, display_frame, cv::Size(640, 480));
            break;
    }
    
    return display_frame;
}

bool StereoVision::enable_video_stream(bool enable) {
    if (stream_enabled_ != enable) {
        stream_enabled_ = enable;
        if (enable && !gst_pipeline_) {
            if (!initialize_video_stream()) {
                std::cerr << "⚠️ 视频流初始化失败，但仍将启用流标志用于调试" << std::endl;
                // 即使初始化失败，也启用流标志，这样可以看到详细的错误信息
                stream_enabled_ = true;
            }
        }
        std::cout << "🎥 立体视觉流输出: " << (stream_enabled_ ? "已启用" : "已禁用") << std::endl;
        std::cout << "📺 GStreamer管道状态: " << (gst_pipeline_ ? "已创建" : "未创建") << std::endl;
        std::cout << "📡 AppSrc状态: " << (gst_appsrc_ ? "已创建" : "未创建") << std::endl;
    }
    return stream_enabled_;
}

void StereoVision::set_display_mode(DisplayMode mode) {
    if (display_mode_ != mode) {
        display_mode_ = mode;
        std::string mode_str = (mode == DisplayMode::SIDE_BY_SIDE) ? "并排显示" : "融合显示";
        std::cout << "立体视觉显示模式: " << mode_str << std::endl;
    }
}

} // namespace vision
} // namespace bamboo_cut