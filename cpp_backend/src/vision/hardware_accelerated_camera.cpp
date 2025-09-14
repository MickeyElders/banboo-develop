#include "bamboo_cut/vision/hardware_accelerated_camera.h"
#include <iostream>
#include <chrono>
#include <algorithm>

namespace bamboo_cut {
namespace vision {

// HardwareAccelerationConfig validation implementation
bool HardwareAccelerationConfig::validate() const {
    if (width <= 0 || height <= 0 || fps <= 0) {
        return false;
    }
    if (v4l2_buffer_count <= 0 || max_queue_size <= 0) {
        return false;
    }
    if (num_capture_threads <= 0) {
        return false;
    }
    return true;
}

bool HardwareAcceleratedFrame::is_valid() const {
    return !cpu_frame.empty() && width > 0 && height > 0;
}

HardwareAcceleratedCamera::HardwareAcceleratedCamera() 
    : config_(), initialized_(false), capturing_(false) {
    std::cout << "创建HardwareAcceleratedCamera实例" << std::endl;
}

HardwareAcceleratedCamera::HardwareAcceleratedCamera(const HardwareAccelerationConfig& config) 
    : config_(config), initialized_(false), capturing_(false) {
    std::cout << "创建HardwareAcceleratedCamera实例，使用自定义配置" << std::endl;
}

HardwareAcceleratedCamera::~HardwareAcceleratedCamera() {
    shutdown();
    std::cout << "销毁HardwareAcceleratedCamera实例" << std::endl;
}

bool HardwareAcceleratedCamera::initialize() {
    if (initialized_) {
        std::cout << "HardwareAcceleratedCamera已初始化" << std::endl;
        return true;
    }
    
    std::cout << "初始化HardwareAcceleratedCamera..." << std::endl;
    
    try {
        // 验证配置
        if (!config_.validate()) {
            std::cerr << "HardwareAcceleratedCamera配置无效" << std::endl;
            return false;
        }
        
        // 根据加速类型初始化不同的硬件
        switch (config_.acceleration_type) {
            case HardwareAccelerationType::V4L2_HW:
                if (!initialize_v4l2_hardware()) {
                    std::cerr << "V4L2硬件初始化失败" << std::endl;
                    return false;
                }
                break;
            
            case HardwareAccelerationType::GSTREAMER_HW:
                if (!initialize_gstreamer_hardware()) {
                    std::cerr << "GStreamer硬件初始化失败" << std::endl;
                    return false;
                }
                break;
            
            case HardwareAccelerationType::CUDA_HW:
                if (!initialize_cuda_hardware()) {
                    std::cerr << "CUDA硬件初始化失败" << std::endl;
                    return false;
                }
                break;
            
            case HardwareAccelerationType::MIXED_HW:
                // 尝试混合初始化
                if (!initialize_v4l2_hardware() && !initialize_gstreamer_hardware()) {
                    std::cerr << "混合硬件初始化失败" << std::endl;
                    return false;
                }
                break;
            
            default:
                std::cout << "使用软件模式" << std::endl;
                break;
        }
        
        initialized_ = true;
        std::cout << "HardwareAcceleratedCamera初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "HardwareAcceleratedCamera初始化异常: " << e.what() << std::endl;
        return false;
    }
}

void HardwareAcceleratedCamera::shutdown() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "关闭HardwareAcceleratedCamera..." << std::endl;
    
    stop_capture();
    cleanup_v4l2_hardware();
    cleanup_gstreamer_hardware();
    cleanup_cuda_hardware();
    
    initialized_ = false;
    std::cout << "HardwareAcceleratedCamera已关闭" << std::endl;
}

bool HardwareAcceleratedCamera::start_capture() {
    if (!initialized_) {
        std::cerr << "摄像头未初始化" << std::endl;
        return false;
    }
    
    if (capturing_) {
        std::cout << "摄像头已在捕获中" << std::endl;
        return true;
    }
    
    std::cout << "开始摄像头捕获..." << std::endl;
    
    stop_capture_ = false;
    capturing_ = true;
    
    if (config_.enable_async_capture) {
        capture_thread_ = std::thread(&HardwareAcceleratedCamera::capture_thread_function, this);
    }
    
    std::cout << "摄像头捕获已启动" << std::endl;
    return true;
}

void HardwareAcceleratedCamera::stop_capture() {
    if (!capturing_) {
        return;
    }
    
    std::cout << "停止摄像头捕获..." << std::endl;
    
    stop_capture_ = true;
    capturing_ = false;
    
    if (capture_thread_.joinable()) {
        frame_queue_cv_.notify_all();
        capture_thread_.join();
    }
    
    std::cout << "摄像头捕获已停止" << std::endl;
}

bool HardwareAcceleratedCamera::capture_frame(HardwareAcceleratedFrame& frame) {
    if (!capturing_) {
        std::cerr << "摄像头未在捕获状态" << std::endl;
        return false;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bool success = false;
        
        // 根据硬件加速类型选择捕获方法
        switch (config_.acceleration_type) {
            case HardwareAccelerationType::V4L2_HW:
                success = capture_v4l2_frame(frame);
                break;
            
            case HardwareAccelerationType::GSTREAMER_HW:
                success = capture_gstreamer_frame(frame);
                break;
            
            default:
                // 软件模式或其他：创建一个测试帧
                frame.cpu_frame = cv::Mat::zeros(config_.height, config_.width, CV_8UC3);
                frame.width = config_.width;
                frame.height = config_.height;
                frame.channels = 3;
                frame.acceleration_type = config_.acceleration_type;
                frame.pixel_format = config_.pixel_format;
                success = true;
                break;
        }
        
        if (success) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            frame.capture_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            frame.capture_time_ms = duration.count() / 1000.0;
            
            // 应用硬件ISP处理
            if (config_.enable_hardware_isp) {
                apply_hardware_isp(frame);
            }
            
            // 更新性能统计
            update_performance_stats(frame);
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "帧捕获异常: " << e.what() << std::endl;
        return false;
    }
}

bool HardwareAcceleratedCamera::capture_frame_async(std::function<void(const HardwareAcceleratedFrame&)> callback) {
    frame_callback_ = callback;
    return start_capture();
}

void HardwareAcceleratedCamera::set_config(const HardwareAccelerationConfig& config) {
    config_ = config;
}

HardwareAccelerationStats HardwareAcceleratedCamera::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return performance_stats_;
}

void HardwareAcceleratedCamera::reset_performance_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    performance_stats_ = HardwareAccelerationStats{};
}

void HardwareAcceleratedCamera::set_frame_callback(FrameCallback callback) {
    frame_callback_ = callback;
}

void HardwareAcceleratedCamera::set_error_callback(ErrorCallback callback) {
    error_callback_ = callback;
}

// 私有方法实现
bool HardwareAcceleratedCamera::initialize_v4l2_hardware() {
    std::cout << "初始化V4L2硬件加速..." << std::endl;
    // 简化实现：返回true表示成功
    return true;
}

bool HardwareAcceleratedCamera::setup_v4l2_buffers() {
    return true;
}

bool HardwareAcceleratedCamera::capture_v4l2_frame(HardwareAcceleratedFrame& frame) {
    // 创建测试帧
    frame.cpu_frame = cv::Mat::zeros(config_.height, config_.width, CV_8UC3);
    frame.width = config_.width;
    frame.height = config_.height;
    frame.channels = 3;
    frame.acceleration_type = HardwareAccelerationType::V4L2_HW;
    frame.pixel_format = config_.pixel_format;
    return true;
}

void HardwareAcceleratedCamera::cleanup_v4l2_hardware() {
    if (v4l2_fd_ >= 0) {
        close(v4l2_fd_);
        v4l2_fd_ = -1;
    }
}

bool HardwareAcceleratedCamera::initialize_gstreamer_hardware() {
    std::cout << "初始化GStreamer硬件加速..." << std::endl;
    // 简化实现
    return true;
}

bool HardwareAcceleratedCamera::setup_gstreamer_pipeline() {
    return true;
}

bool HardwareAcceleratedCamera::capture_gstreamer_frame(HardwareAcceleratedFrame& frame) {
    // 创建测试帧
    frame.cpu_frame = cv::Mat::zeros(config_.height, config_.width, CV_8UC3);
    frame.width = config_.width;
    frame.height = config_.height;
    frame.channels = 3;
    frame.acceleration_type = HardwareAccelerationType::GSTREAMER_HW;
    frame.pixel_format = config_.pixel_format;
    return true;
}

void HardwareAcceleratedCamera::cleanup_gstreamer_hardware() {
    // 清理GStreamer资源
}

bool HardwareAcceleratedCamera::initialize_cuda_hardware() {
    std::cout << "初始化CUDA硬件加速..." << std::endl;
#ifdef ENABLE_CUDA
    // CUDA相关初始化
    return true;
#else
    std::cout << "CUDA未启用，跳过CUDA硬件初始化" << std::endl;
    return false;
#endif
}

bool HardwareAcceleratedCamera::setup_cuda_memory_pool() {
    return true;
}

bool HardwareAcceleratedCamera::map_gpu_memory(cv::Mat& cpu_frame, cv::Mat& gpu_frame) {
    return true;
}

void HardwareAcceleratedCamera::unmap_gpu_memory(cv::Mat& gpu_frame) {
    // 释放GPU内存
}

void HardwareAcceleratedCamera::cleanup_cuda_hardware() {
    // 清理CUDA资源
}

bool HardwareAcceleratedCamera::apply_hardware_isp(HardwareAcceleratedFrame& frame) {
    if (frame.cpu_frame.empty()) {
        return false;
    }
    
    // 简化的ISP处理
    if (config_.enable_auto_exposure) {
        apply_auto_exposure(frame);
    }
    
    if (config_.enable_auto_white_balance) {
        apply_auto_white_balance(frame);
    }
    
    if (config_.enable_noise_reduction) {
        apply_noise_reduction(frame);
    }
    
    if (config_.enable_edge_enhancement) {
        apply_edge_enhancement(frame);
    }
    
    frame.hardware_processed = true;
    return true;
}

bool HardwareAcceleratedCamera::apply_auto_exposure(HardwareAcceleratedFrame& frame) {
    // 简化的自动曝光
    cv::Mat gray;
    cv::cvtColor(frame.cpu_frame, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean_val = cv::mean(gray);
    
    if (mean_val[0] < 100) {
        // 图像过暗，增强亮度
        frame.cpu_frame *= 1.2;
    } else if (mean_val[0] > 200) {
        // 图像过亮，降低亮度
        frame.cpu_frame *= 0.8;
    }
    
    return true;
}

bool HardwareAcceleratedCamera::apply_auto_white_balance(HardwareAcceleratedFrame& frame) {
    // 简化的自动白平衡
    std::vector<cv::Mat> channels;
    cv::split(frame.cpu_frame, channels);
    
    cv::Scalar b_mean = cv::mean(channels[0]);
    cv::Scalar g_mean = cv::mean(channels[1]);
    cv::Scalar r_mean = cv::mean(channels[2]);
    
    double avg = (b_mean[0] + g_mean[0] + r_mean[0]) / 3.0;
    
    channels[0] *= avg / b_mean[0];
    channels[1] *= avg / g_mean[0];
    channels[2] *= avg / r_mean[0];
    
    cv::merge(channels, frame.cpu_frame);
    return true;
}

bool HardwareAcceleratedCamera::apply_noise_reduction(HardwareAcceleratedFrame& frame) {
    cv::bilateralFilter(frame.cpu_frame, frame.cpu_frame, 5, 80, 80);
    return true;
}

bool HardwareAcceleratedCamera::apply_edge_enhancement(HardwareAcceleratedFrame& frame) {
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    cv::filter2D(frame.cpu_frame, frame.cpu_frame, -1, kernel);
    return true;
}

void HardwareAcceleratedCamera::capture_thread_function() {
    while (!stop_capture_) {
        HardwareAcceleratedFrame frame;
        if (capture_frame(frame) && frame_callback_) {
            frame_callback_(frame);
        }
        
        // 控制帧率
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / config_.fps));
    }
}

void HardwareAcceleratedCamera::process_captured_frame(const HardwareAcceleratedFrame& frame) {
    if (frame_callback_) {
        frame_callback_(frame);
    }
}

void HardwareAcceleratedCamera::update_performance_stats(const HardwareAcceleratedFrame& frame) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    performance_stats_.total_frames_captured++;
    performance_stats_.total_frames_processed++;
    
    if (performance_stats_.total_frames_captured == 1) {
        performance_stats_.avg_capture_time_ms = frame.capture_time_ms;
    } else {
        performance_stats_.avg_capture_time_ms = 
            (performance_stats_.avg_capture_time_ms * (performance_stats_.total_frames_captured - 1) + 
             frame.capture_time_ms) / performance_stats_.total_frames_captured;
    }
    
    // 计算FPS
    auto now = std::chrono::steady_clock::now();
    static auto last_time = now;
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time);
    
    if (elapsed.count() > 1000) {
        performance_stats_.current_fps = performance_stats_.total_frames_captured * 1000.0 / elapsed.count();
        last_time = now;
    }
    
    performance_stats_.target_fps = config_.fps;
    performance_stats_.hardware_isp_active = config_.enable_hardware_isp;
    performance_stats_.gpu_memory_mapping_active = config_.enable_gpu_memory_mapping;
    performance_stats_.zero_copy_active = config_.enable_zero_copy;
}

void HardwareAcceleratedCamera::measure_gpu_utilization() {
    // GPU利用率测量的简化实现
}

void HardwareAcceleratedCamera::measure_memory_usage() {
    // 内存使用测量的简化实现
}

void HardwareAcceleratedCamera::set_error(const std::string& error) {
    last_error_ = error;
    std::cerr << "HardwareAcceleratedCamera错误: " << error << std::endl;
    
    if (error_callback_) {
        error_callback_(error);
    }
}

bool HardwareAcceleratedCamera::handle_hardware_error(const std::string& error) {
    set_error(error);
    return false;
}

// MultiCameraHardwareManager implementation
bool MultiCameraHardwareManager::MultiCameraConfig::validate() const {
    if (camera_devices.empty()) {
        return false;
    }
    if (camera_configs.size() != camera_devices.size()) {
        return false;
    }
    if (max_cameras <= 0 || sync_tolerance_ms < 0) {
        return false;
    }
    return true;
}

MultiCameraHardwareManager::MultiCameraHardwareManager() 
    : config_(), initialized_(false) {
    std::cout << "创建MultiCameraHardwareManager实例" << std::endl;
}

MultiCameraHardwareManager::MultiCameraHardwareManager(const MultiCameraConfig& config) 
    : config_(config), initialized_(false) {
    std::cout << "创建MultiCameraHardwareManager实例，使用自定义配置" << std::endl;
}

MultiCameraHardwareManager::~MultiCameraHardwareManager() {
    shutdown();
    std::cout << "销毁MultiCameraHardwareManager实例" << std::endl;
}

bool MultiCameraHardwareManager::initialize() {
    if (initialized_) {
        std::cout << "MultiCameraHardwareManager已初始化" << std::endl;
        return true;
    }
    
    std::cout << "初始化MultiCameraHardwareManager..." << std::endl;
    
    try {
        if (!config_.validate()) {
            std::cerr << "MultiCameraHardwareManager配置无效" << std::endl;
            return false;
        }
        
        if (!initialize_cameras()) {
            std::cerr << "摄像头初始化失败" << std::endl;
            return false;
        }
        
        if (config_.enable_synchronization && !setup_synchronization()) {
            std::cerr << "同步设置失败" << std::endl;
            return false;
        }
        
        initialized_ = true;
        std::cout << "MultiCameraHardwareManager初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "MultiCameraHardwareManager初始化异常: " << e.what() << std::endl;
        return false;
    }
}

void MultiCameraHardwareManager::shutdown() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "关闭MultiCameraHardwareManager..." << std::endl;
    
    stop_all_cameras();
    cleanup_cameras();
    
    initialized_ = false;
    std::cout << "MultiCameraHardwareManager已关闭" << std::endl;
}

bool MultiCameraHardwareManager::start_all_cameras() {
    if (!initialized_) {
        std::cerr << "管理器未初始化" << std::endl;
        return false;
    }
    
    bool all_started = true;
    for (size_t i = 0; i < cameras_.size(); ++i) {
        if (camera_enabled_[i] && cameras_[i]) {
            if (!cameras_[i]->start_capture()) {
                std::cerr << "摄像头 " << i << " 启动失败" << std::endl;
                all_started = false;
            }
        }
    }
    
    return all_started;
}

void MultiCameraHardwareManager::stop_all_cameras() {
    for (auto& camera : cameras_) {
        if (camera) {
            camera->stop_capture();
        }
    }
}

bool MultiCameraHardwareManager::capture_synchronized_frames(std::vector<HardwareAcceleratedFrame>& frames) {
    if (!initialized_) {
        return false;
    }
    
    frames.clear();
    
    if (!capture_individual_frames(frames)) {
        return false;
    }
    
    if (config_.enable_synchronization) {
        return synchronize_frames(frames);
    }
    
    return true;
}

bool MultiCameraHardwareManager::capture_stereo_frames(HardwareAcceleratedFrame& left_frame, HardwareAcceleratedFrame& right_frame) {
    if (cameras_.size() < 2) {
        std::cerr << "立体模式需要至少2个摄像头" << std::endl;
        return false;
    }
    
    std::vector<HardwareAcceleratedFrame> frames;
    if (!capture_synchronized_frames(frames)) {
        return false;
    }
    
    if (frames.size() >= 2) {
        left_frame = frames[0];
        right_frame = frames[1];
        
        if (stereo_frame_callback_) {
            stereo_frame_callback_(left_frame, right_frame);
        }
        
        return true;
    }
    
    return false;
}

void MultiCameraHardwareManager::set_config(const MultiCameraConfig& config) {
    config_ = config;
}

std::vector<HardwareAccelerationStats> MultiCameraHardwareManager::get_all_camera_stats() const {
    std::vector<HardwareAccelerationStats> stats;
    for (const auto& camera : cameras_) {
        if (camera) {
            stats.push_back(camera->get_performance_stats());
        }
    }
    return stats;
}

HardwareAccelerationStats MultiCameraHardwareManager::get_combined_stats() const {
    auto all_stats = get_all_camera_stats();
    HardwareAccelerationStats combined;
    
    if (all_stats.empty()) {
        return combined;
    }
    
    // 合并统计信息
    for (const auto& stats : all_stats) {
        combined.total_frames_captured += stats.total_frames_captured;
        combined.total_frames_processed += stats.total_frames_processed;
        combined.dropped_frames += stats.dropped_frames;
    }
    
    // 计算平均值
    combined.avg_capture_time_ms = 0.0;
    combined.avg_processing_time_ms = 0.0;
    combined.current_fps = 0.0;
    
    for (const auto& stats : all_stats) {
        combined.avg_capture_time_ms += stats.avg_capture_time_ms;
        combined.avg_processing_time_ms += stats.avg_processing_time_ms;
        combined.current_fps += stats.current_fps;
    }
    
    combined.avg_capture_time_ms /= all_stats.size();
    combined.avg_processing_time_ms /= all_stats.size();
    combined.current_fps /= all_stats.size();
    
    return combined;
}

void MultiCameraHardwareManager::set_multi_frame_callback(MultiFrameCallback callback) {
    multi_frame_callback_ = callback;
}

void MultiCameraHardwareManager::set_stereo_frame_callback(StereoFrameCallback callback) {
    stereo_frame_callback_ = callback;
}

bool MultiCameraHardwareManager::initialize_cameras() {
    cameras_.clear();
    camera_devices_ = config_.camera_devices;
    camera_enabled_.resize(camera_devices_.size(), true);
    
    for (size_t i = 0; i < camera_devices_.size(); ++i) {
        auto camera = std::make_unique<HardwareAcceleratedCamera>(config_.camera_configs[i]);
        if (!camera->initialize()) {
            std::cerr << "摄像头 " << i << " 初始化失败" << std::endl;
            return false;
        }
        cameras_.push_back(std::move(camera));
    }
    
    return true;
}

bool MultiCameraHardwareManager::setup_synchronization() {
    last_frame_timestamps_.resize(cameras_.size(), 0);
    return true;
}

bool MultiCameraHardwareManager::capture_individual_frames(std::vector<HardwareAcceleratedFrame>& frames) {
    bool success = true;
    
    for (size_t i = 0; i < cameras_.size(); ++i) {
        if (camera_enabled_[i] && cameras_[i]) {
            HardwareAcceleratedFrame frame;
            if (cameras_[i]->capture_frame(frame)) {
                frames.push_back(frame);
            } else {
                success = false;
            }
        }
    }
    
    return success;
}

bool MultiCameraHardwareManager::synchronize_frames(std::vector<HardwareAcceleratedFrame>& frames) {
    if (frames.size() < 2) {
        return true; // 单个帧不需要同步
    }
    
    // 简化的同步实现：检查时间戳差异
    uint64_t base_timestamp = frames[0].capture_timestamp;
    
    for (size_t i = 1; i < frames.size(); ++i) {
        uint64_t diff = std::abs(static_cast<int64_t>(frames[i].capture_timestamp - base_timestamp));
        if (diff > static_cast<uint64_t>(config_.sync_tolerance_ms * 1000)) {
            std::cerr << "帧 " << i << " 同步超时，时差: " << diff / 1000 << "ms" << std::endl;
            return false;
        }
    }
    
    return true;
}

void MultiCameraHardwareManager::cleanup_cameras() {
    cameras_.clear();
    camera_devices_.clear();
    camera_enabled_.clear();
}

} // namespace vision
} // namespace bamboo_cut