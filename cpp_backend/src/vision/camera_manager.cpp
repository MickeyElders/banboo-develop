#include "bamboo_cut/vision/camera_manager.h"
#include "bamboo_cut/core/logger.h"
#include <opencv2/videoio.hpp>
#include <thread>
#include <chrono>

namespace bamboo_cut {
namespace vision {

CameraManager::CameraManager(const CameraConfig& config) 
    : config_(config), is_running_(false), frame_callback_(nullptr) {
    LOG_INFO("创建CameraManager实例");
}

CameraManager::~CameraManager() {
    stopCapture();
    LOG_INFO("销毁CameraManager实例");
}

bool CameraManager::initialize() {
    LOG_INFO("初始化CameraManager");
    
    try {
        // 检查相机配置
        if (config_.camera_ids.empty()) {
            LOG_ERROR("相机ID列表为空");
            return false;
        }
        
        // 初始化相机
        for (int camera_id : config_.camera_ids) {
            if (!initializeCamera(camera_id)) {
                LOG_ERROR("相机 {} 初始化失败", camera_id);
                return false;
            }
        }
        
        LOG_INFO("CameraManager初始化成功，共 {} 个相机", config_.camera_ids.size());
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("CameraManager初始化异常: {}", e.what());
        return false;
    }
}

bool CameraManager::startCapture() {
    if (is_running_) {
        LOG_WARN("相机捕获已在运行中");
        return true;
    }
    
    LOG_INFO("启动相机捕获");
    
    try {
        is_running_ = true;
        
        // 启动捕获线程
        capture_thread_ = std::thread(&CameraManager::captureLoop, this);
        
        LOG_INFO("相机捕获启动成功");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("启动相机捕获异常: {}", e.what());
        is_running_ = false;
        return false;
    }
}

void CameraManager::stopCapture() {
    if (!is_running_) {
        return;
    }
    
    LOG_INFO("停止相机捕获");
    
    is_running_ = false;
    
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    
    LOG_INFO("相机捕获已停止");
}

void CameraManager::setFrameCallback(FrameCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    frame_callback_ = callback;
    LOG_INFO("设置帧回调函数");
}

CameraInfo CameraManager::getCameraInfo() const {
    CameraInfo info;
    info.camera_count = config_.camera_ids.size();
    info.resolution = config_.resolution;
    info.fps = config_.fps;
    info.is_running = is_running_;
    
    for (int camera_id : config_.camera_ids) {
        info.camera_ids.push_back(camera_id);
    }
    
    return info;
}

bool CameraManager::initializeCamera(int camera_id) {
    try {
        // 创建相机捕获对象
        auto cap = std::make_unique<cv::VideoCapture>(camera_id);
        
        if (!cap->isOpened()) {
            LOG_ERROR("无法打开相机 {}", camera_id);
            return false;
        }
        
        // 设置相机参数
        cap->set(cv::CAP_PROP_FRAME_WIDTH, config_.resolution.width);
        cap->set(cv::CAP_PROP_FRAME_HEIGHT, config_.resolution.height);
        cap->set(cv::CAP_PROP_FPS, config_.fps);
        cap->set(cv::CAP_PROP_BUFFERSIZE, 1);
        
        // 验证设置是否生效
        double actual_width = cap->get(cv::CAP_PROP_FRAME_WIDTH);
        double actual_height = cap->get(cv::CAP_PROP_FRAME_HEIGHT);
        double actual_fps = cap->get(cv::CAP_PROP_FPS);
        
        LOG_INFO("相机 {} 初始化成功: {}x{} @ {}fps", 
                camera_id, actual_width, actual_height, actual_fps);
        
        cameras_[camera_id] = std::move(cap);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("相机 {} 初始化异常: {}", camera_id, e.what());
        return false;
    }
}

void CameraManager::captureLoop() {
    LOG_INFO("相机捕获循环开始");
    
    const auto frame_interval = std::chrono::microseconds(1000000 / config_.fps);
    auto last_frame_time = std::chrono::steady_clock::now();
    
    while (is_running_) {
        auto start_time = std::chrono::steady_clock::now();
        
        // 捕获所有相机的帧
        std::vector<cv::Mat> frames;
        std::vector<int> camera_ids;
        
        for (const auto& [camera_id, cap] : cameras_) {
            cv::Mat frame;
            if (cap->read(frame) && !frame.empty()) {
                frames.push_back(frame.clone());
                camera_ids.push_back(camera_id);
            }
        }
        
        // 如果有帧数据，创建FrameInfo并调用回调
        if (!frames.empty()) {
            FrameInfo frame_info;
            frame_info.timestamp = std::chrono::steady_clock::now();
            frame_info.frames = frames;
            frame_info.camera_ids = camera_ids;
            frame_info.frame_count = frames.size();
            
            // 调用回调函数
            {
                std::lock_guard<std::mutex> lock(callback_mutex_);
                if (frame_callback_) {
                    try {
                        frame_callback_(frame_info);
                    } catch (const std::exception& e) {
                        LOG_ERROR("帧回调函数异常: {}", e.what());
                    }
                }
            }
        }
        
        // 控制帧率
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = end_time - start_time;
        
        if (elapsed < frame_interval) {
            std::this_thread::sleep_for(frame_interval - elapsed);
        }
        
        last_frame_time = end_time;
    }
    
    LOG_INFO("相机捕获循环结束");
}

bool CameraManager::isRunning() const {
    return is_running_;
}

int CameraManager::getCameraCount() const {
    return static_cast<int>(cameras_.size());
}

cv::Mat CameraManager::getLatestFrame(int camera_id) const {
    auto it = cameras_.find(camera_id);
    if (it != cameras_.end()) {
        cv::Mat frame;
        if (it->second->read(frame)) {
            return frame;
        }
    }
    return cv::Mat();
}

std::vector<cv::Mat> CameraManager::getLatestFrames() const {
    std::vector<cv::Mat> frames;
    frames.reserve(cameras_.size());
    
    for (const auto& [camera_id, cap] : cameras_) {
        cv::Mat frame;
        if (cap->read(frame)) {
            frames.push_back(frame);
        }
    }
    
    return frames;
}

} // namespace vision
} // namespace bamboo_cut 