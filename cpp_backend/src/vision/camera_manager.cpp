#include "bamboo_cut/vision/camera_manager.h"
#include "bamboo_cut/core/logger.h"
#include "bamboo_cut/core/system_utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <thread>
#include <chrono>
#include <sstream>
#include <map>
#include <functional>
#include <vector>
#include <cstdlib>

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
        // 检查环境变量中的摄像头类型配置
        const char* camera_type_env = std::getenv("CAMERA_TYPE");
        const char* camera_device_env = std::getenv("CAMERA_DEVICE");
        
        std::string camera_type = camera_type_env ? camera_type_env : "auto";
        std::string camera_device = camera_device_env ? camera_device_env : config_.device_id;
        
        LOG_INFO("摄像头配置 - 类型: {}, 设备: {}", camera_type, camera_device);
        
        // 从device_id中提取相机ID
        std::string device_id = camera_device;
        int camera_id = 0; // 默认相机ID
        
        // 如果device_id包含数字，提取相机ID
        if (device_id.find("video") != std::string::npos) {
            size_t pos = device_id.find_last_of("0123456789");
            if (pos != std::string::npos) {
                std::string id_str = device_id.substr(pos);
                camera_id = core::SystemUtils::safeStringToInt(id_str, 0);
            }
        }
        
        // 更新配置中的设备ID和CSI检测
        config_.device_id = camera_device;
        
        // CSI摄像头检测逻辑：基于环境变量或设备路径
        bool is_csi_camera = false;
        if (camera_type == "csi" || camera_type == "mipi") {
            is_csi_camera = true;
            LOG_INFO("检测到CSI摄像头 (环境变量)");
        } else if (camera_type == "auto") {
            // 自动检测：检查是否存在nvarguscamerasrc或IMX219驱动
            if (core::SystemUtils::commandExists("nvarguscamerasrc")) {
                is_csi_camera = true;
                LOG_INFO("检测到CSI摄像头 (nvarguscamerasrc可用)");
            } else if (core::SystemUtils::isModuleLoaded("imx219")) {
                is_csi_camera = true;
                LOG_INFO("检测到CSI摄像头 (IMX219驱动)");
            }
        }
        
        // 设置CSI标记以供GStreamer pipeline使用
        if (is_csi_camera) {
            config_.device_id = "csi:" + std::to_string(camera_id);
        }
        
        // 尝试多种方法初始化相机
        std::vector<int> camera_ids_to_try = {camera_id};
        if (camera_id == 0) {
            camera_ids_to_try = {0, 1}; // 尝试video0和video1
        }
        
        bool success = false;
        for (int id : camera_ids_to_try) {
            LOG_INFO("尝试初始化相机ID: {}", id);
            
            if (initializeCamera(id)) {
                LOG_INFO("相机 {} 初始化成功", id);
                success = true;
                break;
            } else {
                LOG_WARN("相机 {} 初始化失败，尝试下一个", id);
            }
        }
        
        if (!success) {
            LOG_ERROR("所有相机初始化尝试均失败");
            return false;
        }
        
        LOG_INFO("CameraManager初始化成功");
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
        capture_thread_ = std::make_unique<std::thread>(&CameraManager::captureLoop, this);
        
        LOG_INFO("相机捕获启动成功");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("启动相机捕获异常: {}", e.what());
        is_running_ = false;
        return false;
    }
}

bool CameraManager::stopCapture() {
    if (!is_running_) {
        return true;
    }
    
    LOG_INFO("停止相机捕获");
    
    is_running_ = false;
    
    if (capture_thread_ && capture_thread_->joinable()) {
        capture_thread_->join();
    }
    
    LOG_INFO("相机捕获已停止");
    return true;
}

void CameraManager::setFrameCallback(FrameCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    frame_callback_ = callback;
    LOG_INFO("设置帧回调函数");
}

CameraManager::CameraInfo CameraManager::getCameraInfo() const {
    CameraInfo info;
    info.device_path = config_.device_id;
    info.current_width = config_.width;
    info.current_height = config_.height;
    info.current_fps = static_cast<float>(config_.framerate);
    
    return info;
}

bool CameraManager::initializeCamera(int camera_id) {
    try {
        // 尝试多种初始化方法
        std::vector<std::function<std::unique_ptr<cv::VideoCapture>()>> init_methods;
        
        // 方法1：GStreamer pipeline
        init_methods.push_back([this, camera_id]() -> std::unique_ptr<cv::VideoCapture> {
            std::string pipeline = buildGStreamerPipeline(camera_id);
            LOG_INFO("尝试GStreamer pipeline: {}", pipeline);
            auto cap = std::make_unique<cv::VideoCapture>(pipeline, cv::CAP_GSTREAMER);
            if (cap->isOpened()) {
                return cap;
            }
            return nullptr;
        });
        
        // 方法2：直接V4L2访问
        init_methods.push_back([this, camera_id]() -> std::unique_ptr<cv::VideoCapture> {
            std::string device_path = "/dev/video" + std::to_string(camera_id);
            LOG_INFO("尝试V4L2设备: {}", device_path);
            auto cap = std::make_unique<cv::VideoCapture>(device_path, cv::CAP_V4L2);
            if (cap->isOpened()) {
                // 设置分辨率和帧率
                cap->set(cv::CAP_PROP_FRAME_WIDTH, config_.width);
                cap->set(cv::CAP_PROP_FRAME_HEIGHT, config_.height);
                cap->set(cv::CAP_PROP_FPS, config_.framerate);
                return cap;
            }
            return nullptr;
        });
        
        // 方法3：OpenCV默认后端
        init_methods.push_back([this, camera_id]() -> std::unique_ptr<cv::VideoCapture> {
            LOG_INFO("尝试OpenCV默认后端，相机ID: {}", camera_id);
            auto cap = std::make_unique<cv::VideoCapture>(camera_id);
            if (cap->isOpened()) {
                // 设置分辨率和帧率
                cap->set(cv::CAP_PROP_FRAME_WIDTH, config_.width);
                cap->set(cv::CAP_PROP_FRAME_HEIGHT, config_.height);
                cap->set(cv::CAP_PROP_FPS, config_.framerate);
                return cap;
            }
            return nullptr;
        });
        
        // 方法4：如果是CSI摄像头，尝试简化的nvarguscamerasrc
        if (config_.device_id.find("csi:") != std::string::npos) {
            init_methods.push_back([this, camera_id]() -> std::unique_ptr<cv::VideoCapture> {
                std::stringstream simple_pipeline;
                simple_pipeline << "nvarguscamerasrc sensor-id=" << camera_id
                               << " ! video/x-raw(memory:NVMM), width=" << config_.width
                               << ", height=" << config_.height
                               << ", framerate=" << config_.framerate << "/1"
                               << " ! nvvidconv ! video/x-raw, format=BGR ! appsink";
                
                std::string pipeline_str = simple_pipeline.str();
                LOG_INFO("尝试简化CSI pipeline: {}", pipeline_str);
                auto cap = std::make_unique<cv::VideoCapture>(pipeline_str, cv::CAP_GSTREAMER);
                if (cap->isOpened()) {
                    return cap;
                }
                return nullptr;
            });
        }
        
        // 尝试每种方法
        std::unique_ptr<cv::VideoCapture> successful_cap = nullptr;
        for (size_t i = 0; i < init_methods.size(); ++i) {
            try {
                LOG_INFO("尝试初始化方法 {}/{}", i + 1, init_methods.size());
                successful_cap = init_methods[i]();
                if (successful_cap && successful_cap->isOpened()) {
                    LOG_INFO("方法 {} 成功", i + 1);
                    break;
                }
            } catch (const std::exception& e) {
                LOG_WARN("方法 {} 失败: {}", i + 1, e.what());
            }
        }
        
        if (!successful_cap || !successful_cap->isOpened()) {
            LOG_ERROR("所有初始化方法均失败，相机 {}", camera_id);
            return false;
        }
        
        // 验证设置是否生效
        double actual_width = successful_cap->get(cv::CAP_PROP_FRAME_WIDTH);
        double actual_height = successful_cap->get(cv::CAP_PROP_FRAME_HEIGHT);
        double actual_fps = successful_cap->get(cv::CAP_PROP_FPS);
        
        LOG_INFO("相机 {} 初始化成功: {}x{} @ {}fps",
                camera_id, actual_width, actual_height, actual_fps);
        
        // 测试捕获一帧来验证摄像头工作正常
        cv::Mat test_frame;
        if (successful_cap->read(test_frame) && !test_frame.empty()) {
            LOG_INFO("相机 {} 测试帧捕获成功: {}x{}", camera_id, test_frame.cols, test_frame.rows);
        } else {
            LOG_WARN("相机 {} 无法捕获测试帧，但连接已建立", camera_id);
        }
        
        cameras_[camera_id] = std::move(successful_cap);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("相机 {} 初始化异常: {}", camera_id, e.what());
        return false;
    }
}

void CameraManager::captureLoop() {
    LOG_INFO("相机捕获循环开始");
    
    const auto frame_interval = std::chrono::microseconds(1000000 / config_.framerate);
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
            frame_info.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
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

std::string CameraManager::buildGStreamerPipeline(int camera_id) {
    std::stringstream pipeline;
    
    // 检查是否为 CSI 摄像头
    bool is_csi = config_.device_id.find("csi") != std::string::npos;
    
    // 检查环境变量中的配置
    const char* force_format = std::getenv("CAMERA_FORMAT");
    std::string output_format = force_format ? force_format : "BGR";
    
    // 检查GStreamer能力
    const char* gst_debug = std::getenv("GST_DEBUG");
    bool debug_mode = gst_debug && std::string(gst_debug) != "0";
    
    if (is_csi) {
        // CSI 摄像头优化pipeline - 多种fallback策略
        
        // 策略1：完整硬件加速pipeline（最佳性能）
        pipeline << "nvarguscamerasrc sensor-id=" << camera_id
                << " ! video/x-raw(memory:NVMM),width=" << config_.width
                << ",height=" << config_.height
                << ",framerate=" << config_.framerate << "/1"
                << ",format=NV12";
        
        // 添加可选的图像属性
        const char* exposure = std::getenv("CAMERA_EXPOSURE");
        const char* gain = std::getenv("CAMERA_GAIN");
        if (exposure) {
            pipeline << " exposuretimerange=\"" << exposure << " " << exposure << "\"";
        }
        if (gain) {
            pipeline << " gainrange=\"" << gain << " " << gain << "\"";
        }
        
        // 视频转换和格式处理
        if (output_format == "BGR") {
            pipeline << " ! nvvidconv flip-method=0"
                    << " ! video/x-raw,format=BGRx"
                    << " ! videoconvert"
                    << " ! video/x-raw,format=BGR";
        } else {
            pipeline << " ! nvvidconv flip-method=0"
                    << " ! video/x-raw,format=RGBA"
                    << " ! videoconvert"
                    << " ! video/x-raw,format=RGBA";
        }
        
        pipeline << " ! appsink name=appsink max-buffers=2 drop=true sync=false";
        
        if (debug_mode) {
            LOG_INFO("CSI摄像头pipeline (硬件加速): {}", pipeline.str());
        }
        
    } else {
        // USB/V4L2 摄像头pipeline
        pipeline << "v4l2src device=/dev/video" << camera_id;
        
        // 添加缓冲区设置以提高稳定性
        pipeline << " ! video/x-raw,width=" << config_.width
                << ",height=" << config_.height
                << ",framerate=" << config_.framerate << "/1";
        
        // 尝试自动检测最佳格式
        const char* v4l2_format = std::getenv("V4L2_FORMAT");
        if (v4l2_format) {
            pipeline << ",format=" << v4l2_format;
        }
        
        pipeline << " ! videoconvert"
                << " ! video/x-raw,format=" << output_format
                << " ! appsink name=appsink max-buffers=2 drop=true sync=false";
        
        if (debug_mode) {
            LOG_INFO("V4L2摄像头pipeline: {}", pipeline.str());
        }
    }
    
    return pipeline.str();
}

CameraManager::PerformanceStats CameraManager::getPerformanceStats() const {
    PerformanceStats stats;
    stats.total_frames = 0;
    stats.dropped_frames = 0;
    stats.fps = static_cast<double>(config_.framerate);
    stats.avg_processing_time_ms = 0.0;
    return stats;
}

} // namespace vision
} // namespace bamboo_cut