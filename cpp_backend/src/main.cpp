#include <iostream>
#include <memory>
#include <signal.h>
#include <thread>
#include <chrono>
#include <nlohmann/json.hpp>

// SystemD支持 - 可选依赖
#ifdef ENABLE_SYSTEMD
#include <systemd/sd-daemon.h>
#else
// 如果没有systemd，提供空的实现
static inline int sd_notify(int unset_environment, const char *state) { return 0; }
static inline int sd_watchdog_enabled(int unset_environment, uint64_t *usec) { return 0; }
#endif

#include <bamboo_cut/config.h>
#include <bamboo_cut/core/logger.h>
#include <bamboo_cut/vision/detector.h>
#include <bamboo_cut/vision/camera_manager.h>
#include <bamboo_cut/vision/stereo_vision.h>
#include <bamboo_cut/vision/optimized_detector.h>
#include <bamboo_cut/communication/modbus_server.h>
#include <bamboo_cut/communication/tcp_socket_server.h>

using namespace bamboo_cut;

// 全局变量用于信号处理
std::atomic<bool> g_shutdown_requested{false};
std::chrono::steady_clock::time_point g_shutdown_start_time;

// SystemD watchdog心跳线程
void watchdog_thread() {
#ifdef ENABLE_SYSTEMD
    uint64_t watchdog_usec = 0;
    if (sd_watchdog_enabled(0, &watchdog_usec) > 0) {
        auto interval = std::chrono::microseconds(watchdog_usec / 2); // 发送间隔为超时时间的一半
        while (!g_shutdown_requested) {
            sd_notify(0, "WATCHDOG=1");
            std::this_thread::sleep_for(interval);
        }
    }
#else
    // 没有systemd时，简单的空循环防止线程退出
    while (!g_shutdown_requested) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
#endif
}

// 信号处理函数
void signalHandler(int signal) {
    LOG_INFO("接收到信号 {}, 开始关闭系统...", signal);
    g_shutdown_requested = true;
    g_shutdown_start_time = std::chrono::steady_clock::now();
    
    // 立即通知systemd正在停止
    sd_notify(0, "STOPPING=1");
    
    // 设置4秒超时，避免无限等待（留1秒给systemd）
    alarm(4);
}

// 检查是否需要强制退出
bool should_force_exit() {
    if (!g_shutdown_requested) return false;
    auto elapsed = std::chrono::steady_clock::now() - g_shutdown_start_time;
    return elapsed > std::chrono::seconds(3); // 3秒后强制退出
}

class BambooCutApplication {
public:
    BambooCutApplication() = default;
    ~BambooCutApplication() = default;

    bool initialize() {
        LOG_INFO("=== 智能切竹机后端系统启动 ===");
        LOG_INFO("版本: {}", BAMBOO_CUT_VERSION);
        LOG_INFO("目标架构: {}", TARGET_ARCH);
        
        // 初始化日志系统
        core::Logger::getInstance().init("/var/log/bamboo-cut/backend.log");
        
        // 初始化视觉检测器（非关键模块，失败时继续运行）
        vision_system_available_ = initializeVisionSystem();
        if (!vision_system_available_) {
            LOG_WARN("⚠️ 视觉系统初始化失败，系统将在无视觉检测模式下运行");
        }
        
        // 完全禁用单摄像头系统，只使用立体视觉系统避免设备冲突
        LOG_INFO("🚫 单摄像头系统已禁用，使用立体视觉系统代替");
        camera_system_available_ = false;
        
        // 初始化立体视觉系统（关键模块，用于视觉检测和视频流输出）
        stereo_vision_available_ = initializeStereoVisionSystemWithStreaming();
        if (!stereo_vision_available_) {
            LOG_WARN("⚠️ 立体视觉系统初始化失败，前端将无法获取视频画面");
        }
        
        // 初始化通信系统（关键模块，失败时系统无法运行）
        if (!initializeCommunicationSystem()) {
            LOG_ERROR("❌ 通信系统初始化失败，系统无法运行");
            return false;
        }
        
        // 初始化TCP Socket服务器（关键模块，用于前端通信）
        if (!initializeTcpSocketServer()) {
            LOG_ERROR("❌ TCP Socket服务器初始化失败，前端将无法连接");
            return false;
        }
        
        // 输出系统状态摘要
        LOG_INFO("🎯 系统初始化完成，模块状态:");
        LOG_INFO("   🔍 视觉检测: {}", vision_system_available_ ? "✅ 可用" : "❌ 不可用");
        LOG_INFO("   👁️ 立体视觉(含流): {}", stereo_vision_available_ ? "✅ 可用" : "❌ 不可用");
        LOG_INFO("   🔗 Modbus通信: ✅ 可用");
        LOG_INFO("   📡 前端通信: {}", tcp_server_available_ ? "✅ 可用" : "❌ 不可用");
        
        if (!stereo_vision_available_ && !vision_system_available_) {
            LOG_WARN("⚠️ 系统运行在模拟模式：无摄像头和视觉检测");
            LOG_WARN("⚠️ 可以接收PLC指令但无法进行实际检测");
        }
        
        return true;
    }
    
    void run() {
        LOG_INFO("启动主循环");
        
        // 通知systemd服务已准备就绪
        sd_notify(0, "READY=1");
        
        // 启动SystemD watchdog心跳线程
        std::thread watchdog_th(watchdog_thread);
        
        // 启动所有服务
        LOG_INFO("🔄 准备启动服务...");
        startServices();
        LOG_INFO("🔄 服务启动完成，进入主循环");
        
        // 检查全局退出标志
        LOG_INFO("🔍 检查退出标志: {}", g_shutdown_requested.load() ? "已设置" : "未设置");
        
        // 主循环 - 针对30fps视频流优化
        auto last_stats_time = std::chrono::steady_clock::now();
        auto last_frame_time = std::chrono::steady_clock::now();
        const auto stats_interval = std::chrono::seconds(30);
        const auto target_frame_interval = std::chrono::milliseconds(33);  // 30fps = 33.33ms
        
        LOG_INFO("🔄 开始执行主循环...");
        int loop_count = 0;
        int frame_count = 0;
        
        while (!g_shutdown_requested) {
            loop_count++;
            auto current_time = std::chrono::steady_clock::now();
            
            if (loop_count <= 5 || loop_count % 300 == 0) {  // 减少日志频率
                LOG_INFO("🔄 主循环迭代 #{}, 帧数: {}, 退出标志: {}",
                        loop_count, frame_count, g_shutdown_requested.load() ? "是" : "否");
            }
            
            // 检查是否到达下一帧时间
            if (current_time - last_frame_time >= target_frame_interval) {
                // 处理视觉检测和视频流
                processVision();
                last_frame_time = current_time;
                frame_count++;
                
                // 每10秒输出一次帧率统计
                if (frame_count % 300 == 0) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_stats_time);
                    if (elapsed.count() > 0) {
                        double actual_fps = 300.0 / elapsed.count();
                        LOG_INFO("📹 实际帧率: {:.1f} fps (目标: 30fps)", actual_fps);
                        last_stats_time = current_time;
                    }
                }
            }
            
            // 定期输出性能统计
            auto now = std::chrono::steady_clock::now();
            if (now - last_stats_time >= stats_interval) {
                printPerformanceStats();
                last_stats_time = now;
            }
            
            // 检查强制退出条件
            if (should_force_exit()) {
                LOG_WARN("强制退出主循环：超时3秒");
                break;
            }
            
            // 动态休眠时间，避免占用过多CPU
            auto next_frame_time = last_frame_time + target_frame_interval;
            auto sleep_time = next_frame_time - std::chrono::steady_clock::now();
            
            if (sleep_time > std::chrono::milliseconds(0) && sleep_time < std::chrono::milliseconds(20)) {
                std::this_thread::sleep_for(sleep_time);
            } else {
                // 最小休眠避免CPU占用过高
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        LOG_INFO("主循环结束，退出原因: g_shutdown_requested = {}", g_shutdown_requested.load());
        
        // 等待心跳线程结束
        if (watchdog_th.joinable()) {
            watchdog_th.join();
        }
    }
    
    void shutdown() {
        LOG_INFO("开始关闭系统...");
        
        // 通知systemd服务正在停止
        sd_notify(0, "STOPPING=1");
        
        // 设置关闭标志，停止主循环
        g_shutdown_requested = true;
        
        // 优雅关闭立体视觉系统（最重要）
        if (stereo_vision_) {
            LOG_INFO("关闭立体视觉系统...");
            stereo_vision_->shutdown();
            stereo_vision_.reset();
        }
        
        // 停止TCP Socket服务器
        if (tcp_socket_server_) {
            LOG_INFO("关闭TCP Socket服务器...");
            tcp_socket_server_->stop();
            tcp_socket_server_.reset();
        }
        
        // 停止Modbus服务器
        if (modbus_server_) {
            LOG_INFO("关闭Modbus服务器...");
            modbus_server_->stop();
            modbus_server_.reset();
        }
        
        // 停止摄像头（如果有）
        if (camera_manager_) {
            LOG_INFO("关闭摄像头管理器...");
            camera_manager_->stopCapture();
            camera_manager_.reset();
        }
        
        // 等待一小段时间确保所有线程退出
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        LOG_INFO("系统关闭完成");
    }

private:
    // 核心组件
    std::unique_ptr<vision::BambooDetector> detector_;
    std::unique_ptr<vision::OptimizedDetector> optimized_detector_;
    std::unique_ptr<vision::CameraManager> camera_manager_;
    std::unique_ptr<vision::StereoVision> stereo_vision_;
    std::unique_ptr<communication::ModbusServer> modbus_server_;
    std::unique_ptr<communication::TcpSocketServer> tcp_socket_server_;
    
    // 模块可用性状态
    bool vision_system_available_ = false;
    bool camera_system_available_ = false;
    bool stereo_vision_available_ = false;
    bool communication_system_available_ = false;
    bool tcp_server_available_ = false;
    
    // 当前帧数据
    vision::FrameInfo current_frame_;
    std::mutex frame_mutex_;
    
    bool initializeVisionSystem() {
        LOG_INFO("初始化视觉检测系统...");
        
        // 创建传统检测器配置
        vision::BambooDetector::Config detector_config;
        detector_config.model_path = std::string(MODELS_PATH) + "/bamboo_detection.onnx";
        detector_config.engine_path = std::string(MODELS_PATH) + "/bamboo_detection.trt";
        
#ifdef ENABLE_TENSORRT
        detector_config.use_tensorrt = true;
        LOG_INFO("启用TensorRT加速");
#else
        detector_config.use_tensorrt = false;
        LOG_INFO("使用OpenCV DNN推理");
#endif
        
        detector_ = std::make_unique<vision::BambooDetector>(detector_config);
        
        if (!detector_->initialize()) {
            LOG_ERROR("传统检测器初始化失败");
            return false;
        }
        
        // 创建优化检测器配置
        vision::OptimizedDetectorConfig optimized_config;
        optimized_config.tensorrt_config.model_path = std::string(MODELS_PATH) + "/bamboo_detection.onnx";
        optimized_config.tensorrt_config.enable_fp16 = true;
        optimized_config.confidence_threshold = 0.5f;
        optimized_config.nms_threshold = 0.4f;
        optimized_config.enable_fp16 = true;
        optimized_config.enable_batch_processing = true;
        optimized_config.batch_size = 4;
        
        // 配置 SAHI 切片推理
        optimized_config.enable_sahi_slicing = true;
        optimized_config.sahi_config.slice_height = 512;
        optimized_config.sahi_config.slice_width = 512;
        optimized_config.sahi_config.overlap_ratio = 0.2f;
        optimized_config.sahi_config.slice_strategy = vision::SliceStrategy::ADAPTIVE;
        optimized_config.sahi_config.merge_strategy = vision::MergeStrategy::HYBRID;
        optimized_config.sahi_config.enable_parallel_processing = true;
        
        // 配置硬件加速摄像头
        optimized_config.enable_hardware_acceleration = true;
        optimized_config.camera_config.acceleration_type = vision::HardwareAccelerationType::MIXED_HW;
        optimized_config.camera_config.interface_type = vision::CameraInterfaceType::MIPI_CSI;
        optimized_config.camera_config.width = 1920;
        optimized_config.camera_config.height = 1080;
        optimized_config.camera_config.fps = 30;
        optimized_config.camera_config.pixel_format = "NV12";
        optimized_config.camera_config.enable_gpu_memory_mapping = true;
        optimized_config.camera_config.enable_zero_copy = true;
        optimized_config.camera_config.enable_hardware_isp = true;
        optimized_config.camera_config.enable_async_capture = true;
        
        // 创建优化检测器
        optimized_detector_ = std::make_unique<vision::OptimizedDetector>(optimized_config);
        
        if (!optimized_detector_->initialize()) {
            LOG_WARN("优化检测器初始化失败，将使用传统检测器");
        } else {
            LOG_INFO("优化检测器初始化成功");
        }
        
        LOG_INFO("视觉检测系统初始化完成");
        LOG_INFO("模型信息: {}", detector_->getModelInfo());
        
        return true;
    }
    
    bool initializeCameraSystem() {
        LOG_INFO("初始化摄像头系统...");
        
        vision::CameraConfig camera_config;
        camera_config.device_id = "/dev/video0";
        camera_config.width = 1920;
        camera_config.height = 1080;
        camera_config.framerate = 30;
        
        // 启用GStreamer流输出
        camera_config.enable_stream_output = true;
        camera_config.stream_host = "127.0.0.1";
        camera_config.stream_port = 5000;
        camera_config.stream_format = "H264";
        camera_config.stream_bitrate = 2000000;
        
#ifdef TARGET_ARCH_AARCH64
        camera_config.pipeline = DEFAULT_CAMERA_PIPELINE;
        camera_config.use_hardware_acceleration = true;
        LOG_INFO("使用Jetson硬件加速pipeline");
#else
        camera_config.pipeline = DEFAULT_CAMERA_PIPELINE;
        camera_config.use_hardware_acceleration = false;
        LOG_INFO("使用通用摄像头pipeline");
#endif
        
        camera_manager_ = std::make_unique<vision::CameraManager>(camera_config);
        
        if (!camera_manager_->initialize()) {
            LOG_ERROR("摄像头初始化失败");
            return false;
        }
        
        // 设置帧回调
        camera_manager_->setFrameCallback([this](const vision::FrameInfo& frame) {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            current_frame_ = frame;
        });
        
        LOG_INFO("摄像头系统初始化完成");
        auto camera_info = camera_manager_->getCameraInfo();
        LOG_INFO("摄像头信息: {} @ {}x{}", camera_info.card_name, 
                camera_info.current_width, camera_info.current_height);
        
        return true;
    }
    
    bool initializeStereoVisionSystemWithStreaming() {
        LOG_INFO("初始化带流输出的立体视觉系统...");
        
        // 配置双摄像头参数
        vision::CameraSyncConfig stereo_config;
        stereo_config.left_device = "/dev/video0";   // 左摄像头
        stereo_config.right_device = "/dev/video1";  // 右摄像头
        stereo_config.width = 640;                   // 匹配前端期望分辨率
        stereo_config.height = 480;                  // 匹配前端期望分辨率
        stereo_config.fps = 30;                      // 匹配前端期望帧率
        stereo_config.hardware_sync = false;         // 软件同步
        stereo_config.sync_tolerance_ms = 10;        // 10ms同步容差
        
        stereo_vision_ = std::make_unique<vision::StereoVision>(stereo_config);
        
        if (!stereo_vision_->initialize()) {
            LOG_ERROR("立体视觉系统初始化失败");
            return false;
        }
        
        // 尝试加载标定文件（优先检查项目本地文件）
        std::vector<std::string> calibration_paths = {
            "./config/stereo_calibration.xml",                    // 项目本地配置
            "../config/stereo_calibration.xml",                  // 相对路径配置
            "/opt/bamboo-cut/config/stereo_calibration.xml"      // 系统配置路径
        };
        
        bool calibration_loaded = false;
        std::string used_calibration_file;
        
        for (const auto& calibration_file : calibration_paths) {
            if (stereo_vision_->load_calibration(calibration_file)) {
                LOG_INFO("加载立体标定文件成功: {}", calibration_file);
                auto params = stereo_vision_->get_calibration_params();
                LOG_INFO("基线距离: {:.2f}mm", params.baseline);
                calibration_loaded = true;
                used_calibration_file = calibration_file;
                break;
            }
        }
        
        if (!calibration_loaded) {
            LOG_WARN("未找到标定文件，尝试的路径:");
            for (const auto& path : calibration_paths) {
                LOG_WARN("  - {}", path);
            }
            LOG_INFO("使用单目模式，深度信息不可用");
            LOG_INFO("💡 提示: 将项目根目录的 config/stereo_calibration.xml 复制到系统配置目录");
        }
        
        // 初始化并启用GStreamer视频流输出
        if (stereo_vision_->initialize_video_stream()) {
            LOG_INFO("✅ GStreamer视频流初始化成功");
            // 确保视频流被启用
            bool stream_enabled = stereo_vision_->enable_video_stream(true);
            stereo_vision_->set_display_mode(vision::DisplayMode::SIDE_BY_SIDE);  // 默认并排显示
            LOG_INFO("✅ 立体视觉流输出已启用: {}", stream_enabled ? "成功" : "失败");
        } else {
            LOG_WARN("⚠️ GStreamer视频流初始化失败");
            // 即使初始化失败，也要尝试启用流以便错误诊断
            stereo_vision_->enable_video_stream(true);
        }
        
        LOG_INFO("立体视觉系统（含流输出）初始化完成");
        LOG_INFO("📺 视频流信息:");
        LOG_INFO("   格式: H264, UDP端口: 5000");
        LOG_INFO("   分辨率: 640x480 @ 30fps");
        LOG_INFO("   支持显示模式: 并排显示 | 融合显示");
        
        return true;
    }
    
    bool initializeCommunicationSystem() {
        LOG_INFO("初始化通信系统...");
        
        communication::ModbusConfig modbus_config;
        modbus_config.ip_address = "0.0.0.0";  // 监听所有接口
        modbus_config.port = 502;               // 标准Modbus TCP端口
        modbus_config.max_connections = 10;
        modbus_config.response_timeout_ms = 1000;
        modbus_config.heartbeat_interval_ms = 100;
        
        // 超时设置
        modbus_config.feed_detection_timeout_s = 15;
        modbus_config.clamp_timeout_s = 60;
        modbus_config.cut_execution_timeout_s = 120;
        modbus_config.emergency_response_timeout_ms = 100;
        
        modbus_server_ = std::make_unique<communication::ModbusServer>(modbus_config);
        
        // 设置回调函数
        modbus_server_->set_connection_callback([](bool connected, const std::string& client_ip) {
            if (connected) {
                LOG_INFO("PLC已连接: {}", client_ip);
            } else {
                LOG_WARN("PLC已断开: {}", client_ip);
            }
        });
        
        modbus_server_->set_command_callback([this](communication::PLCCommand command) {
            LOG_INFO("收到PLC指令: {}", static_cast<int>(command));
            handlePLCCommand(command);
        });
        
        modbus_server_->set_emergency_stop_callback([this]() {
            LOG_ERROR("触发紧急停止！");
            handleEmergencyStop();
        });
        
        modbus_server_->set_timeout_callback([](const std::string& timeout_type) {
            LOG_WARN("操作超时: {}", timeout_type);
        });
        
        LOG_INFO("通信系统初始化完成");
        return true;
    }
    
    bool initializeTcpSocketServer() {
        LOG_INFO("初始化TCP Socket服务器...");
        
        // 创建TCP Socket服务器，监听127.0.0.1:8888
        tcp_socket_server_ = std::make_unique<communication::TcpSocketServer>("127.0.0.1", 8888);
        
        // 设置消息回调函数
        tcp_socket_server_->set_message_callback([this](const communication::CommunicationMessage& msg, int client_fd) {
            LOG_INFO("收到前端消息，类型: {}", static_cast<int>(msg.type));
            handleFrontendMessage(msg, client_fd);
        });
        
        // 设置客户端连接回调函数
        tcp_socket_server_->set_client_connected_callback([](int client_fd, const std::string& client_info) {
            LOG_INFO("前端已连接: {} (fd={})", client_info, client_fd);
        });
        
        // 设置客户端断开回调函数
        tcp_socket_server_->set_client_disconnected_callback([](int client_fd) {
            LOG_INFO("前端已断开: fd={}", client_fd);
        });
        
        tcp_server_available_ = true;
        LOG_INFO("TCP Socket服务器初始化完成");
        return true;
    }
    
    void startServices() {
        LOG_INFO("🚀 启动所有可用服务...");
        
        // 启动Modbus服务器（必需服务）
        LOG_INFO("📡 检查Modbus服务器状态: {}", modbus_server_ ? "存在" : "不存在");
        
        if (modbus_server_) {
            LOG_INFO("📡 开始启动Modbus服务器...");
            bool modbus_start_result = modbus_server_->start();
            LOG_INFO("📡 Modbus启动结果: {}", modbus_start_result ? "成功" : "失败");
            
            if (!modbus_start_result) {
                LOG_ERROR("❌ Modbus服务器启动失败");
                return;
            } else {
                LOG_INFO("✅ Modbus服务器启动成功");
                communication_system_available_ = true;
            }
        } else {
            LOG_ERROR("❌ Modbus服务器对象为空");
            return;
        }
        
        LOG_INFO("📋 Modbus启动后的状态检查...");
        LOG_INFO("   communication_system_available_: {}", communication_system_available_ ? "是" : "否");
        
        // 检查立体视觉系统状态（现在是唯一的视频源）
        LOG_INFO("🔍 检查立体视觉系统状态:");
        LOG_INFO("   stereo_vision_存在: {}", stereo_vision_ ? "是" : "否");
        LOG_INFO("   stereo_vision_available_: {}", stereo_vision_available_ ? "是" : "否");
        
        // 立体视觉系统已经在初始化时启动，这里只需要确认状态
        if (stereo_vision_ && stereo_vision_available_) {
            LOG_INFO("✅ 立体视觉系统运行正常");
            LOG_INFO("🎥 立体视觉流输出已启用");
            LOG_INFO("📡 视频流URL: udp://127.0.0.1:5000");
            LOG_INFO("📺 视频格式: H264, 分辨率: 640x480, 帧率: 30fps");
            LOG_INFO("💡 支持显示模式:");
            LOG_INFO("   - 并排显示：显示左右摄像头画面");
            LOG_INFO("   - 融合显示：显示处理后的单一画面");
            LOG_INFO("   - 前端可通过按钮切换显示模式");
        } else {
            LOG_WARN("⚠️ 立体视觉系统不可用");
            LOG_WARN("💡 提示：前端将看不到视频画面");
            LOG_WARN("💡 原因：双摄像头设备(/dev/video0, /dev/video1)不可用");
        }
        
        // 启动TCP Socket服务器（必需服务，用于前端通信）
        LOG_INFO("🔌 检查TCP Socket服务器状态: {}", tcp_socket_server_ ? "存在" : "不存在");
        
        if (tcp_socket_server_ && tcp_server_available_) {
            LOG_INFO("🔌 开始启动TCP Socket服务器...");
            bool tcp_start_result = tcp_socket_server_->start();
            LOG_INFO("🔌 TCP Socket启动结果: {}", tcp_start_result ? "成功" : "失败");
            
            if (!tcp_start_result) {
                LOG_ERROR("❌ TCP Socket服务器启动失败，前端无法连接");
                tcp_server_available_ = false;
            } else {
                LOG_INFO("✅ TCP Socket服务器启动成功");
            }
        } else {
            if (!tcp_socket_server_) {
                LOG_WARN("⚠️ TCP Socket服务器对象为空，跳过启动");
            } else if (!tcp_server_available_) {
                LOG_WARN("⚠️ TCP Socket服务器标记为不可用，跳过启动");
            }
        }
        
        // 设置系统状态（基于立体视觉系统）
        if (modbus_server_) {
            if (stereo_vision_available_ && vision_system_available_) {
                modbus_server_->set_system_status(communication::SystemStatus::RUNNING);
                LOG_INFO("✅ 系统状态：完全运行模式（立体视觉+AI检测）");
            } else if (stereo_vision_available_) {
                modbus_server_->set_system_status(communication::SystemStatus::RUNNING);
                LOG_INFO("✅ 系统状态：立体视觉模式（无AI检测）");
            } else {
                modbus_server_->set_system_status(communication::SystemStatus::MAINTENANCE);
                LOG_WARN("⚠️ 系统状态：有限运行模式（无视觉系统）");
            }
        }
        
        LOG_INFO("🎯 服务启动完成 - 可用服务数: {}/4",
                (communication_system_available_ ? 1 : 0) +
                (tcp_server_available_ ? 1 : 0) +
                (vision_system_available_ ? 1 : 0) +
                (stereo_vision_available_ ? 1 : 0));
    }
    
    void processVision() {
        // 检查是否有可用的视觉系统（只依赖立体视觉系统）
        if (!vision_system_available_ && !stereo_vision_available_) {
            // 在模拟模式下，定期发送模拟数据用于测试
            static auto last_simulation_time = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_simulation_time).count() >= 5) {
                LOG_DEBUG("🎭 模拟模式：发送测试坐标数据");
                if (modbus_server_) {
                    communication::CoordinateData sim_data(1000, communication::BladeNumber::BLADE_1, communication::CutQuality::ABNORMAL);
                    modbus_server_->set_coordinate_data(sim_data);
                }
                last_simulation_time = now;
            }
            return;
        }
        
        // 优先使用立体视觉处理（主要视觉系统）
        if (stereo_vision_available_ && stereo_vision_ && stereo_vision_->is_initialized()) {
            if (processStereovision()) {
                return; // 立体视觉处理成功，直接返回
            }
        }
        
        // 2D视觉检测作为补充（如果立体视觉无法处理）
        if (vision_system_available_ && detector_) {
            // 在没有单摄像头系统的情况下，使用立体视觉的左摄像头进行2D检测
            processStereovisionAs2D();
        }
    }
    
    bool processStereovision() {
        // 捕获立体帧
        vision::StereoFrame stereo_frame;
        static int capture_failures = 0;
        static int successful_captures = 0;
        
        if (!stereo_vision_->capture_stereo_frame(stereo_frame)) {
            capture_failures++;
            if (capture_failures % 100 == 0) {  // 每100次失败输出一次日志
                LOG_WARN("立体帧捕获失败次数: {}, 成功次数: {}", capture_failures, successful_captures);
            }
            
            // 即使捕获失败，也推送一个测试帧保持流活跃
            cv::Mat test_frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
            stereo_vision_->push_frame_to_stream(test_frame);
            return false;
        }
        
        if (!stereo_frame.valid) {
            LOG_DEBUG("立体帧无效，推送测试帧");
            // 推送测试帧
            cv::Mat test_frame(480, 640, CV_8UC3, cv::Scalar(64, 64, 64));
            stereo_vision_->push_frame_to_stream(test_frame);
            return false;
        }
        
        successful_captures++;
        if (successful_captures % 30 == 0) {  // 每30次成功输出一次日志
            LOG_INFO("立体帧捕获成功次数: {}, 失败次数: {}", successful_captures, capture_failures);
        }
        
        // 创建显示帧并推送到视频流
        cv::Mat display_frame = stereo_vision_->create_display_frame(
            stereo_frame.left_image,
            stereo_frame.right_image
        );
        
        if (!display_frame.empty()) {
            stereo_vision_->push_frame_to_stream(display_frame);
        } else {
            LOG_WARN("显示帧为空，推送测试帧");
            cv::Mat test_frame(480, 640, CV_8UC3, cv::Scalar(192, 192, 192));
            stereo_vision_->push_frame_to_stream(test_frame);
        }
        
        // 3D模式 - 使用深度信息过滤检测点
        if (stereo_vision_->is_calibrated() && !stereo_frame.disparity.empty()) {
            auto valid_points = stereo_vision_->detect_bamboo_with_depth(
                stereo_frame.left_image,
                stereo_frame.disparity,
                200.0,   // 最小深度 200mm
                2000.0   // 最大深度 2000mm
            );
            
            if (!valid_points.empty()) {
                // 计算3D坐标
                auto points_3d = stereo_vision_->pixels_to_3d(valid_points, stereo_frame.disparity);
                
                if (!points_3d.empty()) {
                    auto best_point_3d = points_3d[0];
                    
                    // 确定使用的刀片
                    communication::BladeNumber blade = (best_point_3d.x < 0) ?
                        communication::BladeNumber::BLADE_1 : communication::BladeNumber::BLADE_2;
                    
                    // 创建坐标数据
                    communication::CoordinateData coord_data(
                        static_cast<int32_t>(best_point_3d.x * 10),
                        blade,
                        communication::CutQuality::NORMAL
                    );
                    
                    // 更新坐标到Modbus服务器
                    if (modbus_server_) {
                        modbus_server_->set_coordinate_data(coord_data);
                    }
                    
                    LOG_DEBUG("✅ 3D检测: X={:.1f}mm, Y={:.1f}mm, Z={:.1f}mm, 刀片={}",
                             best_point_3d.x, best_point_3d.y, best_point_3d.z, static_cast<int>(blade));
                    return true;
                }
            }
        } else {
            // 无标定模式 - 直接使用左摄像头帧进行基本的视觉处理
            // 即使没有深度信息，也要确保视频流正常工作
            static int frame_count = 0;
            frame_count++;
            if (frame_count % 300 == 0) {  // 每10秒输出一次日志（30fps * 10s = 300帧）
                LOG_INFO("立体视觉流: 无标定模式运行，已处理 {} 帧", frame_count);
            }
        }
        return false;
    }
    
    void processStereovisionAs2D() {
        if (!detector_ || !detector_->is_initialized()) {
            LOG_DEBUG("传统检测器不可用");
            return;
        }
        
        // 从立体视觉系统获取左摄像头帧进行2D检测
        vision::StereoFrame stereo_frame;
        if (!stereo_vision_->capture_stereo_frame(stereo_frame) || !stereo_frame.valid) {
            LOG_DEBUG("无法从立体视觉系统获取帧");
            return;
        }
        
        auto result = detector_->detect(stereo_frame.left_image);
        if (result.success && !result.points.empty()) {
            auto best_point = result.points[0];
            
            // 确定使用的刀片
            communication::BladeNumber blade = (best_point.x < stereo_frame.left_image.cols / 2) ?
                communication::BladeNumber::BLADE_1 : communication::BladeNumber::BLADE_2;
            
            // 创建坐标数据
            float pixel_to_mm = 0.5f; // 像素到mm的比例
            communication::CoordinateData coord_data(
                static_cast<int32_t>(best_point.x * pixel_to_mm * 10),
                blade,
                communication::CutQuality::NORMAL
            );
            
            // 更新坐标到Modbus服务器
            if (modbus_server_) {
                modbus_server_->set_coordinate_data(coord_data);
            }
            
            LOG_DEBUG("✅ 2D检测(立体左摄): X={:.1f}px ({:.1f}mm), 刀片={}, 耗时: {:.2f}ms",
                     best_point.x, best_point.x * pixel_to_mm,
                     static_cast<int>(blade), result.processing_time_ms);
        }
    }
    
    void handlePLCCommand(communication::PLCCommand command) {
        switch (command) {
            case communication::PLCCommand::FEED_DETECTION:
                LOG_INFO("执行进料检测指令");
                modbus_server_->reset_feed_detection_timer();
                // TODO: 启动进料检测逻辑
                break;
                
            case communication::PLCCommand::CUT_PREPARE:
                LOG_INFO("执行切割准备指令");
                modbus_server_->reset_clamp_timer();
                // TODO: 实现夹持固定逻辑
                break;
                
            case communication::PLCCommand::CUT_COMPLETE:
                LOG_INFO("执行切割完成指令");
                modbus_server_->reset_cut_execution_timer();
                // 清除当前坐标数据
                modbus_server_->clear_coordinate_data();
                break;
                
            case communication::PLCCommand::START_FEEDING:
                LOG_INFO("执行启动送料指令");
                modbus_server_->reset_feed_detection_timer();
                // TODO: 实现送料控制逻辑
                break;
                
            case communication::PLCCommand::PAUSE:
                LOG_INFO("执行暂停指令");
                modbus_server_->set_system_status(communication::SystemStatus::PAUSED);
                break;
                
            case communication::PLCCommand::EMERGENCY_STOP:
                LOG_WARN("执行紧急停止指令");
                modbus_server_->trigger_emergency_stop();
                break;
                
            case communication::PLCCommand::RESUME:
                LOG_INFO("执行恢复运行指令");
                modbus_server_->set_system_status(communication::SystemStatus::RUNNING);
                break;
                
            default:
                LOG_WARN("未知PLC指令: {}", static_cast<int>(command));
                break;
        }
    }
    
    void handleEmergencyStop() {
        LOG_ERROR("系统紧急停止！");
        
        // 停止摄像头
        if (camera_manager_) {
            camera_manager_->stopCapture();
        }
        
        // 设置系统健康状态为严重错误
        modbus_server_->set_system_health(communication::SystemHealth::CRITICAL_ERROR);
        
        // TODO: 实现硬件急停逻辑
        // - 停止所有电机
        // - 关闭气动系统
        // - 激活安全制动
    }
    
    void handleFrontendMessage(const communication::CommunicationMessage& msg, int client_fd) {
        LOG_INFO("处理前端消息，类型: {}", static_cast<int>(msg.type));
        
        switch (msg.type) {
            case communication::MessageType::STATUS_REQUEST:
                // 发送系统状态给前端
                sendSystemStatusToFrontend(client_fd);
                break;
                
            case communication::MessageType::PLC_COMMAND:
                // 处理前端发送的PLC命令
                handleFrontendCommand(msg, client_fd);
                break;
                
            default:
                LOG_WARN("未知的前端消息类型: {}", static_cast<int>(msg.type));
                break;
        }
    }
    
    void sendSystemStatusToFrontend(int client_fd) {
        if (!tcp_socket_server_ || !modbus_server_) {
            return;
        }
        
        // 构造系统状态消息
        communication::CommunicationMessage response;
        response.type = communication::MessageType::STATUS_RESPONSE;
        
        // 获取Modbus服务器状态
        auto modbus_stats = modbus_server_->get_statistics();
        
        // 填充状态数据（使用JSON格式）
        nlohmann::json status_data;
        status_data["plc_connected"] = modbus_server_->is_connected();
        status_data["system_status"] = static_cast<int>(modbus_server_->get_system_status());
        status_data["camera_available"] = camera_system_available_;
        status_data["vision_available"] = vision_system_available_;
        status_data["stereo_available"] = stereo_vision_available_;
        status_data["total_requests"] = modbus_stats.total_requests;
        status_data["total_errors"] = modbus_stats.total_errors;
        status_data["heartbeat_timeouts"] = modbus_stats.heartbeat_timeouts;
        
        // 获取坐标数据
        auto coord_data = modbus_server_->get_coordinate_data();
        status_data["coordinate_x"] = coord_data.x_coordinate;
        status_data["blade_number"] = static_cast<int>(coord_data.blade_number);
        status_data["cut_quality"] = static_cast<int>(coord_data.quality);
        
        // 转换JSON为字符串，并复制到response.data
        std::string json_str = status_data.dump();
        strncpy(response.data, json_str.c_str(), sizeof(response.data) - 1);
        response.data[sizeof(response.data) - 1] = '\0';
        response.data_length = json_str.length();
        
        // 发送响应
        tcp_socket_server_->send_message(client_fd, response);
        LOG_DEBUG("已发送系统状态到前端: fd={}", client_fd);
    }
    
    void handleFrontendCommand(const communication::CommunicationMessage& msg, int client_fd) {
        try {
            nlohmann::json command_data = nlohmann::json::parse(msg.data);
            std::string command = command_data["command"];
            
            if (command == "start_detection") {
                LOG_INFO("前端请求启动检测");
                if (modbus_server_) {
                    modbus_server_->set_system_status(communication::SystemStatus::RUNNING);
                }
            } else if (command == "stop_detection") {
                LOG_INFO("前端请求停止检测");
                if (modbus_server_) {
                    modbus_server_->set_system_status(communication::SystemStatus::PAUSED);
                }
            } else if (command == "emergency_stop") {
                LOG_WARN("前端触发紧急停止");
                handleEmergencyStop();
            } else if (command == "set_display_mode") {
                // 处理显示模式切换
                if (stereo_vision_ && stereo_vision_available_) {
                    std::string mode = command_data.value("mode", "side_by_side");
                    if (mode == "side_by_side") {
                        stereo_vision_->set_display_mode(vision::DisplayMode::SIDE_BY_SIDE);
                        LOG_INFO("前端切换显示模式: 并排显示");
                    } else if (mode == "fused") {
                        stereo_vision_->set_display_mode(vision::DisplayMode::FUSED);
                        LOG_INFO("前端切换显示模式: 融合显示");
                    } else {
                        LOG_WARN("未知的显示模式: {}", mode);
                    }
                } else {
                    LOG_WARN("立体视觉系统不可用，无法切换显示模式");
                }
            } else if (command == "toggle_video_stream") {
                // 处理视频流开关
                if (stereo_vision_ && stereo_vision_available_) {
                    bool enable = command_data.value("enable", true);
                    stereo_vision_->enable_video_stream(enable);
                    LOG_INFO("前端切换视频流: {}", enable ? "启用" : "禁用");
                } else {
                    LOG_WARN("立体视觉系统不可用，无法切换视频流");
                }
            } else {
                LOG_WARN("未知的前端指令: {}", command);
            }
            
            // 发送确认响应
            communication::CommunicationMessage response;
            response.type = communication::MessageType::PLC_RESPONSE;
            nlohmann::json response_data;
            response_data["result"] = "ok";
            response_data["command"] = command;
            
            std::string json_str = response_data.dump();
            strncpy(response.data, json_str.c_str(), sizeof(response.data) - 1);
            response.data[sizeof(response.data) - 1] = '\0';
            response.data_length = json_str.length();
            
            tcp_socket_server_->send_message(client_fd, response);
            
        } catch (const std::exception& e) {
            LOG_ERROR("处理前端指令失败: {}", e.what());
        }
    }
    
    void printPerformanceStats() {
        // 安全地打印检测器性能
        if (detector_ && vision_system_available_) {
            try {
                auto detector_stats = detector_->getPerformanceStats();
                LOG_INFO("检测器性能: {:.1f} FPS, 平均推理时间: {:.2f}ms",
                        detector_stats.fps, detector_stats.avg_inference_time_ms);
            } catch (const std::exception& e) {
                LOG_DEBUG("获取检测器统计信息失败: {}", e.what());
            }
        }
        
        // 安全地打印摄像头性能
        if (camera_manager_ && camera_system_available_) {
            try {
                auto camera_stats = camera_manager_->getPerformanceStats();
                LOG_INFO("摄像头性能: {:.1f} FPS, 丢帧: {}",
                        camera_stats.fps, camera_stats.dropped_frames);
            } catch (const std::exception& e) {
                LOG_DEBUG("获取摄像头统计信息失败: {}", e.what());
            }
        }
        
        // 安全地打印Modbus性能
        if (modbus_server_) {
            try {
                auto modbus_stats = modbus_server_->get_statistics();
                LOG_INFO("Modbus性能: 连接={}, 请求={}, 错误={}, 心跳超时={}",
                        modbus_server_->is_connected() ? 1 : 0,
                        modbus_stats.total_requests,
                        modbus_stats.total_errors,
                        modbus_stats.heartbeat_timeouts);
            } catch (const std::exception& e) {
                LOG_DEBUG("获取Modbus统计信息失败: {}", e.what());
            }
        }
        
        // 打印TCP Socket性能
        if (tcp_socket_server_ && tcp_server_available_) {
            try {
                auto tcp_stats = tcp_socket_server_->get_statistics();
                LOG_INFO("前端通信性能: 连接数={}, 发送消息={}, 接收消息={}",
                        tcp_stats.active_clients,
                        tcp_stats.total_messages_sent,
                        tcp_stats.total_messages_received);
            } catch (const std::exception& e) {
                LOG_DEBUG("获取TCP Socket统计信息失败: {}", e.what());
            }
        }
    }
};

int main(int argc, char* argv[]) {
    // 设置信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    try {
        BambooCutApplication app;
        
        if (!app.initialize()) {
            std::cerr << "应用程序初始化失败" << std::endl;
            return -1;
        }
        
        app.run();
        app.shutdown();
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "应用程序异常: " << e.what() << std::endl;
        return -1;
    }
}