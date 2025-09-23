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
#include <bamboo_cut/communication/unix_socket_server.h>

using namespace bamboo_cut;

// 全局变量用于信号处理
std::atomic<bool> g_shutdown_requested{false};

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
        
        // 初始化相机管理器（非关键模块，失败时继续运行）
        camera_system_available_ = initializeCameraSystem();
        if (!camera_system_available_) {
            LOG_WARN("⚠️ 相机系统初始化失败，系统将在无摄像头模式下运行");
        }
        
        // 初始化立体视觉系统（非关键模块，失败时继续运行）
        stereo_vision_available_ = initializeStereoVisionSystem();
        if (!stereo_vision_available_) {
            LOG_WARN("⚠️ 立体视觉系统初始化失败，系统将在2D模式下运行");
        }
        
        // 初始化通信系统（关键模块，失败时系统无法运行）
        if (!initializeCommunicationSystem()) {
            LOG_ERROR("❌ 通信系统初始化失败，系统无法运行");
            return false;
        }
        
        // 初始化UNIX Socket服务器（关键模块，用于前端通信）
        if (!initializeUnixSocketServer()) {
            LOG_ERROR("❌ UNIX Socket服务器初始化失败，前端将无法连接");
            return false;
        }
        
        // 输出系统状态摘要
        LOG_INFO("🎯 系统初始化完成，模块状态:");
        LOG_INFO("   📹 摄像头系统: {}", camera_system_available_ ? "✅ 可用" : "❌ 不可用");
        LOG_INFO("   🔍 视觉检测: {}", vision_system_available_ ? "✅ 可用" : "❌ 不可用");
        LOG_INFO("   👁️ 立体视觉: {}", stereo_vision_available_ ? "✅ 可用" : "❌ 不可用");
        LOG_INFO("   🔗 Modbus通信: ✅ 可用");
        LOG_INFO("   📡 前端通信: {}", unix_socket_available_ ? "✅ 可用" : "❌ 不可用");
        
        if (!camera_system_available_ && !vision_system_available_) {
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
        
        // 主循环
        auto last_stats_time = std::chrono::steady_clock::now();
        const auto stats_interval = std::chrono::seconds(30);
        
        LOG_INFO("🔄 开始执行主循环...");
        int loop_count = 0;
        
        while (!g_shutdown_requested) {
            loop_count++;
            if (loop_count <= 5 || loop_count % 100 == 0) {
                LOG_INFO("🔄 主循环迭代 #{}, 退出标志: {}", loop_count, g_shutdown_requested.load() ? "是" : "否");
            }
            
            // 处理视觉检测
            processVision();
            
            // 定期输出性能统计
            auto now = std::chrono::steady_clock::now();
            if (now - last_stats_time >= stats_interval) {
                printPerformanceStats();
                last_stats_time = now;
            }
            
            // 短暂休眠避免CPU占用过高
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
        
        // 停止UNIX Socket服务器
        if (unix_socket_server_) {
            unix_socket_server_->stop();
        }
        
        // 停止Modbus服务器
        if (modbus_server_) {
            modbus_server_->stop();
        }
        
        // 停止摄像头
        if (camera_manager_) {
            camera_manager_->stopCapture();
        }
        
        LOG_INFO("系统已关闭");
    }

private:
    // 核心组件
    std::unique_ptr<vision::BambooDetector> detector_;
    std::unique_ptr<vision::OptimizedDetector> optimized_detector_;
    std::unique_ptr<vision::CameraManager> camera_manager_;
    std::unique_ptr<vision::StereoVision> stereo_vision_;
    std::unique_ptr<communication::ModbusServer> modbus_server_;
    std::unique_ptr<communication::UnixSocketServer> unix_socket_server_;
    
    // 模块可用性状态
    bool vision_system_available_ = false;
    bool camera_system_available_ = false;
    bool stereo_vision_available_ = false;
    bool communication_system_available_ = false;
    bool unix_socket_available_ = false;
    
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
    
    bool initializeStereoVisionSystem() {
        LOG_INFO("初始化立体视觉系统...");
        
        // 配置双摄像头参数
        vision::CameraSyncConfig stereo_config;
        stereo_config.left_device = "/dev/video0";   // 左摄像头
        stereo_config.right_device = "/dev/video2";  // 右摄像头
        stereo_config.width = 1280;                  // 降低分辨率以提高处理速度
        stereo_config.height = 720;
        stereo_config.fps = 30;
        stereo_config.hardware_sync = false;         // 软件同步
        stereo_config.sync_tolerance_ms = 10;        // 10ms同步容差
        
        stereo_vision_ = std::make_unique<vision::StereoVision>(stereo_config);
        
        if (!stereo_vision_->initialize()) {
            LOG_ERROR("立体视觉系统初始化失败");
            return false;
        }
        
        // 尝试加载标定文件
        std::string calibration_file = "/opt/bamboo-cut/config/stereo_calibration.xml";
        if (stereo_vision_->load_calibration(calibration_file)) {
            LOG_INFO("加载立体标定文件成功: {}", calibration_file);
            auto params = stereo_vision_->get_calibration_params();
            LOG_INFO("基线距离: {:.2f}mm", params.baseline);
        } else {
            LOG_WARN("未找到标定文件，需要进行标定: {}", calibration_file);
            LOG_INFO("使用单目模式，深度信息不可用");
        }
        
        LOG_INFO("立体视觉系统初始化完成");
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
    
    bool initializeUnixSocketServer() {
        LOG_INFO("初始化UNIX Socket服务器...");
        
        // 创建UNIX Socket服务器，只需要socket路径参数
        unix_socket_server_ = std::make_unique<communication::UnixSocketServer>("/tmp/bamboo_socket");
        
        // 设置消息回调函数
        unix_socket_server_->set_message_callback([this](const communication::CommunicationMessage& msg, int client_fd) {
            LOG_INFO("收到前端消息，类型: {}", static_cast<int>(msg.type));
            handleFrontendMessage(msg, client_fd);
        });
        
        // 设置客户端连接回调函数
        unix_socket_server_->set_client_connected_callback([](int client_fd, const std::string& client_name) {
            LOG_INFO("前端已连接: {} (fd={})", client_name, client_fd);
        });
        
        // 设置客户端断开回调函数
        unix_socket_server_->set_client_disconnected_callback([](int client_fd) {
            LOG_INFO("前端已断开: fd={}", client_fd);
        });
        
        unix_socket_available_ = true;
        LOG_INFO("UNIX Socket服务器初始化完成");
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
        
        // 详细调试摄像头系统状态
        LOG_INFO("🔍 检查摄像头系统状态:");
        LOG_INFO("   camera_manager_存在: {}", camera_manager_ ? "是" : "否");
        LOG_INFO("   camera_system_available_: {}", camera_system_available_ ? "是" : "否");
        
        // 启动摄像头捕获（可选服务）
        if (camera_manager_ && camera_system_available_) {
            LOG_INFO("📹 开始启动摄像头服务...");
            if (camera_manager_->startCapture()) {
                LOG_INFO("✅ 摄像头服务启动成功");
            } else {
                LOG_WARN("⚠️ 摄像头服务启动失败，切换到模拟模式");
                camera_system_available_ = false;
            }
        } else {
            if (!camera_manager_) {
                LOG_WARN("⚠️ 摄像头管理器未初始化，跳过启动");
            } else if (!camera_system_available_) {
                LOG_WARN("⚠️ 摄像头系统标记为不可用，跳过启动");
            } else {
                LOG_WARN("⚠️ 摄像头系统不可用，跳过启动");
            }
        }
        
        // 启动UNIX Socket服务器（必需服务，用于前端通信）
        LOG_INFO("🔌 检查UNIX Socket服务器状态: {}", unix_socket_server_ ? "存在" : "不存在");
        
        if (unix_socket_server_ && unix_socket_available_) {
            LOG_INFO("🔌 开始启动UNIX Socket服务器...");
            bool unix_start_result = unix_socket_server_->start();
            LOG_INFO("🔌 UNIX Socket启动结果: {}", unix_start_result ? "成功" : "失败");
            
            if (!unix_start_result) {
                LOG_ERROR("❌ UNIX Socket服务器启动失败，前端无法连接");
                unix_socket_available_ = false;
            } else {
                LOG_INFO("✅ UNIX Socket服务器启动成功");
            }
        } else {
            if (!unix_socket_server_) {
                LOG_WARN("⚠️ UNIX Socket服务器对象为空，跳过启动");
            } else if (!unix_socket_available_) {
                LOG_WARN("⚠️ UNIX Socket服务器标记为不可用，跳过启动");
            }
        }
        
        // 设置系统状态
        if (modbus_server_) {
            if (camera_system_available_ && vision_system_available_) {
                modbus_server_->set_system_status(communication::SystemStatus::RUNNING);
                LOG_INFO("✅ 系统状态：完全运行模式");
            } else {
                modbus_server_->set_system_status(communication::SystemStatus::MAINTENANCE);
                LOG_WARN("⚠️ 系统状态：有限运行模式（部分功能不可用）");
            }
        }
        
        LOG_INFO("🎯 服务启动完成 - 可用服务数: {}/5",
                (communication_system_available_ ? 1 : 0) +
                (unix_socket_available_ ? 1 : 0) +
                (camera_system_available_ ? 1 : 0) +
                (vision_system_available_ ? 1 : 0) +
                (stereo_vision_available_ ? 1 : 0));
    }
    
    void processVision() {
        // 检查是否有可用的视觉系统
        if (!vision_system_available_ && !camera_system_available_ && !stereo_vision_available_) {
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
        
        // 立体视觉处理
        if (stereo_vision_available_ && stereo_vision_ && stereo_vision_->is_initialized()) {
            if (processStereovision()) {
                return; // 立体视觉处理成功，直接返回
            }
        }
        
        // 传统2D视觉处理备选方案
        if (vision_system_available_ && camera_system_available_) {
            process2DVision();
        }
    }
    
    bool processStereovision() {
        // 捕获立体帧
        vision::StereoFrame stereo_frame;
        if (!stereo_vision_->capture_stereo_frame(stereo_frame)) {
            LOG_DEBUG("无法捕获立体帧");
            return false;
        }
        
        if (!stereo_frame.valid) {
            LOG_DEBUG("立体帧无效");
            return false;
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
        }
        return false;
    }
    
    void process2DVision() {
        if (!detector_ || !detector_->is_initialized()) {
            LOG_DEBUG("传统检测器不可用");
            return;
        }
        
        // 从摄像头获取当前帧
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (current_frame_.frames.empty()) {
            LOG_DEBUG("没有可用的摄像头帧");
            return;
        }
        
        auto result = detector_->detect(current_frame_.frames[0]);
        if (result.success && !result.points.empty()) {
            auto best_point = result.points[0];
            
            // 确定使用的刀片
            communication::BladeNumber blade = (best_point.x < current_frame_.frames[0].cols / 2) ?
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
            
            LOG_DEBUG("✅ 2D检测: X={:.1f}px ({:.1f}mm), 刀片={}, 耗时: {:.2f}ms",
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
        if (!unix_socket_server_ || !modbus_server_) {
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
        unix_socket_server_->send_message(client_fd, response);
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
            } else {
                LOG_WARN("未知的前端指令: {}", command);
            }
            
            // 发送确认响应
            communication::CommunicationMessage response;
            response.type = communication::MessageType::PLC_COMMAND_ACK;
            nlohmann::json response_data;
            response_data["result"] = "ok";
            response_data["command"] = command;
            
            std::string json_str = response_data.dump();
            strncpy(response.data, json_str.c_str(), sizeof(response.data) - 1);
            response.data[sizeof(response.data) - 1] = '\0';
            response.data_length = json_str.length();
            
            unix_socket_server_->send_message(client_fd, response);
            
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
        
        // 打印Unix Socket性能
        if (unix_socket_server_ && unix_socket_available_) {
            try {
                auto unix_stats = unix_socket_server_->get_statistics();
                LOG_INFO("前端通信性能: 连接数={}, 发送消息={}, 接收消息={}",
                        unix_stats.active_clients,
                        unix_stats.total_messages_sent,
                        unix_stats.total_messages_received);
            } catch (const std::exception& e) {
                LOG_DEBUG("获取Unix Socket统计信息失败: {}", e.what());
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