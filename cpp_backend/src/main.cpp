#include <iostream>
#include <memory>
#include <signal.h>
#include <thread>
#include <chrono>

#include <bamboo_cut/config.h>
#include <bamboo_cut/core/logger.h>
#include <bamboo_cut/vision/detector.h>
#include <bamboo_cut/vision/camera_manager.h>
#include <bamboo_cut/vision/stereo_vision.h>
#include <bamboo_cut/vision/optimized_detector.h>
#include <bamboo_cut/communication/modbus_server.h>

using namespace bamboo_cut;

// 全局变量用于信号处理
std::atomic<bool> g_shutdown_requested{false};

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
        
        // 初始化视觉检测器
        if (!initializeVisionSystem()) {
            LOG_ERROR("视觉系统初始化失败");
            return false;
        }
        
        // 初始化相机管理器 (修复: 添加缺失的初始化调用)
        if (!initializeCameraSystem()) {
            LOG_ERROR("相机系统初始化失败");
            return false;
        }
        
        // 初始化立体视觉系统
        if (!initializeStereoVisionSystem()) {
            LOG_ERROR("立体视觉系统初始化失败");
            return false;
        }
        
        // 初始化通信系统
        if (!initializeCommunicationSystem()) {
            LOG_ERROR("通信系统初始化失败");
            return false;
        }
        
        LOG_INFO("系统初始化完成");
        return true;
    }
    
    void run() {
        LOG_INFO("启动主循环");
        
        // 启动所有服务
        startServices();
        
        // 主循环
        auto last_stats_time = std::chrono::steady_clock::now();
        const auto stats_interval = std::chrono::seconds(30);
        
        while (!g_shutdown_requested) {
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
        
        LOG_INFO("主循环结束");
    }
    
    void shutdown() {
        LOG_INFO("开始关闭系统...");
        
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
    
    void startServices() {
        LOG_INFO("启动所有服务...");
        
        // 启动Modbus服务器
        if (!modbus_server_->start()) {
            LOG_ERROR("Modbus服务器启动失败");
            return;
        }
        
        // 启动摄像头捕获
        if (!camera_manager_->startCapture()) {
            LOG_ERROR("摄像头启动失败");
            return;
        }
        
        // 设置系统状态为运行
        modbus_server_->set_system_status(communication::SystemStatus::RUNNING);
        
        LOG_INFO("所有服务启动完成");
    }
    
    void processVision() {
        if (!stereo_vision_->is_initialized()) {
            LOG_WARN("立体视觉系统未初始化，跳过处理");
            return;
        }
        
        // 捕获立体帧
        vision::StereoFrame stereo_frame;
        if (!stereo_vision_->capture_stereo_frame(stereo_frame)) {
            LOG_DEBUG("无法捕获立体帧");
            return;
        }
        
        if (!stereo_frame.valid) {
            LOG_DEBUG("立体帧无效");
            return;
        }
        
        // 优先使用立体视觉系统进行检测
        if (stereo_vision_->is_calibrated() && !stereo_frame.disparity.empty()) {
            // 3D模式 - 使用深度信息过滤检测点
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
                    
                    // 确定使用的刀片（根据X坐标位置，统一使用图像中心作为分界线）
                    communication::BladeNumber blade = (best_point_3d.x < 0) ? 
                        communication::BladeNumber::BLADE_1 : communication::BladeNumber::BLADE_2;
                    
                    // 创建坐标数据 (转换为0.1mm精度)
                    communication::CoordinateData coord_data(
                        static_cast<int32_t>(best_point_3d.x * 10), 
                        blade, 
                        communication::CutQuality::NORMAL
                    );
                    
                    // 更新坐标到Modbus服务器
                    modbus_server_->set_coordinate_data(coord_data);
                    
                    LOG_DEBUG("检测到3D切点: X={:.1f}mm, Y={:.1f}mm, Z={:.1f}mm, 刀片={}, 置信度={:.2f}", 
                             best_point_3d.x, best_point_3d.y, best_point_3d.z, 
                             static_cast<int>(blade), best_point_3d.confidence);
                }
            }
        } else {
            // 2D模式 - 使用传统检测器作为备选
            if (detector_ && detector_->is_initialized()) {
                auto result = detector_->detect(stereo_frame.left_image);
                
                if (result.success && !result.points.empty()) {
                    auto best_point = result.points[0];
                    
                    // 确定使用的刀片（根据X坐标位置，统一使用图像中心作为分界线）
                    communication::BladeNumber blade = (best_point.x < stereo_frame.left_image.cols / 2) ? 
                        communication::BladeNumber::BLADE_1 : communication::BladeNumber::BLADE_2;
                    
                    // 创建坐标数据 (转换为0.1mm精度，假设像素到mm的比例)
                    float pixel_to_mm = 0.5f; // 假设每像素0.5mm，需要根据实际标定
                    communication::CoordinateData coord_data(
                        static_cast<int32_t>(best_point.x * pixel_to_mm * 10), 
                        blade, 
                        communication::CutQuality::NORMAL
                    );
                    
                    // 更新坐标到Modbus服务器
                    modbus_server_->set_coordinate_data(coord_data);
                    
                    LOG_DEBUG("检测到2D切点: X={:.1f}px ({:.1f}mm), 刀片={}, 处理时间: {:.2f}ms", 
                             best_point.x, best_point.x * pixel_to_mm, 
                             static_cast<int>(blade), result.processing_time_ms);
                }
            } else {
                LOG_WARN("传统检测器未初始化，无法进行2D检测");
            }
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
    
    void printPerformanceStats() {
        // 打印检测器性能
        auto detector_stats = detector_->getPerformanceStats();
        LOG_INFO("检测器性能: {:.1f} FPS, 平均推理时间: {:.2f}ms", 
                detector_stats.fps, detector_stats.avg_inference_time_ms);
        
        // 打印摄像头性能
        auto camera_stats = camera_manager_->getPerformanceStats();
        LOG_INFO("摄像头性能: {:.1f} FPS, 丢帧: {}", 
                camera_stats.fps, camera_stats.dropped_frames);
        
        // 打印Modbus性能
        auto modbus_stats = modbus_server_->get_statistics();
        LOG_INFO("Modbus性能: 连接={}, 请求={}, 错误={}, 心跳超时={}", 
                modbus_server_->is_connected() ? 1 : 0,
                modbus_stats.total_requests, 
                modbus_stats.total_errors,
                modbus_stats.heartbeat_timeouts);
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