#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "bamboo_cut/vision/optimized_detector.h"
#include "bamboo_cut/communication/modbus_server.h"
#include "bamboo_cut/vision/stereo_vision.h"

using json = nlohmann::json;

// 全局变量
static std::unique_ptr<bamboo_cut::vision::OptimizedDetector> g_detector;
static std::unique_ptr<bamboo_cut::communication::ModbusServer> g_modbus_server;
static std::unique_ptr<bamboo_cut::vision::StereoVision> g_stereo_vision;
static bool g_initialized = false;
static bool g_camera_capturing = false;

// 内存管理
extern "C" {
    void* allocate_memory(size_t size) {
        return malloc(size);
    }
    
    void free_memory(void* ptr) {
        free(ptr);
    }
}

// 初始化推理服务
extern "C" int32_t initialize_inference_service() {
    try {
        if (g_initialized) {
            return 0; // 已经初始化
        }
        
        // 创建优化检测器配置
        bamboo_cut::vision::OptimizedDetectorConfig config;
        config.model_path = "models/bamboo_detector.onnx";
        config.enable_tensorrt = true;
        config.enable_fp16 = true;
        config.enable_batch_processing = true;
        config.batch_size = 4;
        config.enable_sahi_slicing = true;
        config.enable_hardware_acceleration = true;
        
        // 创建检测器
        g_detector = std::make_unique<bamboo_cut::vision::OptimizedDetector>(config);
        
        // 创建Modbus服务器
        bamboo_cut::communication::ModbusConfig modbus_config;
        modbus_config.server_ip = "0.0.0.0";
        modbus_config.server_port = 502;
        modbus_config.heartbeat_interval_ms = 100;
        
        g_modbus_server = std::make_unique<bamboo_cut::communication::ModbusServer>(modbus_config);
        
        // 创建立体视觉系统
        bamboo_cut::vision::CameraSyncConfig sync_config;
        sync_config.left_device = "/dev/video0";
        sync_config.right_device = "/dev/video1";
        sync_config.width = 1920;
        sync_config.height = 1080;
        sync_config.fps = 30;
        
        g_stereo_vision = std::make_unique<bamboo_cut::vision::StereoVision>(sync_config);
        
        g_initialized = true;
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

// 检测竹子
extern "C" int32_t detect_bamboo(
    uint8_t* image_data,
    int32_t width,
    int32_t height,
    uint8_t** result_data,
    int32_t* result_size
) {
    try {
        if (!g_initialized || !g_detector) {
            return -1;
        }
        
        // 将图像数据转换为OpenCV Mat
        cv::Mat image(height, width, CV_8UC3, image_data);
        
        // 执行检测
        auto result = g_detector->detect(image);
        
        // 转换为JSON格式
        json result_json;
        result_json["success"] = true;
        result_json["detections"] = json::array();
        
        for (const auto& detection : result.detections) {
            json detection_json;
            detection_json["x"] = detection.x;
            detection_json["y"] = detection.y;
            detection_json["width"] = detection.width;
            detection_json["height"] = detection.height;
            detection_json["confidence"] = detection.confidence;
            detection_json["label"] = detection.label;
            result_json["detections"].push_back(detection_json);
        }
        
        // 序列化为字符串
        std::string json_string = result_json.dump();
        
        // 分配内存并复制结果
        *result_size = json_string.size();
        *result_data = static_cast<uint8_t*>(allocate_memory(*result_size));
        std::memcpy(*result_data, json_string.c_str(), *result_size);
        
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

// 获取系统状态
extern "C" int32_t get_system_status(
    uint8_t* status_data,
    int32_t* status_size
) {
    try {
        if (!g_initialized) {
            return -1;
        }
        
        json status_json;
        status_json["system_status"] = 1; // 运行状态
        status_json["fps"] = 30.0;
        status_json["cpu_usage"] = 0.3;
        status_json["memory_usage"] = 0.5;
        status_json["gpu_usage"] = 0.4;
        status_json["plc_connected"] = g_modbus_server ? true : false;
        status_json["camera_connected"] = g_camera_capturing;
        status_json["heartbeat_count"] = 0;
        status_json["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        status_json["emergency_stop"] = false;
        status_json["error_message"] = "";
        status_json["warning_message"] = "";
        
        std::string json_string = status_json.dump();
        
        *status_size = json_string.size();
        std::memcpy(status_data, json_string.c_str(), *status_size);
        
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

// 开始摄像头捕获
extern "C" int32_t start_camera_capture() {
    try {
        if (!g_initialized || !g_stereo_vision) {
            return -1;
        }
        
        g_camera_capturing = true;
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

// 停止摄像头捕获
extern "C" int32_t stop_camera_capture() {
    try {
        g_camera_capturing = false;
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

// 获取摄像头帧
extern "C" int32_t get_camera_frame(
    uint8_t** frame_data,
    int32_t* width,
    int32_t* height,
    int32_t* channels
) {
    try {
        if (!g_initialized || !g_stereo_vision || !g_camera_capturing) {
            return -1;
        }
        
        // 捕获立体帧
        auto stereo_frame = g_stereo_vision->capture_stereo_frame();
        if (!stereo_frame.is_valid) {
            return -1;
        }
        
        // 使用左图像作为主要显示
        cv::Mat& image = stereo_frame.left_image;
        
        *width = image.cols;
        *height = image.rows;
        *channels = image.channels();
        
        int frame_size = image.total() * image.elemSize();
        *frame_data = static_cast<uint8_t*>(allocate_memory(frame_size));
        std::memcpy(*frame_data, image.data, frame_size);
        
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

// 设置切割参数
extern "C" int32_t set_cutting_parameters(
    float x_coordinate,
    int32_t blade_number,
    float quality_threshold
) {
    try {
        if (!g_initialized || !g_modbus_server) {
            return -1;
        }
        
        // 更新Modbus寄存器
        bamboo_cut::communication::CoordinateData coord_data;
        coord_data.x_coordinate = x_coordinate;
        coord_data.blade_number = blade_number;
        coord_data.quality = quality_threshold;
        coord_data.coordinate_ready = true;
        
        g_modbus_server->update_coordinate_data(coord_data);
        
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

// 紧急停止
extern "C" int32_t emergency_stop() {
    try {
        if (!g_initialized) {
            return -1;
        }
        
        // 停止摄像头捕获
        g_camera_capturing = false;
        
        // 更新Modbus状态
        if (g_modbus_server) {
            bamboo_cut::communication::SystemStatus status;
            status.system_status = 4; // 紧急停止
            g_modbus_server->update_system_status(status);
        }
        
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

// 关闭推理服务
extern "C" int32_t shutdown_inference_service() {
    try {
        // 停止摄像头捕获
        g_camera_capturing = false;
        
        // 清理资源
        g_detector.reset();
        g_modbus_server.reset();
        g_stereo_vision.reset();
        
        g_initialized = false;
        
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
} 