/**
 * 配置管理器实现
 */

#include "app/config_manager.h"
#include <stdio.h>
#include <string.h>

ConfigManager::ConfigManager() {
    set_defaults();
}

ConfigManager::~ConfigManager() {
}

bool ConfigManager::load(const char* config_file) {
    printf("加载配置文件: %s\n", config_file ? config_file : "默认配置");
    
    if (config_file && load_from_file(config_file)) {
        printf("配置文件加载成功\n");
        return true;
    }
    
    printf("使用默认配置\n");
    return true;
}

void ConfigManager::set_debug_mode(bool enable) {
    config_.debug_mode = enable;
    printf("调试模式: %s\n", enable ? "启用" : "禁用");
}

bool ConfigManager::load_from_file(const char* filename) {
    // TODO: 实现JSON配置文件解析
    // 暂时返回false，使用默认配置
    return false;
}

void ConfigManager::set_defaults() {
    // 设置默认配置值
    config_.debug_mode = false;
    config_.fullscreen = true;
    config_.log_level = LOG_LEVEL_INFO;
    strcpy(config_.log_file, "/var/log/bamboo_controller.log");
    
    // 摄像头默认配置
    strcpy(config_.camera.device_path, "/dev/video0");
    config_.camera.width = DEFAULT_CAMERA_WIDTH;
    config_.camera.height = DEFAULT_CAMERA_HEIGHT;
    config_.camera.fps = DEFAULT_CAMERA_FPS;
    config_.camera.buffer_count = 4;
    config_.camera.use_hardware_acceleration = true;
    config_.camera.exposure = -1;
    config_.camera.gain = -1;
    
    // AI配置
    strcpy(config_.ai.model_path, "/opt/bamboo/models/yolov8n.onnx");
    strcpy(config_.ai.engine_path, "/opt/bamboo/models/yolov8n.engine");
    config_.ai.input_width = 640;
    config_.ai.input_height = 640;
    config_.ai.confidence_threshold = 0.7f;
    config_.ai.nms_threshold = 0.4f;
    config_.ai.use_tensorrt = true;
    config_.ai.use_int8 = true;
    config_.ai.use_dla = true;
    
    // 显示配置
    strcpy(config_.display.framebuffer_device, "/dev/fb0");
    config_.display.width = DEFAULT_DISPLAY_WIDTH;
    config_.display.height = DEFAULT_DISPLAY_HEIGHT;
    config_.display.bpp = 32;
    config_.display.vsync = true;
    config_.display.brightness = 80;
    
    // 触摸配置
    strcpy(config_.touch.device_path, "/dev/input/event0");
    config_.touch.calibration_enabled = true;
    config_.touch.sensitivity = DEFAULT_TOUCH_SENSITIVITY;
    config_.touch.gesture_enabled = false;
}