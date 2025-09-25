/**
 * @file config_loader.cpp
 * @brief C++ LVGL一体化系统配置加载器实现
 * @version 5.0.0
 * @date 2024
 */

#include "bamboo_cut/utils/config_loader.h"
#include <iostream>
#include <fstream>

namespace bamboo_cut {
namespace utils {

ConfigLoader& ConfigLoader::getInstance() {
    static ConfigLoader instance;
    return instance;
}

bool ConfigLoader::loadFromFile(const std::string& filename) {
    std::cout << "[ConfigLoader] 加载配置文件: " << filename << std::endl;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "[ConfigLoader] 无法打开配置文件: " << filename << std::endl;
        return false;
    }
    
    config_file_ = filename;
    
    // TODO: 解析YAML配置文件
    // 目前使用默认配置
    loadDefaultConfig();
    
    std::cout << "[ConfigLoader] 配置加载完成" << std::endl;
    return true;
}

std::string ConfigLoader::getString(const std::string& key, const std::string& default_value) {
    auto it = string_values_.find(key);
    return (it != string_values_.end()) ? it->second : default_value;
}

int ConfigLoader::getInt(const std::string& key, int default_value) {
    auto it = int_values_.find(key);
    return (it != int_values_.end()) ? it->second : default_value;
}

float ConfigLoader::getFloat(const std::string& key, float default_value) {
    auto it = float_values_.find(key);
    return (it != float_values_.end()) ? it->second : default_value;
}

bool ConfigLoader::getBool(const std::string& key, bool default_value) {
    auto it = bool_values_.find(key);
    return (it != bool_values_.end()) ? it->second : default_value;
}

void ConfigLoader::setString(const std::string& key, const std::string& value) {
    string_values_[key] = value;
}

void ConfigLoader::setInt(const std::string& key, int value) {
    int_values_[key] = value;
}

void ConfigLoader::setFloat(const std::string& key, float value) {
    float_values_[key] = value;
}

void ConfigLoader::setBool(const std::string& key, bool value) {
    bool_values_[key] = value;
}

void ConfigLoader::loadDefaultConfig() {
    // 系统配置
    setString("system.name", "AI Bamboo Recognition System");
    setString("system.version", "5.0.0");
    
    // AI推理配置
    setString("detector.model_path", "models/best.pt");
    setFloat("detector.confidence_threshold", 0.85f);
    setFloat("detector.nms_threshold", 0.45f);
    
    // LVGL界面配置
    setInt("ui.screen_width", 1280);
    setInt("ui.screen_height", 800);
    setInt("ui.refresh_rate", 60);
    
    // Modbus通信配置
    setString("modbus.server_ip", "192.168.1.100");
    setInt("modbus.server_port", 502);
    setInt("modbus.slave_id", 1);
    setBool("modbus.auto_reconnect", true);
    
    // 摄像头配置
    setInt("camera.device_index", 0);
    setInt("camera.width", 1280);
    setInt("camera.height", 720);
    setInt("camera.fps", 30);
}

} // namespace utils
} // namespace bamboo_cut