/**
 * @file lvgl_ui_utils.h
 * @brief LVGL UI 共享工具函数 - 用于不同 UI 实现之间的代码复用
 * 
 * 提取通用的 UI 更新逻辑，避免在 DRM 和 Wayland 版本之间重复代码
 */

#pragma once

#include <memory>
#include <string>
#include <lvgl.h>

// 前向声明（注意：这些类在 bamboo_cut 命名空间下，不在 ui 下）
namespace bamboo_cut {
namespace core {
    class DataBridge;
}

namespace utils {
    class JetsonMonitor;
    struct SystemStats;
}
} // namespace bamboo_cut

namespace bamboo_cut {
namespace ui {

/**
 * @brief UI 控件集合 - 用于动态更新的控件引用
 */
struct LVGLControlWidgets {
    // Jetson 系统监控组件
    lv_obj_t* cpu_bar = nullptr;
    lv_obj_t* cpu_label = nullptr;
    lv_obj_t* cpu_temp_label = nullptr;
    
    lv_obj_t* gpu_bar = nullptr;
    lv_obj_t* gpu_label = nullptr;
    lv_obj_t* gpu_temp_label = nullptr;
    
    lv_obj_t* mem_bar = nullptr;
    lv_obj_t* mem_label = nullptr;
    lv_obj_t* swap_usage_label = nullptr;
    
    // AI 模型监控组件
    lv_obj_t* ai_fps_label = nullptr;
    lv_obj_t* ai_inference_time_label = nullptr;
    lv_obj_t* ai_total_detections_label = nullptr;
    lv_obj_t* ai_confidence_label = nullptr;
    
    // 摄像头状态组件
    lv_obj_t* camera_status_label = nullptr;
    lv_obj_t* camera_fps_label = nullptr;
    
    // Modbus 通信组件
    lv_obj_t* modbus_connection_label = nullptr;
    lv_obj_t* modbus_latency_label = nullptr;
    lv_obj_t* modbus_error_count_label = nullptr;
    
    // 状态和性能
    lv_obj_t* status_label = nullptr;
    lv_obj_t* ui_fps_label = nullptr;
};

/**
 * @brief 主题颜色定义
 */
struct LVGLThemeColors {
    lv_color_t primary;
    lv_color_t secondary;
    lv_color_t success;
    lv_color_t warning;
    lv_color_t error;
    lv_color_t background;
    lv_color_t surface;
    
    // 默认构造函数 - 使用原始 UI 的配色方案
    LVGLThemeColors() {
        background = lv_color_hex(0x1A1F26);
        surface    = lv_color_hex(0x252B35);
        primary    = lv_color_hex(0x5B9BD5);
        secondary  = lv_color_hex(0x70A5DB);
        success    = lv_color_hex(0x7FB069);
        warning    = lv_color_hex(0xE6A055);
        error      = lv_color_hex(0xD67B7B);
    }
};

/**
 * @brief 更新 Jetson 系统监控数据（CPU/GPU/RAM/温度）
 * 
 * @param widgets 控件集合
 * @param jetson_monitor Jetson 监控实例
 * @param colors 主题颜色
 * @return true 更新成功，false 失败（降级到模拟数据）
 */
bool updateJetsonMonitoring(
    LVGLControlWidgets& widgets,
    std::shared_ptr<bamboo_cut::utils::JetsonMonitor> jetson_monitor,
    const LVGLThemeColors& colors);

/**
 * @brief 更新 AI 模型统计数据（FPS、推理时间、检测数量）
 * 
 * @param widgets 控件集合
 * @param data_bridge 数据桥接器
 * @return true 更新成功，false 失败
 */
bool updateAIModelStats(
    LVGLControlWidgets& widgets,
    std::shared_ptr<bamboo_cut::core::DataBridge> data_bridge);

/**
 * @brief 更新摄像头状态（在线状态、FPS、分辨率）
 * 
 * @param widgets 控件集合
 * @param data_bridge 数据桥接器
 * @return true 更新成功，false 失败
 */
bool updateCameraStatus(
    LVGLControlWidgets& widgets,
    std::shared_ptr<bamboo_cut::core::DataBridge> data_bridge);

/**
 * @brief 更新 Modbus 通信状态（连接、延迟、错误计数）
 * 
 * @param widgets 控件集合
 * @param data_bridge 数据桥接器
 * @return true 更新成功，false 失败
 */
bool updateModbusStatus(
    LVGLControlWidgets& widgets,
    std::shared_ptr<bamboo_cut::core::DataBridge> data_bridge);

/**
 * @brief 更新 UI 性能统计（FPS）
 * 
 * @param fps_label FPS 标签控件
 * @param ui_fps 当前 UI 帧率
 */
void updateUIPerformance(lv_obj_t* fps_label, float ui_fps);

/**
 * @brief 格式化温度显示
 * 
 * @param temp_celsius 温度（摄氏度）
 * @return 格式化的字符串（例如 "58°C"）
 */
std::string formatTemperature(int temp_celsius);

/**
 * @brief 格式化百分比显示
 * 
 * @param value 数值
 * @param total 总数
 * @return 格式化的字符串（例如 "45%"）
 */
std::string formatPercentage(int value, int total);

/**
 * @brief 格式化频率显示
 * 
 * @param freq_mhz 频率（MHz）
 * @return 格式化的字符串（例如 "1.9GHz" 或 "624MHz"）
 */
std::string formatFrequency(int freq_mhz);

/**
 * @brief 格式化内存大小显示
 * 
 * @param size_mb 内存大小（MB）
 * @return 格式化的字符串（例如 "3347MB" 或 "3.3GB"）
 */
std::string formatMemorySize(int size_mb);

} // namespace ui
} // namespace bamboo_cut
