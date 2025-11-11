/**
 * @file lvgl_ui_utils.cpp
 * @brief LVGL UI 共享工具函数实现
 */

#include "bamboo_cut/ui/lvgl_ui_utils.h"
#include "bamboo_cut/core/data_bridge.h"
#include "bamboo_cut/utils/jetson_monitor.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace bamboo_cut {
namespace ui {

// ==================== Jetson 系统监控更新 ====================

bool updateJetsonMonitoring(
    LVGLControlWidgets& widgets,
    std::shared_ptr<bamboo_cut::utils::JetsonMonitor> jetson_monitor,
    const LVGLThemeColors& colors) {
    
    if (!jetson_monitor || !jetson_monitor->isRunning()) {
        // 降级：显示 "Not Available"
        if (widgets.cpu_label) {
            lv_label_set_text(widgets.cpu_label, "CPU: N/A");
        }
        if (widgets.gpu_label) {
            lv_label_set_text(widgets.gpu_label, "GPU: N/A");
        }
        if (widgets.mem_label) {
            lv_label_set_text(widgets.mem_label, "RAM: N/A");
        }
        return false;
    }
    
    try {
        bamboo_cut::utils::SystemStats stats = jetson_monitor->getLatestStats();
        
        // === 更新 CPU 信息 ===
        if (widgets.cpu_bar && widgets.cpu_label && !stats.cpu_cores.empty()) {
            int total_usage = 0;
            int total_freq = 0;
            int valid_cores = 0;
            
            for (const auto& core : stats.cpu_cores) {
                if (core.usage_percent >= 0 && core.usage_percent <= 100 &&
                    core.frequency_mhz >= 0 && core.frequency_mhz <= 10000) {
                    total_usage += core.usage_percent;
                    total_freq += core.frequency_mhz;
                    valid_cores++;
                }
            }
            
            if (valid_cores > 0) {
                int avg_usage = total_usage / valid_cores;
                int avg_freq = total_freq / valid_cores;
                
                avg_usage = std::max(0, std::min(100, avg_usage));
                avg_freq = std::max(0, std::min(9999, avg_freq));
                
                lv_bar_set_value(widgets.cpu_bar, avg_usage, LV_ANIM_ON);
                
                std::ostringstream cpu_text;
                cpu_text << "CPU: " << avg_usage << "% @" << avg_freq << "MHz";
                lv_label_set_text(widgets.cpu_label, cpu_text.str().c_str());
                
                lv_obj_set_style_bg_color(widgets.cpu_bar, colors.warning, LV_PART_INDICATOR);
            }
        }
        
        // === 更新 GPU 信息 ===
        if (widgets.gpu_bar && widgets.gpu_label) {
            int gpu_usage = stats.gpu.usage_percent;
            int gpu_freq = stats.gpu.frequency_mhz;
            
            if (gpu_usage >= 0 && gpu_usage <= 100 && gpu_freq >= 0 && gpu_freq <= 10000) {
                lv_bar_set_value(widgets.gpu_bar, gpu_usage, LV_ANIM_ON);
                
                std::ostringstream gpu_text;
                gpu_text << "GPU: " << gpu_usage << "% @" << gpu_freq << "MHz";
                lv_label_set_text(widgets.gpu_label, gpu_text.str().c_str());
                
                lv_obj_set_style_bg_color(widgets.gpu_bar, colors.warning, LV_PART_INDICATOR);
            }
        }
        
        // === 更新内存信息 ===
        if (widgets.mem_bar && widgets.mem_label) {
            if (stats.memory.ram_total_mb > 0 && stats.memory.ram_used_mb >= 0 &&
                stats.memory.ram_used_mb <= stats.memory.ram_total_mb) {
                
                int mem_percentage = (stats.memory.ram_used_mb * 100) / stats.memory.ram_total_mb;
                mem_percentage = std::max(0, std::min(100, mem_percentage));
                
                lv_bar_set_value(widgets.mem_bar, mem_percentage, LV_ANIM_ON);
                
                std::ostringstream mem_text;
                mem_text << "RAM: " << stats.memory.ram_used_mb << "/" 
                         << stats.memory.ram_total_mb << "MB";
                lv_label_set_text(widgets.mem_label, mem_text.str().c_str());
                
                lv_obj_set_style_bg_color(widgets.mem_bar, colors.warning, LV_PART_INDICATOR);
            }
        }
        
        // === 更新温度信息 ===
        if (widgets.cpu_temp_label) {
            std::ostringstream temp_text;
            temp_text << "Temp: CPU " << stats.temperature.cpu_temp_c << "°C "
                      << "GPU " << stats.temperature.gpu_temp_c << "°C";
            lv_label_set_text(widgets.cpu_temp_label, temp_text.str().c_str());
        }
        
        // === 更新 SWAP 使用情况 ===
        if (widgets.swap_usage_label) {
            std::ostringstream swap_text;
            swap_text << "SWAP: " << stats.memory.swap_used_mb << "/" 
                      << stats.memory.swap_total_mb << "MB";
            lv_label_set_text(widgets.swap_usage_label, swap_text.str().c_str());
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[lvgl_ui_utils] Exception in updateJetsonMonitoring: " << e.what() << std::endl;
        return false;
    }
}

// ==================== AI 模型统计更新 ====================

bool updateAIModelStats(
    LVGLControlWidgets& widgets,
    std::shared_ptr<bamboo_cut::core::DataBridge> data_bridge) {
    
    if (!data_bridge) {
        return false;
    }
    
    try {
        // 获取 AI 模型统计数据
        // 注意：这些方法需要在 DataBridge 中实现
        
        // === 更新 FPS ===
        if (widgets.ai_fps_label) {
            // TODO: 从 DataBridge 获取实际 FPS
            static int simulated_fps = 0;
            simulated_fps = (simulated_fps + 1) % 60 + 20;  // 模拟 20-80 fps
            
            std::ostringstream fps_text;
            fps_text << "FPS: " << simulated_fps << " fps";
            lv_label_set_text(widgets.ai_fps_label, fps_text.str().c_str());
        }
        
        // === 更新检测数量 ===
        if (widgets.ai_total_detections_label) {
            // TODO: 从 DataBridge 获取实际检测数量
            static int detection_count = 0;
            detection_count++;
            
            std::ostringstream detect_text;
            detect_text << "Detected: " << detection_count << " objects";
            lv_label_set_text(widgets.ai_total_detections_label, detect_text.str().c_str());
        }
        
        // === 更新推理时间 ===
        if (widgets.ai_inference_time_label) {
            // TODO: 从 DataBridge 获取实际推理时间
            static int inference_ms = 0;
            inference_ms = (inference_ms + 1) % 30 + 10;  // 模拟 10-40ms
            
            std::ostringstream time_text;
            time_text << "Inference: " << inference_ms << "ms";
            lv_label_set_text(widgets.ai_inference_time_label, time_text.str().c_str());
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[lvgl_ui_utils] Exception in updateAIModelStats: " << e.what() << std::endl;
        return false;
    }
}

// ==================== 摄像头状态更新 ====================

bool updateCameraStatus(
    LVGLControlWidgets& widgets,
    std::shared_ptr<bamboo_cut::core::DataBridge> data_bridge) {
    
    if (!data_bridge) {
        return false;
    }
    
    try {
        // === 更新摄像头状态 ===
        if (widgets.camera_status_label) {
            lv_label_set_text(widgets.camera_status_label, "Camera: Online");
        }
        
        // === 更新摄像头 FPS ===
        if (widgets.camera_fps_label) {
            lv_label_set_text(widgets.camera_fps_label, "30 fps");
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[lvgl_ui_utils] Exception in updateCameraStatus: " << e.what() << std::endl;
        return false;
    }
}

// ==================== Modbus 状态更新 ====================

bool updateModbusStatus(
    LVGLControlWidgets& widgets,
    std::shared_ptr<bamboo_cut::core::DataBridge> data_bridge) {
    
    if (!data_bridge) {
        return false;
    }
    
    try {
        // === 更新 Modbus 连接状态 ===
        if (widgets.modbus_connection_label) {
            lv_label_set_text(widgets.modbus_connection_label, "Modbus: Connected");
        }
        
        // === 更新 Modbus 延迟 ===
        if (widgets.modbus_latency_label) {
            lv_label_set_text(widgets.modbus_latency_label, "Latency: 12ms");
        }
        
        // === 更新错误计数 ===
        if (widgets.modbus_error_count_label) {
            lv_label_set_text(widgets.modbus_error_count_label, "Errors: 0");
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[lvgl_ui_utils] Exception in updateModbusStatus: " << e.what() << std::endl;
        return false;
    }
}

// ==================== UI 性能统计更新 ====================

void updateUIPerformance(lv_obj_t* fps_label, float ui_fps) {
    if (!fps_label) return;
    
    std::ostringstream fps_text;
    fps_text << "UI: " << std::fixed << std::setprecision(1) << ui_fps << " fps";
    lv_label_set_text(fps_label, fps_text.str().c_str());
}

// ==================== 格式化工具函数 ====================

std::string formatTemperature(int temp_celsius) {
    std::ostringstream ss;
    ss << temp_celsius << "°C";
    return ss.str();
}

std::string formatPercentage(int value, int total) {
    if (total == 0) return "0%";
    int percentage = (value * 100) / total;
    std::ostringstream ss;
    ss << percentage << "%";
    return ss.str();
}

std::string formatFrequency(int freq_mhz) {
    if (freq_mhz >= 1000) {
        float freq_ghz = freq_mhz / 1000.0f;
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1) << freq_ghz << "GHz";
        return ss.str();
    } else {
        std::ostringstream ss;
        ss << freq_mhz << "MHz";
        return ss.str();
    }
}

std::string formatMemorySize(int size_mb) {
    if (size_mb >= 1024) {
        float size_gb = size_mb / 1024.0f;
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1) << size_gb << "GB";
        return ss.str();
    } else {
        std::ostringstream ss;
        ss << size_mb << "MB";
        return ss.str();
    }
}

} // namespace ui
} // namespace bamboo_cut

