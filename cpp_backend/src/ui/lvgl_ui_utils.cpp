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
            temp_text << "Temp: CPU " << stats.temperature.cpu_temp << "°C "
                      << "GPU " << stats.temperature.gpu_temp << "°C";
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
        // 从 DataBridge 获取真实的 Modbus 寄存器数据
        core::ModbusRegisters modbus_registers = data_bridge->getModbusRegisters();
        
        // === 更新连接状态信息 ===
        if (widgets.modbus_connection_label && lv_obj_is_valid(widgets.modbus_connection_label)) {
            bool is_connected = (modbus_registers.heartbeat > 0);
            std::string connection_text = "PLC连接: " + std::string(is_connected ? "在线 ✓" : "离线 ✗");
            lv_label_set_text(widgets.modbus_connection_label, connection_text.c_str());
            // 注意：颜色需要从 theme_colors 传入，这里暂用默认
        }
        
        // === 更新 Modbus 地址信息 ===
        if (widgets.modbus_address_label && lv_obj_is_valid(widgets.modbus_address_label)) {
            lv_label_set_text(widgets.modbus_address_label, "地址: 192.168.1.100:502");
        }
        
        // === 更新通讯延迟 ===
        if (widgets.modbus_latency_label && lv_obj_is_valid(widgets.modbus_latency_label)) {
            static int latency_ms = 8;
            latency_ms = 5 + (rand() % 10);  // 模拟 5-15ms 延迟
            std::string latency_text = "通讯延迟: " + std::to_string(latency_ms) + "ms";
            lv_label_set_text(widgets.modbus_latency_label, latency_text.c_str());
        }
        
        // === 更新最后通讯时间 ===
        if (widgets.modbus_last_success_label && lv_obj_is_valid(widgets.modbus_last_success_label)) {
            static int last_comm_seconds = 2;
            last_comm_seconds = (rand() % 5) + 1;  // 模拟 1-5 秒前
            std::string last_success_text = "最后通讯: " + std::to_string(last_comm_seconds) + "秒前";
            lv_label_set_text(widgets.modbus_last_success_label, last_success_text.c_str());
        }
        
        // === 更新错误计数 ===
        if (widgets.modbus_error_count_label && lv_obj_is_valid(widgets.modbus_error_count_label)) {
            static int error_count = 0;
            if (rand() % 100 == 0) error_count++;  // 偶尔增加错误计数
            std::string error_count_text = "错误计数: " + std::to_string(error_count);
            lv_label_set_text(widgets.modbus_error_count_label, error_count_text.c_str());
        }
        
        // === 更新消息计数 ===
        if (widgets.modbus_message_count_label && lv_obj_is_valid(widgets.modbus_message_count_label)) {
            static int message_count = 1523;
            message_count += 1 + (rand() % 3);  // 模拟消息增长
            std::string message_count_text = "今日消息: " + std::to_string(message_count);
            lv_label_set_text(widgets.modbus_message_count_label, message_count_text.c_str());
        }
        
        // === 更新数据包计数 ===
        if (widgets.modbus_packets_label && lv_obj_is_valid(widgets.modbus_packets_label)) {
            static int packets = 1247;
            packets += 1 + (rand() % 3);  // 模拟数据包增长
            std::string packets_text = "数据包: " + std::to_string(packets);
            lv_label_set_text(widgets.modbus_packets_label, packets_text.c_str());
        }
        
        // === 更新错误率 ===
        if (widgets.modbus_errors_label && lv_obj_is_valid(widgets.modbus_errors_label)) {
            static float error_rate = 0.02f;
            error_rate = (rand() % 10) / 1000.0f;  // 模拟 0.000-0.010% 错误率
            std::string error_rate_text = "错误率: " + std::to_string(static_cast<int>(error_rate * 1000) / 1000.0) + "%";
            lv_label_set_text(widgets.modbus_errors_label, error_rate_text.c_str());
        }
        
        // === 更新心跳状态 ===
        if (widgets.modbus_heartbeat_label && lv_obj_is_valid(widgets.modbus_heartbeat_label)) {
            bool heartbeat_ok = (modbus_registers.heartbeat > 0);
            lv_label_set_text(widgets.modbus_heartbeat_label, heartbeat_ok ? "心跳: OK" : "心跳: 超时");
        }
        
        // === 更新 Modbus 寄存器状态信息 ===
        
        // 40001: 系统状态
        if (widgets.modbus_system_status_label && lv_obj_is_valid(widgets.modbus_system_status_label)) {
            const char* status_names[] = {"STOP", "RUN", "ERROR", "PAUSE", "EMERGENCY", "MAINT"};
            uint16_t system_status = modbus_registers.system_status;
            const char* status_str = (system_status < 6) ? status_names[system_status] : "UNKNOWN";
            std::string system_status_text = "40001 系统状态: " + std::string(status_str);
            lv_label_set_text(widgets.modbus_system_status_label, system_status_text.c_str());
        }
        
        // 40002: PLC命令
        if (widgets.modbus_plc_command_label && lv_obj_is_valid(widgets.modbus_plc_command_label)) {
            const char* command_names[] = {"IDLE", "FEED_DETECT", "CUT_PREP", "CUT_DONE", "START_FEED",
                                           "PAUSE", "EMERGENCY_STOP", "RESUME", "RECHECK", "TAIL_PROCESS"};
            uint16_t plc_command = modbus_registers.plc_command;
            const char* command_str = (plc_command < 10) ? command_names[plc_command] : "UNKNOWN";
            std::string plc_command_text = "40002 PLC命令: " + std::string(command_str);
            lv_label_set_text(widgets.modbus_plc_command_label, plc_command_text.c_str());
        }
        
        // 40003: 坐标就绪
        if (widgets.modbus_coord_ready_label && lv_obj_is_valid(widgets.modbus_coord_ready_label)) {
            uint16_t coord_ready = modbus_registers.coord_ready;
            const char* ready_str = coord_ready ? "YES" : "NO";
            std::string coord_ready_text = "40003 坐标就绪: " + std::string(ready_str);
            lv_label_set_text(widgets.modbus_coord_ready_label, coord_ready_text.c_str());
        }
        
        // 40004-40005: X坐标 (0.1mm)
        if (widgets.modbus_x_coordinate_label && lv_obj_is_valid(widgets.modbus_x_coordinate_label)) {
            uint32_t x_coord_raw = modbus_registers.x_coordinate;
            float x_coord_mm = static_cast<float>(x_coord_raw) * 0.1f;
            std::string x_coord_text = "40004 X坐标: " + std::to_string(static_cast<int>(x_coord_mm * 10) / 10.0) + "mm";
            lv_label_set_text(widgets.modbus_x_coordinate_label, x_coord_text.c_str());
        }
        
        // 40006: 切割质量
        if (widgets.modbus_cut_quality_label && lv_obj_is_valid(widgets.modbus_cut_quality_label)) {
            const char* quality_names[] = {"OK", "NG"};
            uint16_t cut_quality = modbus_registers.cut_quality;
            const char* quality_str = (cut_quality < 2) ? quality_names[cut_quality] : "UNKNOWN";
            std::string cut_quality_text = "40006 切割质量: " + std::string(quality_str);
            lv_label_set_text(widgets.modbus_cut_quality_label, cut_quality_text.c_str());
        }
        
        // 40009: 刀片编号
        if (widgets.modbus_blade_number_label && lv_obj_is_valid(widgets.modbus_blade_number_label)) {
            const char* blade_names[] = {"NONE", "BLADE1", "BLADE2", "DUAL"};
            uint16_t blade_number = modbus_registers.blade_number;
            const char* blade_str = (blade_number < 4) ? blade_names[blade_number] : "UNKNOWN";
            std::string blade_number_text = "40009 刀片编号: " + std::string(blade_str);
            lv_label_set_text(widgets.modbus_blade_number_label, blade_number_text.c_str());
        }
        
        // 40010: 健康状态
        if (widgets.modbus_health_status_label && lv_obj_is_valid(widgets.modbus_health_status_label)) {
            const char* health_names[] = {"OK", "WARN", "ERROR", "CRITICAL"};
            uint16_t health_status = modbus_registers.health_status;
            const char* health_str = (health_status < 4) ? health_names[health_status] : "UNKNOWN";
            std::string health_status_text = "40010 健康状态: " + std::string(health_str);
            lv_label_set_text(widgets.modbus_health_status_label, health_status_text.c_str());
        }
        
        // 40011: 尾料状态
        if (widgets.modbus_tail_status_label && lv_obj_is_valid(widgets.modbus_tail_status_label)) {
            const char* tail_names[] = {"IDLE", "PROCESSING", "DONE", "RECHECK", "", "", "", "", "", "TAIL", "EJECTED"};
            uint16_t tail_status = modbus_registers.tail_status;
            const char* tail_str = (tail_status < 11 && tail_names[tail_status][0] != '\0') ? tail_names[tail_status] : "UNKNOWN";
            std::string tail_text = "40011 尾料: " + std::string(tail_str);
            lv_label_set_text(widgets.modbus_tail_status_label, tail_text.c_str());
        }
        
        // 40012: PLC报警/扩展
        if (widgets.modbus_plc_alarm_label && lv_obj_is_valid(widgets.modbus_plc_alarm_label)) {
            uint16_t alarm = modbus_registers.plc_ext_alarm;
            std::string alarm_text = "40012 报警: " + std::to_string(alarm);
            lv_label_set_text(widgets.modbus_plc_alarm_label, alarm_text.c_str());
        }
        
        // 40014: 导轨方向
        if (widgets.modbus_rail_direction_label && lv_obj_is_valid(widgets.modbus_rail_direction_label)) {
            const char* dir_names[] = {"FWD", "REV"};
            uint16_t dir = modbus_registers.rail_direction;
            const char* dir_str = (dir < 2) ? dir_names[dir] : "UNKNOWN";
            std::string dir_text = "40014 导轨方向: " + std::string(dir_str);
            lv_label_set_text(widgets.modbus_rail_direction_label, dir_text.c_str());
        }
        
        // 40015-40016: 剩余长度 (0.1mm)
        if (widgets.modbus_remaining_length_label && lv_obj_is_valid(widgets.modbus_remaining_length_label)) {
            float remain_mm = static_cast<float>(modbus_registers.remaining_length) * 0.1f;
            std::string remain_text = "40015 剩余: " + std::to_string(static_cast<int>(remain_mm * 10) / 10.0) + "mm";
            lv_label_set_text(widgets.modbus_remaining_length_label, remain_text.c_str());
        }
        
        // 40017: 覆盖率
        if (widgets.modbus_coverage_label && lv_obj_is_valid(widgets.modbus_coverage_label)) {
            uint16_t coverage = modbus_registers.coverage;
            std::string coverage_text = "40017 覆盖率: " + std::to_string(coverage) + "%";
            lv_label_set_text(widgets.modbus_coverage_label, coverage_text.c_str());
        }
        
        // 40018: 速度档
        if (widgets.modbus_feed_speed_label && lv_obj_is_valid(widgets.modbus_feed_speed_label)) {
            uint16_t speed = modbus_registers.feed_speed_gear;
            std::string speed_text = "40018 速度档: " + std::to_string(speed);
            lv_label_set_text(widgets.modbus_feed_speed_label, speed_text.c_str());
        }
        
        // 40019: 处理模式
        if (widgets.modbus_process_mode_label && lv_obj_is_valid(widgets.modbus_process_mode_label)) {
            const char* mode_names[] = {"AUTO", "MANUAL", "MAINT"};
            uint16_t mode = modbus_registers.process_mode;
            const char* mode_str = (mode < 3) ? mode_names[mode] : "UNKNOWN";
            std::string mode_text = "40019 模式: " + std::string(mode_str);
            lv_label_set_text(widgets.modbus_process_mode_label, mode_text.c_str());
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[lvgl_ui_utils] Exception in updateModbusStatus: " << e.what() << std::endl;
        return false;
    }
}

// ==================== 工作流程状态更新 ====================

bool updateWorkflowStatus(
    LVGLControlWidgets& widgets,
    int current_step,
    const LVGLThemeColors& theme_colors) {
    
    // 检查工作流程按钮是否已初始化
    if (widgets.workflow_buttons.empty()) {
        return false;  // 静默跳过，避免日志过多
    }
    
    try {
        // 更新工作流程步骤指示器
        for(size_t i = 0; i < widgets.workflow_buttons.size(); i++) {
            lv_obj_t* step = widgets.workflow_buttons[i];
            if (!step || !lv_obj_is_valid(step)) continue;  // 跳过空指针或无效对象
            
            bool is_active = (i == static_cast<size_t>(current_step) - 1);
            bool is_completed = (i < static_cast<size_t>(current_step) - 1);
            
            if(is_active) {
                // 当前激活步骤 - 使用主题色
                lv_obj_set_style_bg_color(step, theme_colors.primary, 0);
                lv_obj_set_style_border_color(step, theme_colors.primary, 0);
            } else if(is_completed) {
                // 已完成步骤 - 使用成功色
                lv_obj_set_style_bg_color(step, theme_colors.success, 0);
                lv_obj_set_style_border_color(step, theme_colors.success, 0);
            } else {
                // 未完成步骤 - 使用默认灰色
                lv_obj_set_style_bg_color(step, lv_color_hex(0x2A3441), 0);
                lv_obj_set_style_border_color(step, lv_color_hex(0x4A5461), 0);
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[lvgl_ui_utils] Exception in updateWorkflowStatus: " << e.what() << std::endl;
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

