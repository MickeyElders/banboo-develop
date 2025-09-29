/**
 * @file lvgl_events.cpp
 * @brief LVGL事件处理器实现
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include <iostream>

namespace bamboo_cut {
namespace ui {

// ==================== 事件处理器 ====================

void LVGLInterface::onStartButtonClicked(lv_event_t* e) {
#ifdef ENABLE_LVGL
    LVGLInterface* self = static_cast<LVGLInterface*>(lv_event_get_user_data(e));
    if (self) {
        std::cout << "[LVGLInterface] 启动按钮被点击" << std::endl;
        self->system_running_ = true;
        self->showMessageDialog("系统启动", "系统正在启动...");
    }
#endif
}

void LVGLInterface::onStopButtonClicked(lv_event_t* e) {
#ifdef ENABLE_LVGL
    LVGLInterface* self = static_cast<LVGLInterface*>(lv_event_get_user_data(e));
    if (self) {
        std::cout << "[LVGLInterface] 停止按钮被点击" << std::endl;
        self->system_running_ = false;
        self->showMessageDialog("系统停止", "系统已停止运行");
    }
#endif
}

void LVGLInterface::onPauseButtonClicked(lv_event_t* e) {
#ifdef ENABLE_LVGL
    LVGLInterface* self = static_cast<LVGLInterface*>(lv_event_get_user_data(e));
    if (self) {
        std::cout << "[LVGLInterface] 暂停按钮被点击" << std::endl;
        self->showMessageDialog("系统暂停", "系统已暂停");
    }
#endif
}

void LVGLInterface::onEmergencyButtonClicked(lv_event_t* e) {
#ifdef ENABLE_LVGL
    LVGLInterface* self = static_cast<LVGLInterface*>(lv_event_get_user_data(e));
    if (self) {
        std::cout << "[LVGLInterface] 急停按钮被点击" << std::endl;
        self->emergency_stop_ = true;
        self->system_running_ = false;
        self->showMessageDialog("紧急停止", "系统已紧急停止！");
    }
#endif
}

void LVGLInterface::onBladeSelectionChanged(lv_event_t* e) {
#ifdef ENABLE_LVGL
    LVGLInterface* self = static_cast<LVGLInterface*>(lv_event_get_user_data(e));
    lv_obj_t* btn = static_cast<lv_obj_t*>(lv_event_get_target(e));
    
    if (self && btn) {
        // 找到被点击的刀片编号
        for(size_t i = 0; i < self->control_widgets_.blade_buttons.size(); i++) {
            if (self->control_widgets_.blade_buttons[i] == btn) {
                self->selected_blade_ = i + 1;
                std::cout << "[LVGLInterface] 选择刀片 #" << (i+1) << std::endl;
                
                // 更新所有按钮样式
                for(size_t j = 0; j < self->control_widgets_.blade_buttons.size(); j++) {
                    lv_obj_t* b = self->control_widgets_.blade_buttons[j];
                    if (j == i) {
                        lv_obj_set_style_bg_color(b, self->color_success_, 0);
                        lv_obj_set_style_shadow_color(b, self->color_success_, 0);
                    } else {
                        lv_obj_set_style_bg_color(b, self->color_primary_, 0);
                        lv_obj_set_style_shadow_color(b, self->color_primary_, 0);
                    }
                }
                
                // 更新摄像头面板的刀片显示
                if (self->camera_widgets_.blade_value) {
                    std::string blade_text = LV_SYMBOL_SETTINGS " 刀片: #" + std::to_string(self->selected_blade_);
                    lv_label_set_text(self->camera_widgets_.blade_value, blade_text.c_str());
                }
                
                break;
            }
        }
    }
#endif
}

void LVGLInterface::onSettingsButtonClicked(lv_event_t* e) {
#ifdef ENABLE_LVGL
    LVGLInterface* self = static_cast<LVGLInterface*>(lv_event_get_user_data(e));
    if (self) {
        std::cout << "[LVGLInterface] 设置按钮被点击" << std::endl;
        self->showMessageDialog("系统设置", "设置功能开发中...");
    }
#endif
}

void LVGLInterface::showMessageDialog(const std::string& title, const std::string& message) {
#ifdef ENABLE_LVGL
    lv_obj_t* mbox = lv_msgbox_create(NULL);
    lv_msgbox_add_title(mbox, title.c_str());
    lv_msgbox_add_text(mbox, message.c_str());
    lv_msgbox_add_close_button(mbox);
    lv_obj_center(mbox);
    
    // 添加样式
    lv_obj_set_style_bg_color(mbox, color_surface_, 0);
    lv_obj_set_style_border_color(mbox, color_primary_, 0);
    lv_obj_set_style_border_width(mbox, 2, 0);
    lv_obj_set_style_shadow_width(mbox, 30, 0);
    lv_obj_set_style_shadow_opa(mbox, LV_OPA_50, 0);
#endif
}

void LVGLInterface::updateModbusDisplay() {
#ifdef ENABLE_LVGL
    if (!data_bridge_) {
        std::cout << "[LVGLInterface] DataBridge not initialized, skipping Modbus display update" << std::endl;
        return;
    }
    
    // 添加控制面板空指针保护
    if (!control_panel_) {
        std::cout << "[LVGLInterface] Control panel not initialized, skipping Modbus display update" << std::endl;
        return;
    }
    
    // 检查所有关键的Modbus控件是否已正确初始化
    if (!control_widgets_.modbus_connection_label ||
        !control_widgets_.modbus_address_label ||
        !control_widgets_.modbus_latency_label ||
        !control_widgets_.modbus_last_success_label ||
        !control_widgets_.modbus_error_count_label ||
        !control_widgets_.modbus_message_count_label) {
        std::cout << "[LVGLInterface] Modbus widgets not properly initialized, skipping update" << std::endl;
        return;
    }
    
    // 检查Modbus寄存器相关控件
    if (!control_widgets_.modbus_system_status_label ||
        !control_widgets_.modbus_plc_command_label ||
        !control_widgets_.modbus_coord_ready_label ||
        !control_widgets_.modbus_x_coordinate_label ||
        !control_widgets_.modbus_cut_quality_label ||
        !control_widgets_.modbus_blade_number_label ||
        !control_widgets_.modbus_health_status_label) {
        // 寄存器控件未创建，直接返回（不打印日志避免刷屏）
        return;  // ← 添加这一行
    }
    
    // 从DataBridge获取真实的Modbus寄存器数据
    core::ModbusRegisters modbus_registers = data_bridge_->getModbusRegisters();
    
    // 更新连接状态信息（使用模拟数据） - 添加控件有效性检查
    if (control_widgets_.modbus_connection_label && lv_obj_is_valid(control_widgets_.modbus_connection_label)) {
        try {
            bool is_connected = (modbus_registers.heartbeat > 0);
            std::string connection_text = "PLC连接: " + std::string(is_connected ? "在线 ✓" : "离线 ✗");
            lv_label_set_text(control_widgets_.modbus_connection_label, connection_text.c_str());
            lv_obj_set_style_text_color(control_widgets_.modbus_connection_label,
                is_connected ? color_success_ : color_error_, 0);
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Error updating modbus connection label: " << e.what() << std::endl;
        }
    }
    
    // 更新Modbus地址信息 - 添加控件有效性检查
    if (control_widgets_.modbus_address_label && lv_obj_is_valid(control_widgets_.modbus_address_label)) {
        try {
            lv_label_set_text(control_widgets_.modbus_address_label, "地址: 192.168.1.100:502");
            lv_obj_set_style_text_color(control_widgets_.modbus_address_label, lv_color_hex(0xB0B8C1), 0);
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Error updating modbus address label: " << e.what() << std::endl;
        }
    }
    
    // 更新通讯延迟 - 添加控件有效性检查
    if (control_widgets_.modbus_latency_label && lv_obj_is_valid(control_widgets_.modbus_latency_label)) {
        try {
            static int latency_ms = 8;
            latency_ms = 5 + (rand() % 10);  // 模拟5-15ms延迟
            std::string latency_text = "通讯延迟: " + std::to_string(latency_ms) + "ms";
            lv_label_set_text(control_widgets_.modbus_latency_label, latency_text.c_str());
            
            // 根据延迟设置颜色
            if (latency_ms <= 10) {
                lv_obj_set_style_text_color(control_widgets_.modbus_latency_label, color_success_, 0);
            } else if (latency_ms <= 20) {
                lv_obj_set_style_text_color(control_widgets_.modbus_latency_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.modbus_latency_label, color_error_, 0);
            }
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Error updating modbus latency label: " << e.what() << std::endl;
        }
    }
    
    // 更新最后通讯时间 - 添加控件有效性检查
    if (control_widgets_.modbus_last_success_label && lv_obj_is_valid(control_widgets_.modbus_last_success_label)) {
        try {
            static int last_comm_seconds = 2;
            last_comm_seconds = (rand() % 5) + 1;  // 模拟1-5秒前
            std::string last_success_text = "最后通讯: " + std::to_string(last_comm_seconds) + "秒前";
            lv_label_set_text(control_widgets_.modbus_last_success_label, last_success_text.c_str());
            lv_obj_set_style_text_color(control_widgets_.modbus_last_success_label, color_primary_, 0);
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Error updating modbus last success label: " << e.what() << std::endl;
        }
    }
    
    // 更新错误计数 - 添加控件有效性检查
    if (control_widgets_.modbus_error_count_label && lv_obj_is_valid(control_widgets_.modbus_error_count_label)) {
        try {
            static int error_count = 0;
            if (rand() % 100 == 0) error_count++;  // 偶尔增加错误计数
            std::string error_count_text = "错误计数: " + std::to_string(error_count);
            lv_label_set_text(control_widgets_.modbus_error_count_label, error_count_text.c_str());
            lv_obj_set_style_text_color(control_widgets_.modbus_error_count_label,
                error_count == 0 ? color_success_ : color_warning_, 0);
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Error updating modbus error count label: " << e.what() << std::endl;
        }
    }
    
    // 更新消息计数 - 添加控件有效性检查
    if (control_widgets_.modbus_message_count_label && lv_obj_is_valid(control_widgets_.modbus_message_count_label)) {
        try {
            static int message_count = 1523;
            message_count += 1 + (rand() % 3);  // 模拟消息增长
            std::string message_count_text = "今日消息: " + std::to_string(message_count);
            lv_label_set_text(control_widgets_.modbus_message_count_label, message_count_text.c_str());
            lv_obj_set_style_text_color(control_widgets_.modbus_message_count_label, lv_color_hex(0xB0B8C1), 0);
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Error updating modbus message count label: " << e.what() << std::endl;
        }
    }
    
    if (control_widgets_.modbus_packets_label && lv_obj_is_valid(control_widgets_.modbus_packets_label)) {
        try {
            static int packets = 1247;
            packets += 1 + (rand() % 3);  // 模拟数据包增长
            std::string packets_text = "数据包: " + std::to_string(packets);
            lv_label_set_text(control_widgets_.modbus_packets_label, packets_text.c_str());
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Error updating modbus packets label: " << e.what() << std::endl;
        }
    }
    
    if (control_widgets_.modbus_errors_label && lv_obj_is_valid(control_widgets_.modbus_errors_label)) {
        try {
            static float error_rate = 0.02f;
            error_rate = (rand() % 10) / 1000.0f;  // 模拟0.000-0.010%错误率
            std::string error_rate_text = "错误率: " + std::to_string(static_cast<int>(error_rate * 1000) / 1000.0) + "%%";
            lv_label_set_text(control_widgets_.modbus_errors_label, error_rate_text.c_str());
            
            // 根据错误率设置颜色
            if (error_rate > 1.0f) {
                lv_obj_set_style_text_color(control_widgets_.modbus_errors_label, color_error_, 0);
            } else if (error_rate > 0.1f) {
                lv_obj_set_style_text_color(control_widgets_.modbus_errors_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.modbus_errors_label, color_success_, 0);
            }
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Error updating modbus errors label: " << e.what() << std::endl;
        }
    }
    
    if (control_widgets_.modbus_heartbeat_label && lv_obj_is_valid(control_widgets_.modbus_heartbeat_label)) {
        try {
            bool heartbeat_ok = (modbus_registers.heartbeat > 0);
            lv_label_set_text(control_widgets_.modbus_heartbeat_label,
                heartbeat_ok ? "心跳: OK" : "心跳: 超时");
            lv_obj_set_style_text_color(control_widgets_.modbus_heartbeat_label,
                heartbeat_ok ? color_success_ : color_error_, 0);
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Error updating modbus heartbeat label: " << e.what() << std::endl;
        }
    }
    
    // === 更新Modbus寄存器状态信息 === - 添加安全检查
    try {
        updateModbusRegisters(modbus_registers);
    } catch (const std::exception& e) {
        std::cerr << "[LVGLInterface] Error updating modbus registers: " << e.what() << std::endl;
    }
#endif
}

void LVGLInterface::updateModbusRegisters(const core::ModbusRegisters& modbus_registers) {
#ifdef ENABLE_LVGL
    // 40001: 系统状态
    if (control_widgets_.modbus_system_status_label) {
        const char* status_names[] = {"停止", "运行", "错误", "暂停", "紧急停止"};
        uint16_t system_status = modbus_registers.system_status;
        const char* status_str = (system_status < 5) ? status_names[system_status] : "未知";
        
        std::string system_status_text = "40001 系统状态: " + std::string(status_str);
        lv_label_set_text(control_widgets_.modbus_system_status_label, system_status_text.c_str());
        
        // 根据状态设置颜色
        if (system_status == 1) { // 运行
            lv_obj_set_style_text_color(control_widgets_.modbus_system_status_label, color_success_, 0);
        } else if (system_status == 2 || system_status == 4) { // 错误或紧急停止
            lv_obj_set_style_text_color(control_widgets_.modbus_system_status_label, color_error_, 0);
        } else if (system_status == 3) { // 暂停
            lv_obj_set_style_text_color(control_widgets_.modbus_system_status_label, color_warning_, 0);
        } else { // 停止或未知
            lv_obj_set_style_text_color(control_widgets_.modbus_system_status_label, lv_color_hex(0xB0B8C1), 0);
        }
    }
    
    // 40002: PLC命令
    if (control_widgets_.modbus_plc_command_label) {
        const char* command_names[] = {"无", "进料检测", "切割准备", "切割完成", "启动送料",
                                       "停止送料", "复位系统", "保持", "刀片选择"};
        uint16_t plc_command = modbus_registers.plc_command;
        const char* command_str = (plc_command < 9) ? command_names[plc_command] : "未知命令";
        
        std::string plc_command_text = "40002 PLC命令: " + std::string(command_str);
        lv_label_set_text(control_widgets_.modbus_plc_command_label, plc_command_text.c_str());
        
        // 根据命令类型设置颜色
        if (plc_command >= 1 && plc_command <= 5) { // 正常操作命令
            lv_obj_set_style_text_color(control_widgets_.modbus_plc_command_label, color_primary_, 0);
        } else if (plc_command == 6) { // 复位系统
            lv_obj_set_style_text_color(control_widgets_.modbus_plc_command_label, color_warning_, 0);
        } else { // 无命令或其他
            lv_obj_set_style_text_color(control_widgets_.modbus_plc_command_label, lv_color_hex(0xB0B8C1), 0);
        }
    }
    
    // 40003: 坐标就绪标志
    if (control_widgets_.modbus_coord_ready_label) {
        uint16_t coord_ready = modbus_registers.coord_ready;
        const char* ready_str = coord_ready ? "是" : "否";
        
        std::string coord_ready_text = "40003 坐标就绪: " + std::string(ready_str);
        lv_label_set_text(control_widgets_.modbus_coord_ready_label, coord_ready_text.c_str());
        
        lv_obj_set_style_text_color(control_widgets_.modbus_coord_ready_label,
            coord_ready ? color_success_ : color_warning_, 0);
    }
    
    // 40004-40005: X坐标 (32位，0.1mm精度)
    if (control_widgets_.modbus_x_coordinate_label) {
        uint32_t x_coord_raw = modbus_registers.x_coordinate;
        float x_coord_mm = static_cast<float>(x_coord_raw) * 0.1f; // 0.1mm精度
        
        std::string x_coord_text = "40004 X坐标: " + std::to_string(static_cast<int>(x_coord_mm * 10) / 10.0) + "mm";
        lv_label_set_text(control_widgets_.modbus_x_coordinate_label, x_coord_text.c_str());
        
        // 根据坐标值设置颜色（假设有效范围是0-1000mm）
        if (x_coord_mm >= 0.0f && x_coord_mm <= 1000.0f) {
            lv_obj_set_style_text_color(control_widgets_.modbus_x_coordinate_label, color_primary_, 0);
        } else {
            lv_obj_set_style_text_color(control_widgets_.modbus_x_coordinate_label, color_warning_, 0);
        }
    }
    
    // 40006: 切割质量
    if (control_widgets_.modbus_cut_quality_label) {
        const char* quality_names[] = {"正常", "异常"};
        uint16_t cut_quality = modbus_registers.cut_quality;
        const char* quality_str = (cut_quality < 2) ? quality_names[cut_quality] : "未知";
        
        std::string cut_quality_text = "40006 切割质量: " + std::string(quality_str);
        lv_label_set_text(control_widgets_.modbus_cut_quality_label, cut_quality_text.c_str());
        
        lv_obj_set_style_text_color(control_widgets_.modbus_cut_quality_label,
            (cut_quality == 0) ? color_success_ : color_error_, 0);
    }
    
    // 40009: 刀片编号
    if (control_widgets_.modbus_blade_number_label) {
        const char* blade_names[] = {"无", "刀片1", "刀片2", "双刀片"};
        uint16_t blade_number = modbus_registers.blade_number;
        const char* blade_str = (blade_number < 4) ? blade_names[blade_number] : "未知刀片";
        
        std::string blade_number_text = "40009 刀片编号: " + std::string(blade_str);
        lv_label_set_text(control_widgets_.modbus_blade_number_label, blade_number_text.c_str());
        
        // 根据刀片状态设置颜色
        if (blade_number >= 1 && blade_number <= 3) { // 有刀片
            lv_obj_set_style_text_color(control_widgets_.modbus_blade_number_label, color_success_, 0);
        } else { // 无刀片或未知
            lv_obj_set_style_text_color(control_widgets_.modbus_blade_number_label, color_warning_, 0);
        }
    }
    
    // 40010: 健康状态
    if (control_widgets_.modbus_health_status_label) {
        const char* health_names[] = {"正常", "警告", "错误", "严重"};
        uint16_t health_status = modbus_registers.health_status;
        const char* health_str = (health_status < 4) ? health_names[health_status] : "未知";
        
        std::string health_status_text = "40010 健康状态: " + std::string(health_str);
        lv_label_set_text(control_widgets_.modbus_health_status_label, health_status_text.c_str());
        
        // 根据健康状态设置颜色
        if (health_status == 0) { // 正常
            lv_obj_set_style_text_color(control_widgets_.modbus_health_status_label, color_success_, 0);
        } else if (health_status == 1) { // 警告
            lv_obj_set_style_text_color(control_widgets_.modbus_health_status_label, color_warning_, 0);
        } else if (health_status >= 2) { // 错误或严重
            lv_obj_set_style_text_color(control_widgets_.modbus_health_status_label, color_error_, 0);
        } else { // 未知
            lv_obj_set_style_text_color(control_widgets_.modbus_health_status_label, lv_color_hex(0xB0B8C1), 0);
        }
    }
#endif
}

void LVGLInterface::drawDetectionResults(const core::DetectionResult& result) {
#ifdef ENABLE_LVGL
    // 在canvas上绘制检测框和标签
    // 需要根据实际的DetectionResult结构实现
#endif
}

// ==================== LVGL v9 回调函数 ====================

void input_read_cb(lv_indev_t* indev, lv_indev_data_t* data) {
#ifdef ENABLE_LVGL
    // 简单的输入读取实现
    // 在实际项目中，这里应该读取触摸设备或鼠标输入
    data->state = LV_INDEV_STATE_RELEASED;
    data->point.x = 0;
    data->point.y = 0;
#endif
}

} // namespace ui
} // namespace bamboo_cut