/**
 * @file control_panel.cpp
 * @brief LVGL control panel implementation
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include <iostream>

namespace bamboo_cut {
namespace ui {

lv_obj_t* LVGLInterface::createControlPanel(lv_obj_t* parent) {
#ifdef ENABLE_LVGL
    lv_obj_t* container = parent ? parent : main_screen_;
    
    control_panel_ = lv_obj_create(container);
    // ✅ 移除固定宽度，只用 flex_grow 控制比例
    lv_obj_set_width(control_panel_, LV_SIZE_CONTENT);  // 或者直接不设置宽度
    lv_obj_set_height(control_panel_, lv_pct(100));  // 高度填满父容器
    lv_obj_set_flex_grow(control_panel_, 1);  // ✅ 现在会生效，占1/4空间
    lv_obj_add_style(control_panel_, &style_card, 0);
    lv_obj_set_style_pad_all(control_panel_, 15, 0);  // 减少内边距以容纳更多内容
    lv_obj_set_style_radius(control_panel_, 16, 0);
    lv_obj_set_flex_flow(control_panel_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(control_panel_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(control_panel_, 12, 0);  // 减少组件间距
    
    // Control Panel Title
    lv_obj_t* title = lv_label_create(control_panel_);
    lv_label_set_text(title, LV_SYMBOL_SETTINGS " System Info");
    lv_obj_add_style(title, &style_text_title, 0);
    lv_obj_set_style_text_color(title, color_primary_, 0);
    
    // === Jetson System Monitoring Area ===
    createJetsonMonitoringSection(control_panel_);
    
    // === AI Model Monitoring Area ===
    createAIModelSection(control_panel_);
    
    // === Modbus Communication Statistics Area ===
    createModbusSection(control_panel_);
    
    // === System Version Information Area ===
    createVersionSection(control_panel_);
    
    return control_panel_;
#else
    return nullptr;
#endif
}

void LVGLInterface::createJetsonMonitoringSection(lv_obj_t* parent) {
#ifdef ENABLE_LVGL
    lv_obj_t* jetson_section = lv_obj_create(parent);
    lv_obj_set_width(jetson_section, lv_pct(100));
    lv_obj_set_style_bg_color(jetson_section, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(jetson_section, 12, 0);
    lv_obj_set_style_border_width(jetson_section, 1, 0);
    lv_obj_set_style_border_color(jetson_section, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_pad_all(jetson_section, 12, 0);
    lv_obj_clear_flag(jetson_section, LV_OBJ_FLAG_SCROLLABLE);
    // ✅ 移除flex_grow，改用内容自适应高度
    lv_obj_set_height(jetson_section, LV_SIZE_CONTENT);

    // **关键修复：设置为垂直Flex布局**
    lv_obj_set_flex_flow(jetson_section, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(jetson_section, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(jetson_section, 6, 0);  // 组件间距6px

    lv_obj_t* jetson_label = lv_label_create(jetson_section);
    lv_label_set_text(jetson_label, LV_SYMBOL_CHARGE " Jetson Orin Nano");
    lv_obj_set_style_text_color(jetson_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(jetson_label, &lv_font_montserrat_14, 0);
    
    // === CPU信息行容器 ===
    lv_obj_t* cpu_row = lv_obj_create(jetson_section);
    lv_obj_set_width(cpu_row, lv_pct(100));
    lv_obj_set_height(cpu_row, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(cpu_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(cpu_row, 0, 0);
    lv_obj_set_style_pad_all(cpu_row, 0, 0);
    lv_obj_set_flex_flow(cpu_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(cpu_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.cpu_bar = lv_bar_create(cpu_row);
    lv_obj_set_size(control_widgets_.cpu_bar, lv_pct(45), 16);
    lv_bar_set_range(control_widgets_.cpu_bar, 0, 100);
    lv_bar_set_value(control_widgets_.cpu_bar, 45, LV_ANIM_ON);
    lv_obj_set_style_bg_color(control_widgets_.cpu_bar, color_warning_, LV_PART_INDICATOR);
    
    control_widgets_.cpu_temp_label = lv_label_create(cpu_row);
    lv_label_set_text(control_widgets_.cpu_temp_label, "CPU: 62°C");
    lv_obj_set_style_text_color(control_widgets_.cpu_temp_label, color_warning_, 0);
    lv_obj_set_style_text_font(control_widgets_.cpu_temp_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.cpu_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.cpu_label, "CPU: 45%@1.9GHz");
    lv_obj_set_style_text_color(control_widgets_.cpu_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.cpu_label, &lv_font_montserrat_12, 0);
    
    // === GPU信息行容器 ===
    lv_obj_t* gpu_row = lv_obj_create(jetson_section);
    lv_obj_set_width(gpu_row, lv_pct(100));
    lv_obj_set_height(gpu_row, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(gpu_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(gpu_row, 0, 0);
    lv_obj_set_style_pad_all(gpu_row, 0, 0);
    lv_obj_set_flex_flow(gpu_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(gpu_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.gpu_bar = lv_bar_create(gpu_row);
    lv_obj_set_size(control_widgets_.gpu_bar, lv_pct(45), 16);
    lv_bar_set_range(control_widgets_.gpu_bar, 0, 100);
    lv_bar_set_value(control_widgets_.gpu_bar, 72, LV_ANIM_ON);
    lv_obj_set_style_bg_color(control_widgets_.gpu_bar, color_warning_, LV_PART_INDICATOR);
    
    control_widgets_.gpu_temp_label = lv_label_create(gpu_row);
    lv_label_set_text(control_widgets_.gpu_temp_label, "GPU: 58°C");
    lv_obj_set_style_text_color(control_widgets_.gpu_temp_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.gpu_temp_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.gpu_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.gpu_label, "GPU: 72%@624MHz");
    lv_obj_set_style_text_color(control_widgets_.gpu_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.gpu_label, &lv_font_montserrat_12, 0);
    
    // === 内存信息行容器 ===
    lv_obj_t* mem_row = lv_obj_create(jetson_section);
    lv_obj_set_width(mem_row, lv_pct(100));
    lv_obj_set_height(mem_row, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(mem_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(mem_row, 0, 0);
    lv_obj_set_style_pad_all(mem_row, 0, 0);
    lv_obj_set_flex_flow(mem_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(mem_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.mem_bar = lv_bar_create(mem_row);
    lv_obj_set_size(control_widgets_.mem_bar, lv_pct(45), 16);
    lv_bar_set_range(control_widgets_.mem_bar, 0, 100);
    lv_bar_set_value(control_widgets_.mem_bar, 58, LV_ANIM_ON);
    lv_obj_set_style_bg_color(control_widgets_.mem_bar, color_warning_, LV_PART_INDICATOR);
    
    control_widgets_.swap_usage_label = lv_label_create(mem_row);
    lv_label_set_text(control_widgets_.swap_usage_label, "SWAP: 0/3733MB");
    lv_obj_set_style_text_color(control_widgets_.swap_usage_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.swap_usage_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.mem_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.mem_label, "RAM: 3347/7467MB");
    lv_obj_set_style_text_color(control_widgets_.mem_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.mem_label, &lv_font_montserrat_12, 0);
    
    // === 温度信息行容器 ===
    lv_obj_t* temp_row = lv_obj_create(jetson_section);
    lv_obj_set_width(temp_row, lv_pct(100));
    lv_obj_set_height(temp_row, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(temp_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(temp_row, 0, 0);
    lv_obj_set_style_pad_all(temp_row, 0, 0);
    lv_obj_set_flex_flow(temp_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(temp_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.soc_temp_label = lv_label_create(temp_row);
    lv_label_set_text(control_widgets_.soc_temp_label, "SOC: 48.5°C");
    lv_obj_set_style_text_color(control_widgets_.soc_temp_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.soc_temp_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.thermal_temp_label = lv_label_create(temp_row);
    lv_label_set_text(control_widgets_.thermal_temp_label, "Thermal: 50.2°C");
    lv_obj_set_style_text_color(control_widgets_.thermal_temp_label, color_warning_, 0);
    lv_obj_set_style_text_font(control_widgets_.thermal_temp_label, &lv_font_montserrat_12, 0);
    
    // === 电源信息 ===
    control_widgets_.power_in_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.power_in_label, "VDD_IN: 5336mA/4970mW");
    lv_obj_set_style_text_color(control_widgets_.power_in_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.power_in_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.power_cpu_gpu_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.power_cpu_gpu_label, "CPU_GPU: 2617mA/2336mW");
    lv_obj_set_style_text_color(control_widgets_.power_cpu_gpu_label, color_warning_, 0);
    lv_obj_set_style_text_font(control_widgets_.power_cpu_gpu_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.power_soc_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.power_soc_label, "SOC: 1478mA/1318mW");
    lv_obj_set_style_text_color(control_widgets_.power_soc_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.power_soc_label, &lv_font_montserrat_12, 0);
    
    // === 其他系统信息行容器 ===
    lv_obj_t* sys_row = lv_obj_create(jetson_section);
    lv_obj_set_width(sys_row, lv_pct(100));
    lv_obj_set_height(sys_row, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(sys_row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(sys_row, 0, 0);
    lv_obj_set_style_pad_all(sys_row, 0, 0);
    lv_obj_set_flex_flow(sys_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(sys_row, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.emc_freq_label = lv_label_create(sys_row);
    lv_label_set_text(control_widgets_.emc_freq_label, "EMC: 13%@2133MHz");
    lv_obj_set_style_text_color(control_widgets_.emc_freq_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.emc_freq_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.vic_usage_label = lv_label_create(sys_row);
    lv_label_set_text(control_widgets_.vic_usage_label, "VIC: 0%@115MHz");
    lv_obj_set_style_text_color(control_widgets_.vic_usage_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.vic_usage_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.fan_speed_label = lv_label_create(jetson_section);
    lv_label_set_text(control_widgets_.fan_speed_label, "FAN: N/A");
    lv_obj_set_style_text_color(control_widgets_.fan_speed_label, lv_color_hex(0x8A92A1), 0);
    lv_obj_set_style_text_font(control_widgets_.fan_speed_label, &lv_font_montserrat_12, 0);
#endif
}

void LVGLInterface::createAIModelSection(lv_obj_t* parent) {
#ifdef ENABLE_LVGL
    lv_obj_t* ai_section = lv_obj_create(parent);
    lv_obj_set_width(ai_section, lv_pct(100));
    lv_obj_set_height(ai_section, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_color(ai_section, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(ai_section, 12, 0);
    lv_obj_set_style_border_width(ai_section, 1, 0);
    lv_obj_set_style_border_color(ai_section, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_pad_all(ai_section, 12, 0);
    lv_obj_clear_flag(ai_section, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(ai_section, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(ai_section, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(ai_section, 6, 0);

    // 标题
    lv_obj_t* ai_label = lv_label_create(ai_section);
    lv_label_set_text(ai_label, LV_SYMBOL_EYE_OPEN " AI Model Monitoring");
    lv_obj_set_style_text_color(ai_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(ai_label, &lv_font_montserrat_14, 0);
    
    // === 第1行：模型版本 | 推理时间 ===
    lv_obj_t* row1 = lv_obj_create(ai_section);
    lv_obj_set_width(row1, lv_pct(100));
    lv_obj_set_height(row1, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(row1, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(row1, 0, 0);
    lv_obj_set_style_pad_all(row1, 0, 0);
    lv_obj_set_flex_flow(row1, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(row1, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.ai_model_version_label = lv_label_create(row1);
    lv_label_set_text(control_widgets_.ai_model_version_label, "YOLOv8s-2.1.3");
    lv_obj_set_style_text_color(control_widgets_.ai_model_version_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.ai_model_version_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.ai_inference_time_label = lv_label_create(row1);
    lv_label_set_text(control_widgets_.ai_inference_time_label, "Inference: 18.5ms");
    lv_obj_set_style_text_color(control_widgets_.ai_inference_time_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_inference_time_label, &lv_font_montserrat_12, 0);
    
    // === 第2行：置信阈值 | 检测精度 ===
    lv_obj_t* row2 = lv_obj_create(ai_section);
    lv_obj_set_width(row2, lv_pct(100));
    lv_obj_set_height(row2, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(row2, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(row2, 0, 0);
    lv_obj_set_style_pad_all(row2, 0, 0);
    lv_obj_set_flex_flow(row2, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(row2, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.ai_confidence_threshold_label = lv_label_create(row2);
    lv_label_set_text(control_widgets_.ai_confidence_threshold_label, "Confidence: 0.75");
    lv_obj_set_style_text_color(control_widgets_.ai_confidence_threshold_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_confidence_threshold_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.ai_detection_accuracy_label = lv_label_create(row2);
    lv_label_set_text(control_widgets_.ai_detection_accuracy_label, "Accuracy: 94.2%");
    lv_obj_set_style_text_color(control_widgets_.ai_detection_accuracy_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_detection_accuracy_label, &lv_font_montserrat_12, 0);
    
    // === 第3行：总检测数 | 今日检测 ===
    lv_obj_t* row3 = lv_obj_create(ai_section);
    lv_obj_set_width(row3, lv_pct(100));
    lv_obj_set_height(row3, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(row3, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(row3, 0, 0);
    lv_obj_set_style_pad_all(row3, 0, 0);
    lv_obj_set_flex_flow(row3, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(row3, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.ai_total_detections_label = lv_label_create(row3);
    lv_label_set_text(control_widgets_.ai_total_detections_label, "Total: 1,247");
    lv_obj_set_style_text_color(control_widgets_.ai_total_detections_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.ai_total_detections_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.ai_daily_detections_label = lv_label_create(row3);
    lv_label_set_text(control_widgets_.ai_daily_detections_label, "Daily: 89");
    lv_obj_set_style_text_color(control_widgets_.ai_daily_detections_label, color_warning_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_daily_detections_label, &lv_font_montserrat_12, 0);
    
    // === 当前竹子检测状态（子标题） ===
    lv_obj_t* bamboo_status_label = lv_label_create(ai_section);
    lv_label_set_text(bamboo_status_label, "Current Bamboo Detection:");
    lv_obj_set_style_text_color(bamboo_status_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(bamboo_status_label, &lv_font_montserrat_12, 0);
    
    // === 第4行：直径 | 长度 ===
    lv_obj_t* row4 = lv_obj_create(ai_section);
    lv_obj_set_width(row4, lv_pct(100));
    lv_obj_set_height(row4, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(row4, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(row4, 0, 0);
    lv_obj_set_style_pad_all(row4, 0, 0);
    lv_obj_set_flex_flow(row4, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(row4, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.bamboo_diameter_label = lv_label_create(row4);
    lv_label_set_text(control_widgets_.bamboo_diameter_label, "Ø45.2mm");
    lv_obj_set_style_text_color(control_widgets_.bamboo_diameter_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_diameter_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.bamboo_length_label = lv_label_create(row4);
    lv_label_set_text(control_widgets_.bamboo_length_label, "L2850mm");
    lv_obj_set_style_text_color(control_widgets_.bamboo_length_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_length_label, &lv_font_montserrat_12, 0);
    
    // === 第5行：置信度 | 检测耗时 ===
    lv_obj_t* row5 = lv_obj_create(ai_section);
    lv_obj_set_width(row5, lv_pct(100));
    lv_obj_set_height(row5, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(row5, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(row5, 0, 0);
    lv_obj_set_style_pad_all(row5, 0, 0);
    lv_obj_set_flex_flow(row5, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(row5, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.bamboo_confidence_label = lv_label_create(row5);
    lv_label_set_text(control_widgets_.bamboo_confidence_label, "Confidence: 0.92");
    lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_confidence_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.bamboo_detection_time_label = lv_label_create(row5);
    lv_label_set_text(control_widgets_.bamboo_detection_time_label, "Time: 16.8ms");
    lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_detection_time_label, &lv_font_montserrat_12, 0);
    
    // 预切位置（单独一行，因为内容较长）
    control_widgets_.bamboo_cut_positions_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.bamboo_cut_positions_label, "Cut Positions: [950, 1900]mm");
    lv_obj_set_style_text_color(control_widgets_.bamboo_cut_positions_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_cut_positions_label, &lv_font_montserrat_12, 0);
    
    // === 摄像头状态（子标题） ===
    lv_obj_t* camera_status_label = lv_label_create(ai_section);
    lv_label_set_text(camera_status_label, "Camera Status:");
    lv_obj_set_style_text_color(camera_status_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(camera_status_label, &lv_font_montserrat_12, 0);
    
    // === 摄像头1行组 ===
    control_widgets_.camera1_status_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera1_status_label, "CAM-1: Online✓");
    lv_obj_set_style_text_color(control_widgets_.camera1_status_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_status_label, &lv_font_montserrat_12, 0);
    
    // 摄像头1详细信息行：帧率 | 分辨率
    lv_obj_t* cam1_row1 = lv_obj_create(ai_section);
    lv_obj_set_width(cam1_row1, lv_pct(100));
    lv_obj_set_height(cam1_row1, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(cam1_row1, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(cam1_row1, 0, 0);
    lv_obj_set_style_pad_all(cam1_row1, 0, 0);
    lv_obj_set_flex_flow(cam1_row1, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(cam1_row1, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.camera1_fps_label = lv_label_create(cam1_row1);
    lv_label_set_text(control_widgets_.camera1_fps_label, "  30fps");
    lv_obj_set_style_text_color(control_widgets_.camera1_fps_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_fps_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.camera1_resolution_label = lv_label_create(cam1_row1);
    lv_label_set_text(control_widgets_.camera1_resolution_label, "1920x1080");
    lv_obj_set_style_text_color(control_widgets_.camera1_resolution_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_resolution_label, &lv_font_montserrat_12, 0);
    
    // 摄像头1详细信息行：曝光 | 光照
    lv_obj_t* cam1_row2 = lv_obj_create(ai_section);
    lv_obj_set_width(cam1_row2, lv_pct(100));
    lv_obj_set_height(cam1_row2, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(cam1_row2, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(cam1_row2, 0, 0);
    lv_obj_set_style_pad_all(cam1_row2, 0, 0);
    lv_obj_set_flex_flow(cam1_row2, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(cam1_row2, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.camera1_exposure_label = lv_label_create(cam1_row2);
    lv_label_set_text(control_widgets_.camera1_exposure_label, "  Auto Exposure");
    lv_obj_set_style_text_color(control_widgets_.camera1_exposure_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_exposure_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.camera1_lighting_label = lv_label_create(cam1_row2);
    lv_label_set_text(control_widgets_.camera1_lighting_label, "Good Lighting");
    lv_obj_set_style_text_color(control_widgets_.camera1_lighting_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_lighting_label, &lv_font_montserrat_12, 0);
    
    // === 摄像头2行组（结构同上） ===
    control_widgets_.camera2_status_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera2_status_label, "CAM-2: Online✓");
    lv_obj_set_style_text_color(control_widgets_.camera2_status_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera2_status_label, &lv_font_montserrat_12, 0);
    
    lv_obj_t* cam2_row1 = lv_obj_create(ai_section);
    lv_obj_set_width(cam2_row1, lv_pct(100));
    lv_obj_set_height(cam2_row1, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(cam2_row1, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(cam2_row1, 0, 0);
    lv_obj_set_style_pad_all(cam2_row1, 0, 0);
    lv_obj_set_flex_flow(cam2_row1, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(cam2_row1, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.camera2_fps_label = lv_label_create(cam2_row1);
    lv_label_set_text(control_widgets_.camera2_fps_label, "  30fps");
    lv_obj_set_style_text_color(control_widgets_.camera2_fps_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.camera2_fps_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.camera2_resolution_label = lv_label_create(cam2_row1);
    lv_label_set_text(control_widgets_.camera2_resolution_label, "1920x1080");
    lv_obj_set_style_text_color(control_widgets_.camera2_resolution_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.camera2_resolution_label, &lv_font_montserrat_12, 0);
    
    lv_obj_t* cam2_row2 = lv_obj_create(ai_section);
    lv_obj_set_width(cam2_row2, lv_pct(100));
    lv_obj_set_height(cam2_row2, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(cam2_row2, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(cam2_row2, 0, 0);
    lv_obj_set_style_pad_all(cam2_row2, 0, 0);
    lv_obj_set_flex_flow(cam2_row2, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(cam2_row2, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.camera2_exposure_label = lv_label_create(cam2_row2);
    lv_label_set_text(control_widgets_.camera2_exposure_label, "  Auto Exposure");
    lv_obj_set_style_text_color(control_widgets_.camera2_exposure_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera2_exposure_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.camera2_lighting_label = lv_label_create(cam2_row2);
    lv_label_set_text(control_widgets_.camera2_lighting_label, "Good Lighting");
    lv_obj_set_style_text_color(control_widgets_.camera2_lighting_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera2_lighting_label, &lv_font_montserrat_12, 0);
#endif
}

void LVGLInterface::createModbusSection(lv_obj_t* parent) {
#ifdef ENABLE_LVGL
    lv_obj_t* modbus_section = lv_obj_create(parent);
    lv_obj_set_width(modbus_section, lv_pct(100));
    lv_obj_set_height(modbus_section, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_color(modbus_section, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(modbus_section, 12, 0);
    lv_obj_set_style_border_width(modbus_section, 1, 0);
    lv_obj_set_style_border_color(modbus_section, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_pad_all(modbus_section, 12, 0);
    lv_obj_clear_flag(modbus_section, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(modbus_section, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(modbus_section, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(modbus_section, 6, 0);

    // 标题
    lv_obj_t* modbus_label = lv_label_create(modbus_section);
    lv_label_set_text(modbus_label, LV_SYMBOL_WIFI " Modbus Communication");
    lv_obj_set_style_text_color(modbus_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(modbus_label, &lv_font_montserrat_14, 0);
    
    // === 第1行：PLC连接状态 | 地址 ===
    lv_obj_t* row1 = lv_obj_create(modbus_section);
    lv_obj_set_width(row1, lv_pct(100));
    lv_obj_set_height(row1, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(row1, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(row1, 0, 0);
    lv_obj_set_style_pad_all(row1, 0, 0);
    lv_obj_set_flex_flow(row1, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(row1, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.modbus_connection_label = lv_label_create(row1);
    lv_label_set_text(control_widgets_.modbus_connection_label, "Online✓");
    lv_obj_set_style_text_color(control_widgets_.modbus_connection_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_connection_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.modbus_address_label = lv_label_create(row1);
    lv_label_set_text(control_widgets_.modbus_address_label, "192.168.1.100:502");
    lv_obj_set_style_text_color(control_widgets_.modbus_address_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_address_label, &lv_font_montserrat_12, 0);
    
    // === 第2行：通讯延迟 | 最后通讯时间 ===
    lv_obj_t* row2 = lv_obj_create(modbus_section);
    lv_obj_set_width(row2, lv_pct(100));
    lv_obj_set_height(row2, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(row2, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(row2, 0, 0);
    lv_obj_set_style_pad_all(row2, 0, 0);
    lv_obj_set_flex_flow(row2, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(row2, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.modbus_latency_label = lv_label_create(row2);
    lv_label_set_text(control_widgets_.modbus_latency_label, "Latency: 8ms");
    lv_obj_set_style_text_color(control_widgets_.modbus_latency_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_latency_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.modbus_last_success_label = lv_label_create(row2);
    lv_label_set_text(control_widgets_.modbus_last_success_label, "2s ago");
    lv_obj_set_style_text_color(control_widgets_.modbus_last_success_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_last_success_label, &lv_font_montserrat_12, 0);
    
    // === 第3行：错误计数 | 今日消息数 ===
    lv_obj_t* row3 = lv_obj_create(modbus_section);
    lv_obj_set_width(row3, lv_pct(100));
    lv_obj_set_height(row3, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(row3, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(row3, 0, 0);
    lv_obj_set_style_pad_all(row3, 0, 0);
    lv_obj_set_flex_flow(row3, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(row3, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.modbus_error_count_label = lv_label_create(row3);
    lv_label_set_text(control_widgets_.modbus_error_count_label, "Errors: 0");
    lv_obj_set_style_text_color(control_widgets_.modbus_error_count_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_error_count_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.modbus_message_count_label = lv_label_create(row3);
    lv_label_set_text(control_widgets_.modbus_message_count_label, "Daily: 1,523");
    lv_obj_set_style_text_color(control_widgets_.modbus_message_count_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_message_count_label, &lv_font_montserrat_12, 0);
    
    // 验证所有Modbus控件是否正确初始化
    bool all_modbus_widgets_valid = true;
    
    if (!control_widgets_.modbus_connection_label) {
        std::cerr << "[Control Panel] Modbus连接状态标签初始化失败" << std::endl;
        all_modbus_widgets_valid = false;
    }
    if (!control_widgets_.modbus_address_label) {
        std::cerr << "[Control Panel] Modbus地址标签初始化失败" << std::endl;
        all_modbus_widgets_valid = false;
    }
    if (!control_widgets_.modbus_latency_label) {
        std::cerr << "[Control Panel] Modbus延迟标签初始化失败" << std::endl;
        all_modbus_widgets_valid = false;
    }
    if (!control_widgets_.modbus_last_success_label) {
        std::cerr << "[Control Panel] Modbus最后成功标签初始化失败" << std::endl;
        all_modbus_widgets_valid = false;
    }
    if (!control_widgets_.modbus_error_count_label) {
        std::cerr << "[Control Panel] Modbus错误计数标签初始化失败" << std::endl;
        all_modbus_widgets_valid = false;
    }
    if (!control_widgets_.modbus_message_count_label) {
        std::cerr << "[Control Panel] Modbus消息计数标签初始化失败" << std::endl;
        all_modbus_widgets_valid = false;
    }
    
    if (all_modbus_widgets_valid) {
        std::cout << "[Control Panel] 所有Modbus控件初始化成功" << std::endl;
    } else {
        std::cerr << "[Control Panel] 部分Modbus控件初始化失败" << std::endl;
    }
#endif
}

void LVGLInterface::createVersionSection(lv_obj_t* parent) {
#ifdef ENABLE_LVGL
    lv_obj_t* version_section = lv_obj_create(parent);
    lv_obj_set_width(version_section, lv_pct(100));
    lv_obj_set_height(version_section, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_color(version_section, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(version_section, 12, 0);
    lv_obj_set_style_border_width(version_section, 1, 0);
    lv_obj_set_style_border_color(version_section, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_pad_all(version_section, 12, 0);
    lv_obj_clear_flag(version_section, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(version_section, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(version_section, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(version_section, 6, 0);

    // 标题
    lv_obj_t* version_label = lv_label_create(version_section);
    lv_label_set_text(version_label, LV_SYMBOL_SETTINGS " System Version Info");
    lv_obj_set_style_text_color(version_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(version_label, &lv_font_montserrat_14, 0);
    
    // === 第1行：系统版本 | LVGL版本 ===
    lv_obj_t* row1 = lv_obj_create(version_section);
    lv_obj_set_width(row1, lv_pct(100));
    lv_obj_set_height(row1, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(row1, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(row1, 0, 0);
    lv_obj_set_style_pad_all(row1, 0, 0);
    lv_obj_set_flex_flow(row1, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(row1, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.system_version_label = lv_label_create(row1);
    lv_label_set_text(control_widgets_.system_version_label, "v2.1.3-beta");
    lv_obj_set_style_text_color(control_widgets_.system_version_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.system_version_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.lvgl_version_label = lv_label_create(row1);
    lv_label_set_text(control_widgets_.lvgl_version_label, "LVGL v9.1.0");
    lv_obj_set_style_text_color(control_widgets_.lvgl_version_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.lvgl_version_label, &lv_font_montserrat_12, 0);
    
    // === 第2行：编译时间（单独一行，因为内容较长）===
    control_widgets_.build_time_label = lv_label_create(version_section);
    lv_label_set_text(control_widgets_.build_time_label, "Build: 2024-12-21 14:30");
    lv_obj_set_style_text_color(control_widgets_.build_time_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.build_time_label, &lv_font_montserrat_12, 0);
    
    // === 第3行：Git提交 | JetPack版本 ===
    lv_obj_t* row3 = lv_obj_create(version_section);
    lv_obj_set_width(row3, lv_pct(100));
    lv_obj_set_height(row3, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(row3, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(row3, 0, 0);
    lv_obj_set_style_pad_all(row3, 0, 0);
    lv_obj_set_flex_flow(row3, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(row3, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    control_widgets_.git_commit_label = lv_label_create(row3);
    lv_label_set_text(control_widgets_.git_commit_label, "Git: 2939c00");
    lv_obj_set_style_text_color(control_widgets_.git_commit_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.git_commit_label, &lv_font_montserrat_12, 0);
    
    status_widgets_.jetpack_version = lv_label_create(row3);
    lv_label_set_text(status_widgets_.jetpack_version, "JetPack 5.1.2");
    lv_obj_set_style_text_color(status_widgets_.jetpack_version, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(status_widgets_.jetpack_version, &lv_font_montserrat_12, 0);
    
    // === 第4行：CUDA版本（单独一行）===
    status_widgets_.cuda_version = lv_label_create(version_section);
    lv_label_set_text(status_widgets_.cuda_version, "CUDA: 11.4");
    lv_obj_set_style_text_color(status_widgets_.cuda_version, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(status_widgets_.cuda_version, &lv_font_montserrat_12, 0);
#endif
}
} // namespace ui
} // namespace bamboo_cut