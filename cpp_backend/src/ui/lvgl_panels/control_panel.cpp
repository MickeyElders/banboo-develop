/**
 * @file control_panel.cpp
 * @brief LVGL控制面板实现
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

    lv_obj_t* ai_label = lv_label_create(ai_section);
    lv_label_set_text(ai_label, LV_SYMBOL_EYE_OPEN " AI模型监控");
    lv_obj_set_style_text_color(ai_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(ai_label, &lv_font_montserrat_14, 0);
    
    // AI模型版本
    control_widgets_.ai_model_version_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_model_version_label, "模型版本: YOLOv8s-2.1.3");
    lv_obj_set_style_text_color(control_widgets_.ai_model_version_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(control_widgets_.ai_model_version_label, &lv_font_montserrat_12, 0);
    
    // 推理时间
    control_widgets_.ai_inference_time_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_inference_time_label, "推理时间: 18.5ms");
    lv_obj_set_style_text_color(control_widgets_.ai_inference_time_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_inference_time_label, &lv_font_montserrat_12, 0);
    
    // 置信阈值
    control_widgets_.ai_confidence_threshold_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_confidence_threshold_label, "置信阈值: 0.75");
    lv_obj_set_style_text_color(control_widgets_.ai_confidence_threshold_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_confidence_threshold_label, &lv_font_montserrat_12, 0);
    
    // 检测精度
    control_widgets_.ai_detection_accuracy_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_detection_accuracy_label, "检测精度: 94.2%");
    lv_obj_set_style_text_color(control_widgets_.ai_detection_accuracy_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_detection_accuracy_label, &lv_font_montserrat_12, 0);
    
    // 总检测数
    control_widgets_.ai_total_detections_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_total_detections_label, "总检测数: 1,247");
    lv_obj_set_style_text_color(control_widgets_.ai_total_detections_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.ai_total_detections_label, &lv_font_montserrat_12, 0);
    
    // 今日检测数
    control_widgets_.ai_daily_detections_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.ai_daily_detections_label, "今日检测: 89");
    lv_obj_set_style_text_color(control_widgets_.ai_daily_detections_label, color_warning_, 0);
    lv_obj_set_style_text_font(control_widgets_.ai_daily_detections_label, &lv_font_montserrat_12, 0);
    
    // === 当前竹子检测状态 ===
    lv_obj_t* bamboo_status_label = lv_label_create(ai_section);
    lv_label_set_text(bamboo_status_label, "当前竹子检测:");
    lv_obj_set_style_text_color(bamboo_status_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(bamboo_status_label, &lv_font_montserrat_12, 0);
    
    // 竹子直径
    control_widgets_.bamboo_diameter_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.bamboo_diameter_label, "- 直径：45.2mm");
    lv_obj_set_style_text_color(control_widgets_.bamboo_diameter_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_diameter_label, &lv_font_montserrat_12, 0);
    
    // 竹子长度
    control_widgets_.bamboo_length_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.bamboo_length_label, "- 长度：2850mm");
    lv_obj_set_style_text_color(control_widgets_.bamboo_length_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_length_label, &lv_font_montserrat_12, 0);
    
    // 预切位置
    control_widgets_.bamboo_cut_positions_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.bamboo_cut_positions_label, "- 预切位置：[950mm, 1900mm]");
    lv_obj_set_style_text_color(control_widgets_.bamboo_cut_positions_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_cut_positions_label, &lv_font_montserrat_12, 0);
    
    // 检测置信度
    control_widgets_.bamboo_confidence_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.bamboo_confidence_label, "- 置信度：0.92");
    lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_confidence_label, &lv_font_montserrat_12, 0);
    
    // 检测耗时
    control_widgets_.bamboo_detection_time_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.bamboo_detection_time_label, "- 检测耗时：16.8ms");
    lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.bamboo_detection_time_label, &lv_font_montserrat_12, 0);
    
    // === 摄像头状态 ===
    lv_obj_t* camera_status_label = lv_label_create(ai_section);
    lv_label_set_text(camera_status_label, "摄像头状态:");
    lv_obj_set_style_text_color(camera_status_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(camera_status_label, &lv_font_montserrat_12, 0);
    
    // 摄像头1状态
    control_widgets_.camera1_status_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera1_status_label, "摄像头-1：在线 ✓");
    lv_obj_set_style_text_color(control_widgets_.camera1_status_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_status_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.camera1_fps_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera1_fps_label, "  帧率：30 FPS");
    lv_obj_set_style_text_color(control_widgets_.camera1_fps_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_fps_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.camera1_resolution_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera1_resolution_label, "  分辨率：1920x1080");
    lv_obj_set_style_text_color(control_widgets_.camera1_resolution_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_resolution_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.camera1_exposure_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera1_exposure_label, "  曝光：自动");
    lv_obj_set_style_text_color(control_widgets_.camera1_exposure_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_exposure_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.camera1_lighting_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera1_lighting_label, "  光照评分：良好");
    lv_obj_set_style_text_color(control_widgets_.camera1_lighting_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera1_lighting_label, &lv_font_montserrat_12, 0);
    
    // 摄像头2状态
    control_widgets_.camera2_status_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera2_status_label, "摄像头-2：在线 ✓");
    lv_obj_set_style_text_color(control_widgets_.camera2_status_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera2_status_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.camera2_fps_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera2_fps_label, "  帧率：30 FPS");
    lv_obj_set_style_text_color(control_widgets_.camera2_fps_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.camera2_fps_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.camera2_resolution_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera2_resolution_label, "  分辨率：1920x1080");
    lv_obj_set_style_text_color(control_widgets_.camera2_resolution_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.camera2_resolution_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.camera2_exposure_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera2_exposure_label, "  曝光：自动");
    lv_obj_set_style_text_color(control_widgets_.camera2_exposure_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.camera2_exposure_label, &lv_font_montserrat_12, 0);
    
    control_widgets_.camera2_lighting_label = lv_label_create(ai_section);
    lv_label_set_text(control_widgets_.camera2_lighting_label, "  光照评分：良好");
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

    lv_obj_t* modbus_label = lv_label_create(modbus_section);
    lv_label_set_text(modbus_label, LV_SYMBOL_WIFI " Modbus通讯状态");
    lv_obj_set_style_text_color(modbus_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(modbus_label, &lv_font_montserrat_14, 0);
    
    // PLC连接状态
    control_widgets_.modbus_connection_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_connection_label, "PLC连接: 在线 ✓");
    lv_obj_set_style_text_color(control_widgets_.modbus_connection_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_connection_label, &lv_font_montserrat_12, 0);
    
    // 连接地址和端口
    control_widgets_.modbus_address_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_address_label, "地址: 192.168.1.100:502");
    lv_obj_set_style_text_color(control_widgets_.modbus_address_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_address_label, &lv_font_montserrat_12, 0);
    
    // 通讯延迟
    control_widgets_.modbus_latency_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_latency_label, "通讯延迟: 8ms");
    lv_obj_set_style_text_color(control_widgets_.modbus_latency_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_latency_label, &lv_font_montserrat_12, 0);
    
    // 最后成功通讯时间
    control_widgets_.modbus_last_success_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_last_success_label, "最后通讯: 2秒前");
    lv_obj_set_style_text_color(control_widgets_.modbus_last_success_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_last_success_label, &lv_font_montserrat_12, 0);
    
    // 错误计数
    control_widgets_.modbus_error_count_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_error_count_label, "错误计数: 0");
    lv_obj_set_style_text_color(control_widgets_.modbus_error_count_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_error_count_label, &lv_font_montserrat_12, 0);
    
    // 今日消息数
    control_widgets_.modbus_message_count_label = lv_label_create(modbus_section);
    lv_label_set_text(control_widgets_.modbus_message_count_label, "今日消息: 1,523");
    lv_obj_set_style_text_color(control_widgets_.modbus_message_count_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.modbus_message_count_label, &lv_font_montserrat_12, 0);
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

    lv_obj_t* version_label = lv_label_create(version_section);
    lv_label_set_text(version_label, LV_SYMBOL_SETTINGS " 系统版本信息");
    lv_obj_set_style_text_color(version_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(version_label, &lv_font_montserrat_14, 0);
    
    // 系统版本
    control_widgets_.system_version_label = lv_label_create(version_section);
    lv_label_set_text(control_widgets_.system_version_label, "系统版本: v2.1.3-beta");
    lv_obj_set_style_text_color(control_widgets_.system_version_label, color_primary_, 0);
    lv_obj_set_style_text_font(control_widgets_.system_version_label, &lv_font_montserrat_12, 0);
    
    // LVGL版本
    control_widgets_.lvgl_version_label = lv_label_create(version_section);
    lv_label_set_text(control_widgets_.lvgl_version_label, "LVGL版本: v9.1.0");
    lv_obj_set_style_text_color(control_widgets_.lvgl_version_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.lvgl_version_label, &lv_font_montserrat_12, 0);
    
    // 编译时间
    control_widgets_.build_time_label = lv_label_create(version_section);
    lv_label_set_text(control_widgets_.build_time_label, "编译时间: 2024-12-21 14:30");
    lv_obj_set_style_text_color(control_widgets_.build_time_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.build_time_label, &lv_font_montserrat_12, 0);
    
    // Git提交
    control_widgets_.git_commit_label = lv_label_create(version_section);
    lv_label_set_text(control_widgets_.git_commit_label, "Git提交: 2939c00");
    lv_obj_set_style_text_color(control_widgets_.git_commit_label, color_success_, 0);
    lv_obj_set_style_text_font(control_widgets_.git_commit_label, &lv_font_montserrat_12, 0);
    
    // JetPack版本
    control_widgets_.jetpack_version_label = lv_label_create(version_section);
    lv_label_set_text(control_widgets_.jetpack_version_label, "JetPack: 5.1.2");
    lv_obj_set_style_text_color(control_widgets_.jetpack_version_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.jetpack_version_label, &lv_font_montserrat_12, 0);
    
    // CUDA版本
    control_widgets_.cuda_version_label = lv_label_create(version_section);
    lv_label_set_text(control_widgets_.cuda_version_label, "CUDA: 11.4");
    lv_obj_set_style_text_color(control_widgets_.cuda_version_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(control_widgets_.cuda_version_label, &lv_font_montserrat_12, 0);
#endif
}

} // namespace ui
} // namespace bamboo_cut