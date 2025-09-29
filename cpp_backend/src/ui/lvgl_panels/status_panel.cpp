/**
 * @file status_panel.cpp
 * @brief LVGL status panel implementation
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include <iostream>

namespace bamboo_cut {
namespace ui {

lv_obj_t* LVGLInterface::createStatusPanel() {
#ifdef ENABLE_LVGL
    status_panel_ = lv_obj_create(main_screen_);
    lv_obj_set_size(status_panel_, lv_pct(96), lv_pct(20));  // 底部状态面板
    lv_obj_align(status_panel_, LV_ALIGN_BOTTOM_MID, 0, -100);
    lv_obj_add_style(status_panel_, &style_card, 0);
    lv_obj_set_style_pad_all(status_panel_, 16, 0);
    lv_obj_set_style_radius(status_panel_, 12, 0);
    lv_obj_clear_flag(status_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 使用flex布局分为左右两部分
    lv_obj_set_flex_flow(status_panel_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(status_panel_, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(status_panel_, 20, 0);
    
    // === 左侧：系统指标3列网格 ===
    lv_obj_t* metrics_container = lv_obj_create(status_panel_);
    lv_obj_set_size(metrics_container, lv_pct(70), lv_pct(100));
    lv_obj_set_style_bg_opa(metrics_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(metrics_container, 0, 0);
    lv_obj_set_style_pad_all(metrics_container, 0, 0);
    lv_obj_clear_flag(metrics_container, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* metrics_title = lv_label_create(metrics_container);
    lv_label_set_text(metrics_title, LV_SYMBOL_SETTINGS " System Metrics");
    lv_obj_set_style_text_color(metrics_title, color_primary_, 0);
    lv_obj_set_style_text_font(metrics_title, &lv_font_montserrat_14, 0);
    lv_obj_align(metrics_title, LV_ALIGN_TOP_LEFT, 0, 0);
    
    // 3 Column Grid Container
    lv_obj_t* grid_container = lv_obj_create(metrics_container);
    lv_obj_set_size(grid_container, lv_pct(100), 80);
    lv_obj_align(grid_container, LV_ALIGN_TOP_LEFT, 0, 25);
    lv_obj_set_style_bg_opa(grid_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(grid_container, 0, 0);
    lv_obj_set_style_pad_all(grid_container, 0, 0);
    lv_obj_set_flex_flow(grid_container, LV_FLEX_FLOW_ROW_WRAP);
    lv_obj_set_flex_align(grid_container, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_clear_flag(grid_container, LV_OBJ_FLAG_SCROLLABLE);
    
    // 12 System Metrics Data
    const char* metric_names[] = {
        "CPU Freq", "GPU Freq", "MEM Freq",
        "CPU Temp", "GPU Temp", "SYS Temp",
        "Fan Speed", "Power V", "Power A",
        "Net Ping", "Disk Read", "Disk Write"
    };
    const char* metric_values[] = {
        "1.9GHz", "1.3GHz", "1600MHz",
        "62°C", "58°C", "45°C",
        "3200RPM", "19.2V", "2.1A",
        "12ms", "450MB/s", "380MB/s"
    };
    const lv_color_t metric_colors[] = {
        color_success_, color_warning_, color_primary_,
        color_warning_, color_success_, color_success_,
        color_primary_, color_success_, color_success_,
        color_success_, color_primary_, color_primary_
    };
    
    for(int i = 0; i < 12; i++) {
        lv_obj_t* metric_item = lv_obj_create(grid_container);
        lv_obj_set_size(metric_item, lv_pct(32), 35);  // 3列布局，每列32%宽度
        lv_obj_set_style_bg_color(metric_item, lv_color_hex(0x0F1419), 0);
        lv_obj_set_style_radius(metric_item, 6, 0);
        lv_obj_set_style_border_width(metric_item, 1, 0);
        lv_obj_set_style_border_color(metric_item, lv_color_hex(0x2A3441), 0);
        lv_obj_set_style_pad_all(metric_item, 8, 0);
        lv_obj_clear_flag(metric_item, LV_OBJ_FLAG_SCROLLABLE);
        
        lv_obj_t* name_label = lv_label_create(metric_item);
        lv_label_set_text(name_label, metric_names[i]);
        lv_obj_set_style_text_color(name_label, lv_color_hex(0x8A92A1), 0);
        lv_obj_set_style_text_font(name_label, &lv_font_montserrat_12, 0);
        lv_obj_align(name_label, LV_ALIGN_TOP_LEFT, 0, 0);
        
        lv_obj_t* value_label = lv_label_create(metric_item);
        lv_label_set_text(value_label, metric_values[i]);
        lv_obj_set_style_text_color(value_label, metric_colors[i], 0);
        lv_obj_set_style_text_font(value_label, &lv_font_montserrat_14, 0);
        lv_obj_align(value_label, LV_ALIGN_BOTTOM_RIGHT, 0, 0);
        
        status_widgets_.metric_labels.push_back(value_label);
    }
    
    // === 右侧：版本信息 ===
    lv_obj_t* version_container = lv_obj_create(status_panel_);
    lv_obj_set_size(version_container, lv_pct(28), lv_pct(100));
    lv_obj_set_style_bg_color(version_container, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(version_container, 8, 0);
    lv_obj_set_style_border_width(version_container, 1, 0);
    lv_obj_set_style_border_color(version_container, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_pad_all(version_container, 12, 0);
    lv_obj_clear_flag(version_container, LV_OBJ_FLAG_SCROLLABLE);
    
    lv_obj_t* version_title = lv_label_create(version_container);
    lv_label_set_text(version_title, LV_SYMBOL_LIST " System Version");
    lv_obj_set_style_text_color(version_title, lv_color_hex(0xE6A055), 0);
    lv_obj_set_style_text_font(version_title, &lv_font_montserrat_14, 0);
    lv_obj_align(version_title, LV_ALIGN_TOP_LEFT, 0, 0);
    
    // JetPack版本
    status_widgets_.jetpack_version = lv_label_create(version_container);
    lv_label_set_text(status_widgets_.jetpack_version, "JetPack: 5.1.2");
    lv_obj_set_style_text_color(status_widgets_.jetpack_version, lv_color_white(), 0);
    lv_obj_set_style_text_font(status_widgets_.jetpack_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.jetpack_version, LV_ALIGN_TOP_LEFT, 0, 22);
    
    // CUDA版本
    status_widgets_.cuda_version = lv_label_create(version_container);
    lv_label_set_text(status_widgets_.cuda_version, "CUDA: 11.4.315");
    lv_obj_set_style_text_color(status_widgets_.cuda_version, color_success_, 0);
    lv_obj_set_style_text_font(status_widgets_.cuda_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.cuda_version, LV_ALIGN_TOP_LEFT, 0, 40);
    
    // TensorRT版本
    status_widgets_.tensorrt_version = lv_label_create(version_container);
    lv_label_set_text(status_widgets_.tensorrt_version, "TensorRT: 8.5.2");
    lv_obj_set_style_text_color(status_widgets_.tensorrt_version, color_primary_, 0);
    lv_obj_set_style_text_font(status_widgets_.tensorrt_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.tensorrt_version, LV_ALIGN_TOP_LEFT, 0, 58);
    
    // OpenCV版本
    status_widgets_.opencv_version = lv_label_create(version_container);
    lv_label_set_text(status_widgets_.opencv_version, "OpenCV: 4.8.0");
    lv_obj_set_style_text_color(status_widgets_.opencv_version, color_warning_, 0);
    lv_obj_set_style_text_font(status_widgets_.opencv_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.opencv_version, LV_ALIGN_TOP_LEFT, 0, 76);
    
    // Ubuntu版本
    status_widgets_.ubuntu_version = lv_label_create(version_container);
    lv_label_set_text(status_widgets_.ubuntu_version, "Ubuntu: 20.04.6");
    lv_obj_set_style_text_color(status_widgets_.ubuntu_version, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(status_widgets_.ubuntu_version, &lv_font_montserrat_12, 0);
    lv_obj_align(status_widgets_.ubuntu_version, LV_ALIGN_TOP_LEFT, 0, 94);
    
    return status_panel_;
#else
    return nullptr;
#endif
}

} // namespace ui
} // namespace bamboo_cut