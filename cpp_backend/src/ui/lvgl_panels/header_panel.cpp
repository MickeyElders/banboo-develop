/**
 * @file header_panel.cpp
 * @brief LVGL header panel implementation
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include <iostream>

namespace bamboo_cut {
namespace ui {

lv_obj_t* LVGLInterface::createHeaderPanel() {
#ifdef ENABLE_LVGL
    header_panel_ = lv_obj_create(main_screen_);
    lv_obj_set_width(header_panel_, lv_pct(100));
    lv_obj_set_height(header_panel_, 70);
    // 移除 lv_obj_align，使用父容器的Flex布局控制位置
    
    // 简洁的背景样式
    lv_obj_set_style_bg_color(header_panel_, color_surface_, 0);
    lv_obj_set_style_bg_opa(header_panel_, LV_OPA_90, 0);
    lv_obj_set_style_radius(header_panel_, 0, 0);
    lv_obj_set_style_border_width(header_panel_, 1, 0);
    lv_obj_set_style_border_side(header_panel_, LV_BORDER_SIDE_BOTTOM, 0);
    lv_obj_set_style_border_color(header_panel_, lv_color_hex(0x3A4048), 0);
    lv_obj_set_style_border_opa(header_panel_, LV_OPA_50, 0);
    lv_obj_set_style_pad_all(header_panel_, 12, 0);
    lv_obj_clear_flag(header_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // Flex布局
    lv_obj_set_flex_flow(header_panel_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(header_panel_, LV_FLEX_ALIGN_SPACE_BETWEEN,
                          LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // System Title
    header_widgets_.system_title = lv_label_create(header_panel_);
    lv_label_set_text(header_widgets_.system_title,
        LV_SYMBOL_IMAGE " Bamboo Intelligent Cutting");
    lv_obj_add_style(header_widgets_.system_title, &style_text_title, 0);
    lv_obj_set_style_text_color(header_widgets_.system_title, color_primary_, 0);
    
    // Compact workflow indicator container
    lv_obj_t* workflow_container = lv_obj_create(header_panel_);
    lv_obj_set_size(workflow_container, 350, 50);  // More compact
    lv_obj_set_style_bg_opa(workflow_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(workflow_container, 0, 0);
    lv_obj_set_style_pad_all(workflow_container, 8, 0);
    lv_obj_set_flex_flow(workflow_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(workflow_container, LV_FLEX_ALIGN_SPACE_AROUND,
                          LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_clear_flag(workflow_container, LV_OBJ_FLAG_SCROLLABLE);
    
    const char* steps[] = {"Feed", "Detect", "Transfer", "Prepare", "Cut"};
    const char* icons[] = {
        LV_SYMBOL_UPLOAD, LV_SYMBOL_EYE_OPEN, LV_SYMBOL_GPS,
        LV_SYMBOL_SETTINGS, LV_SYMBOL_CUT
    };
    
    for(int i = 0; i < 5; i++) {
        lv_obj_t* step = lv_obj_create(workflow_container);
        lv_obj_set_size(step, 40, 40);  // 更小更紧凑
        lv_obj_set_style_radius(step, 20, 0);
        lv_obj_set_style_border_width(step, 1, 0);
        lv_obj_set_style_pad_all(step, 0, 0);
        lv_obj_clear_flag(step, LV_OBJ_FLAG_SCROLLABLE);
        
        bool is_active = (i == (int)current_step_ - 1);
        bool is_completed = (i < (int)current_step_ - 1);
        
        if(is_active) {
            lv_obj_set_style_bg_color(step, color_primary_, 0);
            lv_obj_set_style_border_color(step, color_primary_, 0);
            lv_obj_set_style_shadow_width(step, 6, 0);
            lv_obj_set_style_shadow_color(step, color_primary_, 0);
            lv_obj_set_style_shadow_opa(step, LV_OPA_20, 0);
        } else if(is_completed) {
            lv_obj_set_style_bg_color(step, color_success_, 0);
            lv_obj_set_style_border_color(step, color_success_, 0);
            lv_obj_set_style_shadow_width(step, 4, 0);
            lv_obj_set_style_shadow_opa(step, LV_OPA_10, 0);
            lv_obj_set_style_shadow_color(step, lv_color_black(), 0);
        } else {
            lv_obj_set_style_bg_color(step, lv_color_hex(0x3A4048), 0);
            lv_obj_set_style_border_color(step, lv_color_hex(0x4A5058), 0);
            lv_obj_set_style_shadow_width(step, 0, 0);
        }
        
        lv_obj_t* icon = lv_label_create(step);
        lv_label_set_text(icon, icons[i]);
        lv_obj_set_style_text_font(icon, &lv_font_montserrat_16, 0);  // 稍小字体
        lv_obj_set_style_text_color(icon, lv_color_white(), 0);
        lv_obj_center(icon);
        
        header_widgets_.workflow_buttons.push_back(step);
    }
    
    // Status container (使用Flex布局替代固定位置)
    lv_obj_t* status_container = lv_obj_create(header_panel_);
    lv_obj_set_size(status_container, LV_SIZE_CONTENT, 50);
    lv_obj_set_style_bg_opa(status_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(status_container, 0, 0);
    lv_obj_set_style_pad_all(status_container, 0, 0);
    lv_obj_set_flex_flow(status_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(status_container, LV_FLEX_ALIGN_END, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(status_container, 10, 0);
    lv_obj_clear_flag(status_container, LV_OBJ_FLAG_SCROLLABLE);
    
    // Heartbeat Status
    lv_obj_t* heartbeat_container = lv_obj_create(status_container);
    lv_obj_set_size(heartbeat_container, 160, 50);
    lv_obj_set_style_bg_color(heartbeat_container, lv_color_hex(0x0F1419), 0);
    lv_obj_set_style_radius(heartbeat_container, 25, 0);
    lv_obj_set_style_border_width(heartbeat_container, 2, 0);
    lv_obj_set_style_border_color(heartbeat_container, color_success_, 0);
    lv_obj_set_style_pad_all(heartbeat_container, 8, 0);
    lv_obj_clear_flag(heartbeat_container, LV_OBJ_FLAG_SCROLLABLE);
    
    // 设置heartbeat_container为Flex容器
    lv_obj_set_flex_flow(heartbeat_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(heartbeat_container, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // Response Time
    header_widgets_.response_label = lv_label_create(status_container);
    lv_label_set_text(header_widgets_.response_label, LV_SYMBOL_LOOP " 15ms");
    lv_obj_set_style_text_color(header_widgets_.response_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(header_widgets_.response_label, &lv_font_montserrat_14, 0);
    // 移除 lv_obj_align，使用Flex布局控制位置
    
    return header_panel_;
#else
    return nullptr;
#endif
}

} // namespace ui
} // namespace bamboo_cut