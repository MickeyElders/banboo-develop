/**
 * @file footer_panel.cpp
 * @brief LVGL footer panel implementation
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include <iostream>

namespace bamboo_cut {
namespace ui {

lv_obj_t* LVGLInterface::createFooterPanel() {
#ifdef ENABLE_LVGL
    footer_panel_ = lv_obj_create(main_screen_);
    lv_obj_set_width(footer_panel_, lv_pct(100));
    lv_obj_set_height(footer_panel_, 80);
    // 移除 lv_obj_align，使用父容器的Flex布局控制位置
    
    // 现代简洁的背景样式
    lv_obj_set_style_bg_color(footer_panel_, color_surface_, 0);
    lv_obj_set_style_radius(footer_panel_, 20, 0);  // 降低圆角
    lv_obj_set_style_border_width(footer_panel_, 1, 0);
    lv_obj_set_style_border_color(footer_panel_, lv_color_hex(0x3A4048), 0);
    lv_obj_set_style_border_opa(footer_panel_, LV_OPA_40, 0);
    lv_obj_set_style_shadow_width(footer_panel_, 16, 0);  // 减少阴影
    lv_obj_set_style_shadow_color(footer_panel_, lv_color_black(), 0);
    lv_obj_set_style_shadow_opa(footer_panel_, LV_OPA_20, 0);
    lv_obj_set_style_pad_all(footer_panel_, 16, 0);
    lv_obj_clear_flag(footer_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 使用flex布局，增加间距
    lv_obj_set_flex_flow(footer_panel_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(footer_panel_, LV_FLEX_ALIGN_SPACE_AROUND,
                          LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(footer_panel_, 12, 0);
    
    // 主操作区域容器
    lv_obj_t* main_controls = lv_obj_create(footer_panel_);
    lv_obj_set_size(main_controls, lv_pct(70), lv_pct(100));
    lv_obj_set_style_bg_opa(main_controls, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(main_controls, 0, 0);
    lv_obj_set_style_pad_all(main_controls, 0, 0);
    lv_obj_set_flex_flow(main_controls, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(main_controls, LV_FLEX_ALIGN_SPACE_EVENLY,
                          LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // Start Button
    footer_widgets_.start_btn = lv_btn_create(main_controls);
    lv_obj_set_size(footer_widgets_.start_btn, 110, 48);  // More compact
    lv_obj_add_style(footer_widgets_.start_btn, &style_btn_success, 0);
    lv_obj_add_style(footer_widgets_.start_btn, &style_btn_pressed, LV_STATE_PRESSED);
    
    lv_obj_t* start_label = lv_label_create(footer_widgets_.start_btn);
    lv_label_set_text(start_label, LV_SYMBOL_PLAY " START");
    lv_obj_set_style_text_font(start_label, &lv_font_montserrat_16, 0);
    lv_obj_center(start_label);
    lv_obj_add_event_cb(footer_widgets_.start_btn, onStartButtonClicked,
                        LV_EVENT_CLICKED, this);
    
    // Pause Button
    footer_widgets_.pause_btn = lv_btn_create(main_controls);
    lv_obj_set_size(footer_widgets_.pause_btn, 110, 48);
    lv_obj_add_style(footer_widgets_.pause_btn, &style_btn_warning, 0);
    lv_obj_add_style(footer_widgets_.pause_btn, &style_btn_pressed, LV_STATE_PRESSED);
    
    lv_obj_t* pause_label = lv_label_create(footer_widgets_.pause_btn);
    lv_label_set_text(pause_label, LV_SYMBOL_PAUSE " PAUSE");
    lv_obj_set_style_text_font(pause_label, &lv_font_montserrat_16, 0);
    lv_obj_center(pause_label);
    lv_obj_add_event_cb(footer_widgets_.pause_btn, onPauseButtonClicked,
                        LV_EVENT_CLICKED, this);
    
    // Stop Button
    footer_widgets_.stop_btn = lv_btn_create(main_controls);
    lv_obj_set_size(footer_widgets_.stop_btn, 110, 48);
    lv_obj_add_style(footer_widgets_.stop_btn, &style_btn_primary, 0);
    lv_obj_add_style(footer_widgets_.stop_btn, &style_btn_pressed, LV_STATE_PRESSED);
    lv_obj_set_style_bg_color(footer_widgets_.stop_btn, lv_color_hex(0x6B7280), 0);
    
    lv_obj_t* stop_label = lv_label_create(footer_widgets_.stop_btn);
    lv_label_set_text(stop_label, LV_SYMBOL_STOP " STOP");
    lv_obj_set_style_text_font(stop_label, &lv_font_montserrat_16, 0);
    lv_obj_center(stop_label);
    lv_obj_add_event_cb(footer_widgets_.stop_btn, onStopButtonClicked,
                        LV_EVENT_CLICKED, this);
    
    // 危险操作区域
    lv_obj_t* danger_zone = lv_obj_create(footer_panel_);
    lv_obj_set_size(danger_zone, 70, lv_pct(100));
    lv_obj_set_style_bg_opa(danger_zone, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(danger_zone, 0, 0);
    lv_obj_set_style_pad_all(danger_zone, 0, 0);
    
    // 设置危险操作区域为Flex容器
    lv_obj_set_flex_flow(danger_zone, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(danger_zone, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // 急停按钮（独立突出）
    footer_widgets_.emergency_btn = lv_btn_create(danger_zone);
    lv_obj_set_size(footer_widgets_.emergency_btn, 60, 60);
    lv_obj_add_style(footer_widgets_.emergency_btn, &style_btn_danger, 0);
    lv_obj_add_style(footer_widgets_.emergency_btn, &style_btn_pressed, LV_STATE_PRESSED);
    // 移除 lv_obj_center，使用Flex布局自动居中
    
    lv_obj_t* emergency_label = lv_label_create(footer_widgets_.emergency_btn);
    lv_label_set_text(emergency_label, LV_SYMBOL_WARNING);
    lv_obj_set_style_text_font(emergency_label, &lv_font_montserrat_24, 0);
    // 移除 lv_obj_center，使用Flex布局自动居中
    lv_obj_add_event_cb(footer_widgets_.emergency_btn, onEmergencyButtonClicked,
                        LV_EVENT_CLICKED, this);
    
    // 辅助操作区域
    lv_obj_t* aux_controls = lv_obj_create(footer_panel_);
    lv_obj_set_size(aux_controls, lv_pct(20), lv_pct(100));
    lv_obj_set_style_bg_opa(aux_controls, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(aux_controls, 0, 0);
    lv_obj_set_style_pad_all(aux_controls, 0, 0);
    lv_obj_set_flex_flow(aux_controls, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(aux_controls, LV_FLEX_ALIGN_CENTER,
                          LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // 设置按钮
    footer_widgets_.power_btn = lv_btn_create(aux_controls);
    lv_obj_set_size(footer_widgets_.power_btn, 48, 48);
    lv_obj_add_style(footer_widgets_.power_btn, &style_btn_primary, 0);
    lv_obj_add_style(footer_widgets_.power_btn, &style_btn_pressed, LV_STATE_PRESSED);
    
    lv_obj_t* power_label = lv_label_create(footer_widgets_.power_btn);
    lv_label_set_text(power_label, LV_SYMBOL_SETTINGS);
    lv_obj_set_style_text_font(power_label, &lv_font_montserrat_16, 0);
    // 移除 lv_obj_center，按钮内的标签会自动居中
    lv_obj_add_event_cb(footer_widgets_.power_btn, onSettingsButtonClicked,
                        LV_EVENT_CLICKED, this);
    
    return footer_panel_;
#else
    return nullptr;
#endif
}

} // namespace ui
} // namespace bamboo_cut