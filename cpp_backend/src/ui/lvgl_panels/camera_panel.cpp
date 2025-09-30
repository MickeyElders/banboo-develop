/**
 * @file camera_panel.cpp
 * @brief LVGL camera panel implementation
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include <iostream>

namespace bamboo_cut {
namespace ui {

lv_obj_t* LVGLInterface::createCameraPanel(lv_obj_t* parent) {
#ifdef ENABLE_LVGL
    lv_obj_t* container = parent ? parent : main_screen_;
    
    camera_panel_ = lv_obj_create(container);
    // 设置大小和布局
    lv_obj_set_width(camera_panel_, LV_SIZE_CONTENT);
    lv_obj_set_height(camera_panel_, lv_pct(100));
    lv_obj_set_flex_grow(camera_panel_, 3);  // 占3/4空间
    
    // 临时添加可见背景调试布局 - 生产环境需要设置为透明
    lv_obj_set_style_bg_opa(camera_panel_, LV_OPA_30, 0);  // 30%透明度便于调试
    lv_obj_set_style_bg_color(camera_panel_, lv_color_hex(0x1E2329), 0);  // 深色背景
    lv_obj_set_style_border_width(camera_panel_, 2, 0);
    lv_obj_set_style_border_color(camera_panel_, lv_color_hex(0x00FF00), 0);  // 绿色边框便于识别
    lv_obj_set_style_border_opa(camera_panel_, LV_OPA_60, 0);
    // lv_obj_set_style_shadow_opa(camera_panel_, LV_OPA_TRANSP, 0);
    // lv_obj_set_style_outline_opa(camera_panel_, LV_OPA_TRANSP, 0);
    lv_obj_clear_flag(camera_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 设置摄像头面板为垂直Flex布局
    lv_obj_set_flex_flow(camera_panel_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(camera_panel_, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_all(camera_panel_, 0, 0);  // 移除所有内边距
    lv_obj_set_style_pad_gap(camera_panel_, 0, 0);  // 移除间隔
    
    // 创建完全透明的视频显示区域占位符
    camera_canvas_ = lv_obj_create(camera_panel_);
    lv_obj_set_width(camera_canvas_, lv_pct(100));
    lv_obj_set_flex_grow(camera_canvas_, 1);  // 占据大部分空间
    
    // 设置为完全透明 - 这里只是占位，实际视频由 nvdrmvideosink 显示
    lv_obj_set_style_bg_opa(camera_canvas_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(camera_canvas_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_shadow_opa(camera_canvas_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_outline_opa(camera_canvas_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(camera_canvas_, 0, 0);
    lv_obj_clear_flag(camera_canvas_, LV_OBJ_FLAG_SCROLLABLE);
    
    std::cout << "Camera panel set to fully transparent for nvdrmvideosink hardware layer" << std::endl;
    
    // 双摄切换按钮容器 - 半透明覆盖层
    lv_obj_t* control_overlay = lv_obj_create(camera_panel_);
    lv_obj_set_width(control_overlay, lv_pct(100));
    lv_obj_set_height(control_overlay, 50);
    lv_obj_set_flex_grow(control_overlay, 0);  // 固定高度
    lv_obj_set_style_bg_color(control_overlay, lv_color_black(), 0);
    lv_obj_set_style_bg_opa(control_overlay, LV_OPA_50, 0);  // 半透明
    lv_obj_set_style_border_width(control_overlay, 0, 0);
    lv_obj_set_style_radius(control_overlay, 8, 0);
    lv_obj_set_style_pad_all(control_overlay, 8, 0);
    lv_obj_clear_flag(control_overlay, LV_OBJ_FLAG_SCROLLABLE);
    
    // 设置控制覆盖层为水平Flex布局
    lv_obj_set_flex_flow(control_overlay, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(control_overlay, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(control_overlay, 10, 0);
    
    // 双摄模式切换按钮组
    camera_widgets_.mode_label = lv_label_create(control_overlay);
    lv_label_set_text(camera_widgets_.mode_label, LV_SYMBOL_VIDEO " 模式:");
    lv_obj_set_style_text_color(camera_widgets_.mode_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(camera_widgets_.mode_label, &lv_font_montserrat_14, 0);
    
    // 单摄按钮
    camera_widgets_.single_btn = lv_btn_create(control_overlay);
    lv_obj_set_size(camera_widgets_.single_btn, 60, 30);
    lv_obj_t* single_label = lv_label_create(camera_widgets_.single_btn);
    lv_label_set_text(single_label, "单摄");
    lv_obj_center(single_label);
    lv_obj_set_style_text_font(single_label, &lv_font_montserrat_12, 0);
    lv_obj_add_event_cb(camera_widgets_.single_btn, onSingleCameraButtonClicked, LV_EVENT_CLICKED, this);
    
    // 并排按钮
    camera_widgets_.split_btn = lv_btn_create(control_overlay);
    lv_obj_set_size(camera_widgets_.split_btn, 60, 30);
    lv_obj_t* split_label = lv_label_create(camera_widgets_.split_btn);
    lv_label_set_text(split_label, "并排");
    lv_obj_center(split_label);
    lv_obj_set_style_text_font(split_label, &lv_font_montserrat_12, 0);
    lv_obj_add_event_cb(camera_widgets_.split_btn, onSplitScreenButtonClicked, LV_EVENT_CLICKED, this);
    
    // 立体按钮
    camera_widgets_.stereo_btn = lv_btn_create(control_overlay);
    lv_obj_set_size(camera_widgets_.stereo_btn, 60, 30);
    lv_obj_t* stereo_label = lv_label_create(camera_widgets_.stereo_btn);
    lv_label_set_text(stereo_label, "立体");
    lv_obj_center(stereo_label);
    lv_obj_set_style_text_font(stereo_label, &lv_font_montserrat_12, 0);
    lv_obj_add_event_cb(camera_widgets_.stereo_btn, onStereoVisionButtonClicked, LV_EVENT_CLICKED, this);
    
    // 状态指示
    camera_widgets_.status_value = lv_label_create(control_overlay);
    lv_label_set_text(camera_widgets_.status_value, LV_SYMBOL_OK " 就绪");
    lv_obj_set_style_text_color(camera_widgets_.status_value, lv_color_hex(0x00FF00), 0);
    lv_obj_set_style_text_font(camera_widgets_.status_value, &lv_font_montserrat_12, 0);
    
    return camera_panel_;
#else
    return nullptr;
#endif
}

} // namespace ui
} // namespace bamboo_cut