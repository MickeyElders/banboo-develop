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
    // 强制设置固定大小确保可见性
    lv_obj_set_width(camera_panel_, lv_pct(75));  // 改为固定75%宽度
    lv_obj_set_height(camera_panel_, lv_pct(100));
    lv_obj_set_flex_grow(camera_panel_, 3);  // 占3/4空间
    
    // 强化可见背景调试布局 - 确保面板可见
    lv_obj_set_style_bg_opa(camera_panel_, LV_OPA_80, 0);  // 提高到80%透明度
    lv_obj_set_style_bg_color(camera_panel_, lv_color_hex(0xFF0000), 0);  // 改为红色背景更醒目
    lv_obj_set_style_border_width(camera_panel_, 5, 0);  // 加粗边框
    lv_obj_set_style_border_color(camera_panel_, lv_color_hex(0x00FF00), 0);  // 绿色边框
    lv_obj_set_style_border_opa(camera_panel_, LV_OPA_100, 0);  // 边框完全不透明
    lv_obj_clear_flag(camera_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    std::cout << "Camera panel created with red background and green border for visibility test" << std::endl;
    
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
    
    // 双摄切换按钮容器 - 强化可见性
    lv_obj_t* control_overlay = lv_obj_create(camera_panel_);
    lv_obj_set_width(control_overlay, lv_pct(100));
    lv_obj_set_height(control_overlay, 80);  // 增加高度
    lv_obj_set_flex_grow(control_overlay, 0);  // 固定高度
    lv_obj_set_style_bg_color(control_overlay, lv_color_hex(0x0000FF), 0);  // 蓝色背景
    lv_obj_set_style_bg_opa(control_overlay, LV_OPA_80, 0);  // 提高不透明度
    lv_obj_set_style_border_width(control_overlay, 3, 0);  // 加粗边框
    lv_obj_set_style_border_color(control_overlay, lv_color_hex(0xFFFF00), 0);  // 黄色边框
    lv_obj_set_style_border_opa(control_overlay, LV_OPA_100, 0);
    lv_obj_set_style_radius(control_overlay, 8, 0);
    lv_obj_set_style_pad_all(control_overlay, 8, 0);
    lv_obj_clear_flag(control_overlay, LV_OBJ_FLAG_SCROLLABLE);
    
    std::cout << "Control overlay created with blue background and yellow border" << std::endl;
    
    // 设置控制覆盖层为水平Flex布局
    lv_obj_set_flex_flow(control_overlay, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(control_overlay, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(control_overlay, 10, 0);
    
    // 双摄模式切换按钮组
    camera_widgets_.mode_label = lv_label_create(control_overlay);
    lv_label_set_text(camera_widgets_.mode_label, LV_SYMBOL_VIDEO " 双摄模式:");
    lv_obj_set_style_text_color(camera_widgets_.mode_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(camera_widgets_.mode_label, &lv_font_montserrat_14, 0);
    
    // 并排按钮 - 强化可见性
    camera_widgets_.split_btn = lv_btn_create(control_overlay);
    lv_obj_set_size(camera_widgets_.split_btn, 100, 45);  // 增大尺寸给更多空间
    lv_obj_set_style_bg_color(camera_widgets_.split_btn, lv_color_hex(0x00FFFF), 0);  // 青色背景
    lv_obj_t* split_label = lv_label_create(camera_widgets_.split_btn);
    lv_label_set_text(split_label, "并排显示");
    lv_obj_center(split_label);
    lv_obj_set_style_text_color(split_label, lv_color_black(), 0);
    lv_obj_set_style_text_font(split_label, &lv_font_montserrat_14, 0);
    lv_obj_add_event_cb(camera_widgets_.split_btn, onSplitScreenButtonClicked, LV_EVENT_CLICKED, this);
    std::cout << "Split screen button created with cyan background" << std::endl;
    
    // 立体按钮 - 强化可见性
    camera_widgets_.stereo_btn = lv_btn_create(control_overlay);
    lv_obj_set_size(camera_widgets_.stereo_btn, 100, 45);  // 增大尺寸给更多空间
    lv_obj_set_style_bg_color(camera_widgets_.stereo_btn, lv_color_hex(0xFFFF00), 0);  // 黄色背景
    lv_obj_t* stereo_label = lv_label_create(camera_widgets_.stereo_btn);
    lv_label_set_text(stereo_label, "立体视觉");
    lv_obj_center(stereo_label);
    lv_obj_set_style_text_color(stereo_label, lv_color_black(), 0);
    lv_obj_set_style_text_font(stereo_label, &lv_font_montserrat_14, 0);
    lv_obj_add_event_cb(camera_widgets_.stereo_btn, onStereoVisionButtonClicked, LV_EVENT_CLICKED, this);
    std::cout << "Stereo vision button created with yellow background" << std::endl;
    
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