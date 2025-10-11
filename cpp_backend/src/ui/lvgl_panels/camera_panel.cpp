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
    
    // 设置为完全透明 - 允许 DeepStream 视频透过显示
    lv_obj_set_style_bg_opa(camera_panel_, LV_OPA_TRANSP, 0);  // 完全透明背景
    lv_obj_set_style_border_opa(camera_panel_, LV_OPA_TRANSP, 0);  // 透明边框
    lv_obj_set_style_shadow_opa(camera_panel_, LV_OPA_TRANSP, 0);  // 透明阴影
    lv_obj_set_style_outline_opa(camera_panel_, LV_OPA_TRANSP, 0);  // 透明轮廓
    lv_obj_clear_flag(camera_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    std::cout << "Camera panel created with transparent background for DeepStream video overlay" << std::endl;
    
    // 设置摄像头面板为垂直Flex布局
    lv_obj_set_flex_flow(camera_panel_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(camera_panel_, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_all(camera_panel_, 0, 0);  // 移除所有内边距
    lv_obj_set_style_pad_gap(camera_panel_, 0, 0);  // 移除间隔
    
    // 创建真正的LVGL canvas用于appsink软件合成
    camera_canvas_ = lv_canvas_create(camera_panel_);
    lv_obj_set_width(camera_canvas_, lv_pct(100));
    lv_obj_set_flex_grow(camera_canvas_, 1);  // 占据大部分空间
    
    // 为canvas分配缓冲区 (960x640, BGRA格式, 32位/像素)
    static uint32_t canvas_buffer[960 * 640];  // 静态缓冲区避免栈溢出
    lv_canvas_set_buffer(camera_canvas_, canvas_buffer, 960, 640, LV_IMG_CF_TRUE_COLOR_ALPHA);
    
    // 设置canvas样式
    lv_obj_set_style_bg_opa(camera_canvas_, LV_OPA_COVER, 0);
    lv_obj_set_style_border_opa(camera_canvas_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_shadow_opa(camera_canvas_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_outline_opa(camera_canvas_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(camera_canvas_, 0, 0);
    lv_obj_clear_flag(camera_canvas_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 初始化canvas为黑色背景
    lv_canvas_fill_bg(camera_canvas_, lv_color_black(), LV_OPA_COVER);
    
    std::cout << "Camera canvas created for appsink software composition (960x640 BGRA)" << std::endl;
    
    // 半透明控制覆盖层 - 保持控件可见但不遮挡视频
    lv_obj_t* control_overlay = lv_obj_create(camera_panel_);
    lv_obj_set_width(control_overlay, lv_pct(100));
    lv_obj_set_height(control_overlay, 80);  // 固定高度
    lv_obj_set_flex_grow(control_overlay, 0);  // 固定高度
    lv_obj_set_style_bg_color(control_overlay, lv_color_hex(0x000000), 0);  // 黑色背景
    lv_obj_set_style_bg_opa(control_overlay, LV_OPA_40, 0);  // 40%透明度
    lv_obj_set_style_border_opa(control_overlay, LV_OPA_TRANSP, 0);  // 透明边框
    lv_obj_set_style_radius(control_overlay, 8, 0);
    lv_obj_set_style_pad_all(control_overlay, 8, 0);
    lv_obj_clear_flag(control_overlay, LV_OBJ_FLAG_SCROLLABLE);
    
    std::cout << "Control overlay created with semi-transparent background" << std::endl;
    
    // 设置控制覆盖层为水平Flex布局
    lv_obj_set_flex_flow(control_overlay, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(control_overlay, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(control_overlay, 10, 0);
    
    // Dual camera mode control buttons
    camera_widgets_.mode_label = lv_label_create(control_overlay);
    lv_label_set_text(camera_widgets_.mode_label, LV_SYMBOL_VIDEO " Dual Camera:");
    lv_obj_set_style_text_color(camera_widgets_.mode_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(camera_widgets_.mode_label, &lv_font_montserrat_14, 0);
    
    // Split screen button - semi-transparent design
    camera_widgets_.split_btn = lv_btn_create(control_overlay);
    lv_obj_set_size(camera_widgets_.split_btn, 100, 45);
    lv_obj_set_style_bg_color(camera_widgets_.split_btn, lv_color_hex(0x4CAF50), 0);  // Green background
    lv_obj_set_style_bg_opa(camera_widgets_.split_btn, LV_OPA_80, 0);  // Semi-transparent
    lv_obj_t* split_label = lv_label_create(camera_widgets_.split_btn);
    lv_label_set_text(split_label, "Split View");
    lv_obj_center(split_label);
    lv_obj_set_style_text_color(split_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(split_label, &lv_font_montserrat_14, 0);
    lv_obj_add_event_cb(camera_widgets_.split_btn, onSplitScreenButtonClicked, LV_EVENT_CLICKED, this);
    std::cout << "Split screen button created with semi-transparent green background" << std::endl;
    
    // Stereo vision button - semi-transparent design
    camera_widgets_.stereo_btn = lv_btn_create(control_overlay);
    lv_obj_set_size(camera_widgets_.stereo_btn, 100, 45);
    lv_obj_set_style_bg_color(camera_widgets_.stereo_btn, lv_color_hex(0x2196F3), 0);  // Blue background
    lv_obj_set_style_bg_opa(camera_widgets_.stereo_btn, LV_OPA_80, 0);  // Semi-transparent
    lv_obj_t* stereo_label = lv_label_create(camera_widgets_.stereo_btn);
    lv_label_set_text(stereo_label, "Stereo 3D");
    lv_obj_center(stereo_label);
    lv_obj_set_style_text_color(stereo_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(stereo_label, &lv_font_montserrat_14, 0);
    lv_obj_add_event_cb(camera_widgets_.stereo_btn, onStereoVisionButtonClicked, LV_EVENT_CLICKED, this);
    std::cout << "Stereo vision button created with semi-transparent blue background" << std::endl;
    
    // Status indicator
    camera_widgets_.status_value = lv_label_create(control_overlay);
    lv_label_set_text(camera_widgets_.status_value, LV_SYMBOL_OK " Ready");
    lv_obj_set_style_text_color(camera_widgets_.status_value, lv_color_hex(0x00FF00), 0);
    lv_obj_set_style_text_font(camera_widgets_.status_value, &lv_font_montserrat_12, 0);
    
    return camera_panel_;
#else
    return nullptr;
#endif
}

} // namespace ui
} // namespace bamboo_cut