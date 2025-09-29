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
    // ✅ 移除固定宽度，只用 flex_grow 控制比例
    lv_obj_set_width(camera_panel_, LV_SIZE_CONTENT);  // 或者直接不设置宽度
    lv_obj_set_height(camera_panel_, lv_pct(100));  // 高度填满父容器
    lv_obj_set_flex_grow(camera_panel_, 3);  // ✅ 现在会生效，占3/4空间
    lv_obj_add_style(camera_panel_, &style_card, 0);
    
    // 简洁优雅的边框
    lv_obj_set_style_bg_color(camera_panel_, lv_color_hex(0x1E2329), 0);
    lv_obj_set_style_border_width(camera_panel_, 1, 0);
    lv_obj_set_style_border_color(camera_panel_, lv_color_hex(0x3A4048), 0);
    lv_obj_set_style_border_opa(camera_panel_, LV_OPA_60, 0);
    lv_obj_set_style_shadow_width(camera_panel_, 12, 0);
    lv_obj_set_style_shadow_color(camera_panel_, lv_color_black(), 0);
    lv_obj_set_style_shadow_opa(camera_panel_, LV_OPA_10, 0);
    lv_obj_set_style_radius(camera_panel_, 16, 0);
    lv_obj_clear_flag(camera_panel_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 设置摄像头面板为垂直Flex布局
    lv_obj_set_flex_flow(camera_panel_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(camera_panel_, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(camera_panel_, 5, 0);
    
    // Canvas画布容器
    lv_obj_t* canvas_container = lv_obj_create(camera_panel_);
    lv_obj_set_width(canvas_container, lv_pct(100));
    lv_obj_set_flex_grow(canvas_container, 1);  // 占据剩余空间
    lv_obj_set_style_bg_opa(canvas_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(canvas_container, 0, 0);
    lv_obj_set_style_pad_all(canvas_container, 5, 0);
    lv_obj_clear_flag(canvas_container, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(canvas_container, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(canvas_container, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // 创建透明视频显示区域 - 用于 nvdrmvideosink 硬件层显示
    camera_canvas_ = lv_obj_create(canvas_container);
    
    // 设置视频显示区域大小（对应摄像头在屏幕上的显示位置）
    const int video_width = 640;   // 视频显示宽度
    const int video_height = 360;  // 视频显示高度
    
    // 设置对象大小
    lv_obj_set_width(camera_canvas_, video_width);
    lv_obj_set_height(camera_canvas_, video_height);
    
    // 设置为完全透明，让下层的 nvdrmvideosink 视频显示出来
    lv_obj_set_style_bg_opa(camera_canvas_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(camera_canvas_, LV_OPA_TRANSP, 0);
    lv_obj_clear_flag(camera_canvas_, LV_OBJ_FLAG_SCROLLABLE);
    
    // 可选：添加微弱的边框以标示视频区域（调试用）
    // lv_obj_set_style_border_width(camera_canvas_, 1, 0);
    // lv_obj_set_style_border_color(camera_canvas_, lv_color_hex(0x00FF00), 0);
    // lv_obj_set_style_border_opa(camera_canvas_, LV_OPA_30, 0);
    
    std::cout << "Camera panel configured for nvdrmvideosink transparent overlay" << std::endl;
    
    // 信息覆盖层
    lv_obj_t* info_overlay = lv_obj_create(camera_panel_);
    lv_obj_set_width(info_overlay, lv_pct(100));
    lv_obj_set_height(info_overlay, 60);
    lv_obj_set_flex_grow(info_overlay, 0);  // 不允许增长
    lv_obj_set_style_bg_color(info_overlay, lv_color_black(), 0);
    lv_obj_set_style_bg_opa(info_overlay, LV_OPA_70, 0);
    lv_obj_set_style_border_width(info_overlay, 0, 0);
    lv_obj_set_style_radius(info_overlay, 8, 0);
    lv_obj_set_style_pad_all(info_overlay, 10, 0);
    lv_obj_clear_flag(info_overlay, LV_OBJ_FLAG_SCROLLABLE);
    
    // 设置信息覆盖层为水平Flex布局
    lv_obj_set_flex_flow(info_overlay, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(info_overlay, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(info_overlay, 10, 0);
    
    // Coordinate Information
    camera_widgets_.coord_value = lv_label_create(info_overlay);
    lv_label_set_text(camera_widgets_.coord_value, LV_SYMBOL_GPS " X: 0.00 Y: 0.00 Z: 0.00");
    lv_obj_set_style_text_color(camera_widgets_.coord_value, color_primary_, 0);
    lv_obj_set_style_text_font(camera_widgets_.coord_value, &lv_font_montserrat_14, 0);
    // 移除 lv_obj_align，使用Flex布局控制位置
    
    // Quality Score
    camera_widgets_.quality_value = lv_label_create(info_overlay);
    lv_label_set_text(camera_widgets_.quality_value, LV_SYMBOL_IMAGE " Quality: 95%");
    lv_obj_set_style_text_color(camera_widgets_.quality_value, color_success_, 0);
    lv_obj_set_style_text_font(camera_widgets_.quality_value, &lv_font_montserrat_14, 0);
    // 移除 lv_obj_align，使用Flex布局控制位置
    
    // Blade Information
    camera_widgets_.blade_value = lv_label_create(info_overlay);
    lv_label_set_text(camera_widgets_.blade_value, LV_SYMBOL_SETTINGS " Blade: #3");
    lv_obj_set_style_text_color(camera_widgets_.blade_value, color_warning_, 0);
    lv_obj_set_style_text_font(camera_widgets_.blade_value, &lv_font_montserrat_14, 0);
    // 移除 lv_obj_align，使用Flex布局控制位置
    
    return camera_panel_;
#else
    return nullptr;
#endif
}

} // namespace ui
} // namespace bamboo_cut