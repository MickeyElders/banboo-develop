#ifndef GUI_VIDEO_VIEW_H
#define GUI_VIDEO_VIEW_H

#include "lvgl.h"
#include "common/types.h"

class Video_view {
public:
    Video_view();
    ~Video_view();
    bool initialize();
    
    // 获取视频容器
    lv_obj_t* get_container() const { return container_; }
    
    // 更新视频显示
    void update_camera_frame(const frame_info_t& frame);
    void update_detection_info(float fps, float inference_time);
    void update_coordinate_display(float x_mm, const char* quality, const char* blade);
    void update_cutting_position(float x_mm, float max_range = 1000.0f);

private:
    lv_obj_t* container_;           // 主容器
    lv_obj_t* camera_title_;        // 摄像头标题容器
    lv_obj_t* camera_canvas_;       // 摄像头画布
    lv_obj_t* coordinate_display_;  // 坐标显示区域
    lv_obj_t* rail_indicator_;      // 导轨指示器
    lv_obj_t* cutting_position_;    // 切割位置指示器
    
    // 标题区域组件
    lv_obj_t* title_label_;         // 标题标签
    lv_obj_t* detection_info_;      // 检测信息标签
    
    // 坐标显示组件
    lv_obj_t* x_coord_label_;       // X坐标标签
    lv_obj_t* quality_label_;       // 质量标签
    lv_obj_t* blade_label_;         // 刀片标签
    
    // 样式
    lv_style_t style_container_;    // 容器样式
    lv_style_t style_canvas_;       // 画布样式
    lv_style_t style_coord_;        // 坐标显示样式
    lv_style_t style_rail_;         // 导轨样式
    lv_style_t style_cutting_;      // 切割位置样式
    
    void create_layout();
    void setup_styles();
    void create_camera_title();
    void create_camera_canvas();
    void create_coordinate_display();
};

#endif