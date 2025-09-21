#ifndef GUI_SETTINGS_PAGE_H
#define GUI_SETTINGS_PAGE_H

#include "lvgl.h"
#include "common/types.h"

class Settings_page {
public:
    Settings_page();
    ~Settings_page();
    bool initialize();
    
    // 获取设置页面容器
    lv_obj_t* get_container() const { return container_; }
    
    // 创建主布局整合器 - 整合所有GUI组件
    void create_main_layout(class Status_bar* status_bar, class Video_view* video_view, class Control_panel* control_panel);
    void update_layout_positions();

private:
    lv_obj_t* container_;           // 主容器
    lv_obj_t* main_grid_;           // 主网格容器
    lv_obj_t* footer_panel_;        // 底部操作面板
    
    // 底部操作按钮
    lv_obj_t* control_buttons_;     // 控制按钮容器
    lv_obj_t* status_info_;         // 状态信息容器
    lv_obj_t* emergency_buttons_;   // 紧急操作按钮容器
    
    // 按钮
    lv_obj_t* start_btn_;           // 启动按钮
    lv_obj_t* pause_btn_;           // 暂停按钮
    lv_obj_t* stop_btn_;            // 停止按钮
    lv_obj_t* emergency_btn_;       // 紧急停止按钮
    lv_obj_t* power_btn_;           // 关机按钮
    
    // 状态标签
    lv_obj_t* current_process_label_; // 当前工序标签
    lv_obj_t* status_info_label_;    // 状态信息标签
    
    // 样式
    lv_style_t style_main_;         // 主容器样式
    lv_style_t style_footer_;       // 底部面板样式
    lv_style_t style_btn_;          // 按钮样式
    lv_style_t style_btn_start_;    // 启动按钮样式
    lv_style_t style_btn_emergency_; // 紧急按钮样式
    
    void setup_styles();
    void create_footer_panel();
    
    // 事件处理
    static void button_event_cb(lv_event_t* e);
    static void emergency_event_cb(lv_event_t* e);
};

#endif