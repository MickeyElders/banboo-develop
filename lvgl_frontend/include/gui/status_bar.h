#ifndef GUI_STATUS_BAR_H
#define GUI_STATUS_BAR_H

#include "lvgl.h"
#include "common/types.h"

class Status_bar {
public:
    Status_bar();
    ~Status_bar();
    bool initialize();
    
    // 获取状态栏容器
    lv_obj_t* get_container() const { return container_; }
    
    // 更新状态信息
    void update_system_title(const char* title);
    void update_workflow_status(int step);
    void update_heartbeat(uint32_t count, uint32_t response_ms);

private:
    lv_obj_t* container_;           // 主容器
    lv_obj_t* system_title_;        // 系统标题
    lv_obj_t* workflow_container_;  // 工作流程状态容器
    lv_obj_t* heartbeat_monitor_;   // 心跳监控
    lv_obj_t* workflow_steps_[5];   // 工作流程步骤
    lv_obj_t* heartbeat_label_;     // 心跳标签
    lv_obj_t* response_label_;      // 响应时间标签
    
    lv_style_t style_container_;    // 容器样式
    lv_style_t style_step_;         // 步骤样式
    lv_style_t style_step_active_;  // 激活步骤样式
    lv_style_t style_step_completed_; // 完成步骤样式
    
    void create_layout();
    void setup_styles();
    void create_workflow_steps();
    void create_heartbeat_monitor();
};
#endif