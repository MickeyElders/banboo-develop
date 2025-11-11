/**
 * @file lvgl_panel_utils.h
 * @brief LVGL 面板创建工具函数 - 用于 DRM 和 Wayland 版本共享
 * 
 * 提取通用的面板创建逻辑，避免在不同 UI 实现之间重复代码
 */

#pragma once

#include "lvgl_ui_utils.h"
#include <memory>

namespace bamboo_cut {
namespace ui {

/**
 * @brief 面板创建上下文 - 传递创建面板所需的参数
 */
struct PanelContext {
    lv_obj_t* parent;                        // 父容器
    LVGLControlWidgets& widgets;             // 控件引用
    const LVGLThemeColors& colors;           // 主题颜色
    int screen_width;                        // 屏幕宽度
    int screen_height;                       // 屏幕高度
};

/**
 * @brief 创建头部面板（系统标题、心跳、响应时间、工作流按钮）
 * 
 * @param ctx 面板创建上下文
 * @return 创建的头部面板对象
 */
lv_obj_t* createHeaderPanel(const PanelContext& ctx);

/**
 * @brief 创建摄像头面板（视频显示区域、坐标信息、质量信息、双摄切换）
 * 
 * @param ctx 面板创建上下文
 * @return 创建的摄像头面板对象
 */
lv_obj_t* createCameraPanel(const PanelContext& ctx);

/**
 * @brief 创建控制面板（Jetson监控、AI模型、Modbus、工作流程、系统版本）
 * 
 * @param ctx 面板创建上下文
 * @return 创建的控制面板对象
 */
lv_obj_t* createControlPanel(const PanelContext& ctx);

/**
 * @brief 创建状态面板（系统指标、版本信息）
 * 
 * @param ctx 面板创建上下文
 * @return 创建的状态面板对象
 */
lv_obj_t* createStatusPanel(const PanelContext& ctx);

/**
 * @brief 创建底部面板（控制按钮、进程信息、统计信息）
 * 
 * @param ctx 面板创建上下文
 * @return 创建的底部面板对象
 */
lv_obj_t* createFooterPanel(const PanelContext& ctx);

} // namespace ui
} // namespace bamboo_cut

