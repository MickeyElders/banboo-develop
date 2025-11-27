/**
 * @file ui_components.h
 * @brief UI 组件构建（与显示后端解耦）
 */

#pragma once

#include "bamboo_cut/ui/ui_context.h"
#include "bamboo_cut/ui/lvgl_wayland_interface.h" // 复用配置结构

namespace bamboo_cut {
namespace ui {

/**
 * @brief 构建基础 UI（头部 + 摄像头面板 + 控制面板）
 */
bool build_basic_ui(UIContext& ctx, const LVGLWaylandConfig& config);
bool build_full_ui(UIContext& ctx, const LVGLWaylandConfig& config, bool debug_camera_panel_opaque);

} // namespace ui
} // namespace bamboo_cut
