/**
 * @file ui_context.h
 * @brief LVGL UI 上下文封装，独立于显示后端
 */

#pragma once

#include <lvgl.h>
#include "bamboo_cut/ui/lvgl_ui_utils.h"

namespace bamboo_cut {
namespace ui {

struct UIContext {
    lv_obj_t* main_screen = nullptr;
    lv_obj_t* camera_panel = nullptr;
    LVGLControlWidgets widgets;
};

} // namespace ui
} // namespace bamboo_cut
