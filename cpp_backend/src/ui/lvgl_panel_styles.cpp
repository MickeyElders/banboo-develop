/**
 * @file lvgl_panel_styles.cpp
 * @brief LVGL面板样式定义 - 兼容性实现
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include <lvgl.h>

namespace bamboo_cut {
namespace ui {

// 全局样式变量定义
lv_style_t style_btn_success;
lv_style_t style_btn_warning;
lv_style_t style_btn_primary;
lv_style_t style_btn_danger;
lv_style_t style_btn_pressed;

// LVGLInterface类的空实现 - 兼容性
LVGLInterface::LVGLInterface(std::shared_ptr<core::DataBridge> data_bridge)
    : data_bridge_(data_bridge) {
}

LVGLInterface::~LVGLInterface() {
}

bool LVGLInterface::initialize(const LVGLConfig& config) {
    // 初始化样式
    initializeStyles();
    return true;
}

bool LVGLInterface::start() {
    return true;
}

void LVGLInterface::stop() {
}

bool LVGLInterface::isRunning() const {
    return false;
}

void LVGLInterface::initializeStyles() {
    // 初始化成功按钮样式
    lv_style_init(&style_btn_success);
    lv_style_set_bg_color(&style_btn_success, lv_color_hex(0x22C55E));
    lv_style_set_bg_opa(&style_btn_success, LV_OPA_COVER);
    lv_style_set_text_color(&style_btn_success, lv_color_white());
    lv_style_set_radius(&style_btn_success, 8);
    
    // 初始化警告按钮样式
    lv_style_init(&style_btn_warning);
    lv_style_set_bg_color(&style_btn_warning, lv_color_hex(0xF59E0B));
    lv_style_set_bg_opa(&style_btn_warning, LV_OPA_COVER);
    lv_style_set_text_color(&style_btn_warning, lv_color_white());
    lv_style_set_radius(&style_btn_warning, 8);
    
    // 初始化主要按钮样式
    lv_style_init(&style_btn_primary);
    lv_style_set_bg_color(&style_btn_primary, lv_color_hex(0x3B82F6));
    lv_style_set_bg_opa(&style_btn_primary, LV_OPA_COVER);
    lv_style_set_text_color(&style_btn_primary, lv_color_white());
    lv_style_set_radius(&style_btn_primary, 8);
    
    // 初始化危险按钮样式
    lv_style_init(&style_btn_danger);
    lv_style_set_bg_color(&style_btn_danger, lv_color_hex(0xEF4444));
    lv_style_set_bg_opa(&style_btn_danger, LV_OPA_COVER);
    lv_style_set_text_color(&style_btn_danger, lv_color_white());
    lv_style_set_radius(&style_btn_danger, 8);
    
    // 初始化按下状态样式
    lv_style_init(&style_btn_pressed);
    lv_style_set_bg_opa(&style_btn_pressed, LV_OPA_80);
    lv_style_set_transform_zoom(&style_btn_pressed, 250);
}

// 成员变量初始化
lv_style_t LVGLInterface::style_text_title;
lv_style_t LVGLInterface::style_card;

// LVGLInterface的面板创建方法 - 空实现
lv_obj_t* LVGLInterface::createHeaderPanel() {
    return nullptr;
}

lv_obj_t* LVGLInterface::createControlPanel(lv_obj_t* parent) {
    return nullptr;
}

lv_obj_t* LVGLInterface::createFooterPanel() {
    return nullptr;
}

lv_obj_t* LVGLInterface::createStatusPanel() {
    return nullptr;
}

} // namespace ui
} // namespace bamboo_cut