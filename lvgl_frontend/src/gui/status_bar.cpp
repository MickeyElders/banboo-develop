#include "gui/status_bar.h"
#include <stdio.h>
#include <string.h>

Status_bar::Status_bar() : container_(nullptr) {
    // Initialize workflow step pointers
    for(int i = 0; i < 5; i++) {
        workflow_steps_[i] = nullptr;
    }
}

Status_bar::~Status_bar() {
    // LVGL objects are automatically cleaned up
}

bool Status_bar::initialize() {
    printf("Initializing status_bar with Grid/Flex layout\n");
    
    // Create main container
    container_ = lv_obj_create(lv_scr_act());
    if (!container_) {
        printf("Error: Unable to create status bar container\n");
        return false;
    }
    
    // Setup styles and layout
    setup_styles();
    create_layout();
    
    printf("Status bar initialization completed\n");
    return true;
}

void Status_bar::setup_styles() {
    // Container style - corresponds to header-panel in HTML
    lv_style_init(&style_container_);
    lv_style_set_bg_color(&style_container_, lv_color_hex(0x2D2D2D));
    lv_style_set_border_color(&style_container_, lv_color_hex(0x404040));
    lv_style_set_border_width(&style_container_, 2);
    lv_style_set_border_side(&style_container_, LV_BORDER_SIDE_BOTTOM);
    lv_style_set_pad_all(&style_container_, 20);
    
    // Workflow step style
    lv_style_init(&style_step_);
    lv_style_set_bg_color(&style_step_, lv_color_hex(0x1E1E1E));
    lv_style_set_border_color(&style_step_, lv_color_hex(0x404040));
    lv_style_set_border_width(&style_step_, 1);
    lv_style_set_radius(&style_step_, 15);
    lv_style_set_pad_ver(&style_step_, 5);
    lv_style_set_pad_hor(&style_step_, 10);
    lv_style_set_text_color(&style_step_, lv_color_hex(0xFFFFFF));
    
    // Active state style
    lv_style_init(&style_step_active_);
    lv_style_set_bg_color(&style_step_active_, lv_color_hex(0xFF6B35));
    lv_style_set_border_color(&style_step_active_, lv_color_hex(0xFF6B35));
    lv_style_set_shadow_color(&style_step_active_, lv_color_hex(0xFF6B35));
    lv_style_set_shadow_width(&style_step_active_, 10);
    lv_style_set_shadow_opa(&style_step_active_, 128);
    
    // Completed state style
    lv_style_init(&style_step_completed_);
    lv_style_set_bg_color(&style_step_completed_, lv_color_hex(0x4CAF50));
    lv_style_set_border_color(&style_step_completed_, lv_color_hex(0x4CAF50));
}

void Status_bar::create_layout() {
    // Set container to Grid layout - corresponds to header-panel flex layout in HTML
    lv_obj_set_size(container_, LV_PCT(100), 60);
    lv_obj_set_pos(container_, 0, 0);
    lv_obj_add_style(container_, &style_container_, 0);
    lv_obj_set_flex_flow(container_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(container_, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // Create system title
    system_title_ = lv_label_create(container_);
    lv_label_set_text(system_title_, "AI Bamboo Cutting System v2.1 - Modbus TCP");
    lv_obj_set_style_text_color(system_title_, lv_color_hex(0xFFFFFF), 0);
    
    // Create workflow status container
    create_workflow_steps();
    
    // Create heartbeat monitor
    create_heartbeat_monitor();
}

void Status_bar::create_workflow_steps() {
    // Create workflow container - corresponds to workflow-status in HTML
    workflow_container_ = lv_obj_create(container_);
    lv_obj_set_size(workflow_container_, LV_SIZE_CONTENT, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(workflow_container_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(workflow_container_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(workflow_container_, 0, LV_PART_MAIN);
    
    // Set Flex layout
    lv_obj_set_flex_flow(workflow_container_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(workflow_container_, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(workflow_container_, 15, 0);
    
    // Create workflow steps based on PLC.md document
    const char* step_names[] = {"送料检测", "视觉识别", "坐标定位", "夹持切割", "完成推出"};
    
    for(int i = 0; i < 5; i++) {
        workflow_steps_[i] = lv_label_create(workflow_container_);
        lv_label_set_text(workflow_steps_[i], step_names[i]);
        lv_obj_add_style(workflow_steps_[i], &style_step_, 0);
    }
}

void Status_bar::create_heartbeat_monitor() {
    // Create heartbeat monitor container - corresponds to heartbeat-monitor in HTML
    heartbeat_monitor_ = lv_obj_create(container_);
    lv_obj_set_size(heartbeat_monitor_, LV_SIZE_CONTENT, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(heartbeat_monitor_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(heartbeat_monitor_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(heartbeat_monitor_, 0, LV_PART_MAIN);
    
    // Set Flex layout
    lv_obj_set_flex_flow(heartbeat_monitor_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(heartbeat_monitor_, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(heartbeat_monitor_, 8, 0);
    
    // Create heartbeat indicator dot
    lv_obj_t* heartbeat_dot = lv_obj_create(heartbeat_monitor_);
    lv_obj_set_size(heartbeat_dot, 8, 8);
    lv_obj_set_style_radius(heartbeat_dot, LV_RADIUS_CIRCLE, 0);
    lv_obj_set_style_bg_color(heartbeat_dot, lv_color_hex(0x4CAF50), 0);
    lv_obj_set_style_border_opa(heartbeat_dot, LV_OPA_TRANSP, 0);
    
    // Create heartbeat label
    heartbeat_label_ = lv_label_create(heartbeat_monitor_);
    lv_label_set_text(heartbeat_label_, "Beat: 12345");
    lv_obj_set_style_text_color(heartbeat_label_, lv_color_hex(0x4CAF50), 0);
    
    // Create response time label
    response_label_ = lv_label_create(heartbeat_monitor_);
    lv_label_set_text(response_label_, "Resp: 12ms");
    lv_obj_set_style_text_color(response_label_, lv_color_hex(0x4CAF50), 0);
    lv_obj_set_style_pad_left(response_label_, 15, 0);
}

void Status_bar::update_system_title(const char* title) {
    if (system_title_ && title) {
        lv_label_set_text(system_title_, title);
    }
}

void Status_bar::update_workflow_status(int step) {
    if (step < 1 || step > 5) return;
    
    // Clear all step special styles
    for(int i = 0; i < 5; i++) {
        if (workflow_steps_[i]) {
            lv_obj_remove_style(workflow_steps_[i], &style_step_active_, 0);
            lv_obj_remove_style(workflow_steps_[i], &style_step_completed_, 0);
        }
    }
    
    // Set current step and completed step styles
    for(int i = 0; i < 5; i++) {
        if (workflow_steps_[i]) {
            if (i < step - 1) {
                // Completed steps
                lv_obj_add_style(workflow_steps_[i], &style_step_completed_, 0);
            } else if (i == step - 1) {
                // Currently active step
                lv_obj_add_style(workflow_steps_[i], &style_step_active_, 0);
            }
        }
    }
}

void Status_bar::update_heartbeat(uint32_t count, uint32_t response_ms) {
    char buffer[64];
    
    if (heartbeat_label_) {
        snprintf(buffer, sizeof(buffer), "Beat: %u", count);
        lv_label_set_text(heartbeat_label_, buffer);
    }
    
    if (response_label_) {
        snprintf(buffer, sizeof(buffer), "Resp: %ums", response_ms);
        lv_label_set_text(response_label_, buffer);
    }
}