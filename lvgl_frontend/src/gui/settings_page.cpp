#include "gui/settings_page.h"
#include "gui/status_bar.h"
#include "gui/video_view.h"
#include "gui/control_panel.h"
#include <stdio.h>

Settings_page::Settings_page() : container_(nullptr) {
}

Settings_page::~Settings_page() {
    // LVGL objects are automatically cleaned up
}

bool Settings_page::initialize() {
    printf("Initializing settings_page as main layout manager\n");
    
    // Create main container as root container for entire application
    container_ = lv_obj_create(lv_scr_act());
    if (!container_) {
        printf("Error: Unable to create main layout container\n");
        return false;
    }
    
    // Set to full screen
    lv_obj_set_size(container_, LV_PCT(100), LV_PCT(100));
    lv_obj_set_pos(container_, 0, 0);
    
    // Setup styles and layout
    setup_styles();
    
    printf("Main layout manager initialization completed\n");
    return true;
}

void Settings_page::setup_styles() {
    // Main container style - corresponds to main-container in HTML
    lv_style_init(&style_main_);
    lv_style_set_bg_color(&style_main_, lv_color_hex(0x1E1E1E));
    lv_style_set_border_opa(&style_main_, LV_OPA_TRANSP);
    lv_style_set_pad_all(&style_main_, 0);
    
    // Footer panel style - corresponds to footer-panel in HTML
    lv_style_init(&style_footer_);
    lv_style_set_bg_color(&style_footer_, lv_color_hex(0x2D2D2D));
    lv_style_set_border_color(&style_footer_, lv_color_hex(0x404040));
    lv_style_set_border_width(&style_footer_, 2);
    lv_style_set_border_side(&style_footer_, LV_BORDER_SIDE_TOP);
    lv_style_set_pad_hor(&style_footer_, 20);
    lv_style_set_pad_ver(&style_footer_, 0);
    
    // Button style
    lv_style_init(&style_btn_);
    lv_style_set_bg_color(&style_btn_, lv_color_hex(0x2D2D2D));
    lv_style_set_border_color(&style_btn_, lv_color_hex(0xFFFFFF));
    lv_style_set_border_width(&style_btn_, 2);
    lv_style_set_radius(&style_btn_, 6);
    lv_style_set_text_color(&style_btn_, lv_color_hex(0xFFFFFF));
    lv_style_set_pad_hor(&style_btn_, 20);
    lv_style_set_pad_ver(&style_btn_, 12);
    lv_style_set_min_width(&style_btn_, 120);
    
    // Start button style
    lv_style_init(&style_btn_start_);
    lv_style_set_bg_color(&style_btn_start_, lv_color_hex(0x4CAF50));
    lv_style_set_border_color(&style_btn_start_, lv_color_hex(0x4CAF50));
    
    // Emergency button style
    lv_style_init(&style_btn_emergency_);
    lv_style_set_bg_color(&style_btn_emergency_, lv_color_hex(0xFF1744));
    lv_style_set_border_color(&style_btn_emergency_, lv_color_hex(0xFF1744));
}

void Settings_page::create_main_layout(Status_bar* status_bar, Video_view* video_view, Control_panel* control_panel) {
    if (!container_) return;
    
    // Apply main container style
    lv_obj_add_style(container_, &style_main_, 0);
    
    // Create main Grid layout - corresponds to main-container grid layout in HTML
    main_grid_ = lv_obj_create(container_);
    lv_obj_set_size(main_grid_, LV_PCT(100), LV_PCT(100));
    lv_obj_set_pos(main_grid_, 0, 0);
    lv_obj_set_style_bg_opa(main_grid_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(main_grid_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(main_grid_, 10, LV_PART_MAIN);
    lv_obj_set_style_pad_top(main_grid_, 0, 0);
    lv_obj_set_style_pad_gap(main_grid_, 10, 0);
    
    // Set Grid layout - 3 rows: header(60px), main(1fr), footer(80px)
    static lv_coord_t row_dsc[] = {60, LV_GRID_FR(1), 80, LV_GRID_TEMPLATE_LAST};
    static lv_coord_t col_dsc[] = {LV_GRID_FR(1), LV_GRID_TEMPLATE_LAST};
    lv_obj_set_grid_dsc_array(main_grid_, col_dsc, row_dsc);
    
    // Add status bar to first row of grid
    if (status_bar && status_bar->get_container()) {
        lv_obj_set_parent(status_bar->get_container(), main_grid_);
        lv_obj_set_grid_cell(status_bar->get_container(), LV_GRID_ALIGN_STRETCH, 0, 1, LV_GRID_ALIGN_STRETCH, 0, 1);
    }
    
    // Create main content area - second row, containing camera area and control panel
    lv_obj_t* main_content = lv_obj_create(main_grid_);
    lv_obj_set_grid_cell(main_content, LV_GRID_ALIGN_STRETCH, 0, 1, LV_GRID_ALIGN_STRETCH, 1, 1);
    lv_obj_set_style_bg_opa(main_content, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(main_content, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(main_content, 0, LV_PART_MAIN);
    lv_obj_set_style_pad_gap(main_content, 10, 0);
    
    // Set main content area Grid layout - 2 columns: camera(1fr), control(380px)
    static lv_coord_t main_col_dsc[] = {LV_GRID_FR(1), 380, LV_GRID_TEMPLATE_LAST};
    static lv_coord_t main_row_dsc[] = {LV_GRID_FR(1), LV_GRID_TEMPLATE_LAST};
    lv_obj_set_grid_dsc_array(main_content, main_col_dsc, main_row_dsc);
    
    // Add video view to left column of main content area
    if (video_view && video_view->get_container()) {
        lv_obj_set_parent(video_view->get_container(), main_content);
        lv_obj_set_grid_cell(video_view->get_container(), LV_GRID_ALIGN_STRETCH, 0, 1, LV_GRID_ALIGN_STRETCH, 0, 1);
    }
    
    // Add control panel to right column of main content area
    if (control_panel && control_panel->get_container()) {
        lv_obj_set_parent(control_panel->get_container(), main_content);
        lv_obj_set_grid_cell(control_panel->get_container(), LV_GRID_ALIGN_STRETCH, 1, 1, LV_GRID_ALIGN_STRETCH, 0, 1);
    }
    
    // Create footer operation panel
    create_footer_panel();
}

void Settings_page::create_footer_panel() {
    // Create footer panel - corresponds to footer-panel in HTML
    footer_panel_ = lv_obj_create(main_grid_);
    lv_obj_set_grid_cell(footer_panel_, LV_GRID_ALIGN_STRETCH, 0, 1, LV_GRID_ALIGN_STRETCH, 2, 1);
    lv_obj_add_style(footer_panel_, &style_footer_, 0);
    
    // Set Flex layout
    lv_obj_set_flex_flow(footer_panel_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(footer_panel_, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // Create control button container - corresponds to control-buttons in HTML
    control_buttons_ = lv_obj_create(footer_panel_);
    lv_obj_set_size(control_buttons_, LV_SIZE_CONTENT, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(control_buttons_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(control_buttons_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(control_buttons_, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(control_buttons_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(control_buttons_, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(control_buttons_, 15, 0);
    
    // Create start button
    start_btn_ = lv_btn_create(control_buttons_);
    lv_obj_add_style(start_btn_, &style_btn_, 0);
    lv_obj_add_style(start_btn_, &style_btn_start_, 0);
    lv_obj_t* start_label = lv_label_create(start_btn_);
    lv_label_set_text(start_label, "▶ Start System");
    lv_obj_center(start_label);
    lv_obj_add_event_cb(start_btn_, button_event_cb, LV_EVENT_CLICKED, (void*)0);
    
    // Create pause button
    pause_btn_ = lv_btn_create(control_buttons_);
    lv_obj_add_style(pause_btn_, &style_btn_, 0);
    lv_obj_set_style_border_color(pause_btn_, lv_color_hex(0xFFC107), 0);
    lv_obj_set_style_text_color(pause_btn_, lv_color_hex(0xFFC107), 0);
    lv_obj_t* pause_label = lv_label_create(pause_btn_);
    lv_label_set_text(pause_label, "⏸ Pause");
    lv_obj_center(pause_label);
    lv_obj_add_event_cb(pause_btn_, button_event_cb, LV_EVENT_CLICKED, (void*)1);
    
    // Create stop button
    stop_btn_ = lv_btn_create(control_buttons_);
    lv_obj_add_style(stop_btn_, &style_btn_, 0);
    lv_obj_set_style_border_color(stop_btn_, lv_color_hex(0xF44336), 0);
    lv_obj_set_style_text_color(stop_btn_, lv_color_hex(0xF44336), 0);
    lv_obj_t* stop_label = lv_label_create(stop_btn_);
    lv_label_set_text(stop_label, "⏹ Stop");
    lv_obj_center(stop_label);
    lv_obj_add_event_cb(stop_btn_, button_event_cb, LV_EVENT_CLICKED, (void*)2);
    
    // Create status info container
    status_info_ = lv_obj_create(footer_panel_);
    lv_obj_set_size(status_info_, LV_SIZE_CONTENT, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(status_info_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(status_info_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(status_info_, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(status_info_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(status_info_, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(status_info_, 5, 0);
    
    // Current process label
    current_process_label_ = lv_label_create(status_info_);
    lv_label_set_text(current_process_label_, "Current Process: Feed Detection");
    lv_obj_set_style_text_color(current_process_label_, lv_color_hex(0xB0B0B0), 0);
    
    // Status info label
    status_info_label_ = lv_label_create(status_info_);
    lv_label_set_text(status_info_label_, "Last Cut: 14:25:33 | Today: 89 cuts | Efficiency: 94.2%");
    lv_obj_set_style_text_color(status_info_label_, lv_color_hex(0xB0B0B0), 0);
    
    // Create emergency operation button container - corresponds to emergency-section in HTML
    emergency_buttons_ = lv_obj_create(footer_panel_);
    lv_obj_set_size(emergency_buttons_, LV_SIZE_CONTENT, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(emergency_buttons_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(emergency_buttons_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(emergency_buttons_, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(emergency_buttons_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(emergency_buttons_, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(emergency_buttons_, 10, 0);
    
    // Create emergency stop button
    emergency_btn_ = lv_btn_create(emergency_buttons_);
    lv_obj_add_style(emergency_btn_, &style_btn_, 0);
    lv_obj_add_style(emergency_btn_, &style_btn_emergency_, 0);
    lv_obj_t* emergency_label = lv_label_create(emergency_btn_);
    lv_label_set_text(emergency_label, "🚨 Emergency Stop");
    lv_obj_center(emergency_label);
    lv_obj_add_event_cb(emergency_btn_, emergency_event_cb, LV_EVENT_CLICKED, (void*)0);
    
    // Create power button
    power_btn_ = lv_btn_create(emergency_buttons_);
    lv_obj_add_style(power_btn_, &style_btn_, 0);
    lv_obj_set_style_border_color(power_btn_, lv_color_hex(0x9C27B0), 0);
    lv_obj_set_style_text_color(power_btn_, lv_color_hex(0x9C27B0), 0);
    lv_obj_t* power_label = lv_label_create(power_btn_);
    lv_label_set_text(power_label, "⏻ Shutdown");
    lv_obj_center(power_label);
    lv_obj_add_event_cb(power_btn_, emergency_event_cb, LV_EVENT_CLICKED, (void*)1);
}

void Settings_page::update_layout_positions() {
    // Responsive adjustment, update layout when screen size changes
    // Currently maintains fixed layout, can add dynamic adjustment logic here
}

// Event handling
void Settings_page::button_event_cb(lv_event_t* e) {
    int button_id = (int)(intptr_t)lv_event_get_user_data(e);
    
    switch(button_id) {
        case 0: // Start
            printf("Start system button clicked\n");
            break;
        case 1: // Pause
            printf("Pause button clicked\n");
            break;
        case 2: // Stop
            printf("Stop button clicked\n");
            break;
    }
}

void Settings_page::emergency_event_cb(lv_event_t* e) {
    int button_id = (int)(intptr_t)lv_event_get_user_data(e);
    
    switch(button_id) {
        case 0: // Emergency stop
            printf("Emergency stop button clicked\n");
            break;
        case 1: // Shutdown
            printf("Shutdown button clicked\n");
            break;
    }
}