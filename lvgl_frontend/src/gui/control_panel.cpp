#include "gui/control_panel.h"
#include <stdio.h>
#include <string.h>

Control_panel::Control_panel() : container_(nullptr) {
    // Initialize pointer arrays
    for(int i = 0; i < 8; i++) {
        register_labels_[i] = nullptr;
    }
    for(int i = 0; i < 3; i++) {
        blade_buttons_[i] = nullptr;
    }
    for(int i = 0; i < 12; i++) {
        system_info_labels_[i] = nullptr;
    }
}

Control_panel::~Control_panel() {
    // LVGL objects are automatically cleaned up
}

bool Control_panel::initialize() {
    printf("Initializing control_panel with Flex layout\n");
    
    // Create main container
    container_ = lv_obj_create(lv_scr_act());
    if (!container_) {
        printf("Error: Unable to create control panel container\n");
        return false;
    }
    
    // Setup styles and layout
    setup_styles();
    create_layout();
    
    printf("Control panel initialization completed\n");
    return true;
}

void Control_panel::setup_styles() {
    // Main container style - corresponds to control-panel in HTML
    lv_style_init(&style_container_);
    lv_style_set_bg_color(&style_container_, lv_color_hex(0x2D2D2D));
    lv_style_set_border_color(&style_container_, lv_color_hex(0x404040));
    lv_style_set_border_width(&style_container_, 2);
    lv_style_set_radius(&style_container_, 8);
    lv_style_set_pad_all(&style_container_, 15);
    
    // Section style - corresponds to panel-section in HTML
    lv_style_init(&style_section_);
    lv_style_set_bg_color(&style_section_, lv_color_hex(0x1A1A1A));
    lv_style_set_border_width(&style_section_, 1);
    lv_style_set_radius(&style_section_, 5);
    lv_style_set_pad_all(&style_section_, 8);
    
    // Modbus section style
    lv_style_init(&style_modbus_);
    lv_style_set_border_color(&style_modbus_, lv_color_hex(0x2196F3));
    
    // PLC section style
    lv_style_init(&style_plc_);
    lv_style_set_border_color(&style_plc_, lv_color_hex(0x4CAF50));
    
    // Jetson section style
    lv_style_init(&style_jetson_);
    lv_style_set_border_color(&style_jetson_, lv_color_hex(0x76B900));
    
    // Table style
    lv_style_init(&style_table_);
    lv_style_set_border_color(&style_table_, lv_color_hex(0x404040));
    lv_style_set_border_width(&style_table_, 1);
    
    // Progress bar style
    lv_style_init(&style_progress_);
    lv_style_set_bg_color(&style_progress_, lv_color_hex(0x404040));
    lv_style_set_radius(&style_progress_, 3);
    
    // Button style
    lv_style_init(&style_button_);
    lv_style_set_bg_color(&style_button_, lv_color_hex(0x1E1E1E));
    lv_style_set_border_color(&style_button_, lv_color_hex(0x404040));
    lv_style_set_border_width(&style_button_, 1);
    lv_style_set_radius(&style_button_, 3);
    lv_style_set_text_color(&style_button_, lv_color_hex(0xB0B0B0));
    
    // Active button style
    lv_style_init(&style_button_active_);
    lv_style_set_bg_color(&style_button_active_, lv_color_hex(0xFF6B35));
    lv_style_set_border_color(&style_button_active_, lv_color_hex(0xFF6B35));
    lv_style_set_text_color(&style_button_active_, lv_color_hex(0xFFFFFF));
}

void Control_panel::create_layout() {
    // Set container to Flex layout, vertical arrangement, with scrolling
    lv_obj_add_style(container_, &style_container_, 0);
    lv_obj_set_flex_flow(container_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(container_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(container_, 6, 0);
    lv_obj_set_scroll_dir(container_, LV_DIR_VER);
    
    // Create various functional sections
    create_modbus_section();
    create_plc_section();
    create_jetson_section();
    create_ai_section();
    create_comm_section();
}

void Control_panel::create_modbus_section() {
    // Create Modbus register status section
    modbus_section_ = lv_obj_create(container_);
    lv_obj_set_size(modbus_section_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_add_style(modbus_section_, &style_section_, 0);
    lv_obj_add_style(modbus_section_, &style_modbus_, 0);
    
    // Set Flex layout
    lv_obj_set_flex_flow(modbus_section_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(modbus_section_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(modbus_section_, 6, 0);
    
    // Create title
    lv_obj_t* title = lv_label_create(modbus_section_);
    lv_label_set_text(title, "Modbus Registers");
    lv_obj_set_style_text_color(title, lv_color_hex(0x2196F3), 0);
    
    // Create register table
    modbus_table_ = lv_table_create(modbus_section_);
    lv_obj_set_size(modbus_table_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_add_style(modbus_table_, &style_table_, 0);
    
    // Set table content
    lv_table_set_col_cnt(modbus_table_, 3);
    lv_table_set_row_cnt(modbus_table_, 9); // 1 header + 8 data rows
    
    // Set table header
    lv_table_set_cell_value(modbus_table_, 0, 0, "Address");
    lv_table_set_cell_value(modbus_table_, 0, 1, "Description");
    lv_table_set_cell_value(modbus_table_, 0, 2, "Value");
    
    // Set data rows - Extended display according to PLC.md
    const char* register_addrs[] = {"40001", "40002", "40003", "40004-05", "40006", "40007-08", "40009", "40010", "40011", "40014", "40015-16", "40017", "40018", "40019"};
    const char* register_descs[] = {"System Status", "PLC Command", "Coord Ready", "X Coordinate", "Cut Quality", "Heartbeat", "Blade Number", "System Health", "Waste Detection", "Rail Direction", "Remaining Length", "Coverage", "Detection Status", "Process Mode"};
    const char* register_values[] = {"1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"};
    
    // Create larger table to display all registers
    lv_table_set_row_cnt(modbus_table_, 15); // 1 header + 14 data rows
    
    for(int i = 0; i < 14; i++) {
        lv_table_set_cell_value(modbus_table_, i + 1, 0, register_addrs[i]);
        lv_table_set_cell_value(modbus_table_, i + 1, 1, register_descs[i]);
        lv_table_set_cell_value(modbus_table_, i + 1, 2, register_values[i]);
        
        // Set value column with specific symbols for different statuses
        if (i == 0) {  // System State - OK
            lv_table_set_cell_value(modbus_table_, i + 1, 2, LV_SYMBOL_OK);
        } else if (i == 4) {  // Cut Quality - Good quality symbol
            lv_table_set_cell_value(modbus_table_, i + 1, 2, LV_SYMBOL_OK " Normal");
        } else {  // Other statuses with their values
            lv_table_set_cell_value(modbus_table_, i + 1, 2, register_values[i]);
        }
    }
    
    // Set table style - Configure table and cell background colors
    lv_obj_set_style_bg_color(modbus_table_, lv_color_hex(0x1A237E), 0);  // Deep blue background for table
    lv_obj_set_style_bg_color(modbus_table_, lv_color_hex(0x1A237E), LV_PART_ITEMS);  // Deep blue background for cells
    lv_obj_set_style_text_color(modbus_table_, lv_color_hex(0xFFFFFF), 0);  // White text for table
    lv_obj_set_style_text_color(modbus_table_, lv_color_hex(0xFFFFFF), LV_PART_ITEMS);  // White text for cells
    lv_obj_set_style_border_color(modbus_table_, lv_color_hex(0x404040), LV_PART_ITEMS);  // Dark border for cells
    lv_obj_set_style_border_width(modbus_table_, 1, LV_PART_ITEMS);  // Border width for cells
}

void Control_panel::create_plc_section() {
    // Create PLC communication status section
    plc_section_ = lv_obj_create(container_);
    lv_obj_set_size(plc_section_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_add_style(plc_section_, &style_section_, 0);
    lv_obj_add_style(plc_section_, &style_plc_, 0);
    
    // Set Flex layout
    lv_obj_set_flex_flow(plc_section_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(plc_section_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(plc_section_, 6, 0);
    
    // Create title
    lv_obj_t* title = lv_label_create(plc_section_);
    lv_label_set_text(title, LV_SYMBOL_WIFI " PLC Communication");
    lv_obj_set_style_text_color(title, lv_color_hex(0x4CAF50), 0);
    
    // Create status grid container
    lv_obj_t* status_grid = lv_obj_create(plc_section_);
    lv_obj_set_size(status_grid, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(status_grid, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(status_grid, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(status_grid, 0, LV_PART_MAIN);
    
    // Set 2x2 grid layout
    static lv_coord_t col_dsc[] = {LV_GRID_FR(1), LV_GRID_FR(1), LV_GRID_TEMPLATE_LAST};
    static lv_coord_t row_dsc[] = {LV_SIZE_CONTENT, LV_SIZE_CONTENT, LV_GRID_TEMPLATE_LAST};
    lv_obj_set_grid_dsc_array(status_grid, col_dsc, row_dsc);
    lv_obj_set_style_pad_gap(status_grid, 4, 0);
    
    // Create status items
    const char* status_labels[] = {"Connection:", "PLC Address:", "Response:", "Total Cuts:"};
    const char* status_values[] = {"Connected", "192.168.1.100", "12ms", "1,247"};
    lv_obj_t** status_value_labels[] = {&plc_status_label_, &plc_address_label_, &plc_response_label_, &total_cuts_label_};
    
    for(int i = 0; i < 4; i++) {
        lv_obj_t* status_item = lv_obj_create(status_grid);
        lv_obj_set_grid_cell(status_item, LV_GRID_ALIGN_STRETCH, i % 2, 1, LV_GRID_ALIGN_CENTER, i / 2, 1);
        lv_obj_set_style_bg_color(status_item, lv_color_hex(0x0D0D0D), 0);
        lv_obj_set_style_border_opa(status_item, LV_OPA_TRANSP, 0);
        lv_obj_set_style_radius(status_item, 3, 0);
        lv_obj_set_style_pad_all(status_item, 3, 0);  // 减少内边距从6px到3px
        lv_obj_set_style_pad_gap(status_item, 1, 0);  // 设置标签之间的间距为1px
        lv_obj_set_flex_flow(status_item, LV_FLEX_FLOW_COLUMN);
        lv_obj_set_flex_align(status_item, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);  // 垂直居中对齐
        
        lv_obj_t* label = lv_label_create(status_item);
        lv_label_set_text(label, status_labels[i]);
        lv_obj_set_style_text_color(label, lv_color_hex(0xB0B0B0), 0);
        
        *status_value_labels[i] = lv_label_create(status_item);
        lv_label_set_text(*status_value_labels[i], status_values[i]);
        lv_obj_set_style_text_color(*status_value_labels[i], lv_color_hex(0xFFFFFF), 0);
        
        // Connection status displayed in green
        if(i == 0) {
            lv_obj_set_style_text_color(*status_value_labels[i], lv_color_hex(0x4CAF50), 0);
        }
    }
    
    // Create blade selector
    lv_obj_t* blade_container = lv_obj_create(plc_section_);
    lv_obj_set_size(blade_container, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(blade_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(blade_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(blade_container, 0, LV_PART_MAIN);
    lv_obj_set_style_pad_top(blade_container, 4, 0);
    
    lv_obj_set_flex_flow(blade_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(blade_container, LV_FLEX_ALIGN_SPACE_EVENLY, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(blade_container, 3, 0);
    
    // 根据PLC.md文档创建PLC控制按键
    const char* plc_control_names[] = {"Start Feed", "Feed Detect", "Cut Prepare"};
    const char* plc_control_symbols[] = {LV_SYMBOL_PLAY, LV_SYMBOL_EYE_OPEN, LV_SYMBOL_SETTINGS};
    
    for(int i = 0; i < 3; i++) {
        blade_buttons_[i] = lv_btn_create(blade_container);
        lv_obj_set_flex_grow(blade_buttons_[i], 1);
        lv_obj_set_height(blade_buttons_[i], LV_SIZE_CONTENT);
        lv_obj_add_style(blade_buttons_[i], &style_button_, 0);
        lv_obj_set_style_pad_ver(blade_buttons_[i], 3, 0);
        
        lv_obj_t* label = lv_label_create(blade_buttons_[i]);
        char button_text[64];
        snprintf(button_text, sizeof(button_text), "%s %s", plc_control_symbols[i], plc_control_names[i]);
        lv_label_set_text(label, button_text);
        lv_obj_center(label);
        
        // Set event callback for PLC commands
        lv_obj_add_event_cb(blade_buttons_[i], plc_command_button_event_cb, LV_EVENT_CLICKED, this);
        lv_obj_set_user_data(blade_buttons_[i], (void*)(intptr_t)(i + 4)); // PLC命令值: 4=启动送料, 1=进料检测, 2=切割准备
    }
    
    // 添加废料处理和紧急停止按钮
    lv_obj_t* emergency_container = lv_obj_create(plc_section_);
    lv_obj_set_size(emergency_container, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(emergency_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(emergency_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(emergency_container, 0, LV_PART_MAIN);
    lv_obj_set_style_pad_top(emergency_container, 4, 0);
    
    lv_obj_set_flex_flow(emergency_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(emergency_container, LV_FLEX_ALIGN_SPACE_EVENLY, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(emergency_container, 3, 0);
    
    // 废料推出按钮
    lv_obj_t* waste_btn = lv_btn_create(emergency_container);
    lv_obj_set_flex_grow(waste_btn, 1);
    lv_obj_set_height(waste_btn, LV_SIZE_CONTENT);
    lv_obj_add_style(waste_btn, &style_button_, 0);
    lv_obj_set_style_pad_ver(waste_btn, 3, 0);
    lv_obj_set_style_bg_color(waste_btn, lv_color_hex(0xFFC107), 0); // 黄色
    
    lv_obj_t* waste_label = lv_label_create(waste_btn);
    lv_label_set_text(waste_label, LV_SYMBOL_TRASH " 废料推出");
    lv_obj_center(waste_label);
    lv_obj_add_event_cb(waste_btn, plc_command_button_event_cb, LV_EVENT_CLICKED, this);
    lv_obj_set_user_data(waste_btn, (void*)(intptr_t)8); // PLC命令: 8=废料推出
    
    // 紧急停止按钮
    lv_obj_t* emergency_btn = lv_btn_create(emergency_container);
    lv_obj_set_flex_grow(emergency_btn, 1);
    lv_obj_set_height(emergency_btn, LV_SIZE_CONTENT);
    lv_obj_add_style(emergency_btn, &style_button_, 0);
    lv_obj_set_style_pad_ver(emergency_btn, 3, 0);
    lv_obj_set_style_bg_color(emergency_btn, lv_color_hex(0xDC3545), 0); // 红色
    
    lv_obj_t* emergency_label = lv_label_create(emergency_btn);
    lv_label_set_text(emergency_label, LV_SYMBOL_STOP " 紧急停止");
    lv_obj_center(emergency_label);
    lv_obj_add_event_cb(emergency_btn, plc_command_button_event_cb, LV_EVENT_CLICKED, this);
    lv_obj_set_user_data(emergency_btn, (void*)(intptr_t)6); // PLC命令: 6=紧急停止
}

void Control_panel::create_jetson_section() {
    // Create Jetson system info section
    jetson_section_ = lv_obj_create(container_);
    lv_obj_set_size(jetson_section_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_add_style(jetson_section_, &style_section_, 0);
    lv_obj_add_style(jetson_section_, &style_jetson_, 0);
    
    // Set Flex layout
    lv_obj_set_flex_flow(jetson_section_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(jetson_section_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(jetson_section_, 4, 0);
    
    // Create title
    lv_obj_t* title = lv_label_create(jetson_section_);
    lv_label_set_text(title, LV_SYMBOL_CHARGE " Jetson Orin Nano 8GB 15W");
    lv_obj_set_style_text_color(title, lv_color_hex(0x76B900), 0);
    
    // Create CPU progress bar
    lv_obj_t* cpu_container = lv_obj_create(jetson_section_);
    lv_obj_set_size(cpu_container, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(cpu_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(cpu_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(cpu_container, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(cpu_container, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_gap(cpu_container, 2, 0);
    
    lv_obj_t* cpu_label_container = lv_obj_create(cpu_container);
    lv_obj_set_size(cpu_label_container, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(cpu_label_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(cpu_label_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(cpu_label_container, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(cpu_label_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(cpu_label_container, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    lv_obj_t* cpu_desc = lv_label_create(cpu_label_container);
    lv_label_set_text(cpu_desc, "CPU (6-core ARM Cortex-A78AE)");
    lv_obj_set_style_text_color(cpu_desc, lv_color_hex(0xB0B0B0), 0);
    
    cpu_usage_label_ = lv_label_create(cpu_label_container);
    lv_label_set_text(cpu_usage_label_, "45%");
    lv_obj_set_style_text_color(cpu_usage_label_, lv_color_hex(0xFFFFFF), 0);
    
    cpu_progress_ = lv_bar_create(cpu_container);
    lv_obj_set_size(cpu_progress_, LV_PCT(100), 6);
    lv_obj_add_style(cpu_progress_, &style_progress_, 0);
    lv_obj_set_style_bg_color(cpu_progress_, lv_color_hex(0x76B900), LV_PART_INDICATOR);
    lv_bar_set_value(cpu_progress_, 45, LV_ANIM_OFF);
    
    // Create GPU progress bar
    lv_obj_t* gpu_container = lv_obj_create(jetson_section_);
    lv_obj_set_size(gpu_container, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(gpu_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(gpu_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(gpu_container, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(gpu_container, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_gap(gpu_container, 2, 0);
    
    lv_obj_t* gpu_label_container = lv_obj_create(gpu_container);
    lv_obj_set_size(gpu_label_container, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(gpu_label_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(gpu_label_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(gpu_label_container, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(gpu_label_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(gpu_label_container, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    lv_obj_t* gpu_desc = lv_label_create(gpu_label_container);
    lv_label_set_text(gpu_desc, "GPU (1024-core NVIDIA Ampere)");
    lv_obj_set_style_text_color(gpu_desc, lv_color_hex(0xB0B0B0), 0);
    
    gpu_usage_label_ = lv_label_create(gpu_label_container);
    lv_label_set_text(gpu_usage_label_, "32%");
    lv_obj_set_style_text_color(gpu_usage_label_, lv_color_hex(0xFFFFFF), 0);
    
    gpu_progress_ = lv_bar_create(gpu_container);
    lv_obj_set_size(gpu_progress_, LV_PCT(100), 6);
    lv_obj_add_style(gpu_progress_, &style_progress_, 0);
    lv_obj_set_style_bg_color(gpu_progress_, lv_color_hex(0xFF6B35), LV_PART_INDICATOR);
    lv_bar_set_value(gpu_progress_, 32, LV_ANIM_OFF);
    
    // Create memory progress bar
    lv_obj_t* memory_container = lv_obj_create(jetson_section_);
    lv_obj_set_size(memory_container, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(memory_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(memory_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(memory_container, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(memory_container, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_gap(memory_container, 2, 0);
    
    lv_obj_t* memory_label_container = lv_obj_create(memory_container);
    lv_obj_set_size(memory_label_container, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(memory_label_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(memory_label_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(memory_label_container, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(memory_label_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(memory_label_container, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    lv_obj_t* memory_desc = lv_label_create(memory_label_container);
    lv_label_set_text(memory_desc, "Memory (LPDDR5)");
    lv_obj_set_style_text_color(memory_desc, lv_color_hex(0xB0B0B0), 0);
    
    memory_usage_label_ = lv_label_create(memory_label_container);
    lv_label_set_text(memory_usage_label_, "2.1GB/8GB");
    lv_obj_set_style_text_color(memory_usage_label_, lv_color_hex(0xFFFFFF), 0);
    
    memory_progress_ = lv_bar_create(memory_container);
    lv_obj_set_size(memory_progress_, LV_PCT(100), 6);
    lv_obj_add_style(memory_progress_, &style_progress_, 0);
    lv_obj_set_style_bg_color(memory_progress_, lv_color_hex(0xFFC107), LV_PART_INDICATOR);
    lv_bar_set_value(memory_progress_, 26, LV_ANIM_OFF);
}

void Control_panel::create_ai_section() {
    // Create AI model status section
    ai_section_ = lv_obj_create(container_);
    lv_obj_set_size(ai_section_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_add_style(ai_section_, &style_section_, 0);
    lv_obj_set_style_border_color(ai_section_, lv_color_hex(0xFF6B35), 0);
    
    // Set Flex layout
    lv_obj_set_flex_flow(ai_section_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(ai_section_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(ai_section_, 4, 0);
    
    // Create title
    lv_obj_t* title = lv_label_create(ai_section_);
    lv_label_set_text(title, LV_SYMBOL_SETTINGS " AI Model Status");
    lv_obj_set_style_text_color(title, lv_color_hex(0xFF6B35), 0);
    
    // Create status grid container
    lv_obj_t* ai_grid = lv_obj_create(ai_section_);
    lv_obj_set_size(ai_grid, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(ai_grid, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(ai_grid, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(ai_grid, 0, LV_PART_MAIN);
    
    // Set 2x3 grid layout
    static lv_coord_t ai_col_dsc[] = {LV_GRID_FR(1), LV_GRID_FR(1), LV_GRID_TEMPLATE_LAST};
    static lv_coord_t ai_row_dsc[] = {LV_SIZE_CONTENT, LV_SIZE_CONTENT, LV_SIZE_CONTENT, LV_GRID_TEMPLATE_LAST};
    lv_obj_set_grid_dsc_array(ai_grid, ai_col_dsc, ai_row_dsc);
    lv_obj_set_style_pad_gap(ai_grid, 4, 0);
    
    // Create AI status items
    const char* ai_labels[] = {"Model Ver:", "Inference:", "Confidence:", "Accuracy:", "Total Det:", "Today Det:"};
    const char* ai_values[] = {"YOLOv8n", "15.3ms", "0.85", "94.2%", "15,432", "89"};
    lv_obj_t** ai_value_labels[] = {nullptr, &inference_time_label_, nullptr, &accuracy_label_, &total_detections_label_, &today_detections_label_};
    
    for(int i = 0; i < 6; i++) {
        lv_obj_t* ai_item = lv_obj_create(ai_grid);
        lv_obj_set_grid_cell(ai_item, LV_GRID_ALIGN_STRETCH, i % 2, 1, LV_GRID_ALIGN_CENTER, i / 2, 1);
        lv_obj_set_style_bg_color(ai_item, lv_color_hex(0x0D0D0D), 0);
        lv_obj_set_style_border_opa(ai_item, LV_OPA_TRANSP, 0);
        lv_obj_set_style_radius(ai_item, 3, 0);
        lv_obj_set_style_pad_all(ai_item, 3, 0);  // 减少内边距从6px到3px
        lv_obj_set_style_pad_gap(ai_item, 1, 0);  // 设置标签之间的间距为1px
        lv_obj_set_flex_flow(ai_item, LV_FLEX_FLOW_COLUMN);
        lv_obj_set_flex_align(ai_item, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);  // 垂直居中对齐
        
        lv_obj_t* label = lv_label_create(ai_item);
        lv_label_set_text(label, ai_labels[i]);
        lv_obj_set_style_text_color(label, lv_color_hex(0xB0B0B0), 0);
        
        lv_obj_t* value_label = lv_label_create(ai_item);
        lv_label_set_text(value_label, ai_values[i]);
        lv_obj_set_style_text_color(value_label, lv_color_hex(0xFFFFFF), 0);
        
        if(ai_value_labels[i]) {
            *ai_value_labels[i] = value_label;
        }
    }
}

void Control_panel::create_comm_section() {
    // Create communication statistics section
    comm_section_ = lv_obj_create(container_);
    lv_obj_set_size(comm_section_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_add_style(comm_section_, &style_section_, 0);
    lv_obj_set_style_border_color(comm_section_, lv_color_hex(0xB0B0B0), 0);
    
    // Set Flex layout
    lv_obj_set_flex_flow(comm_section_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(comm_section_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(comm_section_, 4, 0);
    
    // Create title
    lv_obj_t* title = lv_label_create(comm_section_);
    lv_label_set_text(title, LV_SYMBOL_LIST " Communication Stats");
    lv_obj_set_style_text_color(title, lv_color_hex(0xB0B0B0), 0);
    
    // Create status grid container
    lv_obj_t* comm_grid = lv_obj_create(comm_section_);
    lv_obj_set_size(comm_grid, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(comm_grid, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(comm_grid, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(comm_grid, 0, LV_PART_MAIN);
    
    // Set 2x2 grid layout
    static lv_coord_t comm_col_dsc[] = {LV_GRID_FR(1), LV_GRID_FR(1), LV_GRID_TEMPLATE_LAST};
    static lv_coord_t comm_row_dsc[] = {LV_SIZE_CONTENT, LV_SIZE_CONTENT, LV_GRID_TEMPLATE_LAST};
    lv_obj_set_grid_dsc_array(comm_grid, comm_col_dsc, comm_row_dsc);
    lv_obj_set_style_pad_gap(comm_grid, 4, 0);
    
    // Create communication statistics items
    const char* comm_labels[] = {"Uptime:", "Packets:", "Error Rate:", "Throughput:"};
    const char* comm_values[] = {"2h 15m", "15,432", "0.02%", "1.2KB/s"};
    lv_obj_t** comm_value_labels[] = {&connection_time_label_, &packets_label_, &error_rate_label_, &throughput_label_};
    
    for(int i = 0; i < 4; i++) {
        lv_obj_t* comm_item = lv_obj_create(comm_grid);
        lv_obj_set_grid_cell(comm_item, LV_GRID_ALIGN_STRETCH, i % 2, 1, LV_GRID_ALIGN_CENTER, i / 2, 1);
        lv_obj_set_style_bg_color(comm_item, lv_color_hex(0x0D0D0D), 0);
        lv_obj_set_style_border_opa(comm_item, LV_OPA_TRANSP, 0);
        lv_obj_set_style_radius(comm_item, 3, 0);
        lv_obj_set_style_pad_all(comm_item, 3, 0);  // 减少内边距从6px到3px
        lv_obj_set_style_pad_gap(comm_item, 1, 0);  // 设置标签之间的间距为1px
        lv_obj_set_flex_flow(comm_item, LV_FLEX_FLOW_COLUMN);
        lv_obj_set_flex_align(comm_item, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);  // 垂直居中对齐
        
        lv_obj_t* label = lv_label_create(comm_item);
        lv_label_set_text(label, comm_labels[i]);
        lv_obj_set_style_text_color(label, lv_color_hex(0xB0B0B0), 0);
        
        *comm_value_labels[i] = lv_label_create(comm_item);
        lv_label_set_text(*comm_value_labels[i], comm_values[i]);
        lv_obj_set_style_text_color(*comm_value_labels[i], lv_color_hex(0xFFFFFF), 0);
    }
}

// Update method implementation
void Control_panel::update_modbus_registers(const char* reg_values[]) {
    if(!modbus_table_ || !reg_values) return;
    
    for(int i = 0; i < 8; i++) {
        if(reg_values[i]) {
            lv_table_set_cell_value(modbus_table_, i + 1, 2, reg_values[i]);
        }
    }
}

void Control_panel::update_plc_status(const char* status, const char* address, uint32_t response_ms, uint32_t total_cuts) {
    char buffer[64];
    
    if(plc_status_label_ && status) {
        lv_label_set_text(plc_status_label_, status);
    }
    
    if(plc_address_label_ && address) {
        lv_label_set_text(plc_address_label_, address);
    }
    
    if(plc_response_label_) {
        snprintf(buffer, sizeof(buffer), "%ums", response_ms);
        lv_label_set_text(plc_response_label_, buffer);
    }
    
    if(total_cuts_label_) {
        snprintf(buffer, sizeof(buffer), "%u", total_cuts);
        lv_label_set_text(total_cuts_label_, buffer);
    }
}

void Control_panel::update_jetson_info(const performance_stats_t& stats) {
    char buffer[64];
    
    // 更新CPU使用率
    if(cpu_usage_label_) {
        snprintf(buffer, sizeof(buffer), "%.1f%%", stats.cpu_usage);
        lv_label_set_text(cpu_usage_label_, buffer);
    }
    if(cpu_progress_) {
        lv_bar_set_value(cpu_progress_, (int)stats.cpu_usage, LV_ANIM_ON);
    }
    
    // 更新GPU使用率
    if(gpu_usage_label_) {
        snprintf(buffer, sizeof(buffer), "%.1f%%", stats.gpu_usage);
        lv_label_set_text(gpu_usage_label_, buffer);
    }
    if(gpu_progress_) {
        lv_bar_set_value(gpu_progress_, (int)stats.gpu_usage, LV_ANIM_ON);
    }
    
    // 更新内存使用率
    if(memory_usage_label_) {
        snprintf(buffer, sizeof(buffer), "%.1fGB/8GB", stats.memory_usage_mb / 1024.0f);
        lv_label_set_text(memory_usage_label_, buffer);
    }
    if(memory_progress_) {
        int memory_percent = (int)((stats.memory_usage_mb / 1024.0f / 8.0f) * 100);
        lv_bar_set_value(memory_progress_, memory_percent, LV_ANIM_ON);
    }
}

void Control_panel::update_ai_model_status(float inference_time, float accuracy, uint32_t total_detections, uint32_t today_detections) {
    char buffer[64];
    
    if(inference_time_label_) {
        snprintf(buffer, sizeof(buffer), "%.1fms", inference_time);
        lv_label_set_text(inference_time_label_, buffer);
    }
    
    if(accuracy_label_) {
        snprintf(buffer, sizeof(buffer), "%.1f%%", accuracy);
        lv_label_set_text(accuracy_label_, buffer);
    }
    
    if(total_detections_label_) {
        snprintf(buffer, sizeof(buffer), "%u", total_detections);
        lv_label_set_text(total_detections_label_, buffer);
    }
    
    if(today_detections_label_) {
        snprintf(buffer, sizeof(buffer), "%u", today_detections);
        lv_label_set_text(today_detections_label_, buffer);
    }
}

void Control_panel::update_communication_stats(const char* connection_time, uint32_t packets, float error_rate, const char* throughput) {
    char buffer[64];
    
    if(connection_time_label_ && connection_time) {
        lv_label_set_text(connection_time_label_, connection_time);
    }
    
    if(packets_label_) {
        snprintf(buffer, sizeof(buffer), "%u", packets);
        lv_label_set_text(packets_label_, buffer);
    }
    
    if(error_rate_label_) {
        snprintf(buffer, sizeof(buffer), "%.2f%%", error_rate);
        lv_label_set_text(error_rate_label_, buffer);
    }
    
    if(throughput_label_ && throughput) {
        lv_label_set_text(throughput_label_, throughput);
    }
}

void Control_panel::set_blade_selection(int blade_id) {
    if(blade_id < 0 || blade_id >= 3) return;
    
    // 清除所有按钮的激活状态
    for(int i = 0; i < 3; i++) {
        if(blade_buttons_[i]) {
            lv_obj_remove_style(blade_buttons_[i], &style_button_active_, 0);
        }
    }
    
    // 激活选中的按钮
    if(blade_buttons_[blade_id]) {
        lv_obj_add_style(blade_buttons_[blade_id], &style_button_active_, 0);
    }
}

// Event handling
void Control_panel::blade_button_event_cb(lv_event_t* e) {
    lv_obj_t* btn = lv_event_get_target(e);
    Control_panel* panel = (Control_panel*)lv_event_get_user_data(e);
    int blade_id = (int)(intptr_t)lv_obj_get_user_data(btn);
    
    if(panel) {
        panel->set_blade_selection(blade_id);
        printf("Blade selected: %d\n", blade_id + 1);
    }
}

// PLC命令按钮事件处理 - 根据PLC.md文档实现
void Control_panel::plc_command_button_event_cb(lv_event_t* e) {
    lv_obj_t* btn = lv_event_get_target(e);
    Control_panel* panel = (Control_panel*)lv_event_get_user_data(e);
    int plc_command = (int)(intptr_t)lv_obj_get_user_data(btn);
    
    if(panel) {
        const char* command_names[] = {
            "", "Feed Detect", "Cut Prepare", "Cut Done", "Start Feed",
            "Pause", "Emergency", "Resume", "Waste Out", "Start Detect",
            "Pos Ready", "Grip Done", "Safety Check"
        };
        
        if(plc_command >= 1 && plc_command <= 12) {
            printf("发送PLC命令: %d (%s)\n", plc_command, command_names[plc_command]);
            
            // TODO: 通过Unix Socket客户端发送命令到后端
            // unix_socket_client_send_plc_command(client, plc_command);
            
            // 临时视觉反馈 - 按钮短暂高亮
            lv_obj_add_style(btn, &panel->style_button_active_, 0);
            // TODO: 添加定时器来移除高亮效果
        }
    }
}