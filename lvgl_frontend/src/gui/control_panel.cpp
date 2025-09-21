#include "gui/control_panel.h"
#include "resources/fonts/lv_font_noto_sans_cjk.h"
#include <stdio.h>
#include <string.h>

Control_panel::Control_panel() : container_(nullptr) {
    // 初始化指针数组
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
    // LVGL对象会自动清理，不需要手动删除
}

bool Control_panel::initialize() {
    printf("初始化 control_panel with Flex layout\n");
    
    // 创建主容器
    container_ = lv_obj_create(lv_scr_act());
    if (!container_) {
        printf("错误: 无法创建控制面板容器\n");
        return false;
    }
    
    // 设置样式和布局
    setup_styles();
    create_layout();
    
    printf("控制面板初始化完成\n");
    return true;
}

void Control_panel::setup_styles() {
    // 主容器样式 - 对应HTML中的control-panel
    lv_style_init(&style_container_);
    lv_style_set_bg_color(&style_container_, lv_color_hex(0x2D2D2D));
    lv_style_set_border_color(&style_container_, lv_color_hex(0x404040));
    lv_style_set_border_width(&style_container_, 2);
    lv_style_set_radius(&style_container_, 8);
    lv_style_set_pad_all(&style_container_, 15);
    
    // 区块样式 - 对应HTML中的panel-section
    lv_style_init(&style_section_);
    lv_style_set_bg_color(&style_section_, lv_color_hex(0x1A1A1A));
    lv_style_set_border_width(&style_section_, 1);
    lv_style_set_radius(&style_section_, 5);
    lv_style_set_pad_all(&style_section_, 8);
    
    // Modbus区块样式
    lv_style_init(&style_modbus_);
    lv_style_set_border_color(&style_modbus_, lv_color_hex(0x2196F3));
    
    // PLC区块样式
    lv_style_init(&style_plc_);
    lv_style_set_border_color(&style_plc_, lv_color_hex(0x4CAF50));
    
    // Jetson区块样式
    lv_style_init(&style_jetson_);
    lv_style_set_border_color(&style_jetson_, lv_color_hex(0x76B900));
    
    // 表格样式
    lv_style_init(&style_table_);
    lv_style_set_border_color(&style_table_, lv_color_hex(0x404040));
    lv_style_set_border_width(&style_table_, 1);
    
    // 进度条样式
    lv_style_init(&style_progress_);
    lv_style_set_bg_color(&style_progress_, lv_color_hex(0x404040));
    lv_style_set_radius(&style_progress_, 3);
    
    // 按钮样式
    lv_style_init(&style_button_);
    lv_style_set_bg_color(&style_button_, lv_color_hex(0x1E1E1E));
    lv_style_set_border_color(&style_button_, lv_color_hex(0x404040));
    lv_style_set_border_width(&style_button_, 1);
    lv_style_set_radius(&style_button_, 3);
    lv_style_set_text_color(&style_button_, lv_color_hex(0xB0B0B0));
    
    // 激活按钮样式
    lv_style_init(&style_button_active_);
    lv_style_set_bg_color(&style_button_active_, lv_color_hex(0xFF6B35));
    lv_style_set_border_color(&style_button_active_, lv_color_hex(0xFF6B35));
    lv_style_set_text_color(&style_button_active_, lv_color_hex(0xFFFFFF));
}

void Control_panel::create_layout() {
    // 设置容器为Flex布局，垂直排列，带滚动
    lv_obj_add_style(container_, &style_container_, 0);
    lv_obj_set_flex_flow(container_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(container_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(container_, 6, 0);
    lv_obj_set_scroll_dir(container_, LV_DIR_VER);
    
    // 创建各个功能区块
    create_modbus_section();
    create_plc_section();
    create_jetson_section();
    create_ai_section();
    create_comm_section();
}

void Control_panel::create_modbus_section() {
    // 创建Modbus寄存器状态区块
    modbus_section_ = lv_obj_create(container_);
    lv_obj_set_size(modbus_section_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_add_style(modbus_section_, &style_section_, 0);
    lv_obj_add_style(modbus_section_, &style_modbus_, 0);
    
    // 设置Flex布局
    lv_obj_set_flex_flow(modbus_section_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(modbus_section_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(modbus_section_, 6, 0);
    
    // 创建标题
    lv_obj_t* title = lv_label_create(modbus_section_);
    lv_label_set_text(title, "📊 Modbus寄存器状态");
    lv_obj_set_style_text_color(title, lv_color_hex(0x2196F3), 0);
    lv_obj_set_style_text_font(title, &lv_font_noto_sans_cjk_14, 0);
    
    // 创建寄存器表格
    modbus_table_ = lv_table_create(modbus_section_);
    lv_obj_set_size(modbus_table_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_add_style(modbus_table_, &style_table_, 0);
    
    // 设置表格内容
    lv_table_set_col_cnt(modbus_table_, 3);
    lv_table_set_row_cnt(modbus_table_, 9); // 1个表头 + 8行数据
    
    // 设置表头
    lv_table_set_cell_value(modbus_table_, 0, 0, "地址");
    lv_table_set_cell_value(modbus_table_, 0, 1, "描述");
    lv_table_set_cell_value(modbus_table_, 0, 2, "值");
    
    // 设置数据行
    const char* register_addrs[] = {"40001", "40002", "40003", "40004-40005", "40006", "40007-40008", "40009", "40010"};
    const char* register_descs[] = {"系统状态", "PLC命令", "坐标就绪", "X坐标", "切割质量", "心跳计数", "刀片编号", "健康状态"};
    const char* register_values[] = {"1", "2", "1", "2458", "0", "12345", "3", "0"};
    
    for(int i = 0; i < 8; i++) {
        lv_table_set_cell_value(modbus_table_, i + 1, 0, register_addrs[i]);
        lv_table_set_cell_value(modbus_table_, i + 1, 1, register_descs[i]);
        lv_table_set_cell_value(modbus_table_, i + 1, 2, register_values[i]);
        
        // 设置值列的颜色
        // 设置状态单元格样式（替代 lv_table_set_cell_ctrl）
        if (i == 0) {  // 连接状态为绿色
            lv_table_set_cell_value(modbus_table_, i + 1, 2, "✓");
        } else {  // 其他状态为灰色
            lv_table_set_cell_value(modbus_table_, i + 1, 2, "-");
        }
    }
    
    // 设置表格样式
    lv_obj_set_style_text_font(modbus_table_, &lv_font_noto_sans_cjk_14, 0);
    lv_obj_set_style_text_color(modbus_table_, lv_color_hex(0xFFFFFF), 0);
    lv_obj_set_style_text_color(modbus_table_, lv_color_hex(0xFF6B35), LV_PART_ITEMS | LV_STATE_USER_1);
}

void Control_panel::create_plc_section() {
    // 创建PLC通信状态区块
    plc_section_ = lv_obj_create(container_);
    lv_obj_set_size(plc_section_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_add_style(plc_section_, &style_section_, 0);
    lv_obj_add_style(plc_section_, &style_plc_, 0);
    
    // 设置Flex布局
    lv_obj_set_flex_flow(plc_section_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(plc_section_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(plc_section_, 6, 0);
    
    // 创建标题
    lv_obj_t* title = lv_label_create(plc_section_);
    lv_label_set_text(title, "🔗 PLC通信状态");
    lv_obj_set_style_text_color(title, lv_color_hex(0x4CAF50), 0);
    lv_obj_set_style_text_font(title, &lv_font_noto_sans_cjk_14, 0);
    
    // 创建状态网格容器
    lv_obj_t* status_grid = lv_obj_create(plc_section_);
    lv_obj_set_size(status_grid, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(status_grid, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(status_grid, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(status_grid, 0, LV_PART_MAIN);
    
    // 设置2x2网格布局
    static lv_coord_t col_dsc[] = {LV_GRID_FR(1), LV_GRID_FR(1), LV_GRID_TEMPLATE_LAST};
    static lv_coord_t row_dsc[] = {LV_SIZE_CONTENT, LV_SIZE_CONTENT, LV_GRID_TEMPLATE_LAST};
    lv_obj_set_grid_dsc_array(status_grid, col_dsc, row_dsc);
    lv_obj_set_style_pad_gap(status_grid, 4, 0);
    
    // 创建状态项
    const char* status_labels[] = {"连接状态:", "PLC地址:", "响应时间:", "总切割数:"};
    const char* status_values[] = {"已连接", "192.168.1.100", "12ms", "1,247"};
    lv_obj_t** status_value_labels[] = {&plc_status_label_, &plc_address_label_, &plc_response_label_, &total_cuts_label_};
    
    for(int i = 0; i < 4; i++) {
        lv_obj_t* status_item = lv_obj_create(status_grid);
        lv_obj_set_grid_cell(status_item, LV_GRID_ALIGN_STRETCH, i % 2, 1, LV_GRID_ALIGN_CENTER, i / 2, 1);
        lv_obj_set_style_bg_color(status_item, lv_color_hex(0x0D0D0D), 0);
        lv_obj_set_style_border_opa(status_item, LV_OPA_TRANSP, 0);
        lv_obj_set_style_radius(status_item, 3, 0);
        lv_obj_set_style_pad_all(status_item, 6, 0);
        lv_obj_set_flex_flow(status_item, LV_FLEX_FLOW_COLUMN);
        lv_obj_set_flex_align(status_item, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
        
        lv_obj_t* label = lv_label_create(status_item);
        lv_label_set_text(label, status_labels[i]);
        lv_obj_set_style_text_color(label, lv_color_hex(0xB0B0B0), 0);
        lv_obj_set_style_text_font(label, &lv_font_noto_sans_cjk_14, 0);
        
        *status_value_labels[i] = lv_label_create(status_item);
        lv_label_set_text(*status_value_labels[i], status_values[i]);
        lv_obj_set_style_text_color(*status_value_labels[i], lv_color_hex(0xFFFFFF), 0);
        lv_obj_set_style_text_font(*status_value_labels[i], &lv_font_noto_sans_cjk_14, 0);
        
        // 连接状态显示为绿色
        if(i == 0) {
            lv_obj_set_style_text_color(*status_value_labels[i], lv_color_hex(0x4CAF50), 0);
        }
    }
    
    // 创建刀片选择器
    lv_obj_t* blade_container = lv_obj_create(plc_section_);
    lv_obj_set_size(blade_container, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(blade_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(blade_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(blade_container, 0, LV_PART_MAIN);
    lv_obj_set_style_pad_top(blade_container, 4, 0);
    
    lv_obj_set_flex_flow(blade_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(blade_container, LV_FLEX_ALIGN_SPACE_EVENLY, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(blade_container, 3, 0);
    
    const char* blade_names[] = {"刀片1", "刀片2", "双刀片"};
    for(int i = 0; i < 3; i++) {
        blade_buttons_[i] = lv_btn_create(blade_container);
        lv_obj_set_flex_grow(blade_buttons_[i], 1);
        lv_obj_set_height(blade_buttons_[i], LV_SIZE_CONTENT);
        lv_obj_add_style(blade_buttons_[i], &style_button_, 0);
        lv_obj_set_style_pad_ver(blade_buttons_[i], 3, 0);
        
        lv_obj_t* label = lv_label_create(blade_buttons_[i]);
        lv_label_set_text(label, blade_names[i]);
        lv_obj_set_style_text_font(label, &lv_font_noto_sans_cjk_14, 0);
        lv_obj_center(label);
        
        // 设置事件回调
        lv_obj_add_event_cb(blade_buttons_[i], blade_button_event_cb, LV_EVENT_CLICKED, this);
        lv_obj_set_user_data(blade_buttons_[i], (void*)(intptr_t)i);
        
        // 默认激活第三个按钮（双刀片）
        if(i == 2) {
            lv_obj_add_style(blade_buttons_[i], &style_button_active_, 0);
        }
    }
}

void Control_panel::create_jetson_section() {
    // 创建Jetson系统信息区块
    jetson_section_ = lv_obj_create(container_);
    lv_obj_set_size(jetson_section_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_add_style(jetson_section_, &style_section_, 0);
    lv_obj_add_style(jetson_section_, &style_jetson_, 0);
    
    // 设置Flex布局
    lv_obj_set_flex_flow(jetson_section_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(jetson_section_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(jetson_section_, 4, 0);
    
    // 创建标题
    lv_obj_t* title = lv_label_create(jetson_section_);
    lv_label_set_text(title, "🟢 Jetson Orin Nano 8GB  15W");
    lv_obj_set_style_text_color(title, lv_color_hex(0x76B900), 0);
    lv_obj_set_style_text_font(title, &lv_font_noto_sans_cjk_14, 0);
    
    // 创建CPU进度条
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
    lv_label_set_text(cpu_desc, "CPU (6核 ARM Cortex-A78AE)");
    lv_obj_set_style_text_color(cpu_desc, lv_color_hex(0xB0B0B0), 0);
    lv_obj_set_style_text_font(cpu_desc, &lv_font_noto_sans_cjk_14, 0);
    
    cpu_usage_label_ = lv_label_create(cpu_label_container);
    lv_label_set_text(cpu_usage_label_, "45%");
    lv_obj_set_style_text_color(cpu_usage_label_, lv_color_hex(0xFFFFFF), 0);
    lv_obj_set_style_text_font(cpu_usage_label_, &lv_font_noto_sans_cjk_14, 0);
    
    cpu_progress_ = lv_bar_create(cpu_container);
    lv_obj_set_size(cpu_progress_, LV_PCT(100), 6);
    lv_obj_add_style(cpu_progress_, &style_progress_, 0);
    lv_obj_set_style_bg_color(cpu_progress_, lv_color_hex(0x76B900), LV_PART_INDICATOR);
    lv_bar_set_value(cpu_progress_, 45, LV_ANIM_OFF);
    
    // 创建GPU进度条
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
    lv_obj_set_style_text_font(gpu_desc, &lv_font_noto_sans_cjk_14, 0);
    
    gpu_usage_label_ = lv_label_create(gpu_label_container);
    lv_label_set_text(gpu_usage_label_, "32%");
    lv_obj_set_style_text_color(gpu_usage_label_, lv_color_hex(0xFFFFFF), 0);
    lv_obj_set_style_text_font(gpu_usage_label_, &lv_font_noto_sans_cjk_14, 0);
    
    gpu_progress_ = lv_bar_create(gpu_container);
    lv_obj_set_size(gpu_progress_, LV_PCT(100), 6);
    lv_obj_add_style(gpu_progress_, &style_progress_, 0);
    lv_obj_set_style_bg_color(gpu_progress_, lv_color_hex(0xFF6B35), LV_PART_INDICATOR);
    lv_bar_set_value(gpu_progress_, 32, LV_ANIM_OFF);
    
    // 创建内存进度条
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
    lv_label_set_text(memory_desc, "内存 (LPDDR5)");
    lv_obj_set_style_text_color(memory_desc, lv_color_hex(0xB0B0B0), 0);
    lv_obj_set_style_text_font(memory_desc, &lv_font_noto_sans_cjk_14, 0);
    
    memory_usage_label_ = lv_label_create(memory_label_container);
    lv_label_set_text(memory_usage_label_, "2.1GB/8GB");
    lv_obj_set_style_text_color(memory_usage_label_, lv_color_hex(0xFFFFFF), 0);
    lv_obj_set_style_text_font(memory_usage_label_, &lv_font_noto_sans_cjk_14, 0);
    
    memory_progress_ = lv_bar_create(memory_container);
    lv_obj_set_size(memory_progress_, LV_PCT(100), 6);
    lv_obj_add_style(memory_progress_, &style_progress_, 0);
    lv_obj_set_style_bg_color(memory_progress_, lv_color_hex(0xFFC107), LV_PART_INDICATOR);
    lv_bar_set_value(memory_progress_, 26, LV_ANIM_OFF);
}

void Control_panel::create_ai_section() {
    // 创建AI模型状态区块
    ai_section_ = lv_obj_create(container_);
    lv_obj_set_size(ai_section_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_add_style(ai_section_, &style_section_, 0);
    lv_obj_set_style_border_color(ai_section_, lv_color_hex(0xFF6B35), 0);
    
    // 设置Flex布局
    lv_obj_set_flex_flow(ai_section_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(ai_section_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(ai_section_, 4, 0);
    
    // 创建标题
    lv_obj_t* title = lv_label_create(ai_section_);
    lv_label_set_text(title, "🤖 AI模型状态");
    lv_obj_set_style_text_color(title, lv_color_hex(0xFF6B35), 0);
    lv_obj_set_style_text_font(title, &lv_font_noto_sans_cjk_14, 0);
    
    // 创建状态网格容器
    lv_obj_t* ai_grid = lv_obj_create(ai_section_);
    lv_obj_set_size(ai_grid, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(ai_grid, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(ai_grid, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(ai_grid, 0, LV_PART_MAIN);
    
    // 设置2x3网格布局
    static lv_coord_t ai_col_dsc[] = {LV_GRID_FR(1), LV_GRID_FR(1), LV_GRID_TEMPLATE_LAST};
    static lv_coord_t ai_row_dsc[] = {LV_SIZE_CONTENT, LV_SIZE_CONTENT, LV_SIZE_CONTENT, LV_GRID_TEMPLATE_LAST};
    lv_obj_set_grid_dsc_array(ai_grid, ai_col_dsc, ai_row_dsc);
    lv_obj_set_style_pad_gap(ai_grid, 4, 0);
    
    // 创建AI状态项
    const char* ai_labels[] = {"模型版本:", "推理时间:", "置信阈值:", "检测精度:", "总检测数:", "今日检测:"};
    const char* ai_values[] = {"YOLOv8n", "15.3ms", "0.85", "94.2%", "15,432", "89"};
    lv_obj_t** ai_value_labels[] = {nullptr, &inference_time_label_, nullptr, &accuracy_label_, &total_detections_label_, &today_detections_label_};
    
    for(int i = 0; i < 6; i++) {
        lv_obj_t* ai_item = lv_obj_create(ai_grid);
        lv_obj_set_grid_cell(ai_item, LV_GRID_ALIGN_STRETCH, i % 2, 1, LV_GRID_ALIGN_CENTER, i / 2, 1);
        lv_obj_set_style_bg_color(ai_item, lv_color_hex(0x0D0D0D), 0);
        lv_obj_set_style_border_opa(ai_item, LV_OPA_TRANSP, 0);
        lv_obj_set_style_radius(ai_item, 3, 0);
        lv_obj_set_style_pad_all(ai_item, 6, 0);
        lv_obj_set_flex_flow(ai_item, LV_FLEX_FLOW_COLUMN);
        lv_obj_set_flex_align(ai_item, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
        
        lv_obj_t* label = lv_label_create(ai_item);
        lv_label_set_text(label, ai_labels[i]);
        lv_obj_set_style_text_color(label, lv_color_hex(0xB0B0B0), 0);
        lv_obj_set_style_text_font(label, &lv_font_noto_sans_cjk_14, 0);
        
        lv_obj_t* value_label = lv_label_create(ai_item);
        lv_label_set_text(value_label, ai_values[i]);
        lv_obj_set_style_text_color(value_label, lv_color_hex(0xFFFFFF), 0);
        lv_obj_set_style_text_font(value_label, &lv_font_noto_sans_cjk_14, 0);
        
        if(ai_value_labels[i]) {
            *ai_value_labels[i] = value_label;
        }
    }
}

void Control_panel::create_comm_section() {
    // 创建通信统计区块
    comm_section_ = lv_obj_create(container_);
    lv_obj_set_size(comm_section_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_add_style(comm_section_, &style_section_, 0);
    lv_obj_set_style_border_color(comm_section_, lv_color_hex(0xB0B0B0), 0);
    
    // 设置Flex布局
    lv_obj_set_flex_flow(comm_section_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(comm_section_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(comm_section_, 4, 0);
    
    // 创建标题
    lv_obj_t* title = lv_label_create(comm_section_);
    lv_label_set_text(title, "📈 通信统计");
    lv_obj_set_style_text_color(title, lv_color_hex(0xB0B0B0), 0);
    lv_obj_set_style_text_font(title, &lv_font_noto_sans_cjk_14, 0);
    
    // 创建状态网格容器
    lv_obj_t* comm_grid = lv_obj_create(comm_section_);
    lv_obj_set_size(comm_grid, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(comm_grid, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(comm_grid, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(comm_grid, 0, LV_PART_MAIN);
    
    // 设置2x2网格布局
    static lv_coord_t comm_col_dsc[] = {LV_GRID_FR(1), LV_GRID_FR(1), LV_GRID_TEMPLATE_LAST};
    static lv_coord_t comm_row_dsc[] = {LV_SIZE_CONTENT, LV_SIZE_CONTENT, LV_GRID_TEMPLATE_LAST};
    lv_obj_set_grid_dsc_array(comm_grid, comm_col_dsc, comm_row_dsc);
    lv_obj_set_style_pad_gap(comm_grid, 4, 0);
    
    // 创建通信统计项
    const char* comm_labels[] = {"连接时长:", "数据包:", "错误率:", "吞吐量:"};
    const char* comm_values[] = {"2h 15m", "15,432", "0.02%", "1.2KB/s"};
    lv_obj_t** comm_value_labels[] = {&connection_time_label_, &packets_label_, &error_rate_label_, &throughput_label_};
    
    for(int i = 0; i < 4; i++) {
        lv_obj_t* comm_item = lv_obj_create(comm_grid);
        lv_obj_set_grid_cell(comm_item, LV_GRID_ALIGN_STRETCH, i % 2, 1, LV_GRID_ALIGN_CENTER, i / 2, 1);
        lv_obj_set_style_bg_color(comm_item, lv_color_hex(0x0D0D0D), 0);
        lv_obj_set_style_border_opa(comm_item, LV_OPA_TRANSP, 0);
        lv_obj_set_style_radius(comm_item, 3, 0);
        lv_obj_set_style_pad_all(comm_item, 6, 0);
        lv_obj_set_flex_flow(comm_item, LV_FLEX_FLOW_COLUMN);
        lv_obj_set_flex_align(comm_item, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
        
        lv_obj_t* label = lv_label_create(comm_item);
        lv_label_set_text(label, comm_labels[i]);
        lv_obj_set_style_text_color(label, lv_color_hex(0xB0B0B0), 0);
        lv_obj_set_style_text_font(label, &lv_font_noto_sans_cjk_14, 0);
        
        *comm_value_labels[i] = lv_label_create(comm_item);
        lv_label_set_text(*comm_value_labels[i], comm_values[i]);
        lv_obj_set_style_text_color(*comm_value_labels[i], lv_color_hex(0xFFFFFF), 0);
        lv_obj_set_style_text_font(*comm_value_labels[i], &lv_font_noto_sans_cjk_14, 0);
    }
}

// 更新方法实现
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

// 事件处理
void Control_panel::blade_button_event_cb(lv_event_t* e) {
    lv_obj_t* btn = lv_event_get_target(e);
    Control_panel* panel = (Control_panel*)lv_event_get_user_data(e);
    int blade_id = (int)(intptr_t)lv_obj_get_user_data(btn);
    
    if(panel) {
        panel->set_blade_selection(blade_id);
        printf("选择刀片: %d\n", blade_id + 1);
    }
}