#include "gui/video_view.h"
#include "resources/fonts/lv_font_noto_sans_cjk.h"
#include <stdio.h>
#include <string.h>

Video_view::Video_view() : container_(nullptr) {
}

Video_view::~Video_view() {
    // LVGL对象会自动清理，不需要手动删除
}

bool Video_view::initialize() {
    printf("初始化 video_view with Flex layout\n");
    
    // 创建主容器
    container_ = lv_obj_create(lv_scr_act());
    if (!container_) {
        printf("错误: 无法创建视频视图容器\n");
        return false;
    }
    
    // 设置样式和布局
    setup_styles();
    create_layout();
    
    printf("视频视图初始化完成\n");
    return true;
}

void Video_view::setup_styles() {
    // 主容器样式 - 对应HTML中的camera-panel
    lv_style_init(&style_container_);
    lv_style_set_bg_color(&style_container_, lv_color_hex(0x2D2D2D));
    lv_style_set_border_color(&style_container_, lv_color_hex(0x404040));
    lv_style_set_border_width(&style_container_, 2);
    lv_style_set_radius(&style_container_, 8);
    lv_style_set_pad_all(&style_container_, 15);
    
    // 摄像头画布样式
    lv_style_init(&style_canvas_);
    lv_style_set_bg_color(&style_canvas_, lv_color_hex(0x000000));
    lv_style_set_border_color(&style_canvas_, lv_color_hex(0x404040));
    lv_style_set_border_width(&style_canvas_, 1);
    lv_style_set_radius(&style_canvas_, 4);
    
    // 坐标显示样式
    lv_style_init(&style_coord_);
    lv_style_set_bg_color(&style_coord_, lv_color_hex(0x1A1A1A));
    lv_style_set_border_color(&style_coord_, lv_color_hex(0xFF6B35));
    lv_style_set_border_width(&style_coord_, 1);
    lv_style_set_radius(&style_coord_, 4);
    lv_style_set_pad_all(&style_coord_, 10);
    
    // 导轨指示器样式
    lv_style_init(&style_rail_);
    lv_style_set_bg_color(&style_rail_, lv_color_hex(0x2196F3));
    lv_style_set_bg_opa(&style_rail_, 77); // 30% 透明度
    lv_style_set_border_color(&style_rail_, lv_color_hex(0x2196F3));
    lv_style_set_border_width(&style_rail_, 1);
    lv_style_set_radius(&style_rail_, 4);
    
    // 切割位置指示器样式
    lv_style_init(&style_cutting_);
    lv_style_set_bg_color(&style_cutting_, lv_color_hex(0xF44336));
    lv_style_set_border_opa(&style_cutting_, LV_OPA_TRANSP);
}

void Video_view::create_layout() {
    // 设置容器为Flex布局，垂直排列
    lv_obj_add_style(container_, &style_container_, 0);
    lv_obj_set_flex_flow(container_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(container_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(container_, 10, 0);
    
    // 创建摄像头标题
    create_camera_title();
    
    // 创建摄像头画布
    create_camera_canvas();
    
    // 创建坐标显示
    create_coordinate_display();
}

void Video_view::create_camera_title() {
    // 创建标题容器 - 对应HTML中的camera-title
    camera_title_ = lv_obj_create(container_);
    lv_obj_set_size(camera_title_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(camera_title_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(camera_title_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(camera_title_, 0, LV_PART_MAIN);
    
    // 设置Flex布局
    lv_obj_set_flex_flow(camera_title_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(camera_title_, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // 创建标题标签
    title_label_ = lv_label_create(camera_title_);
    lv_label_set_text(title_label_, "📹 实时检测画面");
    lv_obj_set_style_text_color(title_label_, lv_color_hex(0xFF6B35), 0);
    lv_obj_set_style_text_font(title_label_, &lv_font_noto_sans_cjk_14, 0);
    
    // 创建检测信息标签
    detection_info_ = lv_label_create(camera_title_);
    lv_label_set_text(detection_info_, "导轨范围: 0-1000.0mm | 精度: 0.1mm | FPS: 28.5");
    lv_obj_set_style_text_color(detection_info_, lv_color_hex(0xB0B0B0), 0);
    lv_obj_set_style_text_font(detection_info_, &lv_font_noto_sans_cjk_14, 0);
}

void Video_view::create_camera_canvas() {
    // 创建摄像头画布 - 对应HTML中的camera-canvas，占满整个容器
    camera_canvas_ = lv_obj_create(container_);
    lv_obj_set_flex_grow(camera_canvas_, 1); // 占据剩余空间
    lv_obj_set_size(camera_canvas_, LV_PCT(100), LV_PCT(100)); // 完全占满
    lv_obj_set_style_pad_all(camera_canvas_, 2, LV_PART_MAIN); // 最小边距
    lv_obj_set_style_pad_all(camera_canvas_, 0, LV_PART_MAIN); // 无内边距
    lv_obj_add_style(camera_canvas_, &style_canvas_, 0);
    
    // 创建视频画面提示文字
    lv_obj_t* video_label = lv_label_create(camera_canvas_);
    lv_label_set_text(video_label, "竹材检测视野\n1280 x 720 | YOLOv8 推理中\n推理时间: 15.3ms");
    lv_obj_set_style_text_color(video_label, lv_color_hex(0xB0B0B0), 0);
    lv_obj_set_style_text_align(video_label, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_center(video_label);
    
    // 创建导轨指示器 - 对应HTML中的rail-indicator
    rail_indicator_ = lv_obj_create(camera_canvas_);
    lv_obj_set_size(rail_indicator_, LV_PCT(95), 30);
    lv_obj_set_pos(rail_indicator_, LV_PCT(3), LV_PCT(85));
    lv_obj_add_style(rail_indicator_, &style_rail_, 0);
    
    // 导轨标签 - 使用中文字体
    lv_obj_t* rail_label = lv_label_create(rail_indicator_);
    lv_label_set_text(rail_label, "X轴导轨 (0-1000.0mm)");
    lv_obj_set_style_text_color(rail_label, lv_color_hex(0x2196F3), 0);
    lv_obj_set_style_text_font(rail_label, &lv_font_noto_sans_cjk_14, 0);
    lv_obj_center(rail_label);
    
    // 创建切割位置指示器
    cutting_position_ = lv_obj_create(rail_indicator_);
    lv_obj_set_size(cutting_position_, 2, LV_PCT(100));
    lv_obj_set_pos(cutting_position_, LV_PCT(25), 0); // 默认25%位置
    lv_obj_add_style(cutting_position_, &style_cutting_, 0);
}

void Video_view::create_coordinate_display() {
    // 创建坐标显示区域 - 对应HTML中的coordinate-display
    coordinate_display_ = lv_obj_create(container_);
    lv_obj_set_size(coordinate_display_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_add_style(coordinate_display_, &style_coord_, 0);
    
    // 设置Grid布局，3列显示
    static lv_coord_t col_dsc[] = {LV_GRID_FR(1), LV_GRID_FR(1), LV_GRID_FR(1), LV_GRID_TEMPLATE_LAST};
    static lv_coord_t row_dsc[] = {LV_SIZE_CONTENT, LV_SIZE_CONTENT, LV_GRID_TEMPLATE_LAST};
    
    lv_obj_set_grid_dsc_array(coordinate_display_, col_dsc, row_dsc);
    lv_obj_set_style_pad_gap(coordinate_display_, 10, 0);
    
    // 创建X坐标显示
    lv_obj_t* x_coord_container = lv_obj_create(coordinate_display_);
    lv_obj_set_grid_cell(x_coord_container, LV_GRID_ALIGN_CENTER, 0, 1, LV_GRID_ALIGN_CENTER, 0, 2);
    lv_obj_set_style_bg_opa(x_coord_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(x_coord_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(x_coord_container, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(x_coord_container, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(x_coord_container, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    lv_obj_t* x_label = lv_label_create(x_coord_container);
    lv_label_set_text(x_label, "X坐标");
    lv_obj_set_style_text_color(x_label, lv_color_hex(0xB0B0B0), 0);
    lv_obj_set_style_text_font(x_label, &lv_font_noto_sans_cjk_14, 0);
    
    x_coord_label_ = lv_label_create(x_coord_container);
    lv_label_set_text(x_coord_label_, "245.8mm");
    lv_obj_set_style_text_color(x_coord_label_, lv_color_hex(0xFF6B35), 0);
    lv_obj_set_style_text_font(x_coord_label_, &lv_font_noto_sans_cjk_14, 0);
    
    // 创建切割质量显示
    lv_obj_t* quality_container = lv_obj_create(coordinate_display_);
    lv_obj_set_grid_cell(quality_container, LV_GRID_ALIGN_CENTER, 1, 1, LV_GRID_ALIGN_CENTER, 0, 2);
    lv_obj_set_style_bg_opa(quality_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(quality_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(quality_container, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(quality_container, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(quality_container, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    lv_obj_t* quality_desc = lv_label_create(quality_container);
    lv_label_set_text(quality_desc, "切割质量");
    lv_obj_set_style_text_color(quality_desc, lv_color_hex(0xB0B0B0), 0);
    lv_obj_set_style_text_font(quality_desc, &lv_font_noto_sans_cjk_14, 0);
    
    quality_label_ = lv_label_create(quality_container);
    lv_label_set_text(quality_label_, "正常");
    lv_obj_set_style_text_color(quality_label_, lv_color_hex(0x4CAF50), 0);
    lv_obj_set_style_text_font(quality_label_, &lv_font_noto_sans_cjk_14, 0);
    
    // 创建刀片选择显示
    lv_obj_t* blade_container = lv_obj_create(coordinate_display_);
    lv_obj_set_grid_cell(blade_container, LV_GRID_ALIGN_CENTER, 2, 1, LV_GRID_ALIGN_CENTER, 0, 2);
    lv_obj_set_style_bg_opa(blade_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(blade_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(blade_container, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(blade_container, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(blade_container, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    lv_obj_t* blade_desc = lv_label_create(blade_container);
    lv_label_set_text(blade_desc, "刀片选择");
    lv_obj_set_style_text_color(blade_desc, lv_color_hex(0xB0B0B0), 0);
    lv_obj_set_style_text_font(blade_desc, &lv_font_noto_sans_cjk_14, 0);
    
    blade_label_ = lv_label_create(blade_container);
    lv_label_set_text(blade_label_, "双刀片");
    lv_obj_set_style_text_color(blade_label_, lv_color_hex(0xFF6B35), 0);
    lv_obj_set_style_text_font(blade_label_, &lv_font_noto_sans_cjk_14, 0);
}

void Video_view::update_camera_frame(const frame_info_t& frame) {
    // TODO: 实现摄像头帧显示更新
    // 这里可以将OpenCV Mat转换为LVGL图像并显示
}

void Video_view::update_detection_info(float fps, float inference_time) {
    if (detection_info_) {
        char buffer[128];
        snprintf(buffer, sizeof(buffer), "导轨范围: 0-1000.0mm | 精度: 0.1mm | FPS: %.1f", fps);
        lv_label_set_text(detection_info_, buffer);
    }
}

void Video_view::update_coordinate_display(float x_mm, const char* quality, const char* blade) {
    if (x_coord_label_) {
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%.1fmm", x_mm);
        lv_label_set_text(x_coord_label_, buffer);
    }
    
    if (quality_label_ && quality) {
        lv_label_set_text(quality_label_, quality);
        // 根据质量设置颜色
        if (strcmp(quality, "正常") == 0) {
            lv_obj_set_style_text_color(quality_label_, lv_color_hex(0x4CAF50), 0);
        } else {
            lv_obj_set_style_text_color(quality_label_, lv_color_hex(0xF44336), 0);
        }
    }
    
    if (blade_label_ && blade) {
        lv_label_set_text(blade_label_, blade);
    }
}

void Video_view::update_cutting_position(float x_mm, float max_range) {
    if (cutting_position_ && max_range > 0) {
        // 计算位置百分比
        float percentage = (x_mm / max_range) * 100.0f;
        if (percentage < 0) percentage = 0;
        if (percentage > 100) percentage = 100;
        
        // 更新切割位置指示器
        lv_obj_set_x(cutting_position_, lv_pct((int)percentage));
    }
}