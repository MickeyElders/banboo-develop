#include "gui/video_view.h"
#include <stdio.h>
#include <string.h>

Video_view::Video_view() : container_(nullptr) {
}

Video_view::~Video_view() {
    // LVGL objects are automatically cleaned up
}

bool Video_view::initialize() {
    printf("Initializing video_view with Flex layout\n");
    
    // Create main container
    container_ = lv_obj_create(lv_scr_act());
    if (!container_) {
        printf("Error: Unable to create video view container\n");
        return false;
    }
    
    // Setup styles and layout
    setup_styles();
    create_layout();
    
    printf("Video view initialization completed\n");
    return true;
}

void Video_view::setup_styles() {
    // Main container style - corresponds to camera-panel in HTML
    lv_style_init(&style_container_);
    lv_style_set_bg_color(&style_container_, lv_color_hex(0x2D2D2D));
    lv_style_set_border_color(&style_container_, lv_color_hex(0x404040));
    lv_style_set_border_width(&style_container_, 2);
    lv_style_set_radius(&style_container_, 8);
    lv_style_set_pad_all(&style_container_, 15);
    
    // Camera canvas style
    lv_style_init(&style_canvas_);
    lv_style_set_bg_color(&style_canvas_, lv_color_hex(0x000000));
    lv_style_set_border_color(&style_canvas_, lv_color_hex(0x404040));
    lv_style_set_border_width(&style_canvas_, 1);
    lv_style_set_radius(&style_canvas_, 4);
    
    // Coordinate display style
    lv_style_init(&style_coord_);
    lv_style_set_bg_color(&style_coord_, lv_color_hex(0x1A1A1A));
    lv_style_set_border_color(&style_coord_, lv_color_hex(0xFF6B35));
    lv_style_set_border_width(&style_coord_, 1);
    lv_style_set_radius(&style_coord_, 4);
    lv_style_set_pad_all(&style_coord_, 10);
    
    // Rail indicator style
    lv_style_init(&style_rail_);
    lv_style_set_bg_color(&style_rail_, lv_color_hex(0x2196F3));
    lv_style_set_bg_opa(&style_rail_, 77); // 30% transparency
    lv_style_set_border_color(&style_rail_, lv_color_hex(0x2196F3));
    lv_style_set_border_width(&style_rail_, 1);
    lv_style_set_radius(&style_rail_, 4);
    
    // Cutting position indicator style
    lv_style_init(&style_cutting_);
    lv_style_set_bg_color(&style_cutting_, lv_color_hex(0xF44336));
    lv_style_set_border_opa(&style_cutting_, LV_OPA_TRANSP);
}

void Video_view::create_layout() {
    // Set container to Flex layout, vertical arrangement
    lv_obj_add_style(container_, &style_container_, 0);
    lv_obj_set_flex_flow(container_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(container_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(container_, 10, 0);
    
    // Create camera title
    create_camera_title();
    
    // Create camera canvas
    create_camera_canvas();
    
    // Create coordinate display
    create_coordinate_display();
}

void Video_view::create_camera_title() {
    // Create title container - corresponds to camera-title in HTML
    camera_title_ = lv_obj_create(container_);
    lv_obj_set_size(camera_title_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(camera_title_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(camera_title_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(camera_title_, 0, LV_PART_MAIN);
    
    // Set Flex layout
    lv_obj_set_flex_flow(camera_title_, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(camera_title_, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    // Create title label
    title_label_ = lv_label_create(camera_title_);
    lv_label_set_text(title_label_, LV_SYMBOL_VIDEO " Live Detection View");
    lv_obj_set_style_text_color(title_label_, lv_color_hex(0xFF6B35), 0);
    
    // Create detection info label
    detection_info_ = lv_label_create(camera_title_);
    lv_label_set_text(detection_info_, "Rail Range: 0-1000.0mm | Precision: 0.1mm | FPS: 28.5");
    lv_obj_set_style_text_color(detection_info_, lv_color_hex(0xB0B0B0), 0);
}

void Video_view::create_camera_canvas() {
    // Create camera canvas - corresponds to camera-canvas in HTML, fills entire container
    camera_canvas_ = lv_obj_create(container_);
    lv_obj_set_flex_grow(camera_canvas_, 1); // Take remaining space
    lv_obj_set_size(camera_canvas_, LV_PCT(100), LV_PCT(100)); // Fill completely
    lv_obj_set_style_pad_all(camera_canvas_, 2, LV_PART_MAIN); // Minimal margin
    lv_obj_set_style_pad_all(camera_canvas_, 0, LV_PART_MAIN); // No inner padding
    lv_obj_add_style(camera_canvas_, &style_canvas_, 0);
    
    // Create video view hint text
    lv_obj_t* video_label = lv_label_create(camera_canvas_);
    lv_label_set_text(video_label, "Bamboo Detection View\n1280 x 720 | YOLOv8 Inference\nInference Time: 15.3ms");
    lv_obj_set_style_text_color(video_label, lv_color_hex(0xB0B0B0), 0);
    lv_obj_set_style_text_align(video_label, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_center(video_label);
    
    // Create rail indicator - corresponds to rail-indicator in HTML
    rail_indicator_ = lv_obj_create(camera_canvas_);
    lv_obj_set_size(rail_indicator_, LV_PCT(95), 30);
    lv_obj_set_pos(rail_indicator_, LV_PCT(3), LV_PCT(85));
    lv_obj_add_style(rail_indicator_, &style_rail_, 0);
    
    // Rail label
    lv_obj_t* rail_label = lv_label_create(rail_indicator_);
    lv_label_set_text(rail_label, "X-Axis Rail (0-1000.0mm)");
    lv_obj_set_style_text_color(rail_label, lv_color_hex(0x2196F3), 0);
    lv_obj_center(rail_label);
    
    // Create cutting position indicator
    cutting_position_ = lv_obj_create(rail_indicator_);
    lv_obj_set_size(cutting_position_, 2, LV_PCT(100));
    lv_obj_set_pos(cutting_position_, LV_PCT(25), 0); // Default 25% position
    lv_obj_add_style(cutting_position_, &style_cutting_, 0);
}

void Video_view::create_coordinate_display() {
    // Create coordinate display area - corresponds to coordinate-display in HTML
    coordinate_display_ = lv_obj_create(container_);
    lv_obj_set_size(coordinate_display_, LV_PCT(100), LV_SIZE_CONTENT);
    lv_obj_add_style(coordinate_display_, &style_coord_, 0);
    
    // Set Grid layout, 3 columns display
    static lv_coord_t col_dsc[] = {LV_GRID_FR(1), LV_GRID_FR(1), LV_GRID_FR(1), LV_GRID_TEMPLATE_LAST};
    static lv_coord_t row_dsc[] = {LV_SIZE_CONTENT, LV_SIZE_CONTENT, LV_GRID_TEMPLATE_LAST};
    
    lv_obj_set_grid_dsc_array(coordinate_display_, col_dsc, row_dsc);
    lv_obj_set_style_pad_gap(coordinate_display_, 10, 0);
    
    // Create X coordinate display
    lv_obj_t* x_coord_container = lv_obj_create(coordinate_display_);
    lv_obj_set_grid_cell(x_coord_container, LV_GRID_ALIGN_CENTER, 0, 1, LV_GRID_ALIGN_CENTER, 0, 2);
    lv_obj_set_style_bg_opa(x_coord_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(x_coord_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(x_coord_container, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(x_coord_container, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(x_coord_container, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    lv_obj_t* x_label = lv_label_create(x_coord_container);
    lv_label_set_text(x_label, "X Position");
    lv_obj_set_style_text_color(x_label, lv_color_hex(0xB0B0B0), 0);
    
    x_coord_label_ = lv_label_create(x_coord_container);
    lv_label_set_text(x_coord_label_, "245.8mm");
    lv_obj_set_style_text_color(x_coord_label_, lv_color_hex(0xFF6B35), 0);
    
    // Create cutting quality display
    lv_obj_t* quality_container = lv_obj_create(coordinate_display_);
    lv_obj_set_grid_cell(quality_container, LV_GRID_ALIGN_CENTER, 1, 1, LV_GRID_ALIGN_CENTER, 0, 2);
    lv_obj_set_style_bg_opa(quality_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(quality_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(quality_container, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(quality_container, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(quality_container, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    lv_obj_t* quality_desc = lv_label_create(quality_container);
    lv_label_set_text(quality_desc, "Cut Quality");
    lv_obj_set_style_text_color(quality_desc, lv_color_hex(0xB0B0B0), 0);
    
    quality_label_ = lv_label_create(quality_container);
    lv_label_set_text(quality_label_, "Normal");
    lv_obj_set_style_text_color(quality_label_, lv_color_hex(0x4CAF50), 0);
    
    // Create blade selection display
    lv_obj_t* blade_container = lv_obj_create(coordinate_display_);
    lv_obj_set_grid_cell(blade_container, LV_GRID_ALIGN_CENTER, 2, 1, LV_GRID_ALIGN_CENTER, 0, 2);
    lv_obj_set_style_bg_opa(blade_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_opa(blade_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(blade_container, 0, LV_PART_MAIN);
    lv_obj_set_flex_flow(blade_container, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(blade_container, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    
    lv_obj_t* blade_desc = lv_label_create(blade_container);
    lv_label_set_text(blade_desc, "Blade Select");
    lv_obj_set_style_text_color(blade_desc, lv_color_hex(0xB0B0B0), 0);
    
    blade_label_ = lv_label_create(blade_container);
    lv_label_set_text(blade_label_, "Dual Blade");
    lv_obj_set_style_text_color(blade_label_, lv_color_hex(0xFF6B35), 0);
}

void Video_view::update_camera_frame(const frame_info_t& frame) {
    // TODO: Implement camera frame display update
    // Here you can convert OpenCV Mat to LVGL image and display
}

void Video_view::update_detection_info(float fps, float inference_time) {
    if (detection_info_) {
        char buffer[128];
        snprintf(buffer, sizeof(buffer), "Rail Range: 0-1000.0mm | Precision: 0.1mm | FPS: %.1f", fps);
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
        // Set color based on quality
        if (strcmp(quality, "Normal") == 0 || strcmp(quality, "正常") == 0) {
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
        // Calculate position percentage
        float percentage = (x_mm / max_range) * 100.0f;
        if (percentage < 0) percentage = 0;
        if (percentage > 100) percentage = 100;
        
        // Update cutting position indicator
        lv_obj_set_x(cutting_position_, lv_pct((int)percentage));
    }
}