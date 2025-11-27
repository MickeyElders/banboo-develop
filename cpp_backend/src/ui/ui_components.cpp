#include "bamboo_cut/ui/ui_components.h"
#include <lvgl.h>

namespace bamboo_cut {
namespace ui {

bool build_basic_ui(UIContext& ctx, const LVGLWaylandConfig& config) {
    ctx.main_screen = lv_obj_create(nullptr);
    lv_obj_set_size(ctx.main_screen, config.screen_width, config.screen_height);
    lv_obj_set_style_bg_color(ctx.main_screen, lv_color_hex(0x1A1F26), 0);
    lv_obj_set_style_bg_opa(ctx.main_screen, LV_OPA_COVER, 0);
    lv_obj_clear_flag(ctx.main_screen, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(ctx.main_screen, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_all(ctx.main_screen, 0, 0);

    // Header
    lv_obj_t* header = lv_obj_create(ctx.main_screen);
    lv_obj_set_size(header, lv_pct(100), 60);
    lv_obj_set_style_bg_color(header, lv_color_hex(0x252B35), 0);
    lv_obj_clear_flag(header, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(header, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(header, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_all(header, 10, 0);
    ctx.widgets.system_title = lv_label_create(header);
    lv_label_set_text(ctx.widgets.system_title, LV_SYMBOL_HOME " Bamboo System (DRM)");
    ctx.widgets.heartbeat_label = lv_label_create(header);
    lv_label_set_text(ctx.widgets.heartbeat_label, LV_SYMBOL_LOOP " Ready");

    // Main container
    lv_obj_t* main = lv_obj_create(ctx.main_screen);
    lv_obj_set_size(main, lv_pct(100), lv_pct(100));
    lv_obj_set_flex_flow(main, LV_FLEX_FLOW_ROW);
    lv_obj_set_style_bg_opa(main, LV_OPA_TRANSP, 0);
    lv_obj_clear_flag(main, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_style_pad_all(main, 8, 0);
    lv_obj_set_style_pad_gap(main, 8, 0);

    // Camera panel
    ctx.camera_panel = lv_obj_create(main);
    lv_obj_set_flex_grow(ctx.camera_panel, 3);
    lv_obj_set_style_bg_opa(ctx.camera_panel, LV_OPA_TRANSP, 0);
    lv_obj_clear_flag(ctx.camera_panel, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_t* camera_label = lv_label_create(ctx.camera_panel);
    lv_label_set_text(camera_label, LV_SYMBOL_VIDEO " Camera");
    lv_obj_align(camera_label, LV_ALIGN_TOP_LEFT, 6, 6);

    // Control panel
    lv_obj_t* control = lv_obj_create(main);
    lv_obj_set_flex_grow(control, 1);
    lv_obj_set_style_bg_color(control, lv_color_hex(0x252B35), 0);
    lv_obj_set_style_pad_all(control, 8, 0);
    lv_obj_set_flex_flow(control, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_gap(control, 6, 0);
    lv_obj_clear_flag(control, LV_OBJ_FLAG_SCROLLABLE);
    ctx.widgets.ui_fps_label = lv_label_create(control);
    lv_label_set_text(ctx.widgets.ui_fps_label, LV_SYMBOL_EYE_OPEN " UI: -- fps");

    lv_screen_load(ctx.main_screen);
    return true;
}

bool build_full_ui(UIContext& ctx, const LVGLWaylandConfig& config, bool debug_camera_panel_opaque) {
    lv_color_t color_background = lv_color_hex(0x1A1F26);
    lv_color_t color_surface    = lv_color_hex(0x252B35);
    lv_color_t color_primary    = lv_color_hex(0x5B9BD5);
    lv_color_t color_success    = lv_color_hex(0x7FB069);
    lv_color_t color_warning    = lv_color_hex(0xE6A055);
    lv_color_t color_error      = lv_color_hex(0xD67B7B);

    ctx.main_screen = lv_obj_create(nullptr);
    lv_obj_set_size(ctx.main_screen, config.screen_width, config.screen_height);
    lv_obj_set_style_bg_color(ctx.main_screen, color_background, 0);
    lv_obj_set_style_bg_opa(ctx.main_screen, LV_OPA_COVER, 0);
    lv_obj_set_style_pad_all(ctx.main_screen, 0, 0);
    lv_obj_clear_flag(ctx.main_screen, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(ctx.main_screen, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(ctx.main_screen, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(ctx.main_screen, 5, 0);

    // Header
    lv_obj_t* header_panel = lv_obj_create(ctx.main_screen);
    lv_obj_set_width(header_panel, lv_pct(100));
    lv_obj_set_height(header_panel, 60);
    lv_obj_set_flex_grow(header_panel, 0);
    lv_obj_set_style_bg_color(header_panel, color_surface, 0);
    lv_obj_set_style_radius(header_panel, 0, 0);
    lv_obj_set_style_border_width(header_panel, 0, 0);
    lv_obj_set_style_pad_all(header_panel, 10, 0);
    lv_obj_clear_flag(header_panel, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(header_panel, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(header_panel, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_gap(header_panel, 20, 0);

    ctx.widgets.system_title = lv_label_create(header_panel);
    lv_label_set_text(ctx.widgets.system_title, LV_SYMBOL_HOME " Bamboo Recognition System");
    lv_obj_set_style_text_color(ctx.widgets.system_title, lv_color_white(), 0);
    lv_obj_set_style_text_font(ctx.widgets.system_title, &lv_font_montserrat_16, 0);

    ctx.widgets.heartbeat_label = lv_label_create(header_panel);
    lv_label_set_text(ctx.widgets.heartbeat_label, LV_SYMBOL_LOOP " Online");
    lv_obj_set_style_text_color(ctx.widgets.heartbeat_label, color_success, 0);

    ctx.widgets.response_label = lv_label_create(header_panel);
    lv_label_set_text(ctx.widgets.response_label, LV_SYMBOL_CHARGE " 12ms");
    lv_obj_set_style_text_color(ctx.widgets.response_label, color_primary, 0);

    // Main container
    lv_obj_t* main_container = lv_obj_create(ctx.main_screen);
    lv_obj_set_width(main_container, lv_pct(100));
    lv_obj_set_flex_grow(main_container, 1);
    lv_obj_set_style_bg_opa(main_container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(main_container, 0, 0);
    lv_obj_set_style_pad_all(main_container, 5, 0);
    lv_obj_clear_flag(main_container, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(main_container, LV_FLEX_FLOW_ROW);
    lv_obj_set_flex_align(main_container, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(main_container, 10, 0);

    // Camera panel
    ctx.camera_panel = lv_obj_create(main_container);
    lv_obj_set_height(ctx.camera_panel, lv_pct(100));
    lv_obj_set_flex_grow(ctx.camera_panel, 3);
    if (debug_camera_panel_opaque) {
        lv_obj_set_style_bg_color(ctx.camera_panel, lv_color_hex(0x1D2330), 0);
        lv_obj_set_style_bg_opa(ctx.camera_panel, LV_OPA_80, 0);
        lv_obj_set_style_border_color(ctx.camera_panel, lv_color_hex(0x4A90E2), 0);
        lv_obj_set_style_border_opa(ctx.camera_panel, LV_OPA_60, 0);
        lv_obj_set_style_border_width(ctx.camera_panel, 2, 0);
    } else {
        lv_obj_set_style_bg_opa(ctx.camera_panel, LV_OPA_TRANSP, 0);
        lv_obj_set_style_border_opa(ctx.camera_panel, LV_OPA_TRANSP, 0);
    }
    lv_obj_set_style_pad_all(ctx.camera_panel, 10, 0);
    lv_obj_set_style_radius(ctx.camera_panel, 8, 0);
    lv_obj_clear_flag(ctx.camera_panel, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_clear_flag(ctx.camera_panel, LV_OBJ_FLAG_CLICKABLE);
    lv_obj_add_flag(ctx.camera_panel, LV_OBJ_FLAG_EVENT_BUBBLE);
    lv_obj_set_style_opa(ctx.camera_panel, debug_camera_panel_opaque ? LV_OPA_COVER : LV_OPA_0, 0);

    lv_obj_t* video_label = lv_label_create(ctx.camera_panel);
    lv_label_set_text(video_label, LV_SYMBOL_VIDEO " Camera Feed");
    lv_obj_set_style_bg_opa(ctx.camera_panel, LV_OPA_TRANSP, 0);
    lv_obj_set_style_opa(ctx.camera_panel, LV_OPA_0, 0);
    lv_obj_set_style_border_opa(ctx.camera_panel, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(ctx.camera_panel, 0, 0);
    lv_obj_set_style_text_color(video_label, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(video_label, &lv_font_montserrat_14, 0);
    lv_obj_align(video_label, LV_ALIGN_TOP_LEFT, 10, 10);

    // Control panel
    lv_obj_t* control_panel = lv_obj_create(main_container);
    lv_obj_set_height(control_panel, lv_pct(100));
    lv_obj_set_flex_grow(control_panel, 1);
    lv_obj_set_style_bg_color(control_panel, color_surface, 0);
    lv_obj_set_style_radius(control_panel, 12, 0);
    lv_obj_set_style_border_width(control_panel, 2, 0);
    lv_obj_set_style_border_color(control_panel, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_border_opa(control_panel, LV_OPA_50, 0);
    lv_obj_set_style_pad_all(control_panel, 15, 0);
    lv_obj_clear_flag(control_panel, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_flex_flow(control_panel, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(control_panel, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_gap(control_panel, 10, 0);

    lv_obj_t* control_title = lv_label_create(control_panel);
    lv_label_set_text(control_title, LV_SYMBOL_SETTINGS " System Info");
    lv_obj_set_style_text_color(control_title, lv_color_hex(0x5B9BD5), 0);
    lv_obj_set_style_text_font(control_title, &lv_font_montserrat_16, 0);

    lv_obj_t* jetson_section = lv_obj_create(control_panel);
    lv_obj_set_width(jetson_section, lv_pct(100));
    lv_obj_set_height(jetson_section, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_color(jetson_section, lv_color_hex(0x1A1F26), 0);
    lv_obj_set_style_radius(jetson_section, 8, 0);
    lv_obj_set_style_border_width(jetson_section, 1, 0);
    lv_obj_set_style_border_color(jetson_section, lv_color_hex(0x3A4451), 0);
    lv_obj_set_style_pad_all(jetson_section, 10, 0);
    lv_obj_set_flex_flow(jetson_section, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_gap(jetson_section, 6, 0);
    lv_obj_clear_flag(jetson_section, LV_OBJ_FLAG_SCROLLABLE);

    lv_obj_t* jetson_title = lv_label_create(jetson_section);
    lv_label_set_text(jetson_title, LV_SYMBOL_CHARGE " Jetson Orin Nano");
    lv_obj_set_style_text_color(jetson_title, lv_color_hex(0x70A5DB), 0);
    lv_obj_set_style_text_font(jetson_title, &lv_font_montserrat_14, 0);

    ctx.widgets.cpu_label = lv_label_create(jetson_section);
    lv_label_set_text(ctx.widgets.cpu_label, "CPU: --% @ --MHz");
    lv_obj_set_style_text_color(ctx.widgets.cpu_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(ctx.widgets.cpu_label, &lv_font_montserrat_12, 0);

    ctx.widgets.cpu_bar = lv_bar_create(jetson_section);
    lv_obj_set_size(ctx.widgets.cpu_bar, lv_pct(100), 10);
    lv_obj_set_style_bg_color(ctx.widgets.cpu_bar, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_bg_opa(ctx.widgets.cpu_bar, LV_OPA_COVER, 0);
    lv_bar_set_value(ctx.widgets.cpu_bar, 0, LV_ANIM_OFF);

    ctx.widgets.gpu_label = lv_label_create(jetson_section);
    lv_label_set_text(ctx.widgets.gpu_label, "GPU: --% @ --MHz");
    lv_obj_set_style_text_color(ctx.widgets.gpu_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(ctx.widgets.gpu_label, &lv_font_montserrat_12, 0);

    ctx.widgets.gpu_bar = lv_bar_create(jetson_section);
    lv_obj_set_size(ctx.widgets.gpu_bar, lv_pct(100), 10);
    lv_obj_set_style_bg_color(ctx.widgets.gpu_bar, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_bg_opa(ctx.widgets.gpu_bar, LV_OPA_COVER, 0);
    lv_bar_set_value(ctx.widgets.gpu_bar, 0, LV_ANIM_OFF);

    ctx.widgets.mem_label = lv_label_create(jetson_section);
    lv_label_set_text(ctx.widgets.mem_label, "RAM: --MB / --MB");
    lv_obj_set_style_text_color(ctx.widgets.mem_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(ctx.widgets.mem_label, &lv_font_montserrat_12, 0);

    ctx.widgets.mem_bar = lv_bar_create(jetson_section);
    lv_obj_set_size(ctx.widgets.mem_bar, lv_pct(100), 10);
    lv_obj_set_style_bg_color(ctx.widgets.mem_bar, lv_color_hex(0x2A3441), 0);
    lv_obj_set_style_bg_opa(ctx.widgets.mem_bar, LV_OPA_COVER, 0);
    lv_bar_set_value(ctx.widgets.mem_bar, 0, LV_ANIM_OFF);

    ctx.widgets.swap_usage_label = lv_label_create(jetson_section);
    lv_label_set_text(ctx.widgets.swap_usage_label, "SWAP: --MB");
    lv_obj_set_style_text_color(ctx.widgets.swap_usage_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(ctx.widgets.swap_usage_label, &lv_font_montserrat_12, 0);

    ctx.widgets.cpu_temp_label = lv_label_create(jetson_section);
    lv_label_set_text(ctx.widgets.cpu_temp_label, "CPU: --°C");
    lv_obj_set_style_text_color(ctx.widgets.cpu_temp_label, lv_color_hex(0xE6A055), 0);
    lv_obj_set_style_text_font(ctx.widgets.cpu_temp_label, &lv_font_montserrat_12, 0);

    ctx.widgets.gpu_temp_label = lv_label_create(jetson_section);
    lv_label_set_text(ctx.widgets.gpu_temp_label, "GPU: --°C");
    lv_obj_set_style_text_color(ctx.widgets.gpu_temp_label, lv_color_hex(0xE6A055), 0);
    lv_obj_set_style_text_font(ctx.widgets.gpu_temp_label, &lv_font_montserrat_12, 0);

    ctx.widgets.thermal_warning_label = lv_label_create(jetson_section);
    lv_label_set_text(ctx.widgets.thermal_warning_label, "");
    lv_obj_set_style_text_color(ctx.widgets.thermal_warning_label, color_error, 0);
    lv_obj_set_style_text_font(ctx.widgets.thermal_warning_label, &lv_font_montserrat_12, 0);
    lv_obj_add_flag(ctx.widgets.thermal_warning_label, LV_OBJ_FLAG_HIDDEN);

    ctx.widgets.power_total_label = lv_label_create(jetson_section);
    lv_label_set_text(ctx.widgets.power_total_label, "Power: --W");
    lv_obj_set_style_text_color(ctx.widgets.power_total_label, color_primary, 0);
    lv_obj_set_style_text_font(ctx.widgets.power_total_label, &lv_font_montserrat_12, 0);

    // AI section
    lv_obj_t* ai_section = lv_obj_create(control_panel);
    lv_obj_set_width(ai_section, lv_pct(100));
    lv_obj_set_height(ai_section, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_color(ai_section, lv_color_hex(0x1A1F26), 0);
    lv_obj_set_style_radius(ai_section, 8, 0);
    lv_obj_set_style_border_width(ai_section, 1, 0);
    lv_obj_set_style_border_color(ai_section, lv_color_hex(0x3A4451), 0);
    lv_obj_set_style_pad_all(ai_section, 10, 0);
    lv_obj_set_flex_flow(ai_section, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_gap(ai_section, 6, 0);
    lv_obj_clear_flag(ai_section, LV_OBJ_FLAG_SCROLLABLE);

    lv_obj_t* ai_title = lv_label_create(ai_section);
    lv_label_set_text(ai_title, LV_SYMBOL_IMAGE " AI Model");
    lv_obj_set_style_text_color(ai_title, lv_color_hex(0x7FB069), 0);
    lv_obj_set_style_text_font(ai_title, &lv_font_montserrat_14, 0);

    ctx.widgets.ai_model_name_label = lv_label_create(ai_section);
    lv_label_set_text(ctx.widgets.ai_model_name_label, "Model: YOLOv8");
    lv_obj_set_style_text_color(ctx.widgets.ai_model_name_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_set_style_text_font(ctx.widgets.ai_model_name_label, &lv_font_montserrat_12, 0);

    ctx.widgets.ai_fps_label = lv_label_create(ai_section);
    lv_label_set_text(ctx.widgets.ai_fps_label, "FPS: -- fps");
    lv_obj_set_style_text_color(ctx.widgets.ai_fps_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(ctx.widgets.ai_fps_label, &lv_font_montserrat_12, 0);

    ctx.widgets.ai_inference_time_label = lv_label_create(ai_section);
    lv_label_set_text(ctx.widgets.ai_inference_time_label, "Inference: --ms");
    lv_obj_set_style_text_color(ctx.widgets.ai_inference_time_label, color_primary, 0);
    lv_obj_set_style_text_font(ctx.widgets.ai_inference_time_label, &lv_font_montserrat_12, 0);

    ctx.widgets.ai_total_detections_label = lv_label_create(ai_section);
    lv_label_set_text(ctx.widgets.ai_total_detections_label, "Detected: 0 objects");
    lv_obj_set_style_text_color(ctx.widgets.ai_total_detections_label, lv_color_white(), 0);
    lv_obj_set_style_text_font(ctx.widgets.ai_total_detections_label, &lv_font_montserrat_12, 0);

    ctx.widgets.ai_confidence_label = lv_label_create(ai_section);
    lv_label_set_text(ctx.widgets.ai_confidence_label, "Confidence: --%");
    lv_obj_set_style_text_color(ctx.widgets.ai_confidence_label, color_success, 0);
    lv_obj_set_style_text_font(ctx.widgets.ai_confidence_label, &lv_font_montserrat_12, 0);

    // Footer/Modbus/系统控制等部分可以按需继续扩展...

    lv_screen_load(ctx.main_screen);
    return true;
}

} // namespace ui
} // namespace bamboo_cut
