/**
 * 通用工具函数
 */

#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 时间相关工具函数 */
uint64_t get_timestamp_ms(void);
uint64_t get_timestamp_us(void);
void sleep_ms(uint32_t ms);
void sleep_us(uint32_t us);
char* format_timestamp(uint64_t timestamp, char* buffer, size_t buffer_size);

/* 文件操作工具函数 */
bool file_exists(const char* path);
bool directory_exists(const char* path);
bool create_directory(const char* path);
size_t get_file_size(const char* path);
bool read_file_content(const char* path, char* buffer, size_t buffer_size);
bool write_file_content(const char* path, const char* content);

/* 字符串工具函数 */
char* trim_string(char* str);
bool string_starts_with(const char* str, const char* prefix);
bool string_ends_with(const char* str, const char* suffix);
void string_to_lower(char* str);
void string_to_upper(char* str);
char* string_replace(const char* orig, const char* rep, const char* with);

/* 数学工具函数 */
float clamp_float(float value, float min, float max);
int clamp_int(int value, int min, int max);
float lerp_float(float a, float b, float t);
float map_range(float value, float in_min, float in_max, float out_min, float out_max);
float calculate_distance(float x1, float y1, float x2, float y2);

/* 颜色工具函数 */
uint32_t rgb_to_argb(uint8_t r, uint8_t g, uint8_t b);
uint32_t rgba_to_argb(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
void argb_to_rgb(uint32_t argb, uint8_t* r, uint8_t* g, uint8_t* b);
void argb_to_rgba(uint32_t argb, uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a);
uint32_t blend_colors(uint32_t color1, uint32_t color2, float alpha);

/* 坐标转换工具函数 */
void pixel_to_mm(float pixel_x, float pixel_y, float* mm_x, float* mm_y);
void mm_to_pixel(float mm_x, float mm_y, float* pixel_x, float* pixel_y);
void screen_to_touch(int screen_x, int screen_y, int* touch_x, int* touch_y);
void touch_to_screen(int touch_x, int touch_y, int* screen_x, int* screen_y);

/* 内存工具函数 */
void* safe_malloc(size_t size);
void* safe_calloc(size_t num, size_t size);
void* safe_realloc(void* ptr, size_t size);
void safe_free(void** ptr);

/* 日志工具函数 */
void log_message(log_level_t level, const char* format, ...);
void log_debug(const char* format, ...);
void log_info(const char* format, ...);
void log_warn(const char* format, ...);
void log_error(const char* format, ...);
void log_fatal(const char* format, ...);

/* 错误处理工具函数 */
const char* error_code_to_string(error_code_t error_code);
void set_last_error(error_code_t error_code, const char* error_message);
error_code_t get_last_error(char* error_message, size_t buffer_size);
void clear_last_error(void);

/* 性能监控工具函数 */
void perf_counter_start(const char* name);
void perf_counter_end(const char* name);
void perf_counter_print_all(void);
void perf_counter_reset_all(void);

/* CPU和内存监控 */
float get_cpu_usage(void);
float get_memory_usage_mb(void);
float get_gpu_usage(void);
float get_gpu_memory_usage_mb(void);

/* 配置文件解析 */
bool parse_config_file(const char* config_path, system_config_t* config);
bool save_config_file(const char* config_path, const system_config_t* config);
bool validate_config(const system_config_t* config);

/* 检测结果工具函数 */
void init_detection_result(detection_result_t* result);
void copy_detection_result(const detection_result_t* src, detection_result_t* dst);
void print_detection_result(const detection_result_t* result);
bool is_valid_detection_point(const detection_point_t* point);

/* 图像处理工具函数 (需要OpenCV) */
#ifdef __cplusplus
}

#include <opencv2/opencv.hpp>

/* C++图像处理函数 */
namespace utils {
    // 图像格式转换
    cv::Mat bgr_to_rgb(const cv::Mat& bgr_image);
    cv::Mat rgb_to_bgr(const cv::Mat& rgb_image);
    cv::Mat yuv_to_rgb(const cv::Mat& yuv_image);
    cv::Mat rgb_to_yuv(const cv::Mat& rgb_image);
    
    // 图像缩放和裁剪
    cv::Mat resize_image(const cv::Mat& image, int width, int height);
    cv::Mat crop_image(const cv::Mat& image, int x, int y, int width, int height);
    cv::Mat letterbox_image(const cv::Mat& image, int target_width, int target_height);
    
    // 图像增强
    cv::Mat adjust_brightness(const cv::Mat& image, float brightness);
    cv::Mat adjust_contrast(const cv::Mat& image, float contrast);
    cv::Mat adjust_gamma(const cv::Mat& image, float gamma);
    
    // 绘制检测结果
    void draw_detection_points(cv::Mat& image, const detection_result_t* result);
    void draw_bounding_box(cv::Mat& image, int x, int y, int width, int height, 
                          const cv::Scalar& color, int thickness = 2);
    void draw_text(cv::Mat& image, const std::string& text, int x, int y, 
                   const cv::Scalar& color, double font_scale = 1.0);
    
    // 图像转换为LVGL格式
    bool convert_cv_mat_to_lvgl_buffer(const cv::Mat& mat, uint8_t* buffer, 
                                      int buffer_width, int buffer_height);
    bool convert_lvgl_buffer_to_cv_mat(const uint8_t* buffer, cv::Mat& mat,
                                      int width, int height);
}

extern "C" {
#endif

/* 系统信息获取 */
bool get_system_info(char* info_buffer, size_t buffer_size);
bool get_cpu_info(char* info_buffer, size_t buffer_size);
bool get_memory_info(char* info_buffer, size_t buffer_size);
bool get_gpu_info(char* info_buffer, size_t buffer_size);

/* 设备检测 */
bool detect_camera_devices(char device_list[][64], int max_devices);
bool detect_touch_devices(char device_list[][64], int max_devices);
bool detect_display_devices(char device_list[][64], int max_devices);

/* 权限检查 */
bool check_file_permissions(const char* path, int mode);
bool check_device_permissions(const char* device_path);
bool is_running_as_root(void);

/* 信号处理 */
typedef void (*signal_handler_t)(int signal);
bool setup_signal_handlers(signal_handler_t handler);
void cleanup_signal_handlers(void);

/* 进程管理 */
bool is_process_running(const char* process_name);
bool kill_process_by_name(const char* process_name);
bool start_background_process(const char* command);

#ifdef __cplusplus
}
#endif

#endif // COMMON_UTILS_H