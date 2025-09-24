#include "display/lvgl_display.h"
#include "display/framebuffer_driver.h"
#include "camera/camera_manager.h"
#include "backend/backend_client.h"
#include "lvgl.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#include <stdlib.h>

// LVGL显示驱动变量
static lv_disp_drv_t disp_drv;
static lv_disp_t *disp;
static lv_disp_draw_buf_t disp_buf;

// 摄像头管理器
static camera_manager_t *g_camera_manager = nullptr;

// 后端客户端
static backend_client_t *g_backend_client = nullptr;

// 视频面板对象
static lv_obj_t *g_video_panel = nullptr;

// 状态显示对象
static lv_obj_t *g_system_status_label = nullptr;
static lv_obj_t *g_plc_status_label = nullptr;
static lv_obj_t *g_ai_status_label = nullptr;
static lv_obj_t *g_camera_status_label = nullptr;
static lv_obj_t *g_performance_label = nullptr;
static lv_obj_t *g_detection_data_label = nullptr;

// 显示刷新回调函数
void lvgl_disp_flush(lv_disp_drv_t *drv, const lv_area_t *area, lv_color_t *color_p) {
    // 移除调试打印，避免日志洪水
    // printf("LVGL刷新显示区域: (%d,%d) -> (%d,%d)\n",
    //        area->x1, area->y1, area->x2, area->y2);
    
    // 获取framebuffer信息
    struct fb_var_screeninfo *vinfo = get_fb_vinfo();
    if (!vinfo) {
        printf("错误: 无法获取framebuffer信息\n");
        lv_disp_flush_ready(drv);
        return;
    }
    
    // 将LVGL颜色数据转换为32位ARGB格式并刷新到framebuffer
    uint32_t *fb_data = get_lvgl_buffer();
    if (!fb_data) {
        printf("错误: 无法获取LVGL缓冲区\n");
        lv_disp_flush_ready(drv);
        return;
    }
    
    // 复制颜色数据
    int32_t x, y;
    lv_color_t *src = color_p;
    
    for (y = area->y1; y <= area->y2; y++) {
        for (x = area->x1; x <= area->x2; x++) {
            // 将LVGL颜色转换为32位ARGB
            uint32_t color = lv_color_to32(*src);
            fb_data[y * vinfo->xres + x] = color;
            src++;
        }
    }
    
    // 刷新到实际framebuffer
    framebuffer_flush(area->x1, area->y1, area->x2, area->y2, fb_data);
    
    // 通知LVGL刷新完成
    lv_disp_flush_ready(drv);
}

bool lvgl_display_init() {
    printf("初始化 LVGL显示驱动...\n");
    
    // 首先初始化framebuffer驱动
    if (!framebuffer_driver_init()) {
        printf("错误: Framebuffer驱动初始化失败\n");
        return false;
    }
    
    // 获取framebuffer信息
    struct fb_var_screeninfo *vinfo = get_fb_vinfo();
    if (!vinfo) {
        printf("错误: 无法获取framebuffer信息\n");
        return false;
    }
    
    printf("配置LVGL显示驱动:\n");
    printf("  分辨率: %dx%d\n", vinfo->xres, vinfo->yres);
    printf("  色深: %d位\n", vinfo->bits_per_pixel);
    
    // 获取LVGL缓冲区
    uint32_t *buf1 = get_lvgl_buffer();
    uint32_t *buf2 = get_lvgl_buffer2();
    
    if (!buf1 || !buf2) {
        printf("错误: 无法获取LVGL缓冲区\n");
        framebuffer_driver_deinit();
        return false;
    }
    
    // 初始化LVGL显示缓冲区
    lv_disp_draw_buf_init(&disp_buf, buf1, buf2, vinfo->xres * vinfo->yres);
    
    // 初始化显示驱动
    lv_disp_drv_init(&disp_drv);
    disp_drv.draw_buf = &disp_buf;
    disp_drv.flush_cb = lvgl_disp_flush;
    disp_drv.hor_res = vinfo->xres;
    disp_drv.ver_res = vinfo->yres;
    
    // 注册显示驱动
    disp = lv_disp_drv_register(&disp_drv);
    if (!disp) {
        printf("错误: LVGL显示驱动注册失败\n");
        framebuffer_driver_deinit();
        return false;
    }
    
    // 创建测试界面
    create_test_ui();
    
    // 初始化摄像头
    init_camera_system();
    
    // 初始化后端系统
    init_backend_system();
    
    printf("LVGL显示驱动初始化成功\n");
    return true;
}

void lvgl_display_deinit() {
    printf("清理 LVGL显示驱动\n");
    
    // 清理后端系统
    deinit_backend_system();
    
    // 清理摄像头系统
    deinit_camera_system();
    
    if (disp) {
        lv_obj_clean(lv_scr_act());
    }
    
    framebuffer_driver_deinit();
}

// 创建智能切竹机工业控制界面
void create_test_ui() {
    printf("创建智能切竹机工业控制界面...\n");
    
    // 获取当前屏幕
    lv_obj_t *scr = lv_scr_act();
    
    // 设置工业级深蓝色背景
    lv_obj_set_style_bg_color(scr, lv_color_hex(0x1e3a5f), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(scr, LV_OPA_COVER, LV_PART_MAIN);
    
    // ========== 顶部状态栏 ==========
    lv_obj_t *header = lv_obj_create(scr);
    lv_obj_set_size(header, LV_PCT(100), 60);
    lv_obj_align(header, LV_ALIGN_TOP_MID, 0, 0);
    lv_obj_set_style_bg_color(header, lv_color_hex(0x2c5282), LV_PART_MAIN);
    lv_obj_set_style_border_width(header, 0, LV_PART_MAIN);
    
    // 系统标题 (使用英文避免乱码)
    lv_obj_t *title_label = lv_label_create(header);
    lv_label_set_text(title_label, "Bamboo Cutting System v2.0");
    lv_obj_set_style_text_font(title_label, &lv_font_montserrat_14, 0);
    lv_obj_set_style_text_color(title_label, lv_color_white(), 0);
    lv_obj_align(title_label, LV_ALIGN_LEFT_MID, 20, 0);
    
    // 系统状态指示灯
    lv_obj_t *status_led = lv_led_create(header);
    lv_led_set_color(status_led, lv_color_hex(0x00FF00));
    lv_led_on(status_led);
    lv_obj_align(status_led, LV_ALIGN_RIGHT_MID, -80, 0);
    
    // 状态文本 (使用英文避免乱码)
    lv_obj_t *status_text = lv_label_create(header);
    lv_label_set_text(status_text, "System Ready");
    lv_obj_set_style_text_color(status_text, lv_color_white(), 0);
    lv_obj_align(status_text, LV_ALIGN_RIGHT_MID, -20, 0);
    
    // ========== 左侧控制面板 ==========
    lv_obj_t *control_panel = lv_obj_create(scr);
    lv_obj_set_size(control_panel, 300, LV_PCT(70));
    lv_obj_align(control_panel, LV_ALIGN_LEFT_MID, 10, 10);
    lv_obj_set_style_bg_color(control_panel, lv_color_hex(0x2d3748), LV_PART_MAIN);
    lv_obj_set_style_border_color(control_panel, lv_color_hex(0x4a5568), LV_PART_MAIN);
    lv_obj_set_style_border_width(control_panel, 2, LV_PART_MAIN);
    
    // 控制面板标题 (使用英文避免乱码)
    lv_obj_t *panel_title = lv_label_create(control_panel);
    lv_label_set_text(panel_title, "Control Panel");
    lv_obj_set_style_text_color(panel_title, lv_color_hex(0xE2E8F0), 0);
    lv_obj_align(panel_title, LV_ALIGN_TOP_MID, 0, 15);
    
    // 系统状态
    g_system_status_label = lv_label_create(control_panel);
    lv_label_set_text(g_system_status_label, "System: Initializing...");
    lv_obj_set_style_text_color(g_system_status_label, lv_color_hex(0xFBBF24), 0);
    lv_obj_align(g_system_status_label, LV_ALIGN_TOP_LEFT, 20, 50);
    
    // AI检测状态
    g_ai_status_label = lv_label_create(control_panel);
    lv_label_set_text(g_ai_status_label, "AI Engine: Connecting...");
    lv_obj_set_style_text_color(g_ai_status_label, lv_color_hex(0xFBBF24), 0);
    lv_obj_align(g_ai_status_label, LV_ALIGN_TOP_LEFT, 20, 80);
    
    // 摄像头状态
    g_camera_status_label = lv_label_create(control_panel);
    lv_label_set_text(g_camera_status_label, "Camera: 640x480@30fps");
    lv_obj_set_style_text_color(g_camera_status_label, lv_color_hex(0x68D391), 0);
    lv_obj_align(g_camera_status_label, LV_ALIGN_TOP_LEFT, 20, 110);
    
    // PLC通信状态
    g_plc_status_label = lv_label_create(control_panel);
    lv_label_set_text(g_plc_status_label, "PLC: Connecting...");
    lv_obj_set_style_text_color(g_plc_status_label, lv_color_hex(0xFBBF24), 0);
    lv_obj_align(g_plc_status_label, LV_ALIGN_TOP_LEFT, 20, 140);
    
    // 主要控制按钮
    lv_obj_t *start_btn = lv_btn_create(control_panel);
    lv_obj_set_size(start_btn, 250, 50);
    lv_obj_align(start_btn, LV_ALIGN_TOP_MID, 0, 180);
    lv_obj_set_style_bg_color(start_btn, lv_color_hex(0x38A169), LV_PART_MAIN);
    
    lv_obj_t *start_label = lv_label_create(start_btn);
    lv_label_set_text(start_label, "Start AI Detection");
    lv_obj_set_style_text_color(start_label, lv_color_white(), 0);
    lv_obj_center(start_label);
    
    // 停止按钮
    lv_obj_t *stop_btn = lv_btn_create(control_panel);
    lv_obj_set_size(stop_btn, 250, 50);
    lv_obj_align(stop_btn, LV_ALIGN_TOP_MID, 0, 250);
    lv_obj_set_style_bg_color(stop_btn, lv_color_hex(0xE53E3E), LV_PART_MAIN);
    
    lv_obj_t *stop_label = lv_label_create(stop_btn);
    lv_label_set_text(stop_label, "Stop Detection");
    lv_obj_set_style_text_color(stop_label, lv_color_white(), 0);
    lv_obj_center(stop_label);
    
    // ========== 右侧视频显示区域 ==========
    lv_obj_t *video_panel = lv_obj_create(scr);
    lv_obj_set_size(video_panel, LV_PCT(60), LV_PCT(70));
    lv_obj_align(video_panel, LV_ALIGN_RIGHT_MID, -10, 10);
    lv_obj_set_style_bg_color(video_panel, lv_color_hex(0x000000), LV_PART_MAIN);
    lv_obj_set_style_border_color(video_panel, lv_color_hex(0x4a5568), LV_PART_MAIN);
    lv_obj_set_style_border_width(video_panel, 2, LV_PART_MAIN);
    
    // 视频区域标题
    lv_obj_t *video_title = lv_label_create(video_panel);
    lv_label_set_text(video_title, "Live Video Monitor");
    lv_obj_set_style_text_color(video_title, lv_color_white(), 0);
    lv_obj_align(video_title, LV_ALIGN_TOP_MID, 0, 15);
    
    // 保存视频面板引用，用于后续创建摄像头显示
    g_video_panel = video_panel;
    
    // 视频占位符
    lv_obj_t *video_placeholder = lv_label_create(video_panel);
    lv_label_set_text(video_placeholder, "Camera Initializing...\nWaiting for video stream");
    lv_obj_set_style_text_color(video_placeholder, lv_color_hex(0x888888), 0);
    lv_obj_set_style_text_align(video_placeholder, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_center(video_placeholder);
    
    // ========== 底部数据显示区域 ==========
    lv_obj_t *data_panel = lv_obj_create(scr);
    lv_obj_set_size(data_panel, LV_PCT(95), 120);
    lv_obj_align(data_panel, LV_ALIGN_BOTTOM_MID, 0, -10);
    lv_obj_set_style_bg_color(data_panel, lv_color_hex(0x2d3748), LV_PART_MAIN);
    lv_obj_set_style_border_color(data_panel, lv_color_hex(0x4a5568), LV_PART_MAIN);
    lv_obj_set_style_border_width(data_panel, 2, LV_PART_MAIN);
    
    // 数据面板标题
    lv_obj_t *data_title = lv_label_create(data_panel);
    lv_label_set_text(data_title, "Detection Data");
    lv_obj_set_style_text_color(data_title, lv_color_hex(0xE2E8F0), 0);
    lv_obj_align(data_title, LV_ALIGN_TOP_LEFT, 20, 10);
    
    // 检测结果显示
    g_detection_data_label = lv_label_create(data_panel);
    lv_label_set_text(g_detection_data_label, "Cut Points: 0 | Coordinates: Waiting for results...");
    lv_obj_set_style_text_color(g_detection_data_label, lv_color_hex(0xA0AEC0), 0);
    lv_obj_align(g_detection_data_label, LV_ALIGN_TOP_LEFT, 20, 40);
    
    // 性能指标
    g_performance_label = lv_label_create(data_panel);
    lv_label_set_text(g_performance_label, "Performance: Initializing...");
    lv_obj_set_style_text_color(g_performance_label, lv_color_hex(0x68D391), 0);
    lv_obj_align(g_performance_label, LV_ALIGN_BOTTOM_LEFT, 20, -15);
    
    printf("Bamboo cutting industrial control interface created\n");
}

// 释放可能占用摄像头的进程
bool release_camera_device(const char* device_path) {
    printf("尝试释放摄像头设备: %s\n", device_path);
    
    // 查找使用摄像头设备的进程
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "lsof %s 2>/dev/null", device_path);
    
    FILE* fp = popen(cmd, "r");
    if (fp) {
        char line[256];
        bool found_processes = false;
        
        while (fgets(line, sizeof(line), fp)) {
            if (strstr(line, device_path)) {
                found_processes = true;
                printf("发现占用摄像头的进程: %s", line);
            }
        }
        pclose(fp);
        
        if (found_processes) {
            printf("尝试终止占用摄像头的进程...\n");
            // 强制终止占用摄像头的进程
            snprintf(cmd, sizeof(cmd), "fuser -k %s 2>/dev/null", device_path);
            int ret = system(cmd);
            (void)ret; // 忽略返回值
            
            // 等待进程释放设备
            usleep(500000); // 500ms
            return true;
        }
    }
    
    return false;
}

// 检测可用的摄像头设备
const char* detect_available_camera() {
    static const char* camera_devices[] = {
        "/dev/video0",
        "/dev/video1",
        "/dev/video2",
        "/dev/video3",
        NULL
    };
    
    for (int i = 0; camera_devices[i] != NULL; i++) {
        if (access(camera_devices[i], F_OK) == 0) {
            printf("检测到摄像头设备: %s\n", camera_devices[i]);
            
            // 尝试简单打开测试设备是否可用
            int fd = open(camera_devices[i], O_RDWR | O_NONBLOCK);
            if (fd >= 0) {
                close(fd);
                printf("摄像头设备 %s 可用\n", camera_devices[i]);
                return camera_devices[i];
            } else {
                printf("摄像头设备 %s 被占用: %s\n", camera_devices[i], strerror(errno));
            }
        }
    }
    
    return NULL;
}

// 初始化摄像头系统（增强版本，带设备释放和重试机制）
void init_camera_system() {
    printf("初始化摄像头系统...\n");
    
    // 首先检测可用的摄像头设备
    const char* camera_device = detect_available_camera();
    if (!camera_device) {
        printf("错误：未检测到可用的摄像头设备\n");
        return;
    }
    
    // 尝试释放设备（如果被占用）
    bool device_released = false;
    int retry_count = 0;
    const int max_retries = 3;
    
    while (retry_count < max_retries) {
        printf("摄像头初始化尝试 %d/%d\n", retry_count + 1, max_retries);
        
        // 创建摄像头管理器（GStreamer流模式）
        const char* stream_url = "udp://127.0.0.1:5000";  // 默认流URL
        const char* stream_format = "H264";               // 默认流格式
        g_camera_manager = camera_manager_create(stream_url, stream_format, 640, 480, 30);
        if (!g_camera_manager) {
            printf("错误：创建摄像头管理器失败\n");
            break;
        }
        
        // 尝试初始化摄像头
        if (camera_manager_init(g_camera_manager)) {
            printf("摄像头初始化成功\n");
            
            // 在视频面板中创建摄像头显示对象
            if (g_video_panel && camera_manager_create_video_object(g_camera_manager, g_video_panel)) {
                // 开始摄像头捕获
                if (camera_manager_start_capture(g_camera_manager)) {
                    printf("摄像头系统初始化成功\n");
                    
                    // 更新摄像头状态显示
                    if (g_camera_status_label) {
                        char status_text[128];
                        snprintf(status_text, sizeof(status_text), "Camera: %s@30fps", camera_device);
                        lv_label_set_text(g_camera_status_label, status_text);
                        lv_obj_set_style_text_color(g_camera_status_label, lv_color_hex(0x68D391), 0);
                    }
                    return;
                } else {
                    printf("错误：启动摄像头捕获失败\n");
                }
            } else {
                printf("错误：创建摄像头显示对象失败\n");
            }
        } else {
            printf("错误：初始化摄像头失败，设备可能被占用\n");
            
            // 如果还没有尝试释放设备，则尝试释放
            if (!device_released) {
                device_released = release_camera_device(camera_device);
                if (device_released) {
                    printf("已释放摄像头设备，将重试初始化\n");
                }
            }
        }
        
        // 清理失败的摄像头管理器
        if (g_camera_manager) {
            camera_manager_destroy(g_camera_manager);
            g_camera_manager = nullptr;
        }
        
        retry_count++;
        
        // 如果不是最后一次尝试，等待一段时间再重试
        if (retry_count < max_retries) {
            printf("等待 1 秒后重试...\n");
            sleep(1);
        }
    }
    
    printf("错误：摄像头系统初始化失败，已达到最大重试次数\n");
    
    // 更新摄像头状态显示为错误状态
    if (g_camera_status_label) {
        lv_label_set_text(g_camera_status_label, "Camera: Initialization Failed");
        lv_obj_set_style_text_color(g_camera_status_label, lv_color_hex(0xEF4444), 0);
    }
}

// 清理摄像头系统
void deinit_camera_system() {
    if (g_camera_manager) {
        printf("清理摄像头系统\n");
        camera_manager_destroy(g_camera_manager);
        g_camera_manager = nullptr;
    }
}

// 更新摄像头显示
void update_camera_display() {
    if (g_camera_manager && camera_manager_is_running(g_camera_manager)) {
        camera_manager_update_display(g_camera_manager);
    }
}

// 获取摄像头状态
bool is_camera_running() {
    return g_camera_manager && camera_manager_is_running(g_camera_manager);
}

// 获取摄像头帧率
double get_camera_fps() {
    return g_camera_manager ? camera_manager_get_fps(g_camera_manager) : 0.0;
}

// 初始化后端系统（非阻塞模式）
void init_backend_system() {
    printf("初始化后端系统（非阻塞模式）...\n");
    
    // 检查后端进程是否运行（非阻塞）
    if (!backend_client_is_backend_running()) {
        printf("警告：后端进程未运行，前端将在无后端模式下启动\n");
        // 不尝试启动后端进程，避免阻塞前端启动
    }
    
    // 创建后端客户端
    g_backend_client = backend_client_create("127.0.0.1", 8888);
    if (!g_backend_client) {
        printf("警告：创建后端客户端失败，前端将在离线模式下运行\n");
        return;
    }
    
    // 非阻塞连接到后端
    printf("尝试连接后端（非阻塞）...\n");
    if (!backend_client_connect(g_backend_client)) {
        printf("警告：连接后端失败，前端将继续运行并在后台重试\n");
    }
    
    // 启动通信线程（即使连接失败也启动，因为线程内有重连逻辑）
    if (!backend_client_start_communication(g_backend_client)) {
        printf("警告：启动后端通信线程失败\n");
        backend_client_destroy(g_backend_client);
        g_backend_client = nullptr;
        printf("前端将在无后端通信模式下运行\n");
        return;
    }
    
    printf("后端系统初始化完成（可能处于离线状态，将自动重连）\n");
}

// 清理后端系统
void deinit_backend_system() {
    if (g_backend_client) {
        printf("清理后端系统\n");
        backend_client_destroy(g_backend_client);
        g_backend_client = nullptr;
    }
}

// 更新系统状态显示（安全版本，带频率限制）
void update_system_status_display() {
    // 频率限制：每500ms更新一次状态，避免过度刷新
    static uint32_t last_update_time = 0;
    uint32_t current_time = lv_tick_get();
    if (current_time - last_update_time < 500) {
        return; // 未到更新时间，直接返回
    }
    last_update_time = current_time;
    
    // 安全检查：如果后端客户端不存在，直接返回，不进行任何操作
    if (!g_backend_client) {
        // 静默返回，避免在初始化期间产生错误
        return;
    }
    
    // 获取系统健康信息（安全版本）
    system_health_t health;
    memset(&health, 0, sizeof(health)); // 初始化结构体
    
    if (backend_client_get_system_health(g_backend_client, &health)) {
        // 更新性能标签
        if (g_performance_label) {
            char perf_text[256];
            snprintf(perf_text, sizeof(perf_text),
                    "CPU: %.1f%% | Memory: %.1f%% | GPU: %.1f%% | Temp: %.1f°C",
                    health.cpu_usage, health.memory_usage, health.gpu_usage, health.temperature);
            lv_label_set_text(g_performance_label, perf_text);
        }
    }
    
    // 安全获取后端状态
    backend_status_t backend_status = backend_client_get_backend_status(g_backend_client);
    if (g_system_status_label) {
        switch (backend_status) {
            case BACKEND_CONNECTED:
                lv_label_set_text(g_system_status_label, "System: Running");
                lv_obj_set_style_text_color(g_system_status_label, lv_color_hex(0x68D391), 0);
                break;
            case BACKEND_CONNECTING:
                lv_label_set_text(g_system_status_label, "System: Connecting...");
                lv_obj_set_style_text_color(g_system_status_label, lv_color_hex(0xFBBF24), 0);
                break;
            default:
                lv_label_set_text(g_system_status_label, "System: Disconnected");
                lv_obj_set_style_text_color(g_system_status_label, lv_color_hex(0xEF4444), 0);
                break;
        }
    }
    
    // 安全获取PLC状态
    plc_status_t plc_status = backend_client_get_plc_status(g_backend_client);
    if (g_plc_status_label) {
        switch (plc_status) {
            case PLC_CONNECTED:
                lv_label_set_text(g_plc_status_label, "PLC: 192.168.1.10:502 Connected");
                lv_obj_set_style_text_color(g_plc_status_label, lv_color_hex(0x68D391), 0);
                break;
            case PLC_ERROR:
                lv_label_set_text(g_plc_status_label, "PLC: Communication Error");
                lv_obj_set_style_text_color(g_plc_status_label, lv_color_hex(0xEF4444), 0);
                break;
            default:
                lv_label_set_text(g_plc_status_label, "PLC: Disconnected");
                lv_obj_set_style_text_color(g_plc_status_label, lv_color_hex(0xFBBF24), 0);
                break;
        }
    }
    
    // 更新AI状态
    if (g_ai_status_label) {
        if (backend_status == BACKEND_CONNECTED) {
            lv_label_set_text(g_ai_status_label, "AI Engine: TensorRT Ready");
            lv_obj_set_style_text_color(g_ai_status_label, lv_color_hex(0x68D391), 0);
        } else {
            lv_label_set_text(g_ai_status_label, "AI Engine: Offline");
            lv_obj_set_style_text_color(g_ai_status_label, lv_color_hex(0xFBBF24), 0);
        }
    }
    
    // 安全获取切割坐标
    cutting_coordinate_t coordinate;
    memset(&coordinate, 0, sizeof(coordinate)); // 初始化结构体
    
    if (backend_client_get_coordinate(g_backend_client, &coordinate) && g_detection_data_label) {
        if (coordinate.coordinate_ready) {
            char coord_text[256];
            snprintf(coord_text, sizeof(coord_text),
                    "Cut Point: X=%.1fmm | Blade=%d | Quality=%s",
                    coordinate.x_coordinate / 10.0f,
                    coordinate.blade_number,
                    coordinate.cutting_quality == 0 ? "Good" : "Poor");
            lv_label_set_text(g_detection_data_label, coord_text);
        } else {
            lv_label_set_text(g_detection_data_label, "Cut Points: 0 | Coordinates: Waiting for detection...");
        }
    }
}

// 获取后端客户端
void* get_backend_client() {
    return (void*)g_backend_client;
}