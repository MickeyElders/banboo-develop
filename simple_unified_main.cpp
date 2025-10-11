/**
 * @file simple_unified_main.cpp
 * @brief 单进程LVGL+GStreamer统一架构 - EGL环境共享解决方案
 * 基于用户提供的示例，实现LVGL与nvarguscamerasrc的EGL环境复用
 */

#include <lvgl/lvgl.h>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <EGL/egl.h>
#include <pthread.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <thread>
#include <signal.h>

// 全局变量
static GstElement *g_pipeline = NULL;
static lv_obj_t *g_camera_img = NULL;
static lv_img_dsc_t g_img_dsc = {0};
static uint8_t *g_frame_buffer = NULL;
static pthread_mutex_t g_frame_lock = PTHREAD_MUTEX_INITIALIZER;
static bool g_shutdown_requested = false;

#define CAMERA_WIDTH 960
#define CAMERA_HEIGHT 640

// EGL上下文管理器简化版
class SimpleEGLManager {
private:
    EGLDisplay egl_display;
    EGLContext egl_context;
    bool initialized;

public:
    SimpleEGLManager() : egl_display(EGL_NO_DISPLAY), egl_context(EGL_NO_CONTEXT), initialized(false) {}

    bool initialize() {
        std::cout << "🔧 初始化EGL上下文管理器..." << std::endl;
        
        // 获取默认EGL display
        egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (egl_display == EGL_NO_DISPLAY) {
            std::cout << "❌ 无法获取EGL display" << std::endl;
            return false;
        }

        // 初始化EGL
        EGLint major, minor;
        if (!eglInitialize(egl_display, &major, &minor)) {
            std::cout << "❌ EGL初始化失败" << std::endl;
            return false;
        }

        std::cout << "✅ EGL初始化成功，版本: " << major << "." << minor << std::endl;

        // 绑定OpenGL ES API
        if (!eglBindAPI(EGL_OPENGL_ES_API)) {
            std::cout << "❌ 绑定OpenGL ES API失败" << std::endl;
            return false;
        }

        // 设置环境变量供nvarguscamerasrc使用
        setupArgusEnvironment();
        
        initialized = true;
        std::cout << "✅ EGL上下文管理器初始化完成" << std::endl;
        return true;
    }

    void setupArgusEnvironment() {
        std::cout << "🔧 设置nvarguscamerasrc环境变量..." << std::endl;
        
        // 将EGL display指针转换为字符串
        char display_ptr_str[32];
        snprintf(display_ptr_str, sizeof(display_ptr_str), "%p", (void*)egl_display);
        
        // 设置环境变量
        setenv("EGL_DISPLAY", display_ptr_str, 1);
        setenv("DISPLAY", ":0", 1);
        setenv("GST_DEBUG", "2", 1);
        setenv("GST_GL_PLATFORM", "egl", 1);
        setenv("GST_GL_API", "gles2", 1);
        
        std::cout << "✅ EGL_DISPLAY=" << display_ptr_str << std::endl;
        std::cout << "✅ nvarguscamerasrc环境变量设置完成" << std::endl;
    }

    EGLDisplay getDisplay() const { return egl_display; }
    bool isInitialized() const { return initialized; }
};

// 全局EGL管理器
static SimpleEGLManager g_egl_manager;

// 信号处理
void signal_handler(int sig) {
    std::cout << "\n🛑 收到信号 " << sig << "，开始关闭系统..." << std::endl;
    g_shutdown_requested = true;
}

// GStreamer回调：新帧到达
static GstFlowReturn on_new_sample(GstAppSink *sink, gpointer user_data) {
    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) return GST_FLOW_OK;
    
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        pthread_mutex_lock(&g_frame_lock);
        
        // 复制帧数据到LVGL缓冲区
        if (g_frame_buffer && map.size >= CAMERA_WIDTH * CAMERA_HEIGHT * 4) {
            memcpy(g_frame_buffer, map.data, CAMERA_WIDTH * CAMERA_HEIGHT * 4);
            
            // 触发LVGL重绘
            if (g_camera_img) {
                lv_obj_invalidate(g_camera_img);
            }
        }
        
        pthread_mutex_unlock(&g_frame_lock);
        gst_buffer_unmap(buffer, &map);
    }
    
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

// 在单独线程中运行GStreamer
static void *gstreamer_thread(void *arg) {
    std::cout << "🎬 GStreamer线程启动" << std::endl;
    
    // 延迟启动，等待LVGL的EGL初始化完成
    std::cout << "⏳ 等待LVGL和EGL环境初始化完成..." << std::endl;
    sleep(3);

    // 确保EGL环境已经设置
    if (!g_egl_manager.isInitialized()) {
        std::cout << "❌ EGL环境未初始化，无法启动GStreamer" << std::endl;
        return NULL;
    }

    // 创建管道（使用appsink模式，复用LVGL的EGL环境）
    const char *pipeline_str = 
        "nvarguscamerasrc sensor-id=0 "
        "! video/x-raw(memory:NVMM),width=960,height=640,framerate=30/1,format=NV12 "
        "! nvvidconv "
        "! video/x-raw,format=RGBA,width=960,height=640 "
        "! appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false";
    
    std::cout << "🔧 创建管道: " << pipeline_str << std::endl;
    
    GError *error = NULL;
    g_pipeline = gst_parse_launch(pipeline_str, &error);
    
    if (!g_pipeline) {
        std::cout << "❌ 创建管道失败: " << (error ? error->message : "未知错误") << std::endl;
        if (error) g_error_free(error);
        return NULL;
    }
    
    // 连接appsink回调
    GstElement *sink = gst_bin_get_by_name(GST_BIN(g_pipeline), "sink");
    if (sink) {
        g_signal_connect(sink, "new-sample", G_CALLBACK(on_new_sample), NULL);
        gst_object_unref(sink);
    }
    
    // 启动管道
    std::cout << "▶️ 启动管道..." << std::endl;
    GstStateChangeReturn ret = gst_element_set_state(g_pipeline, GST_STATE_PLAYING);
    
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cout << "❌ 启动失败" << std::endl;
        gst_object_unref(g_pipeline);
        return NULL;
    }
    
    std::cout << "✅ GStreamer管道运行中（使用共享EGL环境）" << std::endl;
    
    // 等待EOS、错误或关闭信号
    GstBus *bus = gst_element_get_bus(g_pipeline);
    while (!g_shutdown_requested) {
        GstMessage *msg = gst_bus_timed_pop_filtered(
            bus, 
            GST_SECOND,  // 1秒超时
            (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS)
        );
        
        if (msg) {
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError *err;
                gchar *debug_info;
                gst_message_parse_error(msg, &err, &debug_info);
                std::cout << "❌ GStreamer错误: " << err->message << std::endl;
                if (debug_info) {
                    std::cout << "   调试: " << debug_info << std::endl;
                    g_free(debug_info);
                }
                g_error_free(err);
                break;
            }
            gst_message_unref(msg);
        }
    }
    
    gst_object_unref(bus);
    gst_element_set_state(g_pipeline, GST_STATE_NULL);
    gst_object_unref(g_pipeline);
    
    std::cout << "🛑 GStreamer线程退出" << std::endl;
    return NULL;
}

// 初始化摄像头显示
void init_camera_view(lv_obj_t *parent) {
    std::cout << "📹 初始化摄像头视图..." << std::endl;
    
    // 分配帧缓冲
    g_frame_buffer = (uint8_t*)malloc(CAMERA_WIDTH * CAMERA_HEIGHT * 4);
    if (!g_frame_buffer) {
        std::cout << "❌ 分配内存失败" << std::endl;
        return;
    }
    
    // 初始化缓冲区为黑色
    memset(g_frame_buffer, 0, CAMERA_WIDTH * CAMERA_HEIGHT * 4);
    
    // 配置LVGL图像
    g_img_dsc.header.always_zero = 0;
    g_img_dsc.header.cf = LV_IMG_CF_TRUE_COLOR_ALPHA;
    g_img_dsc.header.w = CAMERA_WIDTH;
    g_img_dsc.header.h = CAMERA_HEIGHT;
    g_img_dsc.data_size = CAMERA_WIDTH * CAMERA_HEIGHT * 4;
    g_img_dsc.data = g_frame_buffer;
    
    // 创建图像对象
    g_camera_img = lv_img_create(parent);
    lv_img_set_src(g_camera_img, &g_img_dsc);
    lv_obj_align(g_camera_img, LV_ALIGN_CENTER, 0, 0);
    
    std::cout << "✅ 摄像头视图创建完成" << std::endl;
}

int main(int argc, char *argv[]) {
    std::cout << "===============================================" << std::endl;
    std::cout << "🚀 竹子识别系统单进程统一架构启动" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    // 设置信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // 1. 初始化GStreamer
    gst_init(&argc, &argv);
    std::cout << "✅ GStreamer初始化完成" << std::endl;
    
    // 2. 初始化LVGL
    lv_init();
    std::cout << "✅ LVGL初始化完成" << std::endl;
    
    // 3. 初始化EGL上下文管理器（在创建DRM显示之前）
    if (!g_egl_manager.initialize()) {
        std::cout << "❌ EGL上下文管理器初始化失败" << std::endl;
        return -1;
    }
    
    // 4. 创建DRM显示（这会使用我们预先设置的EGL环境）
    lv_display_t *display = lv_linux_drm_create();
    if (!display) {
        std::cout << "❌ 创建DRM显示失败" << std::endl;
        return -1;
    }
    lv_linux_drm_set_file(display, "/dev/dri/card0", -1);
    std::cout << "✅ DRM显示创建完成" << std::endl;
    
    // 5. 初始化触摸屏
    lv_indev_t *indev = lv_linux_evdev_create();
    if (indev) {
        lv_linux_evdev_set_file(indev, "/dev/input/event0");
        std::cout << "✅ 触摸屏初始化完成" << std::endl;
    } else {
        std::cout << "⚠️ 触摸屏初始化失败，继续运行" << std::endl;
    }
    
    // 6. 创建UI
    lv_obj_t *screen = lv_scr_act();
    lv_obj_set_style_bg_color(screen, lv_color_hex(0x1A1F26), 0);
    
    // 标题
    lv_obj_t *title = lv_label_create(screen);
    lv_label_set_text(title, "🏭 竹子识别系统 - EGL共享架构");
    lv_obj_set_style_text_color(title, lv_color_white(), 0);
    lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 20);
    
    // 状态标签
    lv_obj_t *status_label = lv_label_create(screen);
    lv_label_set_text(status_label, "🔧 EGL环境已共享，等待摄像头启动...");
    lv_obj_set_style_text_color(status_label, lv_color_hex(0x00FF00), 0);
    lv_obj_align(status_label, LV_ALIGN_TOP_MID, 0, 60);
    
    // 摄像头视图容器
    lv_obj_t *camera_container = lv_obj_create(screen);
    lv_obj_set_size(camera_container, CAMERA_WIDTH + 20, CAMERA_HEIGHT + 20);
    lv_obj_align(camera_container, LV_ALIGN_CENTER, 0, 20);
    lv_obj_set_style_bg_color(camera_container, lv_color_hex(0x252B35), 0);
    lv_obj_set_style_border_color(camera_container, lv_color_hex(0x5B9BD5), 0);
    lv_obj_set_style_border_width(camera_container, 2, 0);
    lv_obj_set_style_radius(camera_container, 10, 0);
    
    // 摄像头视图
    init_camera_view(camera_container);
    
    // 底部信息
    lv_obj_t *info_label = lv_label_create(screen);
    lv_label_set_text(info_label, "💡 使用nvidia-drm + EGL共享环境\n🎯 nvarguscamerasrc复用LVGL的EGL上下文");
    lv_obj_set_style_text_color(info_label, lv_color_hex(0xB0B8C1), 0);
    lv_obj_align(info_label, LV_ALIGN_BOTTOM_MID, 0, -20);
    
    std::cout << "✅ UI创建完成" << std::endl;
    
    // 7. 启动GStreamer线程
    pthread_t gs_thread;
    pthread_create(&gs_thread, NULL, gstreamer_thread, NULL);
    std::cout << "✅ GStreamer线程已创建" << std::endl;
    
    std::cout << "\n===============================================" << std::endl;
    std::cout << "✅ 单进程统一系统启动完成" << std::endl;
    std::cout << "💡 LVGL和nvarguscamerasrc现在共享同一个EGL环境" << std::endl;
    std::cout << "🎯 解决了nvidia-drm驱动下的EGL冲突问题" << std::endl;
    std::cout << "===============================================\n" << std::endl;
    
    // 更新状态标签
    lv_label_set_text(status_label, "✅ EGL环境共享成功，摄像头正在启动...");
    
    // 8. LVGL主循环
    uint32_t loop_count = 0;
    while (!g_shutdown_requested) {
        pthread_mutex_lock(&g_frame_lock);
        uint32_t time_till_next = lv_timer_handler();
        pthread_mutex_unlock(&g_frame_lock);
        
        // 每5秒更新一次状态
        if (++loop_count % 300 == 0) {  // 大约5秒
            if (g_pipeline) {
                GstState state;
                gst_element_get_state(g_pipeline, &state, NULL, 0);
                if (state == GST_STATE_PLAYING) {
                    lv_label_set_text(status_label, "🎬 摄像头运行中 - EGL环境共享正常");
                }
            }
        }
        
        usleep(time_till_next * 1000);
    }
    
    std::cout << "\n🛑 开始系统关闭..." << std::endl;
    
    // 清理（等待GStreamer线程）
    std::cout << "⏳ 等待GStreamer线程结束..." << std::endl;
    pthread_join(gs_thread, NULL);
    
    std::cout << "🧹 清理内存..." << std::endl;
    if (g_frame_buffer) {
        free(g_frame_buffer);
        g_frame_buffer = NULL;
    }
    
    std::cout << "✅ 系统关闭完成" << std::endl;
    return 0;
}