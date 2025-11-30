// main_v9.c —— LVGL v9 + DeepStream 7.1 + CSI + renderD129
#include <lvgl/lvgl.h>
#include <lvgl/drivers/display/linux_drm.h>  // v9 内置头文件
#include <gst/gst.h>
#include <pthread.h>
#include <unistd.h>

static lv_display_t *disp;
static lv_image_descriptor_t video_dsc = {0};  // v9 用 lv_image_descriptor_t
static lv_obj_t *video_img;

static GstElement *pipeline;
static GMainLoop *loop;

// v9 flush 回调（直接写到 DRM/GBM 缓冲区）
static void drm_flush(lv_layer_t *layer) {
    lv_draw_ctx_t *draw_ctx = lv_display_get_draw_ctx(disp);
    // GBM/EGL 自动处理（v9 内置）
    lv_draw_sw_flush(layer, draw_ctx);
    lv_display_flush_ready(disp);  // v9 API
}

// appsink 回调：DeepStream 帧 → LVGL 图像（零拷贝）
static GstFlowReturn new_sample(GstAppSink *sink, gpointer user_data) {
    GstSample *sample = gst_app_sink_pull_sample(sink);
    GstBuffer *buf = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
        video_dsc.header.w = 1920;  // 你的 CSI 分辨率
        video_dsc.header.h = 1080;
        video_dsc.header.format = LV_COLOR_FORMAT_NATIVE;  // RGBA/NV12 适配
        video_dsc.data_size = map.size;
        video_dsc.data = map.data;
        lv_image_set_src(video_img, &video_dsc);  // v9 API
        lv_obj_mark_layout_as_dirty(video_img);
        gst_buffer_unmap(buf, &map);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

static void *lv_timer_thread(void *arg) {
    while (1) {
        lv_timer_handler();
        usleep(5000);
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    // 强制 renderD129
    setenv("GBM_DEVICE", "/dev/dri/renderD129", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
    setenv("__EGL_VENDOR_LIBRARY_NAME", "nvidia", 1);

    gst_init(&argc, &argv);
    lv_init();

    // v9 DRM 创建（自动 GBM/EGL）
    disp = lv_linux_drm_create();  // v9 内置函数
    lv_display_set_resolution(disp, 1920, 1080);  // 你的分辨率
    static lv_color_t buf[1920 * 10];  // 部分缓冲区（v9 推荐）
    lv_display_set_buffers(disp, buf, NULL, sizeof(buf) / sizeof(lv_color_t), LV_DISPLAY_RENDER_MODE_PARTIAL);
    lv_display_set_flush_cb(disp, drm_flush);

    // 全屏视频
    video_img = lv_image_create(lv_screen_active());  // v9 用 lv_screen_active()
    lv_obj_set_size(video_img, 1920, 1080);
    lv_obj_center(video_img);

    // 半透明按钮
    lv_obj_t *btn = lv_button_create(lv_screen_active());
    lv_obj_set_size(btn, 300, 100);
    lv_obj_align(btn, LV_ALIGN_BOTTOM_MID, 0, -50);
    lv_obj_set_style_bg_opa(btn, LV_OPA_60, 0);
    lv_obj_set_style_bg_color(btn, lv_color_hex(0x0066ff), 0);
    lv_obj_t *label = lv_label_create(btn);
    lv_label_set_text(label, "CSI + LVGL v9");

    // CSI 管道
    const char *pipeline_str = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12,framerate=30/1 ! nvvidconv ! video/x-raw,format=RGBA ! appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false";
    GError *error = NULL;
    pipeline = gst_parse_launch(pipeline_str, &error);
    if (!pipeline) { g_printerr("Pipeline 失败: %s\n", error->message); return -1; }
    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    g_signal_connect(appsink, "new-sample", G_CALLBACK(new_sample), NULL);
    gst_object_unref(appsink);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // LVGL 线程
    pthread_t tid;
    pthread_create(&tid, NULL, lv_timer_thread, NULL);

    loop = g_main_loop_new(NULL, FALSE);
    printf("LVGL v9 + DeepStream CSI Demo 启动！\n");
    g_main_loop_run(loop);
    return 0;
}