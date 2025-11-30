// main.c  —— 直接保存为 main.c
#include <gst/gst.h>
#include <lvgl/lvgl.h>
#include <lv_drivers/display/drm.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>

static lv_disp_t *disp;
static lv_obj_t *video_img;
static lv_img_dsc_t video_dsc = {0};

static GstElement *pipeline;
static GMainLoop *loop;

// 每来一帧就直接扔给 LVGL（零拷贝）
static GstFlowReturn new_sample(GstAppSink *sink, gpointer user_data)
{
    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *s = gst_caps_get_structure(caps, 0);

    // 获取宽高（只在第一帧执行一次）
    if (!video_dsc.data) {
        gst_structure_get_int(s, "width", (gint*)&video_dsc.header.w);
        gst_structure_get_int(s, "height", (gint*)&video_dsc.header.h);
        video_dsc.header.cf = LV_IMG_CF_TRUE_COLOR;  // nvargus 给的是 RGBA
        video_dsc.header.always_zero = 0;
    }

    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        video_dsc.data = (void*)map.data;
        video_dsc.data_size = map.size;
        lv_img_set_src(video_img, &video_dsc);
        lv_obj_invalidate(video_img);
        gst_buffer_unmap(buffer, &map);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

static void *lvgl_timer_thread(void *arg)
{
    while(1) {
        lv_timer_handler();
        usleep(5000);  // 5ms
    }
    return NULL;
}

int main(int argc, char *argv[])
{
    // 强制使用你机器上真正的 NVIDIA render node
    setenv("GBM_DEVICE", "/dev/dri/renderD129", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
    setenv("__EGL_VENDOR_LIBRARY_NAME", "nvidia", 1);

    gst_init(&argc, &argv);
    lv_init();

    // LVGL 使用 DRM (自动走 renderD129)
    lv_drm_init();
    disp = lv_drm_create();
    lv_disp_set_default(disp);

    // 全屏视频背景
    video_img = lv_img_create(lv_scr_act());
    lv_obj_set_size(video_img, lv_disp_get_hor_res(NULL), lv_disp_get_ver_res(NULL));
    lv_obj_center(video_img);

    // 加一个半透明按钮证明 UI 在最上层
    lv_obj_t *btn = lv_btn_create(lv_scr_act());
    lv_obj_set_size(btn, 300, 100);
    lv_obj_align(btn, LV_ALIGN_BOTTOM_MID, 0, -50);
    lv_obj_set_style_bg_opa(btn, LV_OPA_60, 0);
    lv_obj_set_style_bg_color(btn, lv_color_hex(0x0066ff), 0);
    lv_obj_t *label = lv_label_create(btn);
    lv_label_set_text(label, "CSI Camera + LVGL");

    // === DeepStream / GStreamer 管道：原生 CSI 摄像头 ===
    const char *pipeline_str = 
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12,framerate=30/1 ! "
        "nvvidconv ! video/x-raw,format=RGBA ! "
        "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false";

    GError *error = NULL;
    pipeline = gst_parse_launch(pipeline_str, &error);
    if (!pipeline || error) {
        g_printerr("Pipeline 创建失败: %s\n", error ? error->message : "unknown");
        return -1;
    }

    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    g_object_set(appsink, "emit-signals", TRUE, NULL);
    g_signal_connect(appsink, "new-sample", G_CALLBACK(new_sample), NULL);
    gst_object_unref(appsink);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // 启动 LVGL 刷新线程
    pthread_t tid;
    pthread_create(&tid, NULL, lvgl_timer_thread, NULL);

    printf("Jetson 原生 CSI 摄像头 + LVGL 完美融合 Demo 启动成功！\n");
    printf("分辨率自动适配，按 Ctrl+C 退出\n");

    loop = g_main_loop_new(NULL, FALSE);
    g_main_loop_run(loop);

    return 0;
}