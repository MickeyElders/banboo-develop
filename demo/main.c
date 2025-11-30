// main.c —— 绝对最终版（适配 LVGL v9.1 2025年最新版）
#include <lvgl/lvgl.h>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

static lv_display_t *disp;
static lv_img_dsc_t video_dsc = {0};
static lv_obj_t *video_img;
static GstElement *pipeline;

static void drm_flush_cb(lv_display_t *display, const lv_area_t *area, uint8_t *px_map)
{
    lv_display_flush_ready(display);
}

static GstFlowReturn new_sample(GstAppSink *sink, gpointer user_data)
{
    GstSample *sample = gst_app_sink_pull_sample(sink);
    if (!sample) return GST_FLOW_ERROR;

    GstBuffer *buf = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
        // v9.1 最新版只有 w / h / cf 三个字段
        video_dsc.header.w = 1920;
        video_dsc.header.h = 1080;
        video_dsc.header.cf = LV_COLOR_FORMAT_ARGB8888;  // RGBA

        video_dsc.data = map.data;
        video_dsc.data_size = map.size;

        lv_img_set_src(video_img, &video_dsc);
        lv_obj_invalidate(video_img);

        gst_buffer_unmap(buf, &map);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

static void *lv_timer_thread(void *arg)
{
    while (1) {
        lv_timer_handler();
        usleep(5000);
    }
    return NULL;
}

int main(int argc, char *argv[])
{
    setenv("GBM_DEVICE", "/dev/dri/renderD129", 1);
    setenv("LV_LINUX_DRM_CARD", "/dev/dri/renderD129", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
    setenv("__EGL_VENDOR_LIBRARY_NAME", "nvidia", 1);

    gst_init(&argc, &argv);
    lv_init();

    disp = lv_linux_drm_create();
    if (!disp) {
        fprintf(stderr, "ERROR: lv_linux_drm_create() 失败！\n");
        return -1;
    }

    lv_display_set_resolution(disp, 1920, 1080);
    static lv_color_t buf[1920 * 20];  // 多开一点缓冲，稳
    lv_display_set_buffers(disp, buf, NULL, 1920*20, LV_DISPLAY_RENDER_MODE_PARTIAL);
    lv_display_set_flush_cb(disp, drm_flush_cb);

    video_img = lv_img_create(lv_scr_act());
    lv_obj_set_size(video_img, 1920, 1080);
    lv_obj_center(video_img);

    lv_obj_t *btn = lv_btn_create(lv_scr_act());
    lv_obj_set_size(btn, 360, 120);
    lv_obj_align(btn, LV_ALIGN_BOTTOM_MID, 0, -80);
    lv_obj_set_style_bg_opa(btn, LV_OPA_70, 0);
    lv_obj_set_style_bg_color(btn, lv_palette_main(LV_PALETTE_BLUE), 0);
    lv_obj_t *label = lv_label_create(btn);
    lv_label_set_text(label, "LVGL v9.1 + CSI 成功！");

    const char *pipeline_str =
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12,framerate=30/1 ! "
        "nvvidconv ! video/x-raw,format=RGBA ! "
        "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false";

    GError *error = NULL;
    pipeline = gst_parse_launch(pipeline_str, &error);
    if (!pipeline) {
        fprintf(stderr, "Pipeline 失败: %s\n", error->message);
        return -1;
    }

    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    g_signal_connect(appsink, "new-sample", G_CALLBACK(new_sample), NULL);
    gst_object_unref(appsink);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    pthread_t tid;
    pthread_create(&tid, NULL, lv_timer_thread, NULL);

    printf("LVGL v9.1 + DeepStream CSI Demo 启动成功！\n");

    GMainLoop *loop = g_main_loop_new(NULL, FALSE);
    g_main_loop_run(loop);

    return 0;
}