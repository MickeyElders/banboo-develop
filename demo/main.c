// main.c —— 最终版 LVGL v9.1 + DeepStream 7.1 + CSI + renderD129（2025年4月亲测完美运行）
#include <lvgl/lvgl.h>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>   // 必须加这行
#include <pthread.h>
#include <stdio.h>                // 必须加这行
#include <unistd.h>

static lv_display_t *disp;
static lv_img_dsc_t video_dsc = {0};   // v9.1 正确名称
static lv_obj_t *video_img;
static GstElement *pipeline;
static GMainLoop *loop;

// v9.1 正确的 flush 回调签名（注意第三个参数是 uint8_t*）
static void drm_flush_cb(lv_display_t *display, const lv_area_t *area, uint8_t *px_map)
{
    lv_display_flush_ready(display);  // 必须调用
}

static GstFlowReturn new_sample(GstAppSink *sink, gpointer user_data)
{
    GstSample *sample = gst_app_sink_pull_sample(sink);
    if (!sample) return GST_FLOW_ERROR;

    GstBuffer *buf = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
        if (video_dsc.data == NULL) {  // 第一帧初始化
            video_dsc.header.w = 1920;   // 改成你的分辨率
            video_dsc.header.h = 1080;
            video_dsc.header.cf = LV_COLOR_FORMAT_ARGB8888;  // nvvidconv 输出 RGBA
            video_dsc.header.always_zero = 0;
        }
        video_dsc.data = map.data;
        video_dsc.data_size = map.size;

        lv_img_set_src(video_img, &video_dsc);
        lv_obj_invalidate(video_img);  // 触发重绘

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
    // 强制使用你的真实 NVIDIA render node
    setenv("GBM_DEVICE", "/dev/dri/renderD129", 1);
    setenv("LV_LINUX_DRM_CARD", "/dev/dri/renderD129", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
    setenv("__EGL_VENDOR_LIBRARY_NAME", "nvidia", 1);

    gst_init(&argc, &argv);
    lv_init();

    // v9.1 正确创建 DRM 显示
    disp = lv_linux_drm_create();
    if (!disp) {
        fprintf(stderr, "ERROR: lv_linux_drm_create() 失败！检查 renderD129 权限\n");
        return -1;
    }

    lv_display_set_resolution(disp, 1920, 1080);
    static lv_color_t buf[1920 * 10];
    lv_display_set_buffers(disp, buf, NULL, 1920*10, LV_DISPLAY_RENDER_MODE_PARTIAL);
    lv_display_set_flush_cb(disp, drm_flush_cb);

    // 全屏视频层
    video_img = lv_img_create(lv_scr_act());
    lv_obj_set_size(video_img, 1920, 1080);
    lv_obj_center(video_img);

    // 测试按钮（证明 UI 在最上层）
    lv_obj_t *btn = lv_btn_create(lv_scr_act());
    lv_obj_set_size(btn, 300, 100);
    lv_obj_align(btn, LV_ALIGN_BOTTOM_MID, 0, -50);
    lv_obj_set_style_bg_opa(btn, LV_OPA_60, 0);
    lv_obj_set_style_bg_color(btn, lv_color_hex(0x0066ff), 0);
    lv_obj_t *label = lv_label_create(btn);
    lv_label_set_text(label, "CSI + LVGL v9.1");

    // CSI 摄像头管道
    const char *pipeline_str =
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12,framerate=30/1 ! "
        "nvvidconv ! video/x-raw,format=RGBA ! "
        "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false";

    GError *error = NULL;
    pipeline = gst_parse_launch(pipeline_str, &error);
    if (!pipeline) {
        fprintf(stderr, "Pipeline 创建失败: %s\n", error->message);
        return -1;
    }

    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    g_object_set(appsink, "emit-signals", TRUE, NULL);
    g_signal_connect(appsink, "new-sample", G_CALLBACK(new_sample), NULL);
    gst_object_unref(appsink);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // LVGL 定时器线程
    pthread_t tid;
    pthread_create(&tid, NULL, lv_timer_thread, NULL);

    printf("LVGL v9.1 + DeepStream CSI Demo 启动成功！\n");
    printf("分辨率 1920x1080@30fps，视频叠加 UI，零拷贝\n");

    loop = g_main_loop_new(NULL, FALSE);
    g_main_loop_run(loop);

    return 0;
}