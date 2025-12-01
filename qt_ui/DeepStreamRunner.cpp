#include "DeepStreamRunner.h"
#include <iostream>

DeepStreamRunner::DeepStreamRunner(QObject *parent) : QObject(parent) {
#if defined(ENABLE_GSTREAMER) && defined(ENABLE_RTSP)
    start({});
#endif
}

DeepStreamRunner::~DeepStreamRunner() {
    stop();
}

bool DeepStreamRunner::start(const QString &pipeline) {
#if !defined(ENABLE_GSTREAMER) || !defined(ENABLE_RTSP)
    Q_UNUSED(pipeline);
    Q_EMIT errorChanged(QStringLiteral("RTSP server unavailable (missing gstreamer-rtsp-server)"));
    return false;
#else
    std::lock_guard<std::mutex> lock(m_mutex);
    stop();

    if (!ensureGstInited()) return false;

    const std::string launch = pipeline.isEmpty()
        ? "( nvarguscamerasrc sensor-id=0 ! "
          "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
          "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
          "m.sink_0 nvstreammux name=m width=1280 height=720 batch-size=1 live-source=1 ! "
          "nvinfer config-file-path=config/nvinfer_config.txt batch-size=1 ! "
          "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
          "x264enc tune=zerolatency bitrate=4000 speed-preset=superfast ! "
          "rtph264pay name=pay0 pt=96 )"
        : pipeline.toStdString();

    if (!buildServer(launch)) {
        Q_EMIT errorChanged(QStringLiteral("启动 RTSP 服务失败"));
        return false;
    }

    m_thread = std::thread(&DeepStreamRunner::runLoop, this);
    return true;
#endif
}

void DeepStreamRunner::stop() {
#if defined(ENABLE_GSTREAMER) && defined(ENABLE_RTSP)
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_loop) {
        g_main_loop_quit(m_loop);
    }
    if (m_thread.joinable()) {
        m_thread.join();
    }
    if (m_factory) {
        g_object_unref(m_factory);
        m_factory = nullptr;
    }
    if (m_server) {
        g_object_unref(m_server);
        m_server = nullptr;
    }
    if (m_loop) {
        g_main_loop_unref(m_loop);
        m_loop = nullptr;
    }
#endif
}

#if defined(ENABLE_GSTREAMER) && defined(ENABLE_RTSP)
bool DeepStreamRunner::ensureGstInited() {
    std::call_once(m_gstOnce, []() {
        gst_init(nullptr, nullptr);
    });
    return true;
}

bool DeepStreamRunner::buildServer(const std::string &launch) {
    m_loop = g_main_loop_new(nullptr, FALSE);
    if (!m_loop) return false;

    m_server = gst_rtsp_server_new();
    if (!m_server) return false;

    GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(m_server);
    if (!mounts) return false;

    m_factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(m_factory, launch.c_str());
    gst_rtsp_media_factory_set_shared(m_factory, TRUE);

    gst_rtsp_mount_points_add_factory(mounts, "/deepstream", m_factory);
    g_object_unref(mounts);

    guint id = gst_rtsp_server_attach(m_server, nullptr);
    if (id == 0) {
        return false;
    }
    return true;
}

void DeepStreamRunner::runLoop() {
    if (m_loop) {
        g_main_loop_run(m_loop);
    }
}
#endif
