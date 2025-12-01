#include "DeepStreamRunner.h"
#include <iostream>
#include <cstdlib>

DeepStreamRunner::DeepStreamRunner(QObject *parent) : QObject(parent) {
#if defined(ENABLE_GSTREAMER) && defined(ENABLE_RTSP)
    const char *autoStart = std::getenv("DS_AUTOSTART");
    if (autoStart && std::strcmp(autoStart, "1") == 0) {
        // Run autostart asynchronously to avoid blocking UI init (argus/Gst may stall in headless)
        m_autostartThread = std::thread([this]() {
            start({});
        });
    } else {
        std::cout << "[deepstream] Auto-start skipped (set DS_AUTOSTART=1 to enable)" << std::endl;
    }
#endif
}

DeepStreamRunner::~DeepStreamRunner() {
    stop();
    if (m_autostartThread.joinable()) {
        m_autostartThread.join();
    }
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

    std::cout << "[deepstream] Building pipeline for sink=" << (pipeline.isEmpty() ? (std::getenv("DS_SINK") ? std::getenv("DS_SINK") : "fakesink") : "custom") << std::endl;

    // If DS_PIPELINE is provided, use it verbatim.
    const char *envPipeline = std::getenv("DS_PIPELINE");
    if (pipeline.isEmpty() && envPipeline && std::strlen(envPipeline) > 0) {
        std::cout << "[deepstream] Using DS_PIPELINE from environment" << std::endl;
        if (!buildServer(std::string(envPipeline))) {
            Q_EMIT errorChanged(QStringLiteral("启动 RTSP 服务失败 (DS_PIPELINE)"));
            return false;
        }
        m_thread = std::thread(&DeepStreamRunner::runLoop, this);
        return true;
    }

    auto sinkEnv = std::getenv("DS_SINK");
    const std::string sink = sinkEnv ? std::string(sinkEnv) : std::string("rtsp");

    std::cout << "[deepstream] start() invoked, sink=" << sink << std::endl;
    // Detect hardware encoder; fall back to x264enc if missing.
    bool hasNvEnc = false;
    if (gst_is_initialized()) {
        GstElementFactory *f = gst_element_factory_find("nvv4l2h264enc");
        if (f) {
            hasNvEnc = true;
            gst_object_unref(f);
        }
    }
    const std::string encoder = hasNvEnc
        ? "nvv4l2h264enc insert-sps-pps=true bitrate=8000000 maxperf-enable=1 iframeinterval=30 preset-level=1 control-rate=1 ! "
          "h264parse config-interval=1 ! "
        : "x264enc tune=zerolatency bitrate=4000 speed-preset=superfast ! "
          "h264parse config-interval=1 ! ";
    if (!hasNvEnc) {
        std::cout << "[deepstream] nvv4l2h264enc not found, falling back to x264enc (CPU)" << std::endl;
    }

    const std::string launch = pipeline.isEmpty()
        ? ([&sink, &encoder]() -> std::string {
              if (sink == "drm") {
                  // Direct push to DRM plane; no display server required.
                  return "nvarguscamerasrc sensor-id=0 ! "
                         "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
                         "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
                         "m.sink_0 nvstreammux name=m width=1280 height=720 batch-size=1 live-source=1 ! "
                         "nvinfer config-file-path=config/nvinfer_config.txt batch-size=1 ! "
                         "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
                         "nvdrmvideosink sync=false plane-id=0 qos=false";
              }
              if (sink == "rtsp") {
                  // Minimal RTSP pipeline with CSI camera -> H264 -> RTSP (no nvinfer to avoid blocking)
                  return "( nvarguscamerasrc sensor-id=0 ispassthrough=true bufapi-version=true do-timestamp=true ! "
                         "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12 ! "
                         "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
                         + encoder +
                         "rtph264pay name=pay0 pt=96 )";
              }
              // Safe headless default: no display, no encoder
              return "( nvarguscamerasrc sensor-id=0 ! "
                     "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
                     "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
                     "m.sink_0 nvstreammux name=m width=1280 height=720 batch-size=1 live-source=1 ! "
                     "nvinfer config-file-path=config/nvinfer_config.txt batch-size=1 ! "
                     "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
                     "fakesink sync=false )";
          })()
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
    // RTSP 使用 8554，避免与 Web 预览 8080 端口冲突
    gst_rtsp_server_set_service(m_server, "8554");

    GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(m_server);
    if (!mounts) return false;

    m_factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(m_factory, launch.c_str());
    gst_rtsp_media_factory_set_shared(m_factory, TRUE);
    std::cout << "[deepstream] RTSP launch: " << launch << std::endl;

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
