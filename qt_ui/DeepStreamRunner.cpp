#include "DeepStreamRunner.h"
#include <iostream>
#include <cstdlib>
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QtWebSockets/QWebSocket>
#include <gst/gstpromise.h>
#include <gst/webrtc/webrtc.h>
#include <gst/sdp/sdp.h>
#include "WebRTCSignaling.h"

DeepStreamRunner::DeepStreamRunner(QObject *parent) : QObject(parent) {}

DeepStreamRunner::~DeepStreamRunner() {
    stop();
    if (m_autostartThread.joinable()) {
        m_autostartThread.join();
    }
    if (m_webrtcThread.joinable()) {
        m_webrtcThread.join();
    }
}

bool DeepStreamRunner::start(const QString &pipeline) {
#if !defined(ENABLE_GSTREAMER) || !defined(ENABLE_RTSP)
    Q_UNUSED(pipeline);
    Q_EMIT errorChanged(QStringLiteral("RTSP server unavailable (missing gstreamer-rtsp-server)"));
    std::cout << "[deepstream] start failed: RTSP not built in" << std::endl;
    return false;
#else
    std::lock_guard<std::mutex> lock(m_mutex);
    stop();

    qInfo() << "[deepstream] start() called, sink env=" << qgetenv("DS_SINK") << "pipeline len=" << pipeline.size();
    if (!ensureGstInited()) {
        qWarning() << "[deepstream] gst_init failed";
        return false;
    }

    std::cout << "[deepstream] Building pipeline for sink=" << (pipeline.isEmpty() ? (std::getenv("DS_SINK") ? std::getenv("DS_SINK") : "fakesink") : "custom") << std::endl;

    // If DS_PIPELINE is provided, use it verbatim.
    const char *envPipeline = std::getenv("DS_PIPELINE");
    if (pipeline.isEmpty() && envPipeline && std::strlen(envPipeline) > 0) {
        std::cout << "[deepstream] Using DS_PIPELINE from environment" << std::endl;
        if (!buildServer(std::string(envPipeline))) {
            Q_EMIT errorChanged(QStringLiteral("启动 RTSP 服务失败 (DS_PIPELINE)"));
            std::cout << "[deepstream] buildServer failed (DS_PIPELINE)" << std::endl;
            return false;
        }
        m_thread = std::thread(&DeepStreamRunner::runLoop, this);
        std::cout << "[deepstream] RTSP server thread started (DS_PIPELINE)" << std::endl;
        return true;
    }

    auto sinkEnv = std::getenv("DS_SINK");
    const std::string sink = sinkEnv ? std::string(sinkEnv) : std::string("rtsp");

    std::cout << "[deepstream] start() invoked, sink=" << sink << std::endl;
    // Detect hardware encoder; fall back to x264enc if missing.
    bool hasNvEnc = false;
    bool hasNvInfer = false;
    bool hasNvOsd = false;
    if (gst_is_initialized()) {
        GstElementFactory *f = gst_element_factory_find("nvv4l2h264enc");
        if (f) {
            hasNvEnc = true;
            gst_object_unref(f);
        }
        if (GstElementFactory *fi = gst_element_factory_find("nvinfer")) {
            hasNvInfer = true;
            gst_object_unref(fi);
        }
        if (GstElementFactory *fo = gst_element_factory_find("nvdsosd")) {
            hasNvOsd = true;
            gst_object_unref(fo);
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
    if (!hasNvInfer) {
        std::cout << "[deepstream] nvinfer not found, running without inference/OSD" << std::endl;
    }
    if (!hasNvOsd) {
        std::cout << "[deepstream] nvdsosd not found, skipping OSD overlay" << std::endl;
    }

    const std::string launch = pipeline.isEmpty()
        ? ([&sink, &encoder, hasNvInfer, hasNvOsd]() -> std::string {
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
                  // RTSP pipeline with optional nvinfer + nvdsosd overlay
                  std::string p = "( nvarguscamerasrc sensor-id=0 ispassthrough=true bufapi-version=true do-timestamp=true ! "
                                  "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12 ! "
                                  "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! ";
                  // If inference/OSD available, use streammux + nvinfer + nvdsosd
                  if (hasNvInfer || hasNvOsd) {
                      p += "m.sink_0 nvstreammux name=m width=1280 height=720 batch-size=1 live-source=1 ! ";
                      if (hasNvInfer) {
                          p += "nvinfer config-file-path=config/nvinfer_config.txt batch-size=1 ! ";
                      }
                      if (hasNvOsd) {
                          p += "nvdsosd name=osd ! ";
                      }
                  }
                  p += "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! ";
                  p += encoder;
                  p += "rtph264pay name=pay0 pt=96 )";
                  return p;
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
        std::cout << "[deepstream] buildServer failed (auto launch)" << std::endl;
        return false;
    }

    m_thread = std::thread(&DeepStreamRunner::runLoop, this);
    std::cout << "[deepstream] RTSP server thread started" << std::endl;

    // Bind signaling to handler (once) and start WebRTC pipeline
    if (m_signaling) {
        QObject::connect(m_signaling, &WebRTCSignaling::messageReceived,
                         this, [this](const QJsonObject &obj) { handleSignalingMessage(obj); },
                         Qt::UniqueConnection);
        if (!buildWebRTCPipeline()) {
            std::cout << "[deepstream] WebRTC pipeline start failed" << std::endl;
        }
    }
    return true;
#endif
}

void DeepStreamRunner::stop() {
#if defined(ENABLE_GSTREAMER) && defined(ENABLE_RTSP)
    std::lock_guard<std::mutex> lock(m_mutex);

    // Drop active RTSP clients to avoid long waits on quit/join
    if (m_server) {
        auto filter = [](GstRTSPServer *, GstRTSPClient *, gpointer) {
            return GST_RTSP_FILTER_REMOVE;
        };
        gst_rtsp_server_client_filter(m_server, filter, nullptr);
    }
    if (m_loop) g_main_loop_quit(m_loop);
    if (m_thread.joinable()) {
        m_thread.join();
    }
    if (m_webrtcLoop) g_main_loop_quit(m_webrtcLoop);
    if (m_webrtcThread.joinable()) {
        m_webrtcThread.join();
    }
    if (m_factory) {
        g_object_unref(m_factory);
        m_factory = nullptr;
    }
    if (m_server) {
        g_object_unref(m_server);
        m_server = nullptr;
    }
    if (m_webrtcPipeline) {
        gst_element_set_state(m_webrtcPipeline, GST_STATE_NULL);
        gst_object_unref(m_webrtcPipeline);
        m_webrtcPipeline = nullptr;
        m_webrtcBin = nullptr;
    }
    if (m_webrtcLoop) {
        g_main_loop_unref(m_webrtcLoop);
        m_webrtcLoop = nullptr;
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

bool DeepStreamRunner::buildWebRTCPipeline() {
    const char *srcUrlEnv = std::getenv("RTSP_SOURCE");
    const std::string srcUrl = srcUrlEnv ? std::string(srcUrlEnv) : std::string("rtsp://127.0.0.1:8554/deepstream");
    std::string desc = "rtspsrc location=" + srcUrl + " latency=200 protocols=tcp ! "
                       "rtph264depay ! h264parse ! rtph264pay pt=96 config-interval=1 ! "
                       "application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
                       "webrtcbin name=webrtcbin stun-server=stun://stun.l.google.com:19302";

    GError *err = nullptr;
    m_webrtcPipeline = gst_parse_launch(desc.c_str(), &err);
    if (err) {
        std::cout << "[webrtc] gst_parse_launch error: " << err->message << std::endl;
        g_error_free(err);
        return false;
    }
    m_webrtcBin = gst_bin_get_by_name(GST_BIN(m_webrtcPipeline), "webrtcbin");
    if (!m_webrtcBin) {
        std::cout << "[webrtc] webrtcbin not found in pipeline" << std::endl;
        return false;
    }
    g_signal_connect(m_webrtcBin, "on-negotiation-needed", G_CALLBACK(DeepStreamRunner::onNegotiationNeeded), this);
    g_signal_connect(m_webrtcBin, "on-ice-candidate", G_CALLBACK(DeepStreamRunner::onIceCandidate), this);

    gst_element_set_state(m_webrtcPipeline, GST_STATE_PLAYING);
    if (!m_webrtcLoop) m_webrtcLoop = g_main_loop_new(nullptr, FALSE);
    m_webrtcThread = std::thread([this]() {
        if (m_webrtcLoop) g_main_loop_run(m_webrtcLoop);
    });
    std::cout << "[webrtc] pipeline started, source=" << srcUrl << std::endl;
    return true;
}

void DeepStreamRunner::onNegotiationNeeded(GstElement *webrtc, gpointer user_data) {
    auto *self = static_cast<DeepStreamRunner *>(user_data);
    GstPromise *promise = gst_promise_new_with_change_func(
        [](GstPromise *p, gpointer u) {
            auto *runner = static_cast<DeepStreamRunner *>(u);
            const GstStructure *s = gst_promise_get_reply(p);
            GstWebRTCSessionDescription *offer = nullptr;
            gst_structure_get(s, "offer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, NULL);
            gst_promise_unref(p);
            if (!offer) return;
            g_signal_emit_by_name(runner->m_webrtcBin, "set-local-description", offer, nullptr);
            runner->sendSdpToPeer(offer, QStringLiteral("offer"));
            gst_webrtc_session_description_free(offer);
        },
        self, nullptr);
    g_signal_emit_by_name(webrtc, "create-offer", nullptr, promise);
}

void DeepStreamRunner::onIceCandidate(GstElement *webrtc, guint mlineindex, gchar *candidate, gchar *mid, gpointer user_data) {
    Q_UNUSED(webrtc);
    auto *self = static_cast<DeepStreamRunner *>(user_data);
    if (!self || !self->m_signaling) return;
    QJsonObject obj;
    obj["type"] = "ice";
    obj["sdpMid"] = QString::fromUtf8(mid ? mid : "");
    obj["sdpMLineIndex"] = int(mlineindex);
    obj["candidate"] = QString::fromUtf8(candidate ? candidate : "");
    self->m_signaling->sendMessage(obj);
}

void DeepStreamRunner::handleSignalingMessage(const QJsonObject &obj) {
    if (!m_webrtcBin) return;
    const QString type = obj.value("type").toString();
    if (type == "answer") {
        const QString sdp = obj.value("sdp").toString();
        GstSDPMessage *sdpMsg = nullptr;
        if (gst_sdp_message_new(&sdpMsg) != GST_SDP_OK) return;
        if (gst_sdp_message_parse_buffer(reinterpret_cast<const guint8 *>(sdp.toUtf8().constData()),
                                         sdp.toUtf8().size(), sdpMsg) != GST_SDP_OK) {
            gst_sdp_message_free(sdpMsg);
            return;
        }
        GstWebRTCSessionDescription *answer =
            gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_ANSWER, sdpMsg);
        g_signal_emit_by_name(m_webrtcBin, "set-remote-description", answer, nullptr);
        gst_webrtc_session_description_free(answer);
    } else if (type == "ice") {
        const QString cand = obj.value("candidate").toString();
        int mline = obj.value("sdpMLineIndex").toInt();
        const QString mid = obj.value("sdpMid").toString();
        g_signal_emit_by_name(m_webrtcBin, "add-ice-candidate",
                              mid.toUtf8().constData(), mline, cand.toUtf8().constData());
    }
}

void DeepStreamRunner::sendSdpToPeer(GstWebRTCSessionDescription *desc, const QString &type) {
    if (!m_signaling || !desc) return;
    gchar *sdpStr = gst_sdp_message_as_text(desc->sdp);
    if (!sdpStr) return;
    QJsonObject obj;
    obj["type"] = type;
    obj["sdp"] = QString::fromUtf8(sdpStr);
    m_signaling->sendMessage(obj);
    g_free(sdpStr);
}

void DeepStreamRunner::startPipeline() {
    const QByteArray autoStart = qgetenv("DS_AUTOSTART");
    if (!autoStart.isEmpty() && autoStart != "1") {
        std::cout << "[deepstream] startPipeline skipped (DS_AUTOSTART!=1)" << std::endl;
        return;
    }
    start({});
}

void DeepStreamRunner::stopPipeline() {
    stop();
}
#endif
