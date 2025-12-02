#pragma once

#ifdef ENABLE_GSTREAMER
#ifndef GST_USE_UNSTABLE_API
#define GST_USE_UNSTABLE_API
#endif
// gst/glib headers must come before Qt to avoid macro clashes with Qt's signals/slots
#include <gst/gst.h>
#ifdef ENABLE_RTSP
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/webrtc/webrtc.h>
#include <gst/sdp/sdp.h>
#endif
#endif

#include <QObject>
#include <QString>
#include <QJsonObject>
#include <thread>
#include <mutex>

class DeepStreamRunner : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString sourceUrl READ sourceUrl NOTIFY sourceUrlChanged)
public:
    explicit DeepStreamRunner(QObject *parent = nullptr);
    ~DeepStreamRunner();

    void setWebRTCSignaling(class WebRTCSignaling *sig);
    QString sourceUrl() const { return m_sourceUrl; }

    Q_INVOKABLE bool start(const QString &pipeline = QString());
    Q_INVOKABLE void stop();

public Q_SLOTS:
    void startPipeline();
    void stopPipeline();

Q_SIGNALS:
    void sourceUrlChanged();
    void errorChanged(const QString &message);

private:
#ifdef ENABLE_GSTREAMER
    bool ensureGstInited();
#ifdef ENABLE_RTSP
    bool buildServer(const std::string &launch);
    bool buildWebRTCPipeline();
    void runLoop();
    bool ensureWebRTCPipeline();

    // WebRTC callbacks
    static void onNegotiationNeeded(GstElement *webrtc, gpointer user_data);
    static void onIceCandidate(GstElement *webrtc, guint mlineindex, gchar *candidate, gchar *mid, gpointer user_data);
    void handleSignalingMessage(const QJsonObject &obj);
    void sendSdpToPeer(GstWebRTCSessionDescription *desc, const QString &type);
    void createAndSendOffer();
    void renegotiateWebRTC();

    GstRTSPServer *m_server{nullptr};
    GstRTSPMediaFactory *m_factory{nullptr};
    GstElement *m_webrtcPipeline{nullptr};
    GstElement *m_webrtcBin{nullptr};
    GMainLoop *m_loop{nullptr};
    GMainLoop *m_webrtcLoop{nullptr};
    std::thread m_thread;
    std::thread m_webrtcThread;
    std::thread m_startThread;
    std::once_flag m_gstOnce;
    std::atomic_bool m_offerInFlight{false};
#endif
#endif
    QString m_sourceUrl{"rtsp://127.0.0.1:8554/deepstream"};
    std::mutex m_mutex;
    class WebRTCSignaling *m_signaling{nullptr};
};
