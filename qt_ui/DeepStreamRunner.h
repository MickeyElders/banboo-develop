#pragma once

#ifdef ENABLE_GSTREAMER
// gst/glib headers must come before Qt to avoid macro clashes with Qt's signals/slots
#include <gst/gst.h>
#ifdef ENABLE_RTSP
#include <gst/rtsp-server/rtsp-server.h>
#endif
#endif

#include <QObject>
#include <QString>
#include <thread>
#include <mutex>

class DeepStreamRunner : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString sourceUrl READ sourceUrl NOTIFY sourceUrlChanged)
public:
    explicit DeepStreamRunner(QObject *parent = nullptr);
    ~DeepStreamRunner();

    QString sourceUrl() const { return m_sourceUrl; }

    // 启动 RTSP 推流（摄像头 -> H264 -> rtsp://127.0.0.1:8554/deepstream）
    Q_INVOKABLE bool start(const QString &pipeline = QString());
    Q_INVOKABLE void stop();

Q_SIGNALS:
    void sourceUrlChanged();
    void errorChanged(const QString &message);

private:
#ifdef ENABLE_GSTREAMER
    bool ensureGstInited();
#ifdef ENABLE_RTSP
    bool buildServer(const std::string &launch);
    void runLoop();

    GstRTSPServer *m_server{nullptr};
    GstRTSPMediaFactory *m_factory{nullptr};
    GMainLoop *m_loop{nullptr};
    std::thread m_thread;
    std::thread m_autostartThread;
    std::once_flag m_gstOnce;
#endif
#endif
    QString m_sourceUrl{"rtsp://127.0.0.1:8554/deepstream"};
    std::mutex m_mutex;
};
