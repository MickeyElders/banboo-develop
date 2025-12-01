#pragma once

#include <QTcpServer>
#include <QTcpSocket>
#include <QQuickWindow>
#include <QTimer>
#include <QByteArray>

// Simple MJPEG HTTP server that captures the QQuickWindow offscreen buffer
// and streams it to connected clients (e.g. curl or browser).
class WebPreview : public QObject {
    Q_OBJECT
public:
    explicit WebPreview(QQuickWindow *window, quint16 port = 8080, QObject *parent = nullptr);

private Q_SLOTS:
    void onNewConnection();
    void onClientDisconnected();
    void onCapture();

private:
    QQuickWindow *m_window{nullptr};
    QTcpServer m_server;
    QList<QTcpSocket *> m_clients;
    QTimer m_timer;
    QByteArray m_boundary{"--frameboundary"};

    void sendFrame(QTcpSocket *client, const QByteArray &jpeg);
    QByteArray grabJpeg();
};
