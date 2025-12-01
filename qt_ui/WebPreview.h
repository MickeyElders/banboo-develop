#pragma once

#include <QTcpServer>
#include <QTcpSocket>
#include <QQuickWindow>
#include <QTimer>
#include <QByteArray>

// Simple HTTP server:
//   GET /      -> serves bamboo.html (or minimal fallback)
//   GET /mjpeg -> MJPEG stream from QQuickWindow::grabWindow
class WebPreview : public QObject {
    Q_OBJECT
public:
    explicit WebPreview(QQuickWindow *window, const QString &htmlPath, quint16 port = 8080, QObject *parent = nullptr);

private Q_SLOTS:
    void onNewConnection();
    void onClientReadyRead();
    void onClientDisconnected();
    void onCapture();

private:
    QQuickWindow *m_window{nullptr};
    QTcpServer m_server;
    QList<QTcpSocket *> m_clients;
    QTimer m_timer;
    QByteArray m_boundary{"--frameboundary"};
    QString m_htmlPath;

    void sendFrame(QTcpSocket *client, const QByteArray &jpeg);
    QByteArray grabJpeg();
    void sendHtml(QTcpSocket *client);
};
