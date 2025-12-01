#include "WebPreview.h"

#include <QBuffer>
#include <QDebug>
#include <QImage>

WebPreview::WebPreview(QQuickWindow *window, quint16 port, QObject *parent)
    : QObject(parent), m_window(window) {
    if (!m_window) {
        qWarning() << "[webpreview] No window provided, MJPEG preview disabled.";
        return;
    }

    connect(&m_server, &QTcpServer::newConnection, this, &WebPreview::onNewConnection);
    if (!m_server.listen(QHostAddress::Any, port)) {
        qWarning() << "[webpreview] Failed to listen on port" << port << ":" << m_server.errorString();
    } else {
        qInfo() << "[webpreview] MJPEG preview listening on port" << port;
    }

    m_timer.setInterval(200);  // ~5 FPS preview
    connect(&m_timer, &QTimer::timeout, this, &WebPreview::onCapture);
    m_timer.start();
}

void WebPreview::onNewConnection() {
    while (m_server.hasPendingConnections()) {
        QTcpSocket *client = m_server.nextPendingConnection();
        if (!client) continue;
        connect(client, &QTcpSocket::disconnected, this, &WebPreview::onClientDisconnected);
        m_clients << client;
        QByteArray header;
        header += "HTTP/1.0 200 OK\r\n";
        header += "Content-Type: multipart/x-mixed-replace;boundary=" + m_boundary + "\r\n";
        header += "Cache-Control: no-cache\r\n";
        header += "Connection: close\r\n\r\n";
        client->write(header);
        client->flush();
        qInfo() << "[webpreview] Client connected, total:" << m_clients.size();
    }
}

void WebPreview::onClientDisconnected() {
    QTcpSocket *client = qobject_cast<QTcpSocket *>(sender());
    if (client) {
        m_clients.removeAll(client);
        client->deleteLater();
        qInfo() << "[webpreview] Client disconnected, total:" << m_clients.size();
    }
}

QByteArray WebPreview::grabJpeg() {
    if (!m_window) return {};
    const QImage img = m_window->grabWindow();
    if (img.isNull()) return {};
    QByteArray buf;
    QBuffer buffer(&buf);
    buffer.open(QIODevice::WriteOnly);
    img.save(&buffer, "JPG", 75);
    return buf;
}

void WebPreview::sendFrame(QTcpSocket *client, const QByteArray &jpeg) {
    if (!client || jpeg.isEmpty()) return;
    QByteArray part;
    part += m_boundary + "\r\n";
    part += "Content-Type: image/jpeg\r\n";
    part += "Content-Length: " + QByteArray::number(jpeg.size()) + "\r\n\r\n";
    part += jpeg;
    part += "\r\n";
    client->write(part);
    client->flush();
}

void WebPreview::onCapture() {
    if (m_clients.isEmpty()) return;
    const QByteArray jpeg = grabJpeg();
    if (jpeg.isEmpty()) return;
    for (QTcpSocket *c : std::as_const(m_clients)) {
        if (c->state() == QAbstractSocket::ConnectedState) {
            sendFrame(c, jpeg);
        }
    }
}
