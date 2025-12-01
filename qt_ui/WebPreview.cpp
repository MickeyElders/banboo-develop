#include "WebPreview.h"

#include <QBuffer>
#include <QCoreApplication>
#include <QDebug>
#include <QFile>
#include <QImage>

WebPreview::WebPreview(QQuickWindow *window, const QString &htmlPath, quint16 port, QObject *parent)
    : QObject(parent), m_window(window), m_htmlPath(htmlPath) {
    if (!m_window) {
        qWarning() << "[webpreview] No window provided, MJPEG preview disabled.";
        return;
    }

    connect(&m_server, &QTcpServer::newConnection, this, &WebPreview::onNewConnection);
    if (!m_server.listen(QHostAddress::Any, port)) {
        qWarning() << "[webpreview] Failed to listen on port" << port << ":" << m_server.errorString();
    } else {
        qInfo() << "[webpreview] Web preview listening on port" << port << "(/ for HTML, /mjpeg for stream)";
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
        connect(client, &QTcpSocket::readyRead, this, &WebPreview::onClientReadyRead);
    }
}

void WebPreview::onClientReadyRead() {
    QTcpSocket *client = qobject_cast<QTcpSocket *>(sender());
    if (!client) return;
    const QByteArray req = client->readAll();
    const QList<QByteArray> lines = req.split('\n');
    if (lines.isEmpty()) return;
    const QList<QByteArray> parts = lines.first().split(' ');
    if (parts.size() < 2) return;
    const QByteArray path = parts.at(1);
    if (path == "/mjpeg") {
        // Switch to MJPEG streaming
        m_clients << client;
        QByteArray header;
        header += "HTTP/1.0 200 OK\r\n";
        header += "Content-Type: multipart/x-mixed-replace;boundary=" + m_boundary + "\r\n";
        header += "Cache-Control: no-cache\r\n";
        header += "Connection: close\r\n\r\n";
        client->write(header);
        client->flush();
        qInfo() << "[webpreview] MJPEG client connected, total:" << m_clients.size();
    } else {
        sendHtml(client);
        client->disconnectFromHost();
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

void WebPreview::sendHtml(QTcpSocket *client) {
    if (!client) return;
    QByteArray body;
    QFile f(m_htmlPath);
    if (f.exists() && f.open(QIODevice::ReadOnly)) {
        body = f.readAll();
    } else {
        body = "<html><body><h3>Bamboo Preview</h3><img src=\"/mjpeg\" /></body></html>";
    }
    QByteArray header;
    header += "HTTP/1.0 200 OK\r\n";
    header += "Content-Type: text/html; charset=utf-8\r\n";
    header += "Content-Length: " + QByteArray::number(body.size()) + "\r\n";
    header += "Connection: close\r\n\r\n";
    client->write(header);
    client->write(body);
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
