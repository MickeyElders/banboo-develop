#include "HttpServer.h"

#include <QDebug>
#include <QFile>

HttpServer::HttpServer(const QString &htmlPath, quint16 port, QObject *parent)
    : QObject(parent), m_htmlPath(htmlPath) {
    connect(&m_server, &QTcpServer::newConnection, this, &HttpServer::onNewConnection);
    if (!m_server.listen(QHostAddress::AnyIPv4, port)) {
        qWarning() << "[http] listen failed on port" << port << ":" << m_server.errorString();
    } else {
        qInfo() << "[http] serving" << htmlPath << "on port" << port;
    }
}

void HttpServer::onNewConnection() {
    while (m_server.hasPendingConnections()) {
        QTcpSocket *sock = m_server.nextPendingConnection();
        if (!sock) continue;
        qInfo() << "[http] connection from" << sock->peerAddress().toString() << ":" << sock->peerPort();
        connect(sock, &QTcpSocket::readyRead, this, &HttpServer::onReadyRead);
        connect(sock, &QTcpSocket::disconnected, this, &HttpServer::onDisconnected);
        // In fast clients (curl), data may already be buffered before readyRead is connected.
        if (sock->bytesAvailable() > 0) {
            onReadyRead();
        }
    }
}

void HttpServer::onReadyRead() {
    QTcpSocket *sock = qobject_cast<QTcpSocket *>(sender());
    if (!sock) return;
    const QByteArray req = sock->readAll();
    qInfo() << "[http] request bytes" << req.size() << "from" << sock->peerAddress().toString();
    if (!req.startsWith("GET")) {
        sock->disconnectFromHost();
        return;
    }
    QByteArray body;
    QByteArray ctype = "text/html; charset=utf-8";
    QFile f(m_htmlPath);
    if (f.exists() && f.open(QIODevice::ReadOnly)) {
        body = f.readAll();
    } else {
        body = "<html><body><h3>bamboo.html not found</h3></body></html>";
    }
    sendResponse(sock, body, ctype);
}

void HttpServer::onDisconnected() {
    QTcpSocket *sock = qobject_cast<QTcpSocket *>(sender());
    if (sock) sock->deleteLater();
}

void HttpServer::sendResponse(QTcpSocket *sock, const QByteArray &body, const QByteArray &contentType) {
    if (!sock) return;
    QByteArray resp;
    resp += "HTTP/1.1 200 OK\r\n";
    resp += "Content-Type: " + contentType + "\r\n";
    resp += "Content-Length: " + QByteArray::number(body.size()) + "\r\n";
    resp += "Connection: close\r\n\r\n";
    resp += body;
    sock->write(resp);
    sock->flush();
    sock->waitForBytesWritten(100);
    sock->disconnectFromHost();
}
