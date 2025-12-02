#include "HttpServer.h"

#include <QDebug>
#include <QFile>
#include <QCoreApplication>
#include <QUrl>

HttpServer::HttpServer(const QString &htmlPath, quint16 port, QObject *parent)
    : QObject(parent), m_htmlPath(htmlPath) {
    m_docRoot = QCoreApplication::applicationDirPath() + "/../www";
    connect(&m_server, &QTcpServer::newConnection, this, &HttpServer::onNewConnection);
    if (!m_server.listen(QHostAddress::AnyIPv4, port)) {
        qWarning() << "[http] listen failed on port" << port << ":" << m_server.errorString();
    } else {
        qInfo() << "[http] serving" << htmlPath << "and static files under" << m_docRoot << "on port" << port;
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
    if (req.isEmpty()) { sock->disconnectFromHost(); return; }

    const QList<QByteArray> lines = req.split('\n');
    QByteArray method = "GET";
    QByteArray path = "/";
    if (!lines.isEmpty()) {
        const QList<QByteArray> parts = lines.first().split(' ');
        if (parts.size() >= 1) method = parts.at(0).trimmed();
        if (parts.size() >= 2) path = parts.at(1).trimmed();
        // Handle absolute-form requests (e.g., curl with full URL)
        if (path.startsWith("http://") || path.startsWith("https://")) {
            QUrl u(QString::fromUtf8(path));
            if (u.isValid()) path = u.path(QUrl::FullyDecoded).toUtf8();
        }
    }
    const bool headOnly = (method == "HEAD");
    if (method != "GET" && method != "HEAD") {
        sendResponse(sock, QByteArray(), "text/plain; charset=utf-8", headOnly);
        return;
    }
    QByteArray body;
    QByteArray ctype = "text/html; charset=utf-8";
    if (path == "/" || path == "/index.html" || path.isEmpty()) {
        QFile f(m_htmlPath);
        if (f.exists() && f.open(QIODevice::ReadOnly)) {
            body = f.readAll();
        } else {
            body = "<html><body><h3>bamboo.html not found</h3></body></html>";
        }
    } else if (path.startsWith("/hls/")) {
        body = serveFile(QString::fromUtf8(path.mid(5)), ctype);  // strip "/hls/"
        if (body.isEmpty()) {
            sendResponse(sock, QByteArray(), "text/plain; charset=utf-8", headOnly);
            return;
        }
    } else {
        // fallback to index
        QFile f(m_htmlPath);
        if (f.exists() && f.open(QIODevice::ReadOnly)) {
            body = f.readAll();
        } else {
            body = "<html><body><h3>bamboo.html not found</h3></body></html>";
        }
    }
    sendResponse(sock, body, ctype, headOnly);
}

void HttpServer::onDisconnected() {
    QTcpSocket *sock = qobject_cast<QTcpSocket *>(sender());
    if (sock) sock->deleteLater();
}

void HttpServer::sendResponse(QTcpSocket *sock, const QByteArray &body, const QByteArray &contentType, bool headOnly) {
    if (!sock) return;
    QByteArray resp;
    resp += "HTTP/1.1 200 OK\r\n";
    resp += "Content-Type: " + contentType + "\r\n";
    resp += "Content-Length: " + QByteArray::number(body.size()) + "\r\n";
    resp += "Access-Control-Allow-Origin: *\r\n";
    resp += "Connection: close\r\n\r\n";
    if (headOnly) {
        sock->write(resp);
    } else {
        resp += body;
        sock->write(resp);
    }
    sock->flush();
    sock->waitForBytesWritten(100);
    sock->disconnectFromHost();
}

QByteArray HttpServer::serveFile(const QString &path, QByteArray &contentType) {
    // Serve static files from doc root (e.g., HLS playlist/segments)
    QFile f(m_docRoot + "/" + path);
    if (!f.exists() || !f.open(QIODevice::ReadOnly)) {
        return {};
    }
    if (path.endsWith(".m3u8")) {
        contentType = "application/vnd.apple.mpegurl";
    } else if (path.endsWith(".ts")) {
        contentType = "video/MP2T";
    } else if (path.endsWith(".mp4")) {
        contentType = "video/mp4";
    } else if (path.endsWith(".js")) {
        contentType = "application/javascript";
    } else {
        contentType = "application/octet-stream";
    }
    return f.readAll();
}
