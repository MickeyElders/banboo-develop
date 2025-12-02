#include "WebRTCSignaling.h"
#include <QDebug>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QJsonObject>

WebRTCSignaling::WebRTCSignaling(quint16 port, QObject *parent)
    : QObject(parent),
      m_server(QStringLiteral("Bamboo WebRTC Signaling"), QWebSocketServer::NonSecureMode, this) {
    if (!m_server.listen(QHostAddress::Any, port)) {
        qWarning() << "[webrtc] signaling listen failed on port" << port << ":" << m_server.errorString();
        return;
    }
    qInfo() << "[webrtc] signaling server listening on port" << port;
    connect(&m_server, &QWebSocketServer::newConnection, this, &WebRTCSignaling::onNewConnection);
}

void WebRTCSignaling::onNewConnection() {
    while (m_server.hasPendingConnections()) {
        QWebSocket *socket = m_server.nextPendingConnection();
        if (!socket) continue;
        connect(socket, &QWebSocket::textMessageReceived, this, &WebRTCSignaling::onTextMessageReceived);
        connect(socket, &QWebSocket::disconnected, this, &WebRTCSignaling::onDisconnected);
        m_clients << socket;
        qInfo() << "[webrtc] signaling client connected, total:" << m_clients.size();
        Q_EMIT clientConnected();
    }
}

void WebRTCSignaling::onTextMessageReceived(const QString &message) {
    qInfo() << "[webrtc] signaling message:" << message;
    QJsonParseError err;
    const QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8(), &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject()) {
        qWarning() << "[webrtc] invalid JSON message";
        return;
    }
    Q_EMIT messageReceived(doc.object());
}

void WebRTCSignaling::onDisconnected() {
    QWebSocket *socket = qobject_cast<QWebSocket *>(sender());
    if (socket) {
        m_clients.removeAll(socket);
        socket->deleteLater();
        qInfo() << "[webrtc] signaling client disconnected, total:" << m_clients.size();
    }
}

void WebRTCSignaling::sendMessage(const QJsonObject &obj) {
    const QByteArray payload = QJsonDocument(obj).toJson(QJsonDocument::Compact);
    // Ensure send happens on this object's thread to avoid cross-thread socket notifiers.
    QMetaObject::invokeMethod(this, [this, payload]() {
        for (QWebSocket *c : std::as_const(m_clients)) {
            if (c && c->isValid()) {
                c->sendTextMessage(QString::fromUtf8(payload));
            }
        }
    }, Qt::QueuedConnection);
}
