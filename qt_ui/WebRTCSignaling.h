#pragma once
#include <QObject>
#include <QWebSocketServer>
#include <QWebSocket>
#include <QJsonObject>

// Minimal WebSocket signaling server for WebRTC.
// Emits parsed JSON messages; sendMessage can broadcast to all clients.
class WebRTCSignaling : public QObject {
    Q_OBJECT
public:
    explicit WebRTCSignaling(quint16 port = 9000, QObject *parent = nullptr);
    void sendMessage(const QJsonObject &obj);

signals:
    void messageReceived(const QJsonObject &obj);

private slots:
    void onNewConnection();
    void onTextMessageReceived(const QString &message);
    void onDisconnected();

private:
    QWebSocketServer m_server;
    QList<QWebSocket *> m_clients;
};
