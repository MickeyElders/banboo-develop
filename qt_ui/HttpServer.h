#pragma once
#include <QObject>
#include <QTcpServer>
#include <QTcpSocket>

// Very small HTTP server serving a single HTML file (bamboo.html by default).
class HttpServer : public QObject {
    Q_OBJECT
public:
    explicit HttpServer(const QString &htmlPath, quint16 port = 8080, QObject *parent = nullptr);

private slots:
    void onNewConnection();
    void onReadyRead();
    void onDisconnected();

private:
    void sendResponse(QTcpSocket *sock, const QByteArray &body, const QByteArray &contentType);
    QString m_htmlPath;
    QTcpServer m_server;
};
