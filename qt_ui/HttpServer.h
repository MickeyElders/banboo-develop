#pragma once
#include <QObject>
#include <QTcpServer>
#include <QTcpSocket>
#include <QHostAddress>

// Very small HTTP server serving a single HTML file (bamboo.html by default).
class HttpServer : public QObject {
    Q_OBJECT
public:
    explicit HttpServer(const QString &htmlPath, quint16 port = 8080, QObject *parent = nullptr);

private Q_SLOTS:
    void onNewConnection();
    void onReadyRead();
    void onDisconnected();

private:
    void sendResponse(QTcpSocket *sock, const QByteArray &body, const QByteArray &contentType);
    QByteArray serveFile(const QString &path, QByteArray &contentType);
    QString m_htmlPath;
    QString m_docRoot;
    QTcpServer m_server;
};
