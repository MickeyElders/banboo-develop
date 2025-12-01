#pragma once

#include <QObject>

// 占位类：后续可替换为真实 DeepStream 控制器（管线启动/停止、元数据传递）。
class DeepStreamStub : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString sourceUrl READ sourceUrl WRITE setSourceUrl NOTIFY changed)
public:
    explicit DeepStreamStub(QObject *parent = nullptr) : QObject(parent) {}
    QString sourceUrl() const { return m_sourceUrl; }
    void setSourceUrl(const QString &url) {
        if (m_sourceUrl == url) return;
        m_sourceUrl = url;
        Q_EMIT changed();
    }

Q_SIGNALS:
    void changed();

private:
    QString m_sourceUrl{"rtsp://127.0.0.1:8554/deepstream"};
};
