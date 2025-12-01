#pragma once

#include <QObject>
#include <QTimer>

// 简易状态模型，后续可替换为真实后端数据源。

class SystemState : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool running READ running NOTIFY stateChanged)
    Q_PROPERTY(int heartbeat READ heartbeat NOTIFY stateChanged)
    Q_PROPERTY(int currentStep READ currentStep NOTIFY stateChanged)
    Q_PROPERTY(QString currentProcess READ currentProcess NOTIFY stateChanged)
    Q_PROPERTY(double xCoordinate READ xCoordinate NOTIFY stateChanged)
    Q_PROPERTY(QString cutQuality READ cutQuality NOTIFY stateChanged)
    Q_PROPERTY(QString bladeSelection READ bladeSelection NOTIFY stateChanged)
public:
    explicit SystemState(QObject *parent = nullptr);
    bool running() const { return m_running; }
    int heartbeat() const { return m_heartbeat; }
    int currentStep() const { return m_currentStep; }
    QString currentProcess() const { return m_currentProcess; }
    double xCoordinate() const { return m_xCoordinate; }
    QString cutQuality() const { return m_cutQuality; }
    QString bladeSelection() const { return m_bladeSelection; }

public Q_SLOTS:
    void start();
    void pause();
    void stop();
    void emergencyStop();

Q_SIGNALS:
    void stateChanged();

private:
    void simulateStep();

    bool m_running{false};
    int m_heartbeat{12345};
    int m_currentStep{1};
    QString m_currentProcess{"进料检测中"};
    double m_xCoordinate{245.8};
    QString m_cutQuality{"正常"};
    QString m_bladeSelection{"双刀片"};
    QTimer m_heartbeatTimer;
    QTimer m_workflowTimer;
};

class JetsonState : public QObject {
    Q_OBJECT
    Q_PROPERTY(double cpuUsage READ cpuUsage NOTIFY changed)
    Q_PROPERTY(double gpuUsage READ gpuUsage NOTIFY changed)
    Q_PROPERTY(double memUsed READ memUsed NOTIFY changed)
    Q_PROPERTY(double memTotal READ memTotal NOTIFY changed)
    Q_PROPERTY(double temp READ temp NOTIFY changed)
    Q_PROPERTY(double fanRpm READ fanRpm NOTIFY changed)
    Q_PROPERTY(QString perfMode READ perfMode NOTIFY changed)
public:
    explicit JetsonState(QObject *parent = nullptr);
    double cpuUsage() const { return m_cpuUsage; }
    double gpuUsage() const { return m_gpuUsage; }
    double memUsed() const { return m_memUsed; }
    double memTotal() const { return m_memTotal; }
    double temp() const { return m_temp; }
    double fanRpm() const { return m_fanRpm; }
    QString perfMode() const { return m_perfMode; }

public Q_SLOTS:
    void setPerfMode(const QString &mode);

Q_SIGNALS:
    void changed();

private:
    QTimer m_timer;
    double m_cpuUsage{45.0};
    double m_gpuUsage{32.0};
    double m_memUsed{2.1};
    double m_memTotal{8.0};
    double m_temp{52.0};
    double m_fanRpm{2150.0};
    QString m_perfMode{"15W"};
};

class AiState : public QObject {
    Q_OBJECT
    Q_PROPERTY(double inferenceMs READ inferenceMs NOTIFY changed)
    Q_PROPERTY(double fps READ fps NOTIFY changed)
    Q_PROPERTY(int total READ total NOTIFY changed)
    Q_PROPERTY(int today READ today NOTIFY changed)
    Q_PROPERTY(double accuracy READ accuracy NOTIFY changed)
public:
    explicit AiState(QObject *parent = nullptr);
    double inferenceMs() const { return m_inferenceMs; }
    double fps() const { return m_fps; }
    int total() const { return m_total; }
    int today() const { return m_today; }
    double accuracy() const { return m_accuracy; }
Q_SIGNALS:
    void changed();

private:
    QTimer m_timer;
    double m_inferenceMs{15.3};
    double m_fps{28.5};
    int m_total{15432};
    int m_today{89};
    double m_accuracy{94.2};
};

class WifiState : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString ssid READ ssid NOTIFY changed)
    Q_PROPERTY(QString mode READ mode NOTIFY changed)
    Q_PROPERTY(QString status READ status NOTIFY changed)
    Q_PROPERTY(int rssi READ rssi NOTIFY changed)
public:
    explicit WifiState(QObject *parent = nullptr);
    QString ssid() const { return m_ssid; }
    QString mode() const { return m_mode; }
    QString status() const { return m_status; }
    int rssi() const { return m_rssi; }

public Q_SLOTS:
    void apply(const QString &ssid, const QString &password, const QString &mode, const QString &ip, const QString &mask, const QString &gw, const QString &dns);
    void check();

Q_SIGNALS:
    void changed();

private:
    QString m_ssid{"FactoryWiFi"};
    QString m_mode{"DHCP"};
    QString m_status{"已连接"};
    int m_rssi{-55};
};
