#ifndef SYSTEMCONTROLLER_H
#define SYSTEMCONTROLLER_H

#include <QtCore/QObject>
#include <QtCore/QTimer>
#include <QtCore/QProcess>
#include <QtSerialPort/QSerialPort>
#include <QtSerialPort/QSerialPortInfo>
#include <QtNetwork/QTcpSocket>
#include <QtCore/QJsonObject>

class SystemController : public QObject
{
    Q_OBJECT

public:
    enum SystemStatus {
        Offline,
        Initializing,
        Standby,
        Operating,
        Error,
        Maintenance
    };

    enum ControlMode {
        Manual,
        Automatic,
        SemiAutomatic
    };

    enum CutterStatus {
        CutterIdle,
        CutterActive,
        CutterError,
        CutterMaintenance
    };

    struct SystemInfo {
        QString version;
        QString buildDate;
        QString deviceId;
        float cpuUsage;
        float memoryUsage;
        float diskUsage;
        float temperature;
        int uptime;
    };

    struct CutterConfig {
        int cutSpeed;           // 切割速度
        int cutDepth;           // 切割深度
        int bladePosition;      // 刀片位置
        bool autoRetract;       // 自动收回
        int retractDelay;       // 收回延时
        float forceThreshold;   // 力阈值
    };

    explicit SystemController(QObject *parent = nullptr);
    ~SystemController();

    bool initialize();
    void shutdown();
    
    SystemStatus status() const { return m_systemStatus; }
    ControlMode controlMode() const { return m_controlMode; }
    CutterStatus cutterStatus() const { return m_cutterStatus; }
    SystemInfo systemInfo() const { return m_systemInfo; }

    void setControlMode(ControlMode mode);
    void setCutterConfig(const CutterConfig& config);
    
    // 控制命令
    void startOperation();
    void stopOperation();
    void emergencyStop();
    void resetSystem();
    void calibrateCutter();
    
    // 手动控制
    void moveCutterUp();
    void moveCutterDown();
    void activateCutter();
    void retractCutter();

signals:
    void systemStatusChanged(SystemStatus status);
    void controlModeChanged(ControlMode mode);
    void cutterStatusChanged(CutterStatus status);
    void systemInfoUpdated(const SystemInfo& info);
    void errorOccurred(const QString& error);
    void operationCompleted();
    void calibrationCompleted(bool success);

private slots:
    void onSerialDataReceived();
    void onSerialError(QSerialPort::SerialPortError error);
    void onNetworkDataReceived();
    void onNetworkError();
    void onSystemMonitorTimer();
    void onHeartbeatTimer();
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
    void setupSerialCommunication();
    void setupNetworkCommunication();
    void updateSystemInfo();
    void sendCommand(const QString& command, const QJsonObject& parameters = QJsonObject());
    void processResponse(const QByteArray& data);
    bool validateCommand(const QString& command) const;
    void handleError(const QString& error);

    // 系统状态
    SystemStatus m_systemStatus;
    ControlMode m_controlMode;
    CutterStatus m_cutterStatus;
    SystemInfo m_systemInfo;
    CutterConfig m_cutterConfig;

    // 通信
    QSerialPort *m_serialPort;
    QTcpSocket *m_tcpSocket;
    QString m_serialPortName;
    QString m_networkAddress;
    int m_networkPort;

    // 定时器
    QTimer *m_systemMonitorTimer;
    QTimer *m_heartbeatTimer;

    // 进程管理
    QProcess *m_systemProcess;
    
    // 安全控制
    bool m_emergencyStopActive;
    qint64 m_lastHeartbeat;
    int m_communicationTimeout;

    // 日志和调试
    void logMessage(const QString& message);
    bool m_debugEnabled;
};

Q_DECLARE_METATYPE(SystemController::SystemStatus)
Q_DECLARE_METATYPE(SystemController::ControlMode)
Q_DECLARE_METATYPE(SystemController::CutterStatus)
Q_DECLARE_METATYPE(SystemController::SystemInfo)

#endif // SYSTEMCONTROLLER_H