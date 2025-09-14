#include "systemcontroller.h"
#include <QtCore/QTimer>
#include <QtCore/QDateTime>
#include <QtCore/QLoggingCategory>
#include <QtCore/QStandardPaths>
#include <QtCore/QDir>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
#include <QtSerialPort/QSerialPortInfo>
#include <QtNetwork/QHostAddress>
#include <QSysInfo>
#include <random>

Q_LOGGING_CATEGORY(systemController, "app.systemcontroller")

SystemController::SystemController(QObject *parent)
    : QObject(parent)
    , m_systemStatus(Offline)
    , m_controlMode(Manual)
    , m_cutterStatus(CutterIdle)
    , m_serialPort(nullptr)
    , m_tcpSocket(nullptr)
    , m_systemMonitorTimer(nullptr)
    , m_heartbeatTimer(nullptr)
    , m_systemProcess(nullptr)
    , m_emergencyStopActive(false)
    , m_lastHeartbeat(0)
    , m_communicationTimeout(5000)
    , m_debugEnabled(true)
{
    qCInfo(systemController) << "Creating SystemController...";
    
    // 初始化系统信息
    m_systemInfo.version = "2.0.0";
    m_systemInfo.buildDate = __DATE__;
    m_systemInfo.deviceId = QSysInfo::machineHostName();
    m_systemInfo.cpuUsage = 0.0f;
    m_systemInfo.memoryUsage = 0.0f;
    m_systemInfo.diskUsage = 0.0f;
    m_systemInfo.temperature = 0.0f;
    m_systemInfo.uptime = 0;
    
    // 初始化切割器配置
    m_cutterConfig.cutSpeed = 50;
    m_cutterConfig.cutDepth = 10;
    m_cutterConfig.bladePosition = 0;
    m_cutterConfig.autoRetract = true;
    m_cutterConfig.retractDelay = 1000;
    m_cutterConfig.forceThreshold = 100.0f;
    
    // 网络配置
    m_serialPortName = "/dev/ttyUSB0";
    m_networkAddress = "192.168.1.100";
    m_networkPort = 8080;
    
    // 创建定时器
    m_systemMonitorTimer = new QTimer(this);
    connect(m_systemMonitorTimer, &QTimer::timeout, this, &SystemController::onSystemMonitorTimer);
    
    m_heartbeatTimer = new QTimer(this);
    connect(m_heartbeatTimer, &QTimer::timeout, this, &SystemController::onHeartbeatTimer);
}

SystemController::~SystemController()
{
    qCInfo(systemController) << "Destroying SystemController...";
    shutdown();
}

bool SystemController::initialize()
{
    qCInfo(systemController) << "Initializing SystemController...";
    
    if (m_systemStatus != Offline) {
        qCWarning(systemController) << "SystemController already initialized";
        return false;
    }
    
    m_systemStatus = Initializing;
    emit systemStatusChanged(m_systemStatus);
    
    // 设置通信
    setupSerialCommunication();
    setupNetworkCommunication();
    
    // 启动系统监控
    m_systemMonitorTimer->start(1000); // 每秒更新一次
    m_heartbeatTimer->start(2000); // 每2秒发送心跳
    
    // 更新系统信息
    updateSystemInfo();
    
    m_systemStatus = Standby;
    emit systemStatusChanged(m_systemStatus);
    
    qCInfo(systemController) << "SystemController initialized successfully";
    return true;
}

void SystemController::shutdown()
{
    qCInfo(systemController) << "Shutting down SystemController...";
    
    if (m_systemStatus == Offline) {
        return;
    }
    
    // 停止所有操作
    stopOperation();
    
    // 停止定时器
    if (m_systemMonitorTimer) {
        m_systemMonitorTimer->stop();
    }
    if (m_heartbeatTimer) {
        m_heartbeatTimer->stop();
    }
    
    // 关闭通信
    if (m_serialPort && m_serialPort->isOpen()) {
        m_serialPort->close();
    }
    if (m_tcpSocket && m_tcpSocket->state() != QTcpSocket::UnconnectedState) {
        m_tcpSocket->disconnectFromHost();
    }
    
    // 终止进程
    if (m_systemProcess && m_systemProcess->state() != QProcess::NotRunning) {
        m_systemProcess->kill();
        m_systemProcess->waitForFinished(3000);
    }
    
    m_systemStatus = Offline;
    emit systemStatusChanged(m_systemStatus);
    
    qCInfo(systemController) << "SystemController shutdown completed";
}

void SystemController::setControlMode(ControlMode mode)
{
    if (m_controlMode == mode) {
        return;
    }
    
    qCInfo(systemController) << "Changing control mode from" << m_controlMode << "to" << mode;
    
    m_controlMode = mode;
    emit controlModeChanged(m_controlMode);
    
    // 发送模式切换命令
    QJsonObject params;
    params["mode"] = static_cast<int>(mode);
    sendCommand("SET_CONTROL_MODE", params);
}

void SystemController::setCutterConfig(const CutterConfig& config)
{
    m_cutterConfig = config;
    
    // 发送配置命令
    QJsonObject params;
    params["cutSpeed"] = config.cutSpeed;
    params["cutDepth"] = config.cutDepth;
    params["bladePosition"] = config.bladePosition;
    params["autoRetract"] = config.autoRetract;
    params["retractDelay"] = config.retractDelay;
    params["forceThreshold"] = config.forceThreshold;
    
    sendCommand("SET_CUTTER_CONFIG", params);
    
    logMessage(QString("Cutter configuration updated: Speed=%1, Depth=%2")
               .arg(config.cutSpeed).arg(config.cutDepth));
}

void SystemController::startOperation()
{
    qCInfo(systemController) << "Starting operation...";
    
    if (m_emergencyStopActive) {
        handleError("Cannot start operation: Emergency stop is active");
        return;
    }
    
    if (m_systemStatus != Standby) {
        handleError("Cannot start operation: System not in standby state");
        return;
    }
    
    m_systemStatus = Operating;
    emit systemStatusChanged(m_systemStatus);
    
    sendCommand("START_OPERATION");
    
    logMessage("Operation started");
}

void SystemController::stopOperation()
{
    qCInfo(systemController) << "Stopping operation...";
    
    if (m_systemStatus == Operating) {
        m_systemStatus = Standby;
        emit systemStatusChanged(m_systemStatus);
    }
    
    sendCommand("STOP_OPERATION");
    
    logMessage("Operation stopped");
}

void SystemController::emergencyStop()
{
    qCCritical(systemController) << "Emergency stop activated!";
    
    m_emergencyStopActive = true;
    m_systemStatus = Error;
    m_cutterStatus = CutterError;
    
    emit systemStatusChanged(m_systemStatus);
    emit cutterStatusChanged(m_cutterStatus);
    
    sendCommand("EMERGENCY_STOP");
    
    // 立即停止所有定时器和操作
    if (m_systemProcess && m_systemProcess->state() != QProcess::NotRunning) {
        m_systemProcess->kill();
    }
    
    logMessage("EMERGENCY STOP ACTIVATED!");
    emit errorOccurred("Emergency stop activated");
}

void SystemController::resetSystem()
{
    qCInfo(systemController) << "Resetting system...";
    
    m_emergencyStopActive = false;
    m_systemStatus = Initializing;
    m_cutterStatus = CutterIdle;
    
    emit systemStatusChanged(m_systemStatus);
    emit cutterStatusChanged(m_cutterStatus);
    
    sendCommand("RESET_SYSTEM");
    
    // 重新初始化
    QTimer::singleShot(2000, this, [this]() {
        m_systemStatus = Standby;
        emit systemStatusChanged(m_systemStatus);
        logMessage("System reset completed");
    });
}

void SystemController::calibrateCutter()
{
    qCInfo(systemController) << "Starting cutter calibration...";
    
    if (m_systemStatus != Standby) {
        handleError("Cannot calibrate: System not in standby state");
        return;
    }
    
    m_systemStatus = Maintenance;
    m_cutterStatus = CutterMaintenance;
    
    emit systemStatusChanged(m_systemStatus);
    emit cutterStatusChanged(m_cutterStatus);
    
    sendCommand("CALIBRATE_CUTTER");
    
    // 模拟校准过程
    QTimer::singleShot(5000, this, [this]() {
        m_systemStatus = Standby;
        m_cutterStatus = CutterIdle;
        emit systemStatusChanged(m_systemStatus);
        emit cutterStatusChanged(m_cutterStatus);
        emit calibrationCompleted(true);
        logMessage("Cutter calibration completed successfully");
    });
}

void SystemController::moveCutterUp()
{
    if (m_controlMode != Manual) {
        handleError("Manual control not available in current mode");
        return;
    }
    
    sendCommand("MOVE_CUTTER_UP");
    logMessage("Manual command: Cutter up");
}

void SystemController::moveCutterDown()
{
    if (m_controlMode != Manual) {
        handleError("Manual control not available in current mode");
        return;
    }
    
    sendCommand("MOVE_CUTTER_DOWN");
    logMessage("Manual command: Cutter down");
}

void SystemController::activateCutter()
{
    if (m_controlMode != Manual) {
        handleError("Manual control not available in current mode");
        return;
    }
    
    if (m_cutterStatus == CutterError) {
        handleError("Cannot activate cutter: Cutter in error state");
        return;
    }
    
    m_cutterStatus = CutterActive;
    emit cutterStatusChanged(m_cutterStatus);
    
    sendCommand("ACTIVATE_CUTTER");
    logMessage("Manual command: Cutter activated");
}

void SystemController::retractCutter()
{
    m_cutterStatus = CutterIdle;
    emit cutterStatusChanged(m_cutterStatus);
    
    sendCommand("RETRACT_CUTTER");
    logMessage("Manual command: Cutter retracted");
}

void SystemController::setupSerialCommunication()
{
    qCInfo(systemController) << "Setting up serial communication...";
    
    if (m_serialPort) {
        delete m_serialPort;
    }
    
    m_serialPort = new QSerialPort(this);
    m_serialPort->setPortName(m_serialPortName);
    m_serialPort->setBaudRate(QSerialPort::Baud115200);
    m_serialPort->setDataBits(QSerialPort::Data8);
    m_serialPort->setParity(QSerialPort::NoParity);
    m_serialPort->setStopBits(QSerialPort::OneStop);
    
    connect(m_serialPort, &QSerialPort::readyRead, this, &SystemController::onSerialDataReceived);
    connect(m_serialPort, &QSerialPort::errorOccurred, this, &SystemController::onSerialError);
    
    // 尝试打开串口
    if (m_serialPort->open(QIODevice::ReadWrite)) {
        qCInfo(systemController) << "Serial port opened successfully:" << m_serialPortName;
    } else {
        qCWarning(systemController) << "Failed to open serial port:" << m_serialPortName 
                                   << m_serialPort->errorString();
    }
}

void SystemController::setupNetworkCommunication()
{
    qCInfo(systemController) << "Setting up network communication...";
    
    if (m_tcpSocket) {
        delete m_tcpSocket;
    }
    
    m_tcpSocket = new QTcpSocket(this);
    
    connect(m_tcpSocket, &QTcpSocket::readyRead, this, &SystemController::onNetworkDataReceived);
    connect(m_tcpSocket, &QTcpSocket::errorOccurred, this, &SystemController::onNetworkError);
    
    // 尝试连接到控制器
    m_tcpSocket->connectToHost(QHostAddress(m_networkAddress), m_networkPort);
    
    if (m_tcpSocket->waitForConnected(3000)) {
        qCInfo(systemController) << "Network connection established:" << m_networkAddress << ":" << m_networkPort;
    } else {
        qCWarning(systemController) << "Failed to connect to network:" << m_networkAddress 
                                   << ":" << m_networkPort << m_tcpSocket->errorString();
    }
}

void SystemController::updateSystemInfo()
{
    // 模拟系统信息更新
    static int uptimeCounter = 0;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> cpu_dis(0, 99);
    static std::uniform_int_distribution<> mem_dis(20, 99);
    static std::uniform_int_distribution<> disk_dis(30, 79);
    static std::uniform_int_distribution<> temp_dis(40, 59);
    
    uptimeCounter++;
    
    m_systemInfo.uptime = uptimeCounter;
    m_systemInfo.cpuUsage = cpu_dis(gen);
    m_systemInfo.memoryUsage = mem_dis(gen);
    m_systemInfo.diskUsage = disk_dis(gen);
    m_systemInfo.temperature = temp_dis(gen); // 40-60度
    
    emit systemInfoUpdated(m_systemInfo);
}

void SystemController::sendCommand(const QString& command, const QJsonObject& parameters)
{
    if (!validateCommand(command)) {
        handleError(QString("Invalid command: %1").arg(command));
        return;
    }
    
    QJsonObject commandObj;
    commandObj["command"] = command;
    commandObj["timestamp"] = QDateTime::currentMSecsSinceEpoch();
    if (!parameters.isEmpty()) {
        commandObj["parameters"] = parameters;
    }
    
    QJsonDocument doc(commandObj);
    QByteArray data = doc.toJson(QJsonDocument::Compact) + "\n";
    
    // 优先使用串口通信
    if (m_serialPort && m_serialPort->isOpen()) {
        qint64 bytesWritten = m_serialPort->write(data);
        if (bytesWritten == -1) {
            qCWarning(systemController) << "Failed to write to serial port:" << m_serialPort->errorString();
        } else {
            logMessage(QString("Command sent via serial: %1").arg(command));
        }
    } else if (m_tcpSocket && m_tcpSocket->state() == QTcpSocket::ConnectedState) {
        qint64 bytesWritten = m_tcpSocket->write(data);
        if (bytesWritten == -1) {
            qCWarning(systemController) << "Failed to write to network socket:" << m_tcpSocket->errorString();
        } else {
            logMessage(QString("Command sent via network: %1").arg(command));
        }
    } else {
        qCWarning(systemController) << "No communication channel available for command:" << command;
    }
}

void SystemController::processResponse(const QByteArray& data)
{
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(data, &error);
    
    if (error.error != QJsonParseError::NoError) {
        qCWarning(systemController) << "JSON parse error:" << error.errorString();
        return;
    }
    
    QJsonObject response = doc.object();
    QString status = response["status"].toString();
    QString message = response["message"].toString();
    
    if (status == "OK") {
        logMessage(QString("Command response: %1").arg(message));
    } else if (status == "ERROR") {
        handleError(message);
    } else {
        // 处理状态更新等其他消息
        logMessage(QString("Received: %1").arg(QString::fromUtf8(data)));
    }
}

bool SystemController::validateCommand(const QString& command) const
{
    static const QStringList validCommands = {
        "START_OPERATION", "STOP_OPERATION", "EMERGENCY_STOP", "RESET_SYSTEM",
        "SET_CONTROL_MODE", "SET_CUTTER_CONFIG", "CALIBRATE_CUTTER",
        "MOVE_CUTTER_UP", "MOVE_CUTTER_DOWN", "ACTIVATE_CUTTER", "RETRACT_CUTTER",
        "GET_STATUS", "HEARTBEAT"
    };
    
    return validCommands.contains(command);
}

void SystemController::handleError(const QString& error)
{
    qCWarning(systemController) << "System error:" << error;
    
    if (m_systemStatus != Error) {
        m_systemStatus = Error;
        emit systemStatusChanged(m_systemStatus);
    }
    
    emit errorOccurred(error);
    logMessage(QString("ERROR: %1").arg(error));
}

void SystemController::onSerialDataReceived()
{
    if (!m_serialPort) return;
    
    QByteArray data = m_serialPort->readAll();
    processResponse(data);
}

void SystemController::onSerialError(QSerialPort::SerialPortError error)
{
    if (error != QSerialPort::NoError) {
        QString errorString = m_serialPort ? m_serialPort->errorString() : "Unknown error";
        handleError(QString("Serial port error: %1").arg(errorString));
    }
}

void SystemController::onNetworkDataReceived()
{
    if (!m_tcpSocket) return;
    
    QByteArray data = m_tcpSocket->readAll();
    processResponse(data);
}

void SystemController::onNetworkError()
{
    QString errorString = m_tcpSocket ? m_tcpSocket->errorString() : "Unknown error";
    handleError(QString("Network error: %1").arg(errorString));
}

void SystemController::onSystemMonitorTimer()
{
    updateSystemInfo();
    
    // 检查通信超时
    qint64 currentTime = QDateTime::currentMSecsSinceEpoch();
    if (m_lastHeartbeat > 0 && (currentTime - m_lastHeartbeat) > m_communicationTimeout) {
        handleError("Communication timeout - no response from controller");
    }
}

void SystemController::onHeartbeatTimer()
{
    sendCommand("HEARTBEAT");
}

void SystemController::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    Q_UNUSED(exitCode)
    Q_UNUSED(exitStatus)
    
    logMessage("System process finished");
}

void SystemController::logMessage(const QString& message)
{
    if (m_debugEnabled) {
        qCDebug(systemController) << message;
    }
    
    // 这里可以添加日志文件写入功能
}