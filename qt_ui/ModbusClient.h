#pragma once

#include <QObject>
#include <QTimer>
#include <cstdint>

struct modbus_t;

// 轻量 Modbus 客户端，轮询 PLC/本地服务器寄存器，供 QML 绑定。
class ModbusClient : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool connected READ connected NOTIFY connectionChanged)
    Q_PROPERTY(int systemStatus READ systemStatus NOTIFY registersUpdated)
    Q_PROPERTY(int plcCommand READ plcCommand NOTIFY registersUpdated)
    Q_PROPERTY(int coordReady READ coordReady NOTIFY registersUpdated)
    Q_PROPERTY(double xCoordinate READ xCoordinate NOTIFY registersUpdated)
    Q_PROPERTY(int cutQuality READ cutQuality NOTIFY registersUpdated)
    Q_PROPERTY(uint32_t heartbeat READ heartbeat NOTIFY registersUpdated)
    Q_PROPERTY(int bladeNumber READ bladeNumber NOTIFY registersUpdated)
    Q_PROPERTY(int healthStatus READ healthStatus NOTIFY registersUpdated)
    Q_PROPERTY(int tailStatus READ tailStatus NOTIFY registersUpdated)
    Q_PROPERTY(int plcExtAlarm READ plcExtAlarm NOTIFY registersUpdated)
    Q_PROPERTY(int railDirection READ railDirection NOTIFY registersUpdated)
    Q_PROPERTY(uint32_t remainingLength READ remainingLength NOTIFY registersUpdated)
    Q_PROPERTY(int coverage READ coverage NOTIFY registersUpdated)
    Q_PROPERTY(int feedSpeedGear READ feedSpeedGear NOTIFY registersUpdated)
    Q_PROPERTY(int processMode READ processMode NOTIFY registersUpdated)
public:
    explicit ModbusClient(QObject *parent = nullptr);
    ~ModbusClient();

    bool connected() const { return connected_; }

    int systemStatus() const { return regs_.system_status; }
    int plcCommand() const { return regs_.plc_command; }
    int coordReady() const { return regs_.coord_ready; }
    double xCoordinate() const { return regs_.x_coordinate / 10.0; } // 0.1mm 精度
    int cutQuality() const { return regs_.cut_quality; }
    uint32_t heartbeat() const { return regs_.heartbeat; }
    int bladeNumber() const { return regs_.blade_number; }
    int healthStatus() const { return regs_.health_status; }
    int tailStatus() const { return regs_.tail_status; }
    int plcExtAlarm() const { return regs_.plc_ext_alarm; }
    int railDirection() const { return regs_.rail_direction; }
    uint32_t remainingLength() const { return regs_.remaining_length; }
    int coverage() const { return regs_.coverage; }
    int feedSpeedGear() const { return regs_.feed_speed_gear; }
    int processMode() const { return regs_.process_mode; }

    Q_INVOKABLE void configure(const QString &host, int port, int slaveId = 1);
    Q_INVOKABLE bool writeRegister(int address, int value);
    Q_INVOKABLE bool sendCommand(int command);
    Q_INVOKABLE bool setSystemStatus(int status);

signals:
    void registersUpdated();
    void connectionChanged(bool connected);
    void errorMessage(const QString &msg);

private:
    void poll();
    bool ensureConnected();
    void disconnect();

    struct Regs {
        uint16_t system_status{0};
        uint16_t plc_command{0};
        uint16_t coord_ready{0};
        uint32_t x_coordinate{0};
        uint16_t cut_quality{0};
        uint32_t heartbeat{0};
        uint16_t blade_number{0};
        uint16_t health_status{0};
        uint16_t tail_status{0};
        uint16_t plc_ext_alarm{0};
        uint16_t rail_direction{0};
        uint32_t remaining_length{0};
        uint16_t coverage{0};
        uint16_t feed_speed_gear{0};
        uint16_t process_mode{0};
    } regs_;

    modbus_t *ctx_{nullptr};
    QString host_{"127.0.0.1"};
    int port_{1502};   // 默认本地 Modbus server，如需 PLC 改为 502
    int slaveId_{1};
    bool connected_{false};
    QTimer pollTimer_;
};
