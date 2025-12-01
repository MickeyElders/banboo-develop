#pragma once

#include <QObject>
#include <QTimer>
#include <cstdint>
#include <modbus/modbus.h>

// 轻量 Modbus 客户端，轮询 PLC/本地服务器寄存器，供 QML 绑定。
// 地址映射基于需求：
// 相机 >> PLC：0x07D0-D2016 区间（主要用 D2000、D2001、D2002/3、D2004）
// 相机 << PLC：0x0834-D2105 区间（D2100-2105）
class ModbusClient : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool connected READ connected NOTIFY connectionChanged)
    // PLC -> 相机
    Q_PROPERTY(int plcPowerRequest READ plcPowerRequest NOTIFY registersUpdated)          // D2100
    Q_PROPERTY(int plcReceiveState READ plcReceiveState NOTIFY registersUpdated)          // D2101
    Q_PROPERTY(double plcServoPosition READ plcServoPosition NOTIFY registersUpdated)     // D2102/3 float
    Q_PROPERTY(double plcCoordFeedback READ plcCoordFeedback NOTIFY registersUpdated)     // D2104/5 float
    // 相机 -> PLC（本地缓存，便于界面显示）
    Q_PROPERTY(int visionCommAck READ visionCommAck NOTIFY registersUpdated)              // D2000
    Q_PROPERTY(int visionStatus READ visionStatus NOTIFY registersUpdated)                // D2001
    Q_PROPERTY(double visionTargetCoord READ visionTargetCoord NOTIFY registersUpdated)   // D2002/3 float
    Q_PROPERTY(int visionTransferResult READ visionTransferResult NOTIFY registersUpdated)// D2004
public:
    explicit ModbusClient(QObject *parent = nullptr);
    ~ModbusClient();

    bool connected() const { return connected_; }

    // PLC -> 相机
    int plcPowerRequest() const { return plc_to_cam_.power_request; }
    int plcReceiveState() const { return plc_to_cam_.receive_state; }
    double plcServoPosition() const { return plc_to_cam_.servo_pos; }
    double plcCoordFeedback() const { return plc_to_cam_.coord_feedback; }

    // 相机 -> PLC（缓存）
    int visionCommAck() const { return cam_to_plc_.comm_ack; }
    int visionStatus() const { return cam_to_plc_.status; }
    double visionTargetCoord() const { return cam_to_plc_.target_coord; }
    int visionTransferResult() const { return cam_to_plc_.transfer_result; }

    Q_INVOKABLE void configure(const QString &host, int port, int slaveId = 1);
    Q_INVOKABLE bool setVisionCommAck(int value);          // 写 D2000
    Q_INVOKABLE bool setVisionStatus(int value);           // 写 D2001 (1正常、2故障、3未运行)
    Q_INVOKABLE bool setVisionTargetCoord(double coord);   // 写 D2002/3 float mm
    Q_INVOKABLE bool setVisionTransferResult(int value);   // 写 D2004 (1成功、2失败)

Q_SIGNALS:
    void registersUpdated();
    void connectionChanged(bool connected);
    void errorMessage(const QString &msg);

private:
    void poll();
    bool ensureConnected();
    void disconnect();
    bool writeFloat(int addr, float value);

    struct PlcToCam {
        int power_request{0};    // D2100
        int receive_state{0};    // D2101
        float servo_pos{0.0f};   // D2102/3
        float coord_feedback{0.0f}; // D2104/5
    } plc_to_cam_;

    struct CamToPlc {
        int comm_ack{0};         // D2000
        int status{0};           // D2001
        float target_coord{0.0f}; // D2002/3
        int transfer_result{0};  // D2004
    } cam_to_plc_;

    modbus_t *ctx_{nullptr};
    QString host_{"127.0.0.1"};
    int port_{1502};   // 默认本地 Modbus server，如需 PLC 改为 502
    int slaveId_{1};
    bool connected_{false};
    QTimer pollTimer_;
};
