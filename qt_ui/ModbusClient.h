#pragma once

#include <QObject>
#include <QTimer>
#include <cstdint>
#include <modbus/modbus.h>

// Lightweight Modbus TCP client for PLC polling.
class ModbusClient : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool connected READ connected NOTIFY connectionChanged)
    // PLC -> Camera
    Q_PROPERTY(int plcPowerRequest READ plcPowerRequest NOTIFY registersUpdated)
    Q_PROPERTY(int plcReceiveState READ plcReceiveState NOTIFY registersUpdated)
    Q_PROPERTY(double plcServoPosition READ plcServoPosition NOTIFY registersUpdated)
    Q_PROPERTY(double plcCoordFeedback READ plcCoordFeedback NOTIFY registersUpdated)
    // Camera -> PLC (cached)
    Q_PROPERTY(int visionCommAck READ visionCommAck NOTIFY registersUpdated)
    Q_PROPERTY(int visionStatus READ visionStatus NOTIFY registersUpdated)
    Q_PROPERTY(double visionTargetCoord READ visionTargetCoord NOTIFY registersUpdated)
    Q_PROPERTY(int visionTransferResult READ visionTransferResult NOTIFY registersUpdated)
public:
    explicit ModbusClient(QObject *parent = nullptr);
    ~ModbusClient();

    bool connected() const { return connected_; }

    // PLC -> Camera
    int plcPowerRequest() const { return plc_to_cam_.power_request; }
    int plcReceiveState() const { return plc_to_cam_.receive_state; }
    double plcServoPosition() const { return plc_to_cam_.servo_pos; }
    double plcCoordFeedback() const { return plc_to_cam_.coord_feedback; }

    // Camera -> PLC (cached)
    int visionCommAck() const { return cam_to_plc_.comm_ack; }
    int visionStatus() const { return cam_to_plc_.status; }
    double visionTargetCoord() const { return cam_to_plc_.target_coord; }
    int visionTransferResult() const { return cam_to_plc_.transfer_result; }

    Q_INVOKABLE void configure(const QString &host, int port, int slaveId = 1);
    Q_INVOKABLE bool setVisionCommAck(int value);
    Q_INVOKABLE bool setVisionStatus(int value);
    Q_INVOKABLE bool setVisionTargetCoord(double coord);
    Q_INVOKABLE bool setVisionTransferResult(int value);
    Q_INVOKABLE void shutdown();

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
        int power_request{0};
        int receive_state{0};
        float servo_pos{0.0f};
        float coord_feedback{0.0f};
    } plc_to_cam_;

    struct CamToPlc {
        int comm_ack{0};
        int status{0};
        float target_coord{0.0f};
        int transfer_result{0};
    } cam_to_plc_;

    modbus_t *ctx_{nullptr};
    QString host_{"127.0.0.1"};
    int port_{1502};
    int slaveId_{1};
    bool connected_{false};
    QTimer pollTimer_;
};
