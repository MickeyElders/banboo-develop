#include "ModbusClient.h"
#include <modbus/modbus.h>
#include <errno.h>
#include <cstring>
#include <QDebug>

namespace {
inline uint32_t combine(uint16_t high, uint16_t low) {
    return (static_cast<uint32_t>(high) << 16) | low;
}
inline void splitFloat(float value, uint16_t &high, uint16_t &low) {
    uint32_t u = 0;
    static_assert(sizeof(float) == sizeof(uint32_t), "float size mismatch");
    memcpy(&u, &value, sizeof(float));
    high = static_cast<uint16_t>((u >> 16) & 0xFFFF);
    low = static_cast<uint16_t>(u & 0xFFFF);
}
inline float toFloat(uint16_t high, uint16_t low) {
    uint32_t u = combine(high, low);
    float f;
    memcpy(&f, &u, sizeof(float));
    return f;
}
} // namespace

ModbusClient::ModbusClient(QObject *parent) : QObject(parent) {
    pollTimer_.setInterval(200); // 5Hz poll，视实际负载可调
    connect(&pollTimer_, &QTimer::timeout, this, &ModbusClient::poll);
    pollTimer_.start();
}

ModbusClient::~ModbusClient() {
    disconnect();
}

void ModbusClient::configure(const QString &host, int port, int slaveId) {
    host_ = host;
    port_ = port;
    slaveId_ = slaveId;
    disconnect();
}

bool ModbusClient::ensureConnected() {
    if (connected_ && ctx_) return true;
    ctx_ = modbus_new_tcp(host_.toStdString().c_str(), port_);
    if (!ctx_) {
        emit errorMessage(tr("Modbus 创建失败"));
        return false;
    }
    modbus_set_slave(ctx_, slaveId_);
    if (modbus_connect(ctx_) == -1) {
        emit errorMessage(tr("Modbus 连接失败: %1").arg(modbus_strerror(errno)));
        modbus_free(ctx_);
        ctx_ = nullptr;
        connected_ = false;
        emit connectionChanged(false);
        return false;
    }
    connected_ = true;
    emit connectionChanged(true);
    return true;
}

void ModbusClient::disconnect() {
    if (ctx_) {
        modbus_close(ctx_);
        modbus_free(ctx_);
        ctx_ = nullptr;
    }
    if (connected_) {
        connected_ = false;
        emit connectionChanged(false);
    }
}

void ModbusClient::poll() {
    if (!ensureConnected()) return;
    // 读取 PLC -> 相机：D2100-2105 (0x0834-0x0839)
    constexpr int start = 0x0834;
    constexpr int count = 6;
    uint16_t buf[count] = {0};
    const int rc = modbus_read_registers(ctx_, start, count, buf);
    if (rc != count) {
        emit errorMessage(tr("Modbus 读失败: %1").arg(modbus_strerror(errno)));
        disconnect();
        return;
    }
    plc_to_cam_.power_request   = buf[0];
    plc_to_cam_.receive_state   = buf[1];
    plc_to_cam_.servo_pos       = toFloat(buf[2], buf[3]);
    plc_to_cam_.coord_feedback  = toFloat(buf[4], buf[5]);
    emit registersUpdated();
}

bool ModbusClient::writeFloat(int addr, float value) {
    if (!ensureConnected()) return false;
    uint16_t high = 0, low = 0;
    splitFloat(value, high, low);
    uint16_t payload[2] = {high, low};
    const int rc = modbus_write_registers(ctx_, addr, 2, payload);
    if (rc == -1) {
        emit errorMessage(tr("写浮点寄存器失败 %1: %2").arg(addr).arg(modbus_strerror(errno)));
        disconnect();
        return false;
    }
    return true;
}

bool ModbusClient::setVisionCommAck(int value) {
    cam_to_plc_.comm_ack = value;
    if (!ensureConnected()) return false;
    const int rc = modbus_write_register(ctx_, 0x07D0, static_cast<uint16_t>(value));
    if (rc == -1) {
        emit errorMessage(tr("写 D2000 失败: %1").arg(modbus_strerror(errno)));
        disconnect();
        return false;
    }
    emit registersUpdated();
    return true;
}

bool ModbusClient::setVisionStatus(int value) {
    cam_to_plc_.status = value;
    if (!ensureConnected()) return false;
    const int rc = modbus_write_register(ctx_, 0x07D1, static_cast<uint16_t>(value));
    if (rc == -1) {
        emit errorMessage(tr("写 D2001 失败: %1").arg(modbus_strerror(errno)));
        disconnect();
        return false;
    }
    emit registersUpdated();
    return true;
}

bool ModbusClient::setVisionTargetCoord(double coord) {
    cam_to_plc_.target_coord = static_cast<float>(coord);
    const bool ok = writeFloat(0x07D2, static_cast<float>(coord));
    if (ok) emit registersUpdated();
    return ok;
}

bool ModbusClient::setVisionTransferResult(int value) {
    cam_to_plc_.transfer_result = value;
    if (!ensureConnected()) return false;
    const int rc = modbus_write_register(ctx_, 0x07D4, static_cast<uint16_t>(value));
    if (rc == -1) {
        emit errorMessage(tr("写 D2004 失败: %1").arg(modbus_strerror(errno)));
        disconnect();
        return false;
    }
    emit registersUpdated();
    return true;
}
