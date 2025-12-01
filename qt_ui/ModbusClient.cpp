#include "ModbusClient.h"
#include <modbus/modbus.h>
#include <errno.h>
#include <QDebug>

namespace {
constexpr int kStartAddress = 40001;
constexpr int kCount = 19; // 40001 - 40019 inclusive

inline uint32_t combine(uint16_t high, uint16_t low) {
    return (static_cast<uint32_t>(high) << 16) | low;
}
}

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
    uint16_t buf[kCount] = {0};
    const int rc = modbus_read_registers(ctx_, kStartAddress - 40001, kCount, buf);
    if (rc != kCount) {
        emit errorMessage(tr("Modbus 读失败: %1").arg(modbus_strerror(errno)));
        disconnect();
        return;
    }
    regs_.system_status   = buf[0];
    regs_.plc_command     = buf[1];
    regs_.coord_ready     = buf[2];
    regs_.x_coordinate    = combine(buf[3], buf[4]);
    regs_.cut_quality     = buf[5];
    regs_.heartbeat       = combine(buf[6], buf[7]);
    regs_.blade_number    = buf[8];
    regs_.health_status   = buf[9];
    regs_.tail_status     = buf[10];
    regs_.plc_ext_alarm   = buf[11];
    regs_.rail_direction  = buf[13];
    regs_.remaining_length= combine(buf[14], buf[15]);
    regs_.coverage        = buf[16];
    regs_.feed_speed_gear = buf[17];
    regs_.process_mode    = buf[18];
    emit registersUpdated();
}

bool ModbusClient::writeRegister(int address, int value) {
    if (!ensureConnected()) return false;
    const int rc = modbus_write_register(ctx_, address - 40001, static_cast<uint16_t>(value));
    if (rc == -1) {
        emit errorMessage(tr("写寄存器失败 %1: %2").arg(address).arg(modbus_strerror(errno)));
        disconnect();
        return false;
    }
    return true;
}

bool ModbusClient::sendCommand(int command) {
    // 写 40002
    return writeRegister(40002, command);
}

bool ModbusClient::setSystemStatus(int status) {
    // 写 40001
    return writeRegister(40001, status);
}
