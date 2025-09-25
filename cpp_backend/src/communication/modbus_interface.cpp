/**
 * @file modbus_interface.cpp
 * @brief C++ LVGL一体化系统Modbus TCP通信接口实现
 * @version 5.0.0
 * @date 2024
 * 
 * 西门子PLC通信 + 寄存器自动同步 + 连接状态监控
 */

#include "bamboo_cut/communication/modbus_interface.h"
#include <iostream>
#include <thread>
#include <chrono>

#ifdef ENABLE_MODBUS
#include <modbus/modbus.h>
#endif

namespace bamboo_cut {
namespace communication {

ModbusInterface::ModbusInterface()
    : connected_(false)
    , ctx_(nullptr) {
}

ModbusInterface::~ModbusInterface() {
    disconnect();
}

bool ModbusInterface::connect(const std::string& ip, int port) {
    std::cout << "[ModbusInterface] 连接到PLC " << ip << ":" << port << std::endl;
    
#ifdef ENABLE_MODBUS
    ctx_ = modbus_new_tcp(ip.c_str(), port);
    if (!ctx_) {
        std::cout << "[ModbusInterface] 创建Modbus上下文失败" << std::endl;
        return false;
    }
    
    if (modbus_connect(ctx_) == -1) {
        std::cout << "[ModbusInterface] 连接PLC失败: " << modbus_strerror(errno) << std::endl;
        modbus_free(ctx_);
        ctx_ = nullptr;
        return false;
    }
    
    connected_ = true;
    server_ip_ = ip;
    server_port_ = port;
    
    std::cout << "[ModbusInterface] PLC连接成功" << std::endl;
    return true;
#else
    std::cout << "[ModbusInterface] Modbus功能未启用" << std::endl;
    return false;
#endif
}

void ModbusInterface::disconnect() {
#ifdef ENABLE_MODBUS
    if (connected_ && ctx_) {
        modbus_close(ctx_);
        modbus_free(ctx_);
        ctx_ = nullptr;
        connected_ = false;
        std::cout << "[ModbusInterface] PLC连接已断开" << std::endl;
    }
#endif
}

bool ModbusInterface::writeRegister(int address, uint16_t value) {
#ifdef ENABLE_MODBUS
    if (!connected_ || !ctx_) {
        return false;
    }
    
    int result = modbus_write_register(ctx_, address, value);
    return result == 1;
#else
    std::cout << "[ModbusInterface] 模拟写入寄存器 " << address << " = " << value << std::endl;
    return true;
#endif
}

bool ModbusInterface::readRegister(int address, uint16_t& value) {
#ifdef ENABLE_MODBUS
    if (!connected_ || !ctx_) {
        return false;
    }
    
    int result = modbus_read_registers(ctx_, address, 1, &value);
    return result == 1;
#else
    // 模拟读取
    value = 0;
    return true;
#endif
}

bool ModbusInterface::isConnected() const {
    return connected_;
}

void ModbusInterface::setConnectionParameters(const std::string& ip, int port, int slave_id) {
    server_ip_ = ip;
    server_port_ = port;
    slave_id_ = slave_id;
}

} // namespace communication
} // namespace bamboo_cut