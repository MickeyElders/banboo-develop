/**
 * @file data_bridge.cpp
 * @brief Thread-safe data bridge implementation.
 */

#include "bamboo_cut/core/data_bridge.h"
#include <chrono>
#include <algorithm>

namespace bamboo_cut {
namespace core {

DataBridge::DataBridge() {
    // Initialize display frame buffer
    display_frame_ = cv::Mat::zeros(480, 640, CV_8UC3);
}

DataBridge::~DataBridge() = default;

void DataBridge::updateFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!frame.empty()) {
        frame.copyTo(latest_frame_);
        cv::resize(frame, display_frame_, cv::Size(640, 480));
        new_frame_available_.store(true);
        frame_cv_.notify_one();
    }
}

bool DataBridge::getLatestFrame(cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!latest_frame_.empty()) {
        latest_frame_.copyTo(frame);
        return true;
    }
    return false;
}

bool DataBridge::getDisplayFrame(cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!display_frame_.empty()) {
        display_frame_.copyTo(frame);
        return true;
    }
    return false;
}

void DataBridge::updateDetection(const DetectionResult& result) {
    std::lock_guard<std::mutex> lock(detection_mutex_);
    detection_data_ = result;
    detection_data_.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    new_detection_available_.store(true);
    detection_cv_.notify_one();
}

bool DataBridge::getDetectionResult(DetectionResult& result) {
    std::lock_guard<std::mutex> lock(detection_mutex_);
    if (detection_data_.valid) {
        result = detection_data_;
        return true;
    }
    return false;
}

void DataBridge::updateStats(const SystemStats& stats) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = stats;
}

SystemStats DataBridge::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void DataBridge::updateModbusRegisters(const ModbusRegisters& registers) {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    modbus_registers_ = registers;
}

ModbusRegisters DataBridge::getModbusRegisters() const {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    return modbus_registers_;
}

void DataBridge::setPLCCommandValue(uint16_t command) {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    modbus_registers_.plc_command = command;
}

void DataBridge::setProcessMode(uint16_t mode) {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    modbus_registers_.process_mode = mode;
}

void DataBridge::setFeedSpeedGear(uint16_t gear) {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    modbus_registers_.feed_speed_gear = gear;
}

void DataBridge::setPLCAlarmCode(uint16_t alarm) {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    modbus_registers_.plc_ext_alarm = alarm;
}

void DataBridge::setSystemRunning(bool running) {
    system_running_.store(running);
    
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    modbus_registers_.system_status = running ? 1 : 0;
}

bool DataBridge::isSystemRunning() const {
    return system_running_.load();
}

void DataBridge::setEmergencyStop(bool stop) {
    emergency_stop_.store(stop);
    
    if (stop) {
        setSystemRunning(false);
        std::lock_guard<std::mutex> lock(modbus_mutex_);
        modbus_registers_.system_status = 4; // emergency stop
    }
}

bool DataBridge::isEmergencyStop() const {
    return emergency_stop_.load();
}

void DataBridge::setCurrentStep(int step) {
    current_step_.store(step);
}

int DataBridge::getCurrentStep() const {
    return current_step_.load();
}

void DataBridge::updateHeartbeat() {
    uint32_t current = heartbeat_counter_.fetch_add(1);
    
    if (current > 4294967290U) {
        heartbeat_counter_.store(0);
    }
    
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    modbus_registers_.heartbeat = heartbeat_counter_.load();
}

uint32_t DataBridge::getHeartbeat() const {
    return heartbeat_counter_.load();
}

void DataBridge::publishCuttingPoint(float x_mm, uint16_t coverage, uint16_t quality,
                                     uint16_t blade, uint16_t tail_status) {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    
    if (x_mm < 0.0f) {
        x_mm = 0.0f;
    }
    // 协议约定 0.1mm，缺少标定时按毫米直写
    modbus_registers_.x_coordinate = static_cast<uint32_t>(x_mm * 10.0f);
    modbus_registers_.coverage = coverage;
    modbus_registers_.cut_quality = quality;
    modbus_registers_.blade_number = blade;
    modbus_registers_.tail_status = tail_status;
    modbus_registers_.coord_ready = 1;
}

void DataBridge::clearCoordinateReady() {
    std::lock_guard<std::mutex> lock(modbus_mutex_);
    modbus_registers_.coord_ready = 0;
    modbus_registers_.x_coordinate = 0;
    modbus_registers_.coverage = 0;
    modbus_registers_.tail_status = 0;
    modbus_registers_.cut_quality = 0;
}

} // namespace core
} // namespace bamboo_cut
