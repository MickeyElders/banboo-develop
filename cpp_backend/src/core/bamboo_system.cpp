/**
 * @file bamboo_system.cpp
 * @brief C++ LVGL一体化竹子识别系统核心控制器实现
 * @version 5.0.0
 * @date 2024
 * 
 * C++推理后端 + LVGL界面 + Modbus通信的完整一体化系统
 */

#include "bamboo_cut/core/bamboo_system.h"
#include "bamboo_cut/utils/logger.h"
#include "bamboo_cut/utils/config_loader.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <fstream>

namespace bamboo_cut {
namespace core {

// 静态实例指针用于信号处理
BambooSystem* BambooSystem::instance_ = nullptr;

BambooSystem::BambooSystem() 
    : current_state_(SystemState::UNINITIALIZED)
    , data_bridge_(std::make_shared<DataBridge>())
    , start_time_(std::chrono::system_clock::now()) {
    
    instance_ = this;
    
    // 初始化系统信息
    system_info_.state = SystemState::UNINITIALIZED;
    system_info_.state_name = "未初始化";
    system_info_.uptime = std::chrono::seconds(0);
    system_info_.cpu_usage = 0.0f;
    system_info_.memory_usage = 0.0f;
    system_info_.current_workflow_step = 0;
    system_info_.ai_inference_active = false;
    system_info_.ui_interface_active = false;
    system_info_.modbus_communication_active = false;
    system_info_.last_error = "";
    system_info_.start_time = start_time_;
    
    // 初始化性能统计
    system_info_.performance.inference_fps = 0.0f;
    system_info_.performance.ui_fps = 0.0f;
    system_info_.performance.total_detections = 0;
    system_info_.performance.total_cuts = 0;
    system_info_.performance.system_efficiency = 0.0f;
    
    // 初始化性能监控
    performance_monitor_.last_performance_check = std::chrono::high_resolution_clock::now();
    performance_monitor_.frame_count = 0;
    performance_monitor_.detection_count = 0;
    performance_monitor_.average_inference_time = 0.0f;
    
    last_stats_update_ = std::chrono::high_resolution_clock::now();
}

BambooSystem::~BambooSystem() {
    stop();
    instance_ = nullptr;
}

bool BambooSystem::initialize(const SystemConfig& config) {
    std::cout << "[BambooSystem] 初始化系统...v" << getVersionInfo().toString() << std::endl;
    
    changeState(SystemState::INITIALIZING);
    
    try {
        config_ = config;
        
        // 初始化日志系统
        utils::Logger::getInstance().initialize("logs/bamboo_system.log");
        utils::Logger::getInstance().log(utils::LogLevel::INFO, "系统初始化开始");
        
        // 初始化所有子系统
        if (!initializeSubsystems()) {
            handleSystemError("子系统初始化失败");
            return false;
        }
        
        changeState(SystemState::READY);
        utils::Logger::getInstance().log(utils::LogLevel::INFO, "系统初始化完成");
        std::cout << "[BambooSystem] 系统初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        handleSystemError("初始化异常: " + std::string(e.what()));
        return false;
    }
}

bool BambooSystem::start() {
    if (current_state_ != SystemState::READY) {
        last_error_ = "系统未就绪，无法启动";
        return false;
    }
    
    std::cout << "[BambooSystem] 启动系统..." << std::endl;
    
    try {
        // 启动所有子系统
        if (!startSubsystems()) {
            handleSystemError("子系统启动失败");
            return false;
        }
        
        should_stop_ = false;
        running_ = true;
        
        // 启动主循环线程
        main_thread_ = std::thread(&BambooSystem::mainLoop, this);
        
        changeState(SystemState::RUNNING);
        utils::Logger::getInstance().log(utils::LogLevel::INFO, "系统启动完成");
        std::cout << "[BambooSystem] 系统启动完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        handleSystemError("启动异常: " + std::string(e.what()));
        return false;
    }
}

void BambooSystem::stop() {
    if (!running_.load()) {
        return;
    }
    
    std::cout << "[BambooSystem] 停止系统..." << std::endl;
    
    should_stop_ = true;
    running_ = false;
    
    // 等待主线程结束
    if (main_thread_.joinable()) {
        main_thread_.join();
    }
    
    // 停止所有子系统
    stopSubsystems();
    
    changeState(SystemState::SHUTDOWN);
    utils::Logger::getInstance().log(utils::LogLevel::INFO, "系统已停止");
    std::cout << "[BambooSystem] 系统已停止" << std::endl;
}

void BambooSystem::pause() {
    if (current_state_ == SystemState::RUNNING) {
        changeState(SystemState::PAUSED);
        data_bridge_->setSystemRunning(false);
        utils::Logger::getInstance().log(utils::LogLevel::INFO, "系统已暂停");
        std::cout << "[BambooSystem] 系统已暂停" << std::endl;
    }
}

void BambooSystem::resume() {
    if (current_state_ == SystemState::PAUSED) {
        changeState(SystemState::RUNNING);
        data_bridge_->setSystemRunning(true);
        data_bridge_->setEmergencyStop(false);
        utils::Logger::getInstance().log(utils::LogLevel::INFO, "系统已恢复");
        std::cout << "[BambooSystem] 系统已恢复" << std::endl;
    }
}

void BambooSystem::emergencyStop() {
    std::cout << "[BambooSystem] 紧急停止！" << std::endl;
    
    changeState(SystemState::EMERGENCY);
    data_bridge_->setEmergencyStop(true);
    data_bridge_->setSystemRunning(false);
    
    // 停止工作流程
    if (workflow_manager_) {
        workflow_manager_->emergencyStop();
    }
    
    utils::Logger::getInstance().log(utils::LogLevel::CRITICAL, "系统紧急停止");
}

int BambooSystem::run() {
    if (!running_.load()) {
        std::cerr << "[BambooSystem] 系统未启动" << std::endl;
        return -1;
    }
    
    std::cout << "[BambooSystem] 进入主运行循环..." << std::endl;
    
    // 等待主线程完成
    if (main_thread_.joinable()) {
        main_thread_.join();
    }
    
    return 0;
}

BambooSystem::SystemInfo BambooSystem::getSystemInfo() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    SystemInfo info = system_info_;
    
    // 计算运行时间
    auto now = std::chrono::system_clock::now();
    info.uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
    
    // 更新当前状态
    info.state = current_state_;
    info.state_name = systemStateToString(current_state_);
    
    return info;
}

bool BambooSystem::reloadConfig(const SystemConfig& config) {
    std::cout << "[BambooSystem] 重新加载配置..." << std::endl;
    
    config_ = config;
    
    // 重新配置子系统
    if (inference_thread_) {
        // TODO: 更新推理配置
    }
    
    if (ui_interface_) {
        // TODO: 更新界面配置
    }
    
    if (modbus_interface_) {
        // TODO: 更新Modbus配置
    }
    
    utils::Logger::getInstance().log(utils::LogLevel::INFO, "配置重新加载完成");
    std::cout << "[BambooSystem] 配置重新加载完成" << std::endl;
    return true;
}

bool BambooSystem::saveState(const std::string& state_file) const {
    try {
        std::ofstream file(state_file);
        if (!file.is_open()) {
            return false;
        }
        
        auto info = getSystemInfo();
        
        file << "# 竹子识别系统状态文件\n";
        file << "state: " << static_cast<int>(info.state) << "\n";
        file << "uptime: " << info.uptime.count() << "\n";
        file << "current_step: " << info.current_workflow_step << "\n";
        file << "total_detections: " << info.performance.total_detections << "\n";
        file << "total_cuts: " << info.performance.total_cuts << "\n";
        
        return true;
    } catch (const std::exception& e) {
        last_error_ = "保存状态失败: " + std::string(e.what());
        return false;
    }
}

bool BambooSystem::loadState(const std::string& state_file) {
    try {
        std::ifstream file(state_file);
        if (!file.is_open()) {
            return false;
        }
        
        // TODO: 实现状态加载逻辑
        
        utils::Logger::getInstance().log(utils::LogLevel::INFO, "系统状态已加载");
        return true;
    } catch (const std::exception& e) {
        handleSystemError("加载状态失败: " + std::string(e.what()));
        return false;
    }
}

void BambooSystem::mainLoop() {
    std::cout << "[BambooSystem] 主循环开始" << std::endl;
    
    while (!should_stop_.load()) {
        try {
            // 更新系统统计信息
            updateSystemStats();
            
            // 监控系统健康状态
            monitorSystemHealth();
            
            // 控制循环频率
            std::this_thread::sleep_for(
                std::chrono::milliseconds(config_.system_params.main_loop_interval_ms));
                
        } catch (const std::exception& e) {
            handleSystemError("主循环异常: " + std::string(e.what()));
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
    
    std::cout << "[BambooSystem] 主循环结束" << std::endl;
}

bool BambooSystem::initializeSubsystems() {
    std::cout << "[BambooSystem] 初始化子系统..." << std::endl;
    
    try {
        // 初始化推理线程
        if (config_.system_params.enable_ai_inference) {
            inference_thread_ = std::make_unique<inference::InferenceThread>(data_bridge_);
            system_info_.ai_inference_active = true;
            std::cout << "[BambooSystem] AI推理模块已初始化" << std::endl;
        }
        
        // 初始化LVGL界面
        if (config_.system_params.enable_ui_interface) {
#ifdef ENABLE_LVGL
            ui_interface_ = std::make_unique<ui::LVGLInterface>(data_bridge_);
            if (ui_interface_->initialize(config_.ui_config)) {
                system_info_.ui_interface_active = true;
                std::cout << "[BambooSystem] LVGL界面模块已初始化" << std::endl;
            } else {
                std::cerr << "[BambooSystem] LVGL界面初始化失败" << std::endl;
            }
#else
            std::cout << "[BambooSystem] LVGL未启用，跳过界面初始化" << std::endl;
#endif
        }
        
        // 初始化Modbus通信
        if (config_.system_params.enable_modbus_communication) {
#ifdef ENABLE_MODBUS
            modbus_interface_ = std::make_unique<communication::ModbusInterface>(data_bridge_);
            if (modbus_interface_->initialize(config_.modbus_config)) {
                system_info_.modbus_communication_active = true;
                std::cout << "[BambooSystem] Modbus通信模块已初始化" << std::endl;
            } else {
                std::cerr << "[BambooSystem] Modbus通信初始化失败" << std::endl;
            }
#else
            std::cout << "[BambooSystem] Modbus未启用，跳过通信初始化" << std::endl;
#endif
        }
        
        // 初始化工作流程管理器
        workflow_manager_ = std::make_unique<WorkflowManager>(data_bridge_);
        
        return true;
        
    } catch (const std::exception& e) {
        handleSystemError("子系统初始化异常: " + std::string(e.what()));
        return false;
    }
}

bool BambooSystem::startSubsystems() {
    std::cout << "[BambooSystem] 启动子系统..." << std::endl;
    
    try {
        // 启动推理线程
        if (inference_thread_ && !inference_thread_->start()) {
            std::cerr << "[BambooSystem] 推理线程启动失败" << std::endl;
            return false;
        }
        
        // 启动LVGL界面
        if (ui_interface_ && !ui_interface_->start()) {
            std::cerr << "[BambooSystem] 界面线程启动失败" << std::endl;
            // 界面启动失败不影响核心功能
        }
        
        // 启动Modbus通信
        if (modbus_interface_ && !modbus_interface_->start()) {
            std::cerr << "[BambooSystem] Modbus通信启动失败" << std::endl;
            // Modbus启动失败不影响核心功能
        }
        
        // 启动工作流程管理器
        if (workflow_manager_) {
            workflow_manager_->startWorkflow();
        }
        
        return true;
        
    } catch (const std::exception& e) {
        handleSystemError("子系统启动异常: " + std::string(e.what()));
        return false;
    }
}

void BambooSystem::stopSubsystems() {
    std::cout << "[BambooSystem] 停止子系统..." << std::endl;
    
    // 停止工作流程管理器
    if (workflow_manager_) {
        workflow_manager_->stopWorkflow();
        workflow_manager_.reset();
    }
    
    // 停止Modbus通信
    if (modbus_interface_) {
        modbus_interface_->stop();
        modbus_interface_.reset();
    }
    
    // 停止LVGL界面
    if (ui_interface_) {
        ui_interface_->stop();
        ui_interface_.reset();
    }
    
    // 停止推理线程
    if (inference_thread_) {
        inference_thread_->stop();
        inference_thread_.reset();
    }
}

void BambooSystem::updateSystemStats() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_stats_update_);
    
    if (duration.count() >= config_.system_params.stats_update_interval_ms) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        
        // 获取数据桥接中的统计信息
        auto stats = data_bridge_->getStats();
        
        // 更新系统性能统计
        system_info_.performance.inference_fps = stats.inference_fps;
        system_info_.performance.ui_fps = stats.camera_fps;  // 暂时用摄像头帧率代替UI帧率
        system_info_.performance.total_detections = stats.total_detections;
        
        // 更新工作流程步骤
        system_info_.current_workflow_step = data_bridge_->getCurrentStep();
        
        // 计算系统效率（简化版本）
        if (performance_monitor_.frame_count > 0) {
            system_info_.performance.system_efficiency = 
                (float)performance_monitor_.detection_count / performance_monitor_.frame_count * 100.0f;
        }
        
        // 更新系统资源使用情况（简化版本）
        system_info_.cpu_usage = stats.jetson.cpu_usage;
        system_info_.memory_usage = stats.jetson.memory_usage;
        
        last_stats_update_ = now;
    }
}

void BambooSystem::monitorSystemHealth() {
    // 检查各子系统健康状态
    bool all_systems_healthy = true;
    
    if (inference_thread_ && !inference_thread_->isRunning()) {
        all_systems_healthy = false;
        handleSystemError("推理线程异常停止");
    }
    
    if (ui_interface_ && !ui_interface_->isRunning()) {
        // 界面异常不影响核心功能
        utils::Logger::getInstance().log(utils::LogLevel::WARNING, "界面线程异常");
    }
    
    if (modbus_interface_ && !modbus_interface_->isConnected()) {
        // Modbus连接异常不影响核心功能
        utils::Logger::getInstance().log(utils::LogLevel::WARNING, "Modbus连接异常");
    }
    
    // 检查紧急停止状态
    if (data_bridge_->isEmergencyStop()) {
        if (current_state_ != SystemState::EMERGENCY) {
            changeState(SystemState::EMERGENCY);
        }
    }
}

void BambooSystem::handleSystemError(const std::string& error_msg) {
    last_error_ = error_msg;
    changeState(SystemState::ERROR);
    
    utils::Logger::getInstance().log(utils::LogLevel::ERROR, error_msg);
    std::cerr << "[BambooSystem] 系统错误: " << error_msg << std::endl;
}

void BambooSystem::changeState(SystemState new_state) {
    SystemState old_state = current_state_;
    current_state_ = new_state;
    
    std::cout << "[BambooSystem] 状态变更: " << systemStateToString(old_state) 
              << " -> " << systemStateToString(new_state) << std::endl;
              
    utils::Logger::getInstance().log(utils::LogLevel::INFO, 
        "状态变更: " + systemStateToString(old_state) + " -> " + systemStateToString(new_state));
}

void BambooSystem::setupSignalHandlers() {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    signal(SIGUSR1, signalHandler);
    signal(SIGUSR2, signalHandler);
    signal(SIGPIPE, SIG_IGN);
}

void BambooSystem::signalHandler(int signal) {
    if (instance_) {
        switch (signal) {
            case SIGINT:
            case SIGTERM:
                instance_->stop();
                break;
            case SIGUSR1:
                instance_->reloadConfig(instance_->config_);
                break;
            case SIGUSR2:
                // 输出系统状态
                auto info = instance_->getSystemInfo();
                std::cout << "系统状态: " << info.state_name << std::endl;
                break;
        }
    }
}

// 系统配置实现
bool SystemConfig::loadFromFile(const std::string& config_file) {
    try {
        utils::ConfigLoader loader;
        if (!loader.loadFromFile(config_file)) {
            return false;
        }
        
        // TODO: 解析配置文件并设置参数
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool SystemConfig::saveToFile(const std::string& config_file) const {
    try {
        // TODO: 保存配置到文件
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

// WorkflowManager实现
WorkflowManager::WorkflowManager(std::shared_ptr<DataBridge> data_bridge)
    : data_bridge_(data_bridge)
    , current_step_(1)
    , step_timeout_ms_(30000) {
}

WorkflowManager::~WorkflowManager() {
    stopWorkflow();
}

void WorkflowManager::startWorkflow() {
    if (workflow_running_.load()) return;
    
    should_stop_ = false;
    workflow_running_ = true;
    workflow_thread_ = std::thread(&WorkflowManager::workflowLoop, this);
    
    std::cout << "[WorkflowManager] 工作流程已启动" << std::endl;
}

void WorkflowManager::stopWorkflow() {
    if (!workflow_running_.load()) return;
    
    should_stop_ = true;
    workflow_running_ = false;
    
    if (workflow_thread_.joinable()) {
        workflow_thread_.join();
    }
    
    std::cout << "[WorkflowManager] 工作流程已停止" << std::endl;
}

void WorkflowManager::pauseWorkflow() {
    workflow_paused_ = true;
    std::cout << "[WorkflowManager] 工作流程已暂停" << std::endl;
}

void WorkflowManager::resumeWorkflow() {
    workflow_paused_ = false;
    std::cout << "[WorkflowManager] 工作流程已恢复" << std::endl;
}

void WorkflowManager::emergencyStop() {
    should_stop_ = true;
    workflow_running_ = false;
    std::cout << "[WorkflowManager] 工作流程紧急停止" << std::endl;
}

int WorkflowManager::getCurrentStep() const {
    return current_step_;
}

void WorkflowManager::workflowLoop() {
    while (!should_stop_.load()) {
        if (!workflow_paused_.load()) {
            executeWorkflowStep(current_step_);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void WorkflowManager::executeWorkflowStep(int step) {
    data_bridge_->setCurrentStep(step);
    
    // TODO: 实现具体的工作流程步骤
    switch (step) {
        case 1: // 进料检测
            break;
        case 2: // 视觉识别
            break;
        case 3: // 坐标传输
            break;
        case 4: // 切割准备
            break;
        case 5: // 执行切割
            break;
    }
}

bool WorkflowManager::isStepCompleted(int step) {
    // TODO: 检查步骤完成条件
    return false;
}

void WorkflowManager::handleStepTimeout(int step) {
    std::cerr << "[WorkflowManager] 步骤 " << step << " 超时" << std::endl;
}

// 辅助函数实现
std::string systemStateToString(SystemState state) {
    switch (state) {
        case SystemState::UNINITIALIZED: return "未初始化";
        case SystemState::INITIALIZING: return "初始化中";
        case SystemState::READY: return "就绪";
        case SystemState::RUNNING: return "运行中";
        case SystemState::PAUSED: return "暂停";
        case SystemState::ERROR: return "错误";
        case SystemState::EMERGENCY: return "紧急停止";
        case SystemState::SHUTDOWN: return "关闭中";
        default: return "未知状态";
    }
}

VersionInfo getVersionInfo() {
    VersionInfo info;
    info.major = 5;
    info.minor = 0;
    info.patch = 0;
    info.build_date = __DATE__ " " __TIME__;
    info.git_commit = "dev";
    return info;
}

std::string VersionInfo::toString() const {
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

} // namespace core
} // namespace bamboo_cut