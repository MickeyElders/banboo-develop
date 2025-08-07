#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include "bamboo_cut/communication/modbus_server.h"

using namespace bamboo_cut::communication;

// 全局变量用于信号处理
std::unique_ptr<ModbusServer> g_server;
std::atomic<bool> g_running{true};

// 信号处理函数
void signal_handler(int signal) {
    std::cout << "\n收到信号 " << signal << "，正在停止服务器..." << std::endl;
    g_running.store(false);
    if (g_server) {
        g_server->stop();
    }
}

// 模拟视觉识别系统
class MockVisionSystem {
public:
    MockVisionSystem(ModbusServer* server) : server_(server) {
        vision_thread_ = std::thread(&MockVisionSystem::vision_loop, this);
    }
    
    ~MockVisionSystem() {
        running_.store(false);
        if (vision_thread_.joinable()) {
            vision_thread_.join();
        }
    }
    
private:
    void vision_loop() {
        std::cout << "视觉识别系统启动" << std::endl;
        
        int cycle_count = 0;
        while (running_.load() && g_running.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            
            // 模拟检测到竹子
            if (server_->get_system_status() == SystemStatus::RUNNING) {
                cycle_count++;
                
                // 模拟识别到竹子坐标
                int32_t x_coordinate = 1000 + (cycle_count % 5) * 100; // 1000, 1100, 1200, 1300, 1400
                BladeNumber blade = (cycle_count % 2) ? BladeNumber::BLADE_1 : BladeNumber::BLADE_2;
                
                CoordinateData coord_data(x_coordinate, blade, CutQuality::NORMAL);
                server_->set_coordinate_data(coord_data);
                
                std::cout << "🔍 视觉检测: X=" << x_coordinate 
                          << "mm, 刀片=" << static_cast<int>(blade) << std::endl;
            }
        }
        
        std::cout << "视觉识别系统停止" << std::endl;
    }
    
    ModbusServer* server_;
    std::atomic<bool> running_{true};
    std::thread vision_thread_;
};

// 事件回调函数
void on_connection_changed(bool connected, const std::string& client_ip) {
    if (connected) {
        std::cout << "✅ PLC连接: " << client_ip << std::endl;
    } else {
        std::cout << "❌ PLC断开: " << client_ip << std::endl;
    }
}

void on_plc_command(PLCCommand command) {
    std::string cmd_name;
    switch (command) {
        case PLCCommand::FEED_DETECTION:
            cmd_name = "进料检测";
            break;
        case PLCCommand::CUT_PREPARE:
            cmd_name = "切割准备";
            break;
        case PLCCommand::CUT_COMPLETE:
            cmd_name = "切割完成";
            break;
        case PLCCommand::START_FEEDING:
            cmd_name = "启动送料";
            break;
        case PLCCommand::PAUSE:
            cmd_name = "暂停";
            break;
        case PLCCommand::EMERGENCY_STOP:
            cmd_name = "紧急停止";
            break;
        case PLCCommand::RESUME:
            cmd_name = "恢复运行";
            break;
        default:
            cmd_name = "未知命令";
            break;
    }
    
    std::cout << "📨 PLC命令: " << cmd_name << " (" << static_cast<int>(command) << ")" << std::endl;
}

void on_emergency_stop() {
    std::cout << "🚨 紧急停止触发！" << std::endl;
}

void on_timeout(const std::string& timeout_type) {
    std::cout << "⏰ 超时: " << timeout_type << std::endl;
}

// 状态监控线程
void status_monitor_thread(ModbusServer* server) {
    std::cout << "状态监控启动" << std::endl;
    
    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        if (!server->is_running()) {
            continue;
        }
        
        auto stats = server->get_statistics();
        std::cout << "\n📊 系统状态报告:" << std::endl;
        std::cout << "  系统状态: ";
        
        switch (server->get_system_status()) {
            case SystemStatus::STOPPED:
                std::cout << "停止"; break;
            case SystemStatus::RUNNING:
                std::cout << "运行"; break;
            case SystemStatus::ERROR:
                std::cout << "错误"; break;
            case SystemStatus::PAUSED:
                std::cout << "暂停"; break;
            case SystemStatus::EMERGENCY_STOP:
                std::cout << "紧急停止"; break;
            case SystemStatus::MAINTENANCE:
                std::cout << "维护"; break;
        }
        std::cout << std::endl;
        
        std::cout << "  PLC连接: " << (server->is_connected() ? "已连接" : "未连接") << std::endl;
        std::cout << "  心跳状态: " << (server->is_heartbeat_active() ? "正常" : "异常") << std::endl;
        std::cout << "  心跳计数: " << server->get_heartbeat_counter() << std::endl;
        std::cout << "  总连接数: " << stats.total_connections << std::endl;
        std::cout << "  总请求数: " << stats.total_requests << std::endl;
        std::cout << "  错误次数: " << stats.total_errors << std::endl;
        std::cout << "  心跳超时: " << stats.heartbeat_timeouts << std::endl;
        
        auto coord = server->get_coordinate_data();
        if (server->get_system_status() == SystemStatus::RUNNING && coord.x_coordinate != 0) {
            std::cout << "  当前坐标: X=" << coord.x_coordinate 
                      << "mm, 刀片=" << static_cast<int>(coord.blade_number) << std::endl;
        }
        
        std::string last_error = server->get_last_error();
        if (!last_error.empty()) {
            std::cout << "  最后错误: " << last_error << std::endl;
        }
        std::cout << std::endl;
    }
    
    std::cout << "状态监控停止" << std::endl;
}

int main() {
    std::cout << "=== 竹子切割机 - Modbus通信服务器示例 ===" << std::endl;
    
    // 设置信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    try {
        // 配置Modbus服务器
        ModbusConfig config;
        config.ip_address = "0.0.0.0";      // 监听所有网络接口
        config.port = 502;                   // 标准Modbus TCP端口
        config.max_connections = 10;         // 最大连接数
        config.response_timeout_ms = 1000;   // 响应超时
        config.heartbeat_interval_ms = 20;   // 心跳间隔20ms
        
        // 超时设置
        config.feed_detection_timeout_s = 15;   // 进料检测超时15秒
        config.clamp_timeout_s = 60;             // 夹持超时60秒
        config.cut_execution_timeout_s = 120;    // 切割执行超时120秒
        config.emergency_response_timeout_ms = 100; // 紧急响应100ms
        
        std::cout << "📝 配置信息:" << std::endl;
        std::cout << "  监听地址: " << config.ip_address << ":" << config.port << std::endl;
        std::cout << "  最大连接: " << config.max_connections << std::endl;
        std::cout << "  心跳间隔: " << config.heartbeat_interval_ms << "ms" << std::endl;
        std::cout << std::endl;
        
        // 创建Modbus服务器
        g_server = std::make_unique<ModbusServer>(config);
        
        // 设置事件回调
        g_server->set_connection_callback(on_connection_changed);
        g_server->set_command_callback(on_plc_command);
        g_server->set_emergency_stop_callback(on_emergency_stop);
        g_server->set_timeout_callback(on_timeout);
        
        // 启动服务器
        if (!g_server->start()) {
            std::cerr << "❌ 启动Modbus服务器失败" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Modbus服务器启动成功" << std::endl;
        
        // 设置初始系统状态
        g_server->set_system_status(SystemStatus::RUNNING);
        g_server->set_system_health(SystemHealth::NORMAL);
        
        std::cout << "🎯 系统状态设置为运行中" << std::endl;
        
        // 启动模拟视觉系统
        std::cout << "🚀 启动模拟视觉识别系统..." << std::endl;
        MockVisionSystem vision_system(g_server.get());
        
        // 启动状态监控线程
        std::thread status_thread(status_monitor_thread, g_server.get());
        
        std::cout << "\n💡 提示: " << std::endl;
        std::cout << "  - 使用Modbus客户端连接到 " << config.ip_address << ":" << config.port << std::endl;
        std::cout << "  - 寄存器地址范围: 40001-40008" << std::endl;
        std::cout << "  - 使用 Ctrl+C 停止服务器" << std::endl;
        std::cout << "  - 系统会每3秒模拟检测到竹子坐标" << std::endl;
        std::cout << "\n📋 寄存器映射:" << std::endl;
        std::cout << "  40001: 系统状态 (0=停止, 1=运行, 2=错误, 3=暂停, 4=紧急停止, 5=维护)" << std::endl;
        std::cout << "  40002: PLC命令 (1=进料检测, 2=切割准备, 3=切割完成, 4=启动送料, 5=暂停, 6=紧急停止, 7=恢复)" << std::endl;
        std::cout << "  40003: 坐标就绪标志 (0=无坐标, 1=有坐标)" << std::endl;
        std::cout << "  40004-40005: X坐标 (32位, 0.1mm精度)" << std::endl;
        std::cout << "  40006: 切割质量 (0=正常, 1=异常)" << std::endl;
        std::cout << "  40007-40008: 心跳计数器 (32位)" << std::endl;
        std::cout << "  40009: 刀片编号 (0=无, 1=刀片1, 2=刀片2, 3=双刀片)" << std::endl;
        std::cout << "  40010: 系统健康状态 (0=正常, 1=警告, 2=错误, 3=严重错误)" << std::endl;
        std::cout << "\n🔄 系统正在运行，等待PLC连接..." << std::endl;
        
        // 主循环
        while (g_running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // 检查紧急停止状态
            if (g_server->is_emergency_stopped()) {
                std::cout << "⚠️  系统处于紧急停止状态" << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        
        // 等待状态监控线程结束
        if (status_thread.joinable()) {
            status_thread.join();
        }
        
        std::cout << "🔄 正在停止服务器..." << std::endl;
        g_server->stop();
        
        std::cout << "✅ 服务器已停止" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 程序异常: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "👋 程序退出" << std::endl;
    return 0;
}