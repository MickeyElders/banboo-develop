#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include "bamboo_cut/communication/modbus_server.h"

extern "C" {
    #include <modbus/modbus.h>
}

using namespace bamboo_cut::communication;

class ModbusServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 使用测试端口避免冲突
        config_.port = 15020;
        config_.ip_address = "127.0.0.1";
        config_.heartbeat_interval_ms = 100; // 加快测试速度
        
        server_ = std::make_unique<ModbusServer>(config_);
    }
    
    void TearDown() override {
        if (server_->is_running()) {
            server_->stop();
        }
        server_.reset();
    }
    
    // 创建测试客户端
    modbus_t* create_test_client() {
        modbus_t* ctx = modbus_new_tcp("127.0.0.1", config_.port);
        if (ctx) {
            modbus_set_response_timeout(ctx, 1, 0); // 1秒超时
        }
        return ctx;
    }
    
    ModbusConfig config_;
    std::unique_ptr<ModbusServer> server_;
};

// 测试服务器启动和停止
TEST_F(ModbusServerTest, StartStopServer) {
    EXPECT_FALSE(server_->is_running());
    
    // 启动服务器
    EXPECT_TRUE(server_->start());
    EXPECT_TRUE(server_->is_running());
    
    // 重复启动应该失败
    EXPECT_FALSE(server_->start());
    
    // 停止服务器
    server_->stop();
    EXPECT_FALSE(server_->is_running());
}

// 测试客户端连接
TEST_F(ModbusServerTest, ClientConnection) {
    ASSERT_TRUE(server_->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    bool connection_received = false;
    std::string connected_ip;
    
    server_->set_connection_callback([&](bool connected, const std::string& ip) {
        connection_received = true;
        connected_ip = ip;
        EXPECT_TRUE(connected);
    });
    
    // 创建客户端连接
    modbus_t* client = create_test_client();
    ASSERT_NE(client, nullptr);
    
    int result = modbus_connect(client);
    EXPECT_EQ(result, 0);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_TRUE(connection_received);
    EXPECT_EQ(connected_ip, "127.0.0.1");
    
    modbus_close(client);
    modbus_free(client);
}

// 测试系统状态寄存器读写
TEST_F(ModbusServerTest, SystemStatusRegisters) {
    ASSERT_TRUE(server_->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 设置系统状态
    server_->set_system_status(SystemStatus::RUNNING);
    server_->set_system_health(SystemHealth::WARNING);
    
    // 创建客户端读取寄存器
    modbus_t* client = create_test_client();
    ASSERT_NE(client, nullptr);
    ASSERT_EQ(modbus_connect(client), 0);
    
    uint16_t registers[10];
    
    // 读取系统状态寄存器 (40001)
    int result = modbus_read_holding_registers(client, 40000, 3, registers);
    EXPECT_EQ(result, 3);
    
    // 验证系统状态
    EXPECT_EQ(registers[0], static_cast<uint16_t>(SystemStatus::RUNNING));
    
    modbus_close(client);
    modbus_free(client);
}

// 测试坐标数据推送
TEST_F(ModbusServerTest, CoordinateDataPush) {
    ASSERT_TRUE(server_->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 设置坐标数据
    CoordinateData coord_data(1500, BladeNumber::BLADE_1, CutQuality::NORMAL);
    server_->set_coordinate_data(coord_data);
    
    // 创建客户端读取坐标
    modbus_t* client = create_test_client();
    ASSERT_NE(client, nullptr);
    ASSERT_EQ(modbus_connect(client), 0);
    
    uint16_t registers[10];
    
    // 读取坐标相关寄存器 (40003-40007)
    int result = modbus_read_holding_registers(client, 40002, 5, registers);
    EXPECT_EQ(result, 5);
    
    // 验证坐标就绪标志
    EXPECT_EQ(registers[0], 1); // REG_COORD_READY
    
    // 验证X坐标 (32位值，占用2个寄存器)
    uint32_t x_coord = registers[1] | (static_cast<uint32_t>(registers[2]) << 16);
    EXPECT_EQ(x_coord, 1500);
    
    // 验证刀片编号
    EXPECT_EQ(registers[4], static_cast<uint16_t>(BladeNumber::BLADE_1));
    
    modbus_close(client);
    modbus_free(client);
}

// 测试PLC命令处理
TEST_F(ModbusServerTest, PLCCommandProcessing) {
    ASSERT_TRUE(server_->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    bool command_received = false;
    PLCCommand received_command = PLCCommand::NONE;
    
    server_->set_command_callback([&](PLCCommand cmd) {
        command_received = true;
        received_command = cmd;
    });
    
    // 创建客户端发送命令
    modbus_t* client = create_test_client();
    ASSERT_NE(client, nullptr);
    ASSERT_EQ(modbus_connect(client), 0);
    
    // 写入PLC命令寄存器 (40002)
    uint16_t command_value = static_cast<uint16_t>(PLCCommand::FEED_DETECTION);
    int result = modbus_write_register(client, 40001, command_value);
    EXPECT_EQ(result, 1);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // 验证命令回调被触发
    EXPECT_TRUE(command_received);
    EXPECT_EQ(received_command, PLCCommand::FEED_DETECTION);
    
    modbus_close(client);
    modbus_free(client);
}

// 测试心跳机制
TEST_F(ModbusServerTest, HeartbeatMechanism) {
    ASSERT_TRUE(server_->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 创建客户端连接
    modbus_t* client = create_test_client();
    ASSERT_NE(client, nullptr);
    ASSERT_EQ(modbus_connect(client), 0);
    
    // 等待几个心跳周期
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    uint16_t registers[10];
    
    // 读取心跳寄存器 (40006-40007)
    int result = modbus_read_holding_registers(client, 40005, 2, registers);
    EXPECT_EQ(result, 2);
    
    // 验证心跳计数器不为0
    uint32_t heartbeat = registers[0] | (static_cast<uint32_t>(registers[1]) << 16);
    EXPECT_GT(heartbeat, 0);
    
    // 验证心跳活跃状态
    EXPECT_TRUE(server_->is_heartbeat_active());
    
    modbus_close(client);
    modbus_free(client);
}

// 测试紧急停止机制
TEST_F(ModbusServerTest, EmergencyStopMechanism) {
    ASSERT_TRUE(server_->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    bool emergency_triggered = false;
    server_->set_emergency_stop_callback([&]() {
        emergency_triggered = true;
    });
    
    // 触发紧急停止
    server_->trigger_emergency_stop();
    
    EXPECT_TRUE(emergency_triggered);
    EXPECT_TRUE(server_->is_emergency_stopped());
    EXPECT_EQ(server_->get_system_status(), SystemStatus::EMERGENCY_STOP);
    
    // 确认紧急停止
    server_->acknowledge_emergency_stop();
    EXPECT_FALSE(server_->is_emergency_stopped());
    EXPECT_EQ(server_->get_system_status(), SystemStatus::STOPPED);
}

// 测试超时管理
TEST_F(ModbusServerTest, TimeoutManagement) {
    // 使用较短的超时时间进行测试
    config_.feed_detection_timeout_s = 1;
    server_ = std::make_unique<ModbusServer>(config_);
    
    ASSERT_TRUE(server_->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    bool timeout_received = false;
    std::string timeout_type;
    
    server_->set_timeout_callback([&](const std::string& type) {
        timeout_received = true;
        timeout_type = type;
    });
    
    // 启动进料检测计时器
    server_->reset_feed_detection_timer();
    
    // 等待超时
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    
    EXPECT_TRUE(timeout_received);
    EXPECT_EQ(timeout_type, "feed_detection_timeout");
}

// 测试统计信息
TEST_F(ModbusServerTest, Statistics) {
    ASSERT_TRUE(server_->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 创建客户端连接并发送请求
    modbus_t* client = create_test_client();
    ASSERT_NE(client, nullptr);
    ASSERT_EQ(modbus_connect(client), 0);
    
    uint16_t registers[5];
    modbus_read_holding_registers(client, 40000, 5, registers);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto stats = server_->get_statistics();
    EXPECT_GT(stats.total_connections, 0);
    EXPECT_GT(stats.total_requests, 0);
    
    modbus_close(client);
    modbus_free(client);
}

// 测试错误处理
TEST_F(ModbusServerTest, ErrorHandling) {
    // 使用无效端口测试启动失败
    ModbusConfig invalid_config;
    invalid_config.port = -1; // 无效端口
    
    ModbusServer invalid_server(invalid_config);
    EXPECT_FALSE(invalid_server.start());
    EXPECT_FALSE(invalid_server.is_running());
}

// 性能测试：多客户端并发连接
TEST_F(ModbusServerTest, ConcurrentClients) {
    ASSERT_TRUE(server_->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    const int num_clients = 5;
    std::vector<std::thread> client_threads;
    std::atomic<int> successful_connections{0};
    
    for (int i = 0; i < num_clients; ++i) {
        client_threads.emplace_back([&, i]() {
            modbus_t* client = create_test_client();
            if (client && modbus_connect(client) == 0) {
                uint16_t registers[5];
                if (modbus_read_holding_registers(client, 40000, 5, registers) == 5) {
                    successful_connections++;
                }
                modbus_close(client);
                modbus_free(client);
            }
        });
    }
    
    for (auto& thread : client_threads) {
        thread.join();
    }
    
    EXPECT_EQ(successful_connections.load(), num_clients);
    
    auto stats = server_->get_statistics();
    EXPECT_GE(stats.total_connections, num_clients);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}