/**
 * 主应用程序类实现
 */

#include "app/main_app.h"
#include "app/event_manager.h"
#include "common/utils.h"
#include "common/types.h"
#include "gui/status_bar.h"
#include "gui/video_view.h"
#include "gui/control_panel.h"
#include "gui/settings_page.h"
#include "input/touch_driver.h"
#include "backend/backend_client.h"
#include "backend/tcp_socket_client.h"
#include "display/lvgl_display.h"
#include <stdio.h>
#include <string.h>
// 其他头文件暂时注释掉，避免不完整类型问题
// #include "app/config_manager.h"
// #include "camera/camera_manager.h"
// #include "ai/yolo_detector.h"

// 全局变量：视频视图组件引用（供外部访问）
Video_view* g_video_view_component = nullptr;

MainApp::MainApp(const system_config_t& config)
    : config_(config)
    , running_(false)
    , initialized_(false)
    , camera_manager_(nullptr)
    , backend_client_(nullptr)
{
}

MainApp::~MainApp() {
    stop();
    
    // 清理TCP Socket客户端
    if (tcp_socket_client_) {
        tcp_socket_client_->disconnect();
        tcp_socket_client_.reset();
    }
    
    // 清理后端客户端
    if (backend_client_) {
        backend_client_destroy(backend_client_);
        backend_client_ = nullptr;
    }
    
    // 清理摄像头管理器
    if (camera_manager_) {
        camera_manager_destroy(camera_manager_);
        camera_manager_ = nullptr;
    }
}

bool MainApp::initialize() {
    printf("初始化主应用程序（独立模式）...\n");
    
    try {
        // 创建事件管理器
        event_manager_ = std::make_unique<EventManager>();
        if (!event_manager_->initialize()) {
            printf("错误: 事件管理器初始化失败\n");
            return false;
        }
        
        // 初始化GUI界面（优先级最高，确保界面能显示）
        printf("初始化GUI界面...\n");
        setup_gui();
        
        // 初始化触摸输入（非阻塞）
        printf("初始化触摸输入...\n");
        setup_touch_input();
        
        // 非阻塞初始化后端通信（降级方案：失败不影响启动）
        printf("初始化后端通信（非阻塞模式）...\n");
        try {
            setup_backend_communication();
        } catch (const std::exception& e) {
            printf("警告: 后端通信初始化失败，前端将在离线模式下运行: %s\n", e.what());
        }
        
        // 非阻塞初始化摄像头（降级方案：失败不影响启动）
        printf("初始化摄像头系统（非阻塞模式）...\n");
        try {
            setup_camera();
        } catch (const std::exception& e) {
            printf("警告: 摄像头初始化失败，前端将在无视频模式下运行: %s\n", e.what());
        }
        
        // TODO: 初始化AI检测器（非阻塞）
        // setup_ai_detector();
        
        initialized_ = true;
        printf("主应用程序初始化成功（独立模式）\n");
        printf("前端已进入独立运行模式，可脱离后端工作\n");
        return true;
        
    } catch (const std::exception& e) {
        printf("错误: 主应用程序初始化异常: %s\n", e.what());
        return false;
    }
}

bool MainApp::start() {
    if (!initialized_) {
        printf("错误: 应用程序未初始化\n");
        return false;
    }
    
    printf("启动主应用程序...\n");
    running_ = true;
    
    // 启动TCP Socket客户端
    if (tcp_socket_client_) {
        if (!tcp_socket_client_->connect()) {
            printf("警告: TCP Socket客户端连接失败\n");
        }
    }
    
    // 启动后端通信（向后兼容）
    if (backend_client_) {
        if (!backend_client_start_communication(backend_client_)) {
            printf("警告: 后端通信启动失败\n");
        }
    }
    
    printf("主应用程序启动成功\n");
    return true;
}

void MainApp::stop() {
    if (!running_) {
        return;
    }
    
    printf("停止主应用程序...\n");
    running_ = false;
    
    // 停止TCP Socket客户端
    if (tcp_socket_client_) {
        tcp_socket_client_->disconnect();
    }
    
    // 停止后端通信（向后兼容）
    if (backend_client_) {
        backend_client_stop_communication(backend_client_);
    }
    
    // 停止摄像头组件
    if (camera_manager_) {
        camera_manager_stop_capture(camera_manager_);
        camera_manager_deinit(camera_manager_);
    }
    
    // 清理触摸驱动
    touch_driver_deinit();
    
    printf("主应用程序已停止\n");
}

void MainApp::process_events() {
    if (!running_ || !event_manager_) {
        return;
    }
    
    // 处理事件队列
    event_manager_->process_events();
}

void MainApp::setup_gui() {
    printf("设置LVGL GUI界面 with Grid/Flex layout...\n");
    
    // 创建GUI组件实例
    #include "gui/status_bar.h"
    #include "gui/video_view.h"
    #include "gui/control_panel.h"
    #include "gui/settings_page.h"
    
    // 创建各个GUI组件
    static Status_bar status_bar;
    static Video_view video_view;
    static Control_panel control_panel;
    static Settings_page main_layout;
    
    // 初始化各个组件
    if (!status_bar.initialize()) {
        printf("错误: 状态栏初始化失败\n");
        return;
    }
    
    if (!video_view.initialize()) {
        printf("错误: 视频视图初始化失败\n");
        return;
    }
    
    // 保存视频视图组件的全局引用，用于摄像头数据桥接
    g_video_view_component = &video_view;
    
    if (!control_panel.initialize()) {
        printf("错误: 控制面板初始化失败\n");
        return;
    }
    
    if (!main_layout.initialize()) {
        printf("错误: 主布局管理器初始化失败\n");
        return;
    }
    
    // 创建主布局整合所有组件
    main_layout.create_main_layout(&status_bar, &video_view, &control_panel);
    
    // 测试数据更新
    status_bar.update_workflow_status(1);
    status_bar.update_heartbeat(12345, 12);
    
    video_view.update_detection_info(28.5f, 15.3f);
    video_view.update_coordinate_display(245.8f, "正常", "双刀片");
    video_view.update_cutting_position(245.8f, 1000.0f);
    
    performance_stats_t test_stats = {0};
    test_stats.cpu_usage = 45.0f;
    test_stats.gpu_usage = 32.0f;
    test_stats.memory_usage_mb = 2150.0f; // 2.1GB in MB
    control_panel.update_jetson_info(test_stats);
    
    control_panel.update_ai_model_status(15.3f, 94.2f, 15432, 89);
    control_panel.update_communication_stats("2h 15m", 15432, 0.02f, "1.2KB/s");
    
    printf("LVGL GUI界面设置完成 - 基于Grid和Flex布局\n");
}

void MainApp::setup_camera() {
    printf("设置摄像头系统（与LVGL显示驱动整合）...\n");
    
    // 重要：不在这里创建摄像头管理器，因为 lvgl_display.cpp 中已经创建了
    // 这样避免重复创建导致的资源冲突和性能问题
    // 摄像头管理器由 lvgl_display.cpp 的 init_camera_system() 函数负责创建和管理
    
    printf("摄像头系统将由LVGL显示驱动管理，避免重复创建\n");
    printf("视频流将通过 update_camera_display() 函数自动更新到GUI\n");
    
    // 摄像头管理器将在 lvgl_display_init() 中初始化，这样确保：
    // 1. 避免重复的摄像头管理器实例
    // 2. 确保摄像头与LVGL显示正确集成
    // 3. 提高系统性能和稳定性
    
    camera_manager_ = nullptr; // 确保不创建重复实例
    
    printf("摄像头系统设置完成（委托给LVGL显示驱动管理）\n");
}

void MainApp::setup_ai_detector() {
    // TODO: 设置AI检测器
}

void MainApp::setup_touch_input() {
    printf("设置LVGL触摸输入设备...\n");
    
    // 检查触摸驱动是否已经初始化（避免重复初始化）
    if (!touch_driver_is_available()) {
        printf("警告: 触摸驱动未初始化，将在无触摸模式下运行\n");
        return;
    }
    
    // 创建LVGL输入设备
    static lv_indev_drv_t indev_drv;
    lv_indev_drv_init(&indev_drv);
    indev_drv.type = LV_INDEV_TYPE_POINTER;
    indev_drv.read_cb = touch_driver_read;
    
    lv_indev_t* touch_indev = lv_indev_drv_register(&indev_drv);
    if (!touch_indev) {
        printf("错误: 无法注册LVGL触摸输入设备\n");
        return;
    }
    
    // 打印触摸设备信息用于调试
    touch_device_info_t* device_info = touch_device_get_info();
    if (device_info) {
        printf("LVGL触摸设备已注册:\n");
        printf("  设备路径: %s\n", device_info->device_path);
        printf("  设备名称: %s\n", device_info->device_name);
        printf("  多点触控: %s\n", device_info->is_multitouch ? "支持" : "不支持");
    }
    
    printf("LVGL触摸输入设备设置完成\n");
}

void MainApp::setup_backend_communication() {
    printf("设置后端通信（非阻塞独立模式）...\n");
    
    try {
        // 创建TCP Socket客户端连接
        const char* server_host = getenv("BAMBOO_BACKEND_HOST") ? getenv("BAMBOO_BACKEND_HOST") : "127.0.0.1";
        const char* server_port_str = getenv("BAMBOO_BACKEND_PORT") ? getenv("BAMBOO_BACKEND_PORT") : "8888";
        int server_port = atoi(server_port_str);
        
        printf("尝试连接TCP Socket后端: %s:%d（非阻塞）\n", server_host, server_port);
        
        // 创建TCP Socket客户端（非阻塞模式）
        tcp_socket_client_ = std::make_unique<TcpSocketClient>(server_host, server_port);
        
        // 设置连接回调（优雅处理连接状态）
        tcp_socket_client_->set_connection_callback([this](ConnectionStatus status) {
            switch (status) {
                case ConnectionStatus::CONNECTED:
                    printf("前端已连接到后端服务器\n");
                    break;
                case ConnectionStatus::DISCONNECTED:
                    printf("前端与后端服务器连接断开（继续独立运行）\n");
                    break;
                case ConnectionStatus::CONNECTING:
                    printf("正在连接后端服务器...\n");
                    break;
                case ConnectionStatus::RECONNECTING:
                    printf("正在重新连接后端服务器...\n");
                    break;
                case ConnectionStatus::CONNECTION_ERROR:
                    printf("后端服务器连接错误（前端继续独立运行）\n");
                    break;
            }
        });
        
        // 设置消息回调
        tcp_socket_client_->set_message_callback([this](const CommunicationMessage& message) {
            printf("收到后端消息\n");
            // TODO: 解析消息并更新UI
        });
        
        // 启用自动重连（后台重试，不阻塞前端）
        tcp_socket_client_->enable_auto_reconnect(true);
        tcp_socket_client_->set_reconnect_interval(5); // 5秒间隔重连
        
        // 非阻塞连接启动（失败不影响前端启动）
        if (!tcp_socket_client_->connect()) {
            printf("警告: TCP Socket客户端初始连接失败\n");
            printf("前端将以独立模式启动，后台将继续尝试连接后端\n");
        } else {
            printf("TCP Socket客户端初始连接成功\n");
        }
        
        printf("后端通信模块启动成功（独立模式，支持后台重连）\n");
        
    } catch (const std::exception& e) {
        printf("警告: 后端通信初始化异常: %s\n", e.what());
        printf("前端将以完全独立模式运行（无后端连接）\n");
        tcp_socket_client_.reset();
    }
    
    // 为了向后兼容，保持backend_client_为nullptr
    backend_client_ = nullptr;
    
    printf("后端通信设置完成（前端可独立工作）\n");
}