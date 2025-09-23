/**
 * 主应用程序类实现
 */

#include "app/main_app.h"
#include "app/event_manager.h"
#include "common/utils.h"
#include "gui/status_bar.h"
#include "gui/video_view.h"
#include "gui/control_panel.h"
#include "gui/settings_page.h"
#include "input/touch_driver.h"
#include "backend/backend_client.h"
#include <stdio.h>
#include <string.h>
// 其他头文件暂时注释掉，避免不完整类型问题
// #include "app/config_manager.h"
// #include "camera/camera_manager.h"
// #include "ai/yolo_detector.h"

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
    printf("初始化主应用程序...\n");
    
    try {
        // 创建事件管理器
        event_manager_ = std::make_unique<EventManager>();
        if (!event_manager_->initialize()) {
            printf("错误: 事件管理器初始化失败\n");
            return false;
        }
        
        // 初始化GUI界面
        setup_gui();
        
        // 初始化触摸输入
        setup_touch_input();
        
        // 初始化后端通信
        setup_backend_communication();
        
        // 初始化摄像头组件
        setup_camera();
        
        // TODO: 初始化AI检测器
        // setup_ai_detector();
        
        initialized_ = true;
        printf("主应用程序初始化成功\n");
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
    
    // 启动后端通信
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
    
    // 停止后端通信
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
    printf("设置异步摄像头管理器（完全非阻塞模式）...\n");
    
    // 使用配置中的显示分辨率创建摄像头管理器
    const camera_config_t& camera_config = config_.camera;
    
    // 如果显示分辨率没有配置，使用摄像头原始分辨率
    int display_width = camera_config.display_width > 0 ? camera_config.display_width : camera_config.width;
    int display_height = camera_config.display_height > 0 ? camera_config.display_height : camera_config.height;
    
    printf("摄像头配置: 源分辨率 %dx%d, 显示分辨率 %dx%d\n",
           camera_config.width, camera_config.height, display_width, display_height);
    
    // 创建摄像头管理器，使用GStreamer流参数
    std::string stream_url = "udp://127.0.0.1:5000";  // 默认流URL
    std::string stream_format = "H264";               // 默认流格式
    
    camera_manager_ = camera_manager_create(
        stream_url.c_str(),
        stream_format.c_str(),
        display_width,
        display_height,
        camera_config.fps
    );
    
    if (!camera_manager_) {
        printf("警告: 创建摄像头管理器失败，界面将在无摄像头模式下运行\n");
        return;
    }
    
    // 完全异步初始化：不等待任何结果，立即启动后台线程
    printf("启动完全异步摄像头管理器...\n");
    printf("UI将立即渲染，摄像头数据将在后台异步获取\n");
    
    // 直接初始化（无阻塞）
    camera_manager_init(camera_manager_);
    
    // 直接启动捕获线程（无阻塞）
    camera_manager_start_capture(camera_manager_);
    
    printf("异步摄像头管理器启动完成 - UI渲染不受影响\n");
    printf("摄像头状态: 后台连接中，有数据时自动显示，无数据时显示黑屏\n");
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
    printf("设置后端通信（非阻塞模式）...\n");
    
    // 创建后端客户端，使用UNIX Domain Socket路径（与后端保持一致）
    const char* socket_path = "/tmp/bamboo_socket";
    backend_client_ = backend_client_create(socket_path);
    
    if (!backend_client_) {
        printf("警告: 创建后端客户端失败，前端将在无后端模式下运行\n");
        return;
    }
    
    // 非阻塞尝试连接到后端，失败也不影响前端启动
    printf("尝试连接后端...\n");
    if (!backend_client_connect(backend_client_)) {
        printf("警告: 连接后端失败，前端将继续启动并在后台自动重试连接\n");
    } else {
        printf("后端连接成功\n");
    }
    
    printf("后端通信设置完成（前端可独立运行）\n");
}