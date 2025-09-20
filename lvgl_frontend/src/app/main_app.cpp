/**
 * 主应用程序类实现
 */

#include "app/main_app.h"
#include "app/event_manager.h"
#include "app/config_manager.h"
#include "camera/camera_manager.h"
#include "ai/yolo_detector.h"
#include "gui/video_view.h"
#include "input/touch_driver.h"
#include "common/utils.h"
#include <stdio.h>

MainApp::MainApp(const system_config_t& config)
    : config_(config)
    , running_(false)
    , initialized_(false)
{
}

MainApp::~MainApp() {
    stop();
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
        
        // TODO: 初始化其他组件
        // setup_camera();
        // setup_ai_detector();
        // setup_touch_input();
        
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
    
    // TODO: 启动各个组件
    
    printf("主应用程序启动成功\n");
    return true;
}

void MainApp::stop() {
    if (!running_) {
        return;
    }
    
    printf("停止主应用程序...\n");
    running_ = false;
    
    // TODO: 停止各个组件
    
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
    printf("设置LVGL GUI界面...\n");
    
    // GUI界面已经在lvgl_display_init()中的create_test_ui()函数中创建
    // 这里可以添加额外的GUI组件或事件处理
    
    printf("LVGL GUI界面设置完成\n");
}

void MainApp::setup_camera() {
    // TODO: 设置摄像头
}

void MainApp::setup_ai_detector() {
    // TODO: 设置AI检测器
}

void MainApp::setup_touch_input() {
    // TODO: 设置触摸输入
}