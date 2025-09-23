/**
 * 主应用程序类
 */

#ifndef APP_MAIN_APP_H
#define APP_MAIN_APP_H

#include <memory>
#include <atomic>
#include "common/types.h"
#include "camera/camera_manager.h"

class EventManager;
class TcpSocketClient;
// 前向声明后端客户端类型
struct backend_client_t;
// 其他类暂时未实现，使用前向声明避免编译错误

class MainApp {
public:
    explicit MainApp(const system_config_t& config);
    ~MainApp();

    bool initialize();
    bool start();
    void stop();
    void process_events();

    bool is_running() const { return running_; }

private:
    system_config_t config_;
    std::atomic<bool> running_;
    std::atomic<bool> initialized_;

    // 核心组件
    std::unique_ptr<EventManager> event_manager_;
    camera_manager_t* camera_manager_;  // C风格的摄像头管理器
    backend_client_t* backend_client_;  // C风格的后端客户端（向后兼容）
    std::unique_ptr<TcpSocketClient> tcp_socket_client_; // TCP Socket客户端
    // TODO: 其他组件
    // std::unique_ptr<Yolo_detector> ai_detector_;
    // std::unique_ptr<Video_view> video_renderer_;

    void setup_gui();
    void setup_camera();
    void setup_ai_detector();
    void setup_touch_input();
    void setup_backend_communication();
};

#endif // APP_MAIN_APP_H