/**
 * 主应用程序类
 */

#ifndef APP_MAIN_APP_H
#define APP_MAIN_APP_H

#include <memory>
#include <atomic>
#include "common/types.h"

class ConfigManager;
class EventManager;
class CameraManager;
class AIDetector;
class VideoRenderer;
class TouchController;

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
    std::unique_ptr<CameraManager> camera_manager_;
    std::unique_ptr<AIDetector> ai_detector_;
    std::unique_ptr<VideoRenderer> video_renderer_;
    std::unique_ptr<TouchController> touch_controller_;

    void setup_gui();
    void setup_camera();
    void setup_ai_detector();
    void setup_touch_input();
};

#endif // APP_MAIN_APP_H