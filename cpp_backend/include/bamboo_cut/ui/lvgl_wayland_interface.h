#pragma once

#include <string>
#include <atomic>

// 轻量占位实现：提供接口以便无 UI/Wayland 环境可编译运行。
// 如需真 LVGL/Wayland 显示，请替换为实际实现。

struct lv_obj_t; // 前向声明，避免依赖 LVGL 头

namespace bamboo_cut {
namespace ui {

struct LVGLWaylandConfig {
    int screen_width{1280};
    int screen_height{800};
    int refresh_rate{60};
    bool enable_touch{true};
    std::string touch_device{"/dev/input/event0"};
    std::string wayland_display{"wayland-0"};
    bool fullscreen{true};
};

class LVGLWaylandInterface {
public:
    LVGLWaylandInterface() = default;
    ~LVGLWaylandInterface() = default;

    bool initialize(const LVGLWaylandConfig& config);
    bool start();
    void stop();
    bool isRunning() const { return running_.load(); }

    // 返回 LVGL 画布对象；占位实现返回 nullptr。
    lv_obj_t* getCameraCanvas() { return nullptr; }

private:
    LVGLWaylandConfig config_;
    std::atomic<bool> running_{false};
};

} // namespace ui
} // namespace bamboo_cut
