#pragma once

#include <string>
#include <atomic>

// 轻量占位实现：提供接口以便无 UI/Wayland 环境可编译运行。
// 如需真 LVGL/Wayland 显示，请替换为实际实现。

// 避免与系统 lvgl 冲突，使用与 lvgl 相同的别名声明。
typedef struct _lv_obj_t lv_obj_t;

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
    bool isFullyInitialized() const { return fully_initialized_.load(); }

    // 返回 LVGL 画布对象；占位实现返回 nullptr。
    lv_obj_t* getCameraCanvas() { return nullptr; }

private:
    LVGLWaylandConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> fully_initialized_{false};
};

} // namespace ui
} // namespace bamboo_cut
