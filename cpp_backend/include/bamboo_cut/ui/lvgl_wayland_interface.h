#pragma once

#include <atomic>
#include <string>

// Stub LVGL interface to satisfy build; no real LVGL dependency.
namespace bamboo_cut {
namespace ui {

struct LVGLWaylandConfig {
    int screen_width{1280};
    int screen_height{800};
    int refresh_rate{60};
    bool enable_touch{false};
    std::string touch_device{"/dev/input/event0"};
    std::string wayland_display{"wayland-0"};
    bool fullscreen{true};
};

class LVGLWaylandInterface {
public:
    LVGLWaylandInterface() = default;
    ~LVGLWaylandInterface() = default;

    bool initialize(const LVGLWaylandConfig&) { return false; }
    bool start() { running_ = true; return true; }
    void stop() { running_ = false; }
    bool isRunning() const { return running_.load(); }
    bool isFullyInitialized() const { return false; }

    void* getCameraCanvas() { return nullptr; }

private:
    std::atomic<bool> running_{false};
};

} // namespace ui
} // namespace bamboo_cut
