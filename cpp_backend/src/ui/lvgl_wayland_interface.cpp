#include "bamboo_cut/ui/lvgl_wayland_interface.h"
#include <iostream>

namespace bamboo_cut {
namespace ui {

bool LVGLWaylandInterface::initialize(const LVGLWaylandConfig& config) {
    config_ = config;
    std::cout << "[LVGLWaylandInterface] initialize stub (no UI) "
              << config.screen_width << "x" << config.screen_height
              << " display=" << config.wayland_display << std::endl;
    fully_initialized_.store(true);
    return true;
}

bool LVGLWaylandInterface::start() {
    running_.store(true);
    std::cout << "[LVGLWaylandInterface] start stub (no UI thread launched)" << std::endl;
    return true;
}

void LVGLWaylandInterface::stop() {
    running_.store(false);
    std::cout << "[LVGLWaylandInterface] stop stub" << std::endl;
}

} // namespace ui
} // namespace bamboo_cut
