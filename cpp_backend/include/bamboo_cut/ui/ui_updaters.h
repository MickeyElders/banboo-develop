/**
 * @file ui_updaters.h
 * @brief UI 数据更新封装
 */

#pragma once

#include "bamboo_cut/ui/ui_context.h"
#include <chrono>
#include <memory>

namespace bamboo_cut {
namespace core { class DataBridge; }
namespace utils { class JetsonMonitor; }
namespace ui {

void update_ui_data(UIContext& ctx,
                    const std::shared_ptr<bamboo_cut::core::DataBridge>& data_bridge,
                    const std::shared_ptr<bamboo_cut::utils::JetsonMonitor>& monitor,
                    int& frame_count,
                    std::chrono::steady_clock::time_point& last_update);

} // namespace ui
} // namespace bamboo_cut
