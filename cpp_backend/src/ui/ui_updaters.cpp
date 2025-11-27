#include "bamboo_cut/ui/ui_updaters.h"
#include "bamboo_cut/ui/lvgl_ui_utils.h"
#include "bamboo_cut/core/data_bridge.h"
#include "bamboo_cut/utils/jetson_monitor.h"
#include <sstream>
#include <iomanip>

namespace bamboo_cut {
namespace ui {

void update_ui_data(UIContext& ctx,
                    const std::shared_ptr<bamboo_cut::core::DataBridge>& data_bridge,
                    const std::shared_ptr<bamboo_cut::utils::JetsonMonitor>& monitor,
                    int& frame_count,
                    std::chrono::steady_clock::time_point& last_update) {
    auto now = std::chrono::steady_clock::now();
    frame_count++;
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
    if (elapsed >= 1000) {
        float ui_fps = (frame_count * 1000.0f) / elapsed;
        frame_count = 0;
        last_update = now;

        if (ctx.widgets.ui_fps_label) {
            std::ostringstream fps;
            fps << LV_SYMBOL_EYE_OPEN << " UI: " << std::fixed << std::setprecision(1) << ui_fps << " fps";
            lv_label_set_text(ctx.widgets.ui_fps_label, fps.str().c_str());
        }
    }

    updateJetsonMonitoring(ctx.widgets, monitor, ui::LVGLThemeColors());
    updateAIModelStats(ctx.widgets, data_bridge);
    updateCameraStatus(ctx.widgets, data_bridge);
    updateModbusStatus(ctx.widgets, data_bridge);
}

} // namespace ui
} // namespace bamboo_cut
