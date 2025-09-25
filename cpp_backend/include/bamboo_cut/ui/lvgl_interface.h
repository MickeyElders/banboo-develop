/**
 * @file lvgl_interface.h
 * @brief C++ LVGL一体化系统界面管理器
 * 工业级竹子识别系统LVGL界面实现
 */

#pragma once

#ifdef ENABLE_LVGL
#include <lvgl/lvgl.h>
#else
// LVGL未启用时的占位符定义
typedef void* lv_obj_t;
typedef void* lv_event_t;
typedef void* lv_disp_drv_t;
typedef void* lv_indev_drv_t;
typedef void* lv_disp_t;
typedef void* lv_indev_t;
typedef void* lv_area_t;
typedef void* lv_color_t;
typedef void* lv_disp_draw_buf_t;
typedef void* lv_indev_data_t;
#endif
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include "bamboo_cut/core/data_bridge.h"

namespace bamboo_cut {
namespace ui {

/**
 * @brief LVGL配置参数
 */
struct LVGLConfig {
    int screen_width;           // 屏幕宽度
    int screen_height;          // 屏幕高度
    int refresh_rate;           // 刷新率(Hz)
    std::string touch_device;   // 触摸设备路径
    std::string display_device; // 显示设备路径
    bool enable_touch;          // 启用触摸
    
    LVGLConfig()
        : screen_width(1280)
        , screen_height(800) 
        , refresh_rate(60)
        , touch_device("/dev/input/event0")
        , display_device("/dev/fb0")
        , enable_touch(true) {}
};

/**
 * @brief 工作流程步骤定义
 */
enum class WorkflowStep {
    FEED_DETECTION = 1,     // 进料检测
    VISION_RECOGNITION = 2, // 视觉识别
    COORDINATE_TRANSFER = 3,// 坐标传输
    CUT_PREPARE = 4,        // 切割准备
    EXECUTE_CUT = 5         // 执行切割
};

/**
 * @brief LVGL界面管理器
 * 负责整个工业界面的创建、更新和事件处理
 */
class LVGLInterface {
public:
    explicit LVGLInterface(std::shared_ptr<core::DataBridge> data_bridge);
    ~LVGLInterface();

    /**
     * @brief 初始化LVGL界面系统
     */
    bool initialize(const LVGLConfig& config);

    /**
     * @brief 启动界面线程
     */
    bool start();

    /**
     * @brief 停止界面线程
     */
    void stop();

    /**
     * @brief 检查界面是否在运行
     */
    bool isRunning() const { return running_.load(); }

    /**
     * @brief 设置全屏模式
     */
    void setFullscreen(bool fullscreen);

private:
    /**
     * @brief 界面线程主循环
     */
    void uiLoop();

    /**
     * @brief 创建主界面
     */
    void createMainInterface();

    /**
     * @brief 创建头部面板
     */
    lv_obj_t* createHeaderPanel();

    /**
     * @brief 创建摄像头显示区域
     */
    lv_obj_t* createCameraPanel();

    /**
     * @brief 创建控制面板
     */
    lv_obj_t* createControlPanel();

    /**
     * @brief 创建状态面板
     */
    lv_obj_t* createStatusPanel();

    /**
     * @brief 创建底部控制栏
     */
    lv_obj_t* createFooterPanel();

    /**
     * @brief 更新界面数据
     */
    void updateInterface();

    /**
     * @brief 更新摄像头画面
     */
    void updateCameraView();

    /**
     * @brief 更新系统状态信息
     */
    void updateSystemStats();

    /**
     * @brief 更新Modbus寄存器显示
     */
    void updateModbusDisplay();

    /**
     * @brief 更新工作流程状态
     */
    void updateWorkflowStatus();

    /**
     * @brief 绘制检测结果标注
     */
    void drawDetectionResults(const core::DetectionResult& result);

    // 事件处理器
    static void onStartButtonClicked(lv_event_t* e);
    static void onStopButtonClicked(lv_event_t* e);
    static void onPauseButtonClicked(lv_event_t* e);
    static void onEmergencyButtonClicked(lv_event_t* e);
    static void onBladeSelectionChanged(lv_event_t* e);
    static void onSettingsButtonClicked(lv_event_t* e);

    /**
     * @brief 显示消息对话框
     */
    void showMessageDialog(const std::string& title, const std::string& message);

    /**
     * @brief 初始化LVGL样式主题
     */
    void initializeTheme();

    /**
     * @brief 初始化显示驱动
     */
    bool initializeDisplay();

    /**
     * @brief 初始化输入设备
     */
    bool initializeInput();

private:
    std::shared_ptr<core::DataBridge> data_bridge_;
    LVGLConfig config_;
    
    std::thread ui_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    
    // LVGL对象引用
    lv_obj_t* main_screen_;
    lv_obj_t* header_panel_;
    lv_obj_t* camera_panel_;
    lv_obj_t* camera_canvas_;
    lv_obj_t* control_panel_;
    lv_obj_t* status_panel_;
    lv_obj_t* footer_panel_;
    
    // 界面组件
    struct {
        lv_obj_t* system_title;
        lv_obj_t* heartbeat_label;
        lv_obj_t* response_label;
        std::vector<lv_obj_t*> workflow_buttons;
    } header_widgets_;
    
    struct {
        lv_obj_t* info_label;
        lv_obj_t* coord_value;
        lv_obj_t* quality_value;
        lv_obj_t* blade_value;
    } camera_widgets_;
    
    struct {
        lv_obj_t* modbus_table;
        lv_obj_t* plc_status_table;
        lv_obj_t* jetson_stats_table;
        lv_obj_t* ai_stats_table;
        std::vector<lv_obj_t*> blade_buttons;
    } control_widgets_;
    
    struct {
        lv_obj_t* start_btn;
        lv_obj_t* pause_btn;
        lv_obj_t* stop_btn;
        lv_obj_t* emergency_btn;
        lv_obj_t* power_btn;
        lv_obj_t* process_label;
        lv_obj_t* stats_label;
    } footer_widgets_;
    
    // LVGL驱动
    lv_disp_drv_t disp_drv_;
    lv_indev_drv_t indev_drv_;
    lv_disp_t* display_;
    lv_indev_t* input_device_;
    
    // 显示缓冲区
    static lv_color_t* disp_buf1_;
    static lv_color_t* disp_buf2_;
    static lv_disp_draw_buf_t draw_buf_;
    
    // 性能监控
    std::chrono::high_resolution_clock::time_point last_update_time_;
    int frame_count_;
    float ui_fps_;
    
    // 当前状态
    WorkflowStep current_step_;
    bool system_running_;
    bool emergency_stop_;
    int selected_blade_;
    
    // 颜色主题
    lv_color_t color_primary_;
    lv_color_t color_secondary_;
    lv_color_t color_success_;
    lv_color_t color_warning_;
    lv_color_t color_error_;
    lv_color_t color_background_;
    lv_color_t color_surface_;
};

/**
 * @brief LVGL显示驱动回调
 */
void lvgl_disp_flush(lv_disp_drv_t* disp_drv, const lv_area_t* area, lv_color_t* color_p);

/**
 * @brief LVGL输入设备回调
 */
void lvgl_input_read(lv_indev_drv_t* indev_drv, lv_indev_data_t* data);

} // namespace ui
} // namespace bamboo_cut