/**
 * @file lvgl_wayland_interface.h
 * @brief LVGL Wayland接口 - Weston合成器架构支持
 * 
 * 替代DRM直接访问，使用标准Wayland客户端协议
 * 支持lv_drivers/wayland后端和多输入设备
 */

#ifndef BAMBOO_CUT_UI_LVGL_WAYLAND_INTERFACE_H
#define BAMBOO_CUT_UI_LVGL_WAYLAND_INTERFACE_H

#include <memory>
#include <thread>
#include <atomic>
#include <string>
#include <opencv2/opencv.hpp>

// 前向声明LVGL类型
struct _lv_obj_t;
typedef struct _lv_obj_t lv_obj_t;
struct _lv_indev_t;
typedef struct _lv_indev_t lv_indev_t;
struct _lv_display_t;
typedef struct _lv_display_t lv_display_t;

namespace bamboo_cut {
namespace ui {

/**
 * @brief LVGL Wayland配置结构
 */
struct LVGLWaylandConfig {
    int screen_width = 1280;
    int screen_height = 800;
    int refresh_rate = 60;
    bool enable_touch = true;
    std::string touch_device = "/dev/input/event0";
    std::string wayland_display = "wayland-0";
    
    // 窗口配置
    bool fullscreen = true;
    int window_x = 0;
    int window_y = 0;
    int window_width = 1280;
    int window_height = 800;
};

/**
 * @brief LVGL Wayland接口类
 *
 * 提供标准Wayland客户端模式的LVGL界面
 * 替代原有的DRM直接访问方案，支持Wayland Subsurface架构
 */
class LVGLWaylandInterface {
public:
    LVGLWaylandInterface();
    ~LVGLWaylandInterface();

    /**
     * @brief 初始化LVGL Wayland界面
     * @param config 配置参数
     * @return 成功返回true
     */
    bool initialize(const LVGLWaylandConfig& config);

    /**
     * @brief 启动UI线程
     * @return 成功返回true
     */
    bool start();

    /**
     * @brief 停止UI线程
     */
    void stop();

    /**
     * @brief 获取摄像头Canvas对象
     * @return Canvas对象指针，用于DeepStream集成
     */
    lv_obj_t* getCameraCanvas();

    /**
     * @brief 检查是否完全初始化
     * @return 完全初始化返回true
     */
    bool isFullyInitialized() const;

    /**
     * @brief 检查是否运行中
     * @return 运行中返回true
     */
    bool isRunning() const;

    /**
     * @brief 更新摄像头Canvas（兼容性接口）
     * @param frame OpenCV帧数据
     */
    void updateCameraCanvas(const cv::Mat& frame);

    /**
     * @brief 检查Wayland环境是否可用
     * @return 可用返回true
     */
    static bool isWaylandEnvironmentAvailable();

    // 🆕 新增：Wayland Subsurface架构支持方法
    
    /**
     * @brief 获取内部实现类指针以供DeepStream访问
     */
    void* getImpl();
    
    /**
     * @brief 获取Wayland Display对象
     * @return Wayland Display指针，用于DeepStream Subsurface创建
     */
    void* getWaylandDisplay();
    
    /**
     * @brief 获取Wayland Compositor对象
     * @return Wayland Compositor指针，用于创建子表面
     */
    void* getWaylandCompositor();
    
    /**
     * @brief 获取Wayland Subcompositor对象
     * @return Wayland Subcompositor指针，用于创建subsurface
     */
    void* getWaylandSubcompositor();
    
    /**
     * @brief 获取Wayland Surface对象（父表面）
     * @return Wayland Surface指针，用作subsurface的父表面
     */
    void* getWaylandSurface();

private:
    // 内部实现指针
    class Impl;
    std::unique_ptr<Impl> pImpl_;

    // 线程安全状态
    std::atomic<bool> running_{false};
    std::atomic<bool> fully_initialized_{false};
    std::thread ui_thread_;

    // UI主循环
    void uiThreadLoop();

    // UI创建方法
    void createUI();
    void createHeaderPanel();
    void createCameraPanel();
    void createControlPanel();
    void createFooterPanel();

    // 事件处理
    void setupEventHandlers();

    // 输入设备初始化
    bool initializeInputDevices();

    // 清理资源
    void cleanup();
};

} // namespace ui
} // namespace bamboo_cut

#endif // BAMBOO_CUT_UI_LVGL_WAYLAND_INTERFACE_H