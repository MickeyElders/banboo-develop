/**
 * @file display_backend.h
 * @brief 显示后端抽象层
 * 支持DRM和Wayland双模式切换
 */

#ifndef BAMBOO_CUT_UI_DISPLAY_BACKEND_H
#define BAMBOO_CUT_UI_DISPLAY_BACKEND_H

#include <memory>
#include <string>

namespace bamboo_cut {
namespace ui {

/**
 * @brief 显示区域定义
 */
struct DisplayArea {
    int x, y;           // 位置
    int width, height;  // 尺寸
    
    DisplayArea(int x = 0, int y = 0, int w = 1280, int h = 800)
        : x(x), y(y), width(w), height(h) {}
};

/**
 * @brief 显示后端类型
 */
enum class DisplayBackendType {
    DRM_DIRECT,      // DRM直接渲染
    WAYLAND_CLIENT,  // Wayland客户端
    AUTO_DETECT,     // 自动检测
    FALLBACK         // 回退模式
};

/**
 * @brief 显示后端抽象接口
 */
class DisplayBackend {
public:
    virtual ~DisplayBackend() = default;
    
    /**
     * @brief 初始化显示后端
     * @param area 显示区域
     * @return true如果初始化成功
     */
    virtual bool initialize(const DisplayArea& area) = 0;
    
    /**
     * @brief 启动显示后端
     * @return true如果启动成功
     */
    virtual bool start() = 0;
    
    /**
     * @brief 停止显示后端
     */
    virtual void stop() = 0;
    
    /**
     * @brief 检查是否正在运行
     * @return true如果正在运行
     */
    virtual bool isRunning() const = 0;
    
    /**
     * @brief 更新显示区域
     * @param area 新的显示区域
     * @return true如果更新成功
     */
    virtual bool updateArea(const DisplayArea& area) = 0;
    
    /**
     * @brief 设置刷新回调
     * @param callback 回调函数指针
     */
    virtual void setFlushCallback(void* callback) = 0;
    
    /**
     * @brief 获取显示句柄
     * @return 显示句柄指针
     */
    virtual void* getDisplayHandle() = 0;
    
protected:
    DisplayArea current_area_;
    bool initialized_ = false;
    bool running_ = false;
    void* flush_callback_ = nullptr;
    void* display_handle_ = nullptr;
};

/**
 * @brief DRM显示后端实现
 */
class DRMDisplayBackend : public DisplayBackend {
public:
    DRMDisplayBackend() = default;
    ~DRMDisplayBackend() override = default;
    
    bool initialize(const DisplayArea& area) override;
    bool start() override;
    void stop() override;
    bool isRunning() const override;
    bool updateArea(const DisplayArea& area) override;
    void setFlushCallback(void* callback) override;
    void* getDisplayHandle() override;
};

/**
 * @brief Wayland显示后端实现
 */
class WaylandDisplayBackend : public DisplayBackend {
public:
    WaylandDisplayBackend() = default;
    ~WaylandDisplayBackend() override = default;
    
    bool initialize(const DisplayArea& area) override;
    bool start() override;
    void stop() override;
    bool isRunning() const override;
    bool updateArea(const DisplayArea& area) override;
    void setFlushCallback(void* callback) override;
    void* getDisplayHandle() override;

private:
    /**
     * @brief 连接到Wayland服务器
     * @return true如果连接成功
     */
    bool connectToWaylandServer();
    
    /**
     * @brief 创建Wayland surface
     * @return true如果创建成功
     */
    bool createWaylandSurface();
    
    /**
     * @brief 处理Wayland事件
     */
    void handleWaylandEvents();
    
#ifdef ENABLE_WAYLAND
    struct wl_display* wl_display_ = nullptr;
    struct wl_surface* wl_surface_ = nullptr;
    struct wl_compositor* wl_compositor_ = nullptr;
#else
    void* wl_display_ = nullptr;
    void* wl_surface_ = nullptr;
    void* wl_compositor_ = nullptr;
#endif
};

/**
 * @brief Wayland检测器
 */
class WaylandDetector {
public:
    /**
     * @brief 检测Wayland支持
     * @return true如果Wayland可用
     */
    static bool detectWaylandSupport();
    
    /**
     * @brief 检测Wayland合成器
     * @return 合成器名称
     */
    static std::string detectCompositor();
    
    /**
     * @brief 检查是否在Wayland会话中
     * @return true如果在Wayland会话中
     */
    static bool isWaylandSession();
};

/**
 * @brief 显示后端工厂
 */
class DisplayBackendFactory {
public:
    /**
     * @brief 创建显示后端
     * @param type 后端类型
     * @return 显示后端实例
     */
    static std::unique_ptr<DisplayBackend> createBackend(DisplayBackendType type);
    
    /**
     * @brief 创建最佳可用后端
     * @return 显示后端实例
     */
    static std::unique_ptr<DisplayBackend> createBestBackend();
    
    /**
     * @brief 检查后端是否可用
     * @param type 后端类型
     * @return true如果可用
     */
    static bool isBackendAvailable(DisplayBackendType type);
    
    /**
     * @brief 获取推荐的后端类型
     * @return 推荐的后端类型
     */
    static DisplayBackendType getRecommendedType();
};

} // namespace ui
} // namespace bamboo_cut

#endif // BAMBOO_CUT_UI_DISPLAY_BACKEND_H