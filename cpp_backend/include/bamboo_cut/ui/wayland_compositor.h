/**
 * @file wayland_compositor.h
 * @brief Wayland合成器检测和支持工具
 * 提供Wayland环境检测和配置功能
 */

#ifndef BAMBOO_CUT_UI_WAYLAND_COMPOSITOR_H
#define BAMBOO_CUT_UI_WAYLAND_COMPOSITOR_H

#include <string>
#include <vector>
#include <memory>

#ifdef ENABLE_WAYLAND
#include <wayland-client.h>
#endif

namespace bamboo_cut {
namespace ui {

/**
 * @brief Wayland检测器
 * 提供Wayland环境和合成器检测功能
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
     * @return 合成器名称，如果未检测到则为空
     */
    static std::string detectCompositor();
    
    /**
     * @brief 检查环境变量
     * @return Wayland display名称
     */
    static std::string getWaylandDisplay();
    
    /**
     * @brief 列出可用的Wayland扩展
     * @return 扩展名称列表
     */
    static std::vector<std::string> listWaylandExtensions();
    
    /**
     * @brief 检查是否在Wayland会话中运行
     * @return true如果在Wayland会话中
     */
    static bool isWaylandSession();

private:
    static bool checkWaylandSocket();
    static bool testWaylandConnection();
};

/**
 * @brief Wayland合成器配置
 * 管理不同合成器的特定配置
 */
class WaylandCompositorConfig {
public:
    enum class CompositorType {
        WESTON,      // 参考实现
        SWAY,        // i3兼容的平铺合成器
        GNOME_SHELL, // GNOME桌面
        KDE_KWIN,    // KDE桌面
        CUSTOM,      // 自定义合成器
        UNKNOWN      // 未知合成器
    };
    
    /**
     * @brief 构造函数
     * @param type 合成器类型
     */
    explicit WaylandCompositorConfig(CompositorType type = CompositorType::UNKNOWN);
    
    /**
     * @brief 获取合成器类型
     */
    CompositorType getType() const { return type_; }
    
    /**
     * @brief 设置合成器特定参数
     * @param key 参数键
     * @param value 参数值
     */
    void setParameter(const std::string& key, const std::string& value);
    
    /**
     * @brief 获取合成器特定参数
     * @param key 参数键
     * @return 参数值，如果不存在则为空字符串
     */
    std::string getParameter(const std::string& key) const;
    
    /**
     * @brief 检查是否支持特定功能
     * @param feature 功能名称
     * @return true如果支持
     */
    bool supportsFeature(const std::string& feature) const;
    
    /**
     * @brief 获取推荐的渲染配置
     * @return 配置字符串
     */
    std::string getRecommendedRenderConfig() const;

private:
    CompositorType type_;
    std::map<std::string, std::string> parameters_;
    
    void initializeDefaults();
};

/**
 * @brief Wayland连接管理器
 * 管理与Wayland服务器的连接
 */
class WaylandConnectionManager {
public:
    WaylandConnectionManager();
    ~WaylandConnectionManager();
    
    /**
     * @brief 连接到Wayland服务器
     * @param display_name 显示器名称，nullptr使用默认
     * @return true如果连接成功
     */
    bool connect(const char* display_name = nullptr);
    
    /**
     * @brief 断开连接
     */
    void disconnect();
    
    /**
     * @brief 检查连接状态
     * @return true如果已连接
     */
    bool isConnected() const;
    
    /**
     * @brief 获取显示器句柄
     * @return Wayland显示器指针
     */
    void* getDisplay() const;
    
    /**
     * @brief 处理事件
     * @return 处理的事件数量
     */
    int dispatchEvents();
    
    /**
     * @brief 刷新输出
     */
    void flush();

private:
#ifdef ENABLE_WAYLAND
    struct wl_display* display_;
    struct wl_registry* registry_;
    struct wl_compositor* compositor_;
    struct wl_shell* shell_;
    
    static void registryHandler(void* data, struct wl_registry* registry,
                               uint32_t id, const char* interface, uint32_t version);
    static void registryRemover(void* data, struct wl_registry* registry, uint32_t id);
    
    static const struct wl_registry_listener registry_listener_;
#else
    void* display_;
#endif
    
    bool connected_;
};

/**
 * @brief Wayland窗口/Surface管理器
 * 创建和管理Wayland surface
 */
class WaylandSurfaceManager {
public:
    explicit WaylandSurfaceManager(WaylandConnectionManager* connection);
    ~WaylandSurfaceManager();
    
    /**
     * @brief 创建surface
     * @param width 宽度
     * @param height 高度
     * @return true如果创建成功
     */
    bool createSurface(int width, int height);
    
    /**
     * @brief 销毁surface
     */
    void destroySurface();
    
    /**
     * @brief 获取surface句柄
     * @return Wayland surface指针
     */
    void* getSurface() const;
    
    /**
     * @brief 提交surface
     */
    void commit();
    
    /**
     * @brief 调整surface大小
     * @param width 新宽度
     * @param height 新高度
     */
    void resize(int width, int height);
    
    /**
     * @brief 设置surface位置
     * @param x X坐标
     * @param y Y坐标
     */
    void setPosition(int x, int y);

private:
#ifdef ENABLE_WAYLAND
    struct wl_surface* surface_;
    struct wl_shell_surface* shell_surface_;
    struct wl_buffer* buffer_;
#else
    void* surface_;
    void* shell_surface_;
    void* buffer_;
#endif
    
    WaylandConnectionManager* connection_;
    int width_, height_;
    
    bool createBuffer(int width, int height);
    void destroyBuffer();
};

/**
 * @brief Wayland事件处理器
 * 处理Wayland协议事件
 */
class WaylandEventHandler {
public:
    virtual ~WaylandEventHandler() = default;
    
    /**
     * @brief 处理surface进入事件
     * @param surface surface指针
     * @param output 输出设备指针
     */
    virtual void onSurfaceEnter(void* surface, void* output) {}
    
    /**
     * @brief 处理surface离开事件
     * @param surface surface指针
     * @param output 输出设备指针
     */
    virtual void onSurfaceLeave(void* surface, void* output) {}
    
    /**
     * @brief 处理surface配置事件
     * @param surface surface指针
     * @param width 新宽度
     * @param height 新高度
     */
    virtual void onSurfaceConfigure(void* surface, int width, int height) {}
    
    /**
     * @brief 处理surface损坏事件
     * @param surface surface指针
     * @param x X坐标
     * @param y Y坐标
     * @param width 宽度
     * @param height 高度
     */
    virtual void onSurfaceDamage(void* surface, int x, int y, int width, int height) {}
};

} // namespace ui
} // namespace bamboo_cut

#endif // BAMBOO_CUT_UI_WAYLAND_COMPOSITOR_H