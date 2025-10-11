/**
 * @file xvfb_manager.h
 * @brief Xvfb虚拟显示管理器头文件 - 解决nvarguscamerasrc EGL初始化问题
 */

#ifndef BAMBOO_CUT_UI_XVFB_MANAGER_H
#define BAMBOO_CUT_UI_XVFB_MANAGER_H

#include <sys/types.h>

namespace bamboo_cut {
namespace ui {

/**
 * @class XvfbManager
 * @brief Xvfb虚拟显示管理器
 * 
 * 解决nvarguscamerasrc在无头环境中的EGL初始化问题
 * nvbufsurftransform需要一个有效的X11显示连接
 */
class XvfbManager {
private:
    static pid_t xvfb_pid_;           ///< Xvfb进程PID
    static bool initialized_;         ///< 初始化状态
    static const char* DISPLAY_NUM;   ///< 虚拟显示编号
    
public:
    /**
     * @brief 启动Xvfb虚拟显示服务
     * @return true 如果启动成功或已运行
     */
    static bool startXvfb();
    
    /**
     * @brief 设置环境变量以使用Xvfb显示
     * 
     * 这个方法会：
     * 1. 确保Xvfb正在运行
     * 2. 设置DISPLAY环境变量
     * 3. 清除XAUTHORITY以避免权限问题
     */
    static void setupEnvironment();
    
    /**
     * @brief 停止Xvfb虚拟显示服务
     */
    static void stopXvfb();
    
    /**
     * @brief 检查Xvfb是否正在运行
     * @return true 如果Xvfb正在运行
     */
    static bool isRunning();
};

} // namespace ui
} // namespace bamboo_cut

#endif // BAMBOO_CUT_UI_XVFB_MANAGER_H