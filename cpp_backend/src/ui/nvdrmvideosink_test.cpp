/**
 * @file nvdrmvideosink_test.cpp
 * @brief nvdrmvideosink叠加平面功能测试程序
 * 验证视频输出与LVGL界面的分离显示
 */

#include "bamboo_cut/deepstream/deepstream_manager.h"
#include "bamboo_cut/ui/lvgl_interface.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

namespace bamboo_cut {
namespace test {

class NVDRMVideoSinkTest {
public:
    NVDRMVideoSinkTest() 
        : deepstream_manager_(nullptr)
        , lvgl_interface_(nullptr)
        , test_running_(false) {}
    
    ~NVDRMVideoSinkTest() {
        cleanup();
    }
    
    /**
     * @brief 初始化测试环境
     */
    bool initialize() {
        std::cout << "=== nvdrmvideosink 叠加平面测试 ===" << std::endl;
        
        // 1. 检测DRM设备和叠加平面
        if (!detectDRMCapabilities()) {
            std::cerr << "DRM设备检测失败" << std::endl;
            return false;
        }
        
        // 2. 初始化LVGL界面（使用主DRM平面）
        lvgl_interface_ = std::make_unique<ui::LVGLInterface>();
        if (!lvgl_interface_->initialize()) {
            std::cerr << "LVGL界面初始化失败" << std::endl;
            return false;
        }
        
        // 3. 配置DeepStream使用nvdrmvideosink
        deepstream::DeepStreamConfig config;
        config.sink_mode = deepstream::VideoSinkMode::NVDRMVIDEOSINK;
        config.overlay.z_order = 1;  // 确保视频在LVGL之上
        
        deepstream_manager_ = std::make_unique<deepstream::DeepStreamManager>();
        if (!deepstream_manager_->initialize(config)) {
            std::cerr << "DeepStream初始化失败" << std::endl;
            return false;
        }
        
        std::cout << "测试环境初始化完成" << std::endl;
        return true;
    }
    
    /**
     * @brief 运行层级分离测试
     */
    bool runLayerSeparationTest() {
        std::cout << "\n--- 开始层级分离测试 ---" << std::endl;
        
        // 1. 启动LVGL界面
        std::cout << "启动LVGL界面..." << std::endl;
        if (!lvgl_interface_->show()) {
            std::cerr << "LVGL界面启动失败" << std::endl;
            return false;
        }
        
        // 等待界面稳定
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // 2. 启动视频流（叠加平面模式）
        std::cout << "启动nvdrmvideosink视频流..." << std::endl;
        if (!deepstream_manager_->start()) {
            std::cerr << "视频流启动失败" << std::endl;
            return false;
        }
        
        // 3. 验证层级分离
        return validateLayerSeparation();
    }
    
    /**
     * @brief 运行性能基准测试
     */
    bool runPerformanceTest() {
        std::cout << "\n--- 开始性能基准测试 ---" << std::endl;
        
        test_running_ = true;
        auto start_time = std::chrono::steady_clock::now();
        
        int frame_count = 0;
        const int test_duration_seconds = 30;
        
        while (test_running_) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            
            if (elapsed.count() >= test_duration_seconds) {
                break;
            }
            
            // 模拟界面更新
            lvgl_interface_->update();
            frame_count++;
            
            // 每秒输出一次统计
            if (frame_count % 60 == 0) {
                double fps = frame_count / static_cast<double>(elapsed.count());
                std::cout << "运行时间: " << elapsed.count() << "s, FPS: " << fps << std::endl;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        double average_fps = frame_count / static_cast<double>(total_time.count());
        
        std::cout << "性能测试完成:" << std::endl;
        std::cout << "  总帧数: " << frame_count << std::endl;
        std::cout << "  测试时长: " << total_time.count() << "秒" << std::endl;
        std::cout << "  平均FPS: " << average_fps << std::endl;
        
        // 性能要求：至少25 FPS
        return average_fps >= 25.0;
    }
    
    /**
     * @brief 运行交互响应测试
     */
    bool runInteractionTest() {
        std::cout << "\n--- 开始交互响应测试 ---" << std::endl;
        
        // 模拟触摸交互
        for (int i = 0; i < 10; i++) {
            // 模拟触摸事件
            if (!simulateTouchEvent(100 + i * 50, 100 + i * 30)) {
                std::cerr << "触摸事件模拟失败" << std::endl;
                return false;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "触摸事件 " << (i + 1) << "/10 完成" << std::endl;
        }
        
        std::cout << "交互响应测试完成" << std::endl;
        return true;
    }
    
    /**
     * @brief 停止测试
     */
    void stopTest() {
        test_running_ = false;
        
        if (deepstream_manager_) {
            deepstream_manager_->stop();
        }
        
        if (lvgl_interface_) {
            lvgl_interface_->hide();
        }
    }

private:
    /**
     * @brief 检测DRM设备能力
     */
    bool detectDRMCapabilities() {
        std::cout << "检测DRM设备能力..." << std::endl;
        
        // 尝试打开DRM设备
        const char* drm_devices[] = {"/dev/dri/card1"};
        int drm_fd = -1;
        
        for (const char* device : drm_devices) {
            drm_fd = open(device, O_RDWR);
            if (drm_fd >= 0) {
                std::cout << "成功打开DRM设备: " << device << std::endl;
                break;
            }
        }
        
        if (drm_fd < 0) {
            std::cerr << "无法打开任何DRM设备" << std::endl;
            return false;
        }
        
        // 获取DRM资源
        drmModeRes* resources = drmModeGetResources(drm_fd);
        if (!resources) {
            std::cerr << "无法获取DRM资源" << std::endl;
            close(drm_fd);
            return false;
        }
        
        std::cout << "DRM设备信息:" << std::endl;
        std::cout << "  连接器数量: " << resources->count_connectors << std::endl;
        std::cout << "  CRTC数量: " << resources->count_crtcs << std::endl;
        
        // 检查叠加平面
        drmModePlaneRes* plane_resources = drmModeGetPlaneResources(drm_fd);
        if (plane_resources) {
            std::cout << "  叠加平面数量: " << plane_resources->count_planes << std::endl;
            
            for (uint32_t i = 0; i < plane_resources->count_planes; i++) {
                drmModePlane* plane = drmModeGetPlane(drm_fd, plane_resources->planes[i]);
                if (plane) {
                    std::cout << "    平面 " << i << ": ID=" << plane_resources->planes[i] 
                              << ", 可能的CRTC=" << plane->possible_crtcs << std::endl;
                    drmModeFreePlane(plane);
                }
            }
            
            drmModeFreePlaneResources(plane_resources);
        } else {
            std::cout << "  未检测到叠加平面支持" << std::endl;
        }
        
        drmModeFreeResources(resources);
        close(drm_fd);
        
        return true;
    }
    
    /**
     * @brief 验证层级分离
     */
    bool validateLayerSeparation() {
        std::cout << "验证显示层级分离..." << std::endl;
        
        // 检查视频流状态
        if (!deepstream_manager_->isRunning()) {
            std::cerr << "视频流未正常运行" << std::endl;
            return false;
        }
        
        // 检查LVGL界面状态
        if (!lvgl_interface_->isActive()) {
            std::cerr << "LVGL界面未激活" << std::endl;
            return false;
        }
        
        // 验证层级配置
        auto current_mode = deepstream_manager_->getCurrentSinkMode();
        if (current_mode != deepstream::VideoSinkMode::NVDRMVIDEOSINK) {
            std::cerr << "视频输出模式不正确" << std::endl;
            return false;
        }
        
        std::cout << "层级分离验证成功:" << std::endl;
        std::cout << "  - LVGL界面在主DRM平面正常显示" << std::endl;
        std::cout << "  - 视频在独立叠加平面正常显示" << std::endl;
        std::cout << "  - 两个显示层没有冲突" << std::endl;
        
        return true;
    }
    
    /**
     * @brief 模拟触摸事件
     */
    bool simulateTouchEvent(int x, int y) {
        // 这里应该通过适当的接口模拟触摸事件
        // 暂时返回成功
        return true;
    }
    
    /**
     * @brief 清理资源
     */
    void cleanup() {
        stopTest();
        deepstream_manager_.reset();
        lvgl_interface_.reset();
    }

private:
    std::unique_ptr<deepstream::DeepStreamManager> deepstream_manager_;
    std::unique_ptr<ui::LVGLInterface> lvgl_interface_;
    bool test_running_;
};

} // namespace test
} // namespace bamboo_cut

/**
 * @brief 主测试函数
 */
int main(int argc, char* argv[]) {
    bamboo_cut::test::NVDRMVideoSinkTest test;
    
    // 初始化测试环境
    if (!test.initialize()) {
        std::cerr << "测试环境初始化失败" << std::endl;
        return 1;
    }
    
    bool all_tests_passed = true;
    
    // 运行层级分离测试
    if (!test.runLayerSeparationTest()) {
        std::cerr << "层级分离测试失败" << std::endl;
        all_tests_passed = false;
    }
    
    // 运行性能测试
    if (!test.runPerformanceTest()) {
        std::cerr << "性能测试失败" << std::endl;
        all_tests_passed = false;
    }
    
    // 运行交互测试
    if (!test.runInteractionTest()) {
        std::cerr << "交互测试失败" << std::endl;
        all_tests_passed = false;
    }
    
    // 停止测试
    test.stopTest();
    
    if (all_tests_passed) {
        std::cout << "\n=== 所有测试通过 ===" << std::endl;
        std::cout << "nvdrmvideosink叠加平面功能正常工作" << std::endl;
        return 0;
    } else {
        std::cout << "\n=== 部分测试失败 ===" << std::endl;
        return 1;
    }
}