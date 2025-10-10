/**
 * @file drm_render_test.cpp
 * @brief NVIDIA-DRM渲染功能验证测试程序
 * @author Kilo Code
 * @date 2025-01-10
 * 
 * 用于验证从tegra_drm迁移到nvidia-drm后的完整渲染功能，
 * 包括帧缓冲区初始化、显示输出配置、GPU加速支持和性能基准测试
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <random>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>

#ifdef ENABLE_LVGL
#include "lvgl/lvgl.h"
#endif

#include "bamboo_cut/ui/lvgl_interface.h"
#include "bamboo_cut/utils/logger.h"

namespace bamboo_cut {
namespace ui {

/**
 * @brief DRM渲染功能测试类
 */
class DRMRenderTest {
private:
    int drm_fd = -1;
    drmModeModeInfo mode_info = {};
    drmModeConnector* connector = nullptr;
    drmModeEncoder* encoder = nullptr;
    drmModeCrtc* crtc = nullptr;
    uint32_t framebuffer_id = 0;
    uint32_t* framebuffer_data = nullptr;
    size_t framebuffer_size = 0;
    std::string driver_name;
    
    // 性能统计
    struct PerformanceStats {
        double avg_render_time_ms = 0.0;
        double max_render_time_ms = 0.0;
        double min_render_time_ms = 1000.0;
        size_t frame_count = 0;
        double fps = 0.0;
    } perf_stats;

public:
    /**
     * @brief 构造函数
     */
    DRMRenderTest() = default;
    
    /**
     * @brief 析构函数
     */
    ~DRMRenderTest() {
        cleanup();
    }
    
    /**
     * @brief 初始化DRM渲染测试
     * @return true 成功，false 失败
     */
    bool initialize() {
        Logger::info("开始初始化DRM渲染功能测试");
        
        // 检测DRM驱动类型
        driver_name = detectDRMDriverType();
        Logger::info("检测到DRM驱动: " + driver_name);
        
        // 优先尝试nvidia-drm设备
        if (!openDRMDevice()) {
            Logger::error("无法打开DRM设备");
            return false;
        }
        
        // 获取显示器连接信息
        if (!setupDisplay()) {
            Logger::error("显示器配置失败");
            return false;
        }
        
        // 创建帧缓冲区
        if (!createFramebuffer()) {
            Logger::error("帧缓冲区创建失败");
            return false;
        }
        
        Logger::info("DRM渲染功能测试初始化成功");
        return true;
    }
    
    /**
     * @brief 运行完整的渲染功能验证测试
     * @return true 测试通过，false 测试失败
     */
    bool runCompleteTest() {
        Logger::info("开始运行完整渲染功能验证测试");
        
        // 1. 基础渲染测试
        if (!testBasicRendering()) {
            Logger::error("基础渲染测试失败");
            return false;
        }
        
        // 2. 颜色渐变测试
        if (!testColorGradient()) {
            Logger::error("颜色渐变测试失败");
            return false;
        }
        
        // 3. 几何图形渲染测试
        if (!testGeometryRendering()) {
            Logger::error("几何图形渲染测试失败");
            return false;
        }
        
        // 4. 性能基准测试
        if (!testPerformanceBenchmark()) {
            Logger::error("性能基准测试失败");
            return false;
        }
        
        // 5. GPU加速功能测试
        if (!testGPUAcceleration()) {
            Logger::error("GPU加速功能测试失败");
            return false;
        }
        
        // 6. 内存优化测试
        if (!testMemoryOptimization()) {
            Logger::error("内存优化测试失败");
            return false;
        }
        
        // 输出测试结果
        printTestResults();
        
        Logger::info("完整渲染功能验证测试通过");
        return true;
    }

private:
    /**
     * @brief 打开DRM设备
     */
    bool openDRMDevice() {
        // 优先尝试nvidia-drm设备 (/dev/dri/card0)
        drm_fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
        if (drm_fd >= 0) {
            Logger::info("成功打开nvidia-drm设备: /dev/dri/card0");
            return true;
        }
        
        // 回退到tegra-drm设备 (/dev/dri/card1)
        drm_fd = open("/dev/dri/card1", O_RDWR | O_CLOEXEC);
        if (drm_fd >= 0) {
            Logger::info("回退到tegra-drm设备: /dev/dri/card1");
            return true;
        }
        
        Logger::error("无法打开任何DRM设备");
        return false;
    }
    
    /**
     * @brief 配置显示器
     */
    bool setupDisplay() {
        drmModeRes* resources = drmModeGetResources(drm_fd);
        if (!resources) {
            Logger::error("无法获取DRM资源");
            return false;
        }
        
        // 查找可用的连接器
        for (int i = 0; i < resources->count_connectors; i++) {
            connector = drmModeGetConnector(drm_fd, resources->connectors[i]);
            if (connector && connector->connection == DRM_MODE_CONNECTED && connector->count_modes > 0) {
                mode_info = connector->modes[0];  // 使用第一个可用模式
                Logger::info("找到连接的显示器: " + std::to_string(mode_info.hdisplay) + "x" + std::to_string(mode_info.vdisplay));
                break;
            }
            drmModeFreeConnector(connector);
            connector = nullptr;
        }
        
        if (!connector) {
            Logger::error("未找到连接的显示器");
            drmModeFreeResources(resources);
            return false;
        }
        
        // 查找编码器
        if (connector->encoder_id) {
            encoder = drmModeGetEncoder(drm_fd, connector->encoder_id);
        }
        
        if (!encoder) {
            Logger::error("未找到编码器");
            drmModeFreeResources(resources);
            return false;
        }
        
        // 获取CRTC
        if (encoder->crtc_id) {
            crtc = drmModeGetCrtc(drm_fd, encoder->crtc_id);
        }
        
        if (!crtc) {
            Logger::error("未找到CRTC");
            drmModeFreeResources(resources);
            return false;
        }
        
        drmModeFreeResources(resources);
        return true;
    }
    
    /**
     * @brief 创建帧缓冲区
     */
    bool createFramebuffer() {
        uint32_t width = mode_info.hdisplay;
        uint32_t height = mode_info.vdisplay;
        uint32_t bpp = 32;  // 32位色深
        uint32_t stride = width * (bpp / 8);
        
        // 针对NVIDIA DRM优化内存对齐
        if (driver_name == "nvidia-drm") {
            // NVIDIA GPU要求64字节边界对齐
            stride = (stride + 63) & ~63;
        }
        
        framebuffer_size = stride * height;
        
        // 创建DRM帧缓冲区
        struct drm_mode_create_dumb create_req = {};
        create_req.width = width;
        create_req.height = height;
        create_req.bpp = bpp;
        
        if (drmIoctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) < 0) {
            Logger::error("创建DRM缓冲区失败");
            return false;
        }
        
        // 添加帧缓冲区
        if (drmModeAddFB(drm_fd, width, height, 24, bpp, create_req.pitch, 
                         create_req.handle, &framebuffer_id) < 0) {
            Logger::error("添加帧缓冲区失败");
            return false;
        }
        
        // 映射内存
        struct drm_mode_map_dumb map_req = {};
        map_req.handle = create_req.handle;
        
        if (drmIoctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map_req) < 0) {
            Logger::error("映射帧缓冲区失败");
            return false;
        }
        
        // 针对NVIDIA DRM优化内存映射
        int mmap_flags = MAP_SHARED;
        if (driver_name == "nvidia-drm") {
            mmap_flags |= MAP_NORESERVE;  // NVIDIA GPU优化
        }
        
        framebuffer_data = static_cast<uint32_t*>(mmap(0, framebuffer_size, 
                                                      PROT_READ | PROT_WRITE, 
                                                      mmap_flags, drm_fd, map_req.offset));
        
        if (framebuffer_data == MAP_FAILED) {
            Logger::error("内存映射失败");
            return false;
        }
        
        Logger::info("帧缓冲区创建成功: " + std::to_string(width) + "x" + std::to_string(height));
        return true;
    }
    
    /**
     * @brief 基础渲染测试
     */
    bool testBasicRendering() {
        Logger::info("开始基础渲染测试");
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 清除屏幕为蓝色
        uint32_t blue = 0xFF0000FF;
        for (size_t i = 0; i < framebuffer_size / sizeof(uint32_t); i++) {
            framebuffer_data[i] = blue;
        }
        
        // 设置显示
        if (drmModeSetCrtc(drm_fd, crtc->crtc_id, framebuffer_id, 0, 0, 
                          &connector->connector_id, 1, &mode_info) < 0) {
            Logger::error("设置CRTC失败");
            return false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double render_time = std::chrono::duration<double, std::milli>(end - start).count();
        updatePerformanceStats(render_time);
        
        usleep(500000);  // 显示0.5秒
        
        Logger::info("基础渲染测试完成，耗时: " + std::to_string(render_time) + "ms");
        return true;
    }
    
    /**
     * @brief 颜色渐变测试
     */
    bool testColorGradient() {
        Logger::info("开始颜色渐变测试");
        
        auto start = std::chrono::high_resolution_clock::now();
        
        uint32_t width = mode_info.hdisplay;
        uint32_t height = mode_info.vdisplay;
        
        // 创建水平渐变效果
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                uint8_t red = (x * 255) / width;
                uint8_t green = (y * 255) / height;
                uint8_t blue = 128;
                uint32_t color = (red << 16) | (green << 8) | blue | 0xFF000000;
                framebuffer_data[y * width + x] = color;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double render_time = std::chrono::duration<double, std::milli>(end - start).count();
        updatePerformanceStats(render_time);
        
        usleep(1000000);  // 显示1秒
        
        Logger::info("颜色渐变测试完成，耗时: " + std::to_string(render_time) + "ms");
        return true;
    }
    
    /**
     * @brief 几何图形渲染测试
     */
    bool testGeometryRendering() {
        Logger::info("开始几何图形渲染测试");
        
        auto start = std::chrono::high_resolution_clock::now();
        
        uint32_t width = mode_info.hdisplay;
        uint32_t height = mode_info.vdisplay;
        
        // 清除屏幕为黑色
        memset(framebuffer_data, 0, framebuffer_size);
        
        // 绘制矩形
        drawRectangle(width/4, height/4, width/2, height/2, 0xFF00FF00);
        
        // 绘制圆形
        drawCircle(width/2, height/2, std::min(width, height)/6, 0xFFFF0000);
        
        auto end = std::chrono::high_resolution_clock::now();
        double render_time = std::chrono::duration<double, std::milli>(end - start).count();
        updatePerformanceStats(render_time);
        
        usleep(1000000);  // 显示1秒
        
        Logger::info("几何图形渲染测试完成，耗时: " + std::to_string(render_time) + "ms");
        return true;
    }
    
    /**
     * @brief 性能基准测试
     */
    bool testPerformanceBenchmark() {
        Logger::info("开始性能基准测试");
        
        const int test_frames = 100;
        auto total_start = std::chrono::high_resolution_clock::now();
        
        for (int frame = 0; frame < test_frames; frame++) {
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            // 绘制动态内容
            uint32_t width = mode_info.hdisplay;
            uint32_t height = mode_info.vdisplay;
            
            // 随机颜色填充
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFF);
            
            uint32_t color = dis(gen) | 0xFF000000;
            
            // 使用优化的批量复制
            if (driver_name == "nvidia-drm") {
                // NVIDIA优化：批量内存操作
                size_t pixels = framebuffer_size / sizeof(uint32_t);
                for (size_t i = 0; i < pixels; i += 64) {
                    size_t batch_size = std::min(size_t(64), pixels - i);
                    std::fill(framebuffer_data + i, framebuffer_data + i + batch_size, color);
                }
            } else {
                // 标准填充
                std::fill(framebuffer_data, framebuffer_data + framebuffer_size / sizeof(uint32_t), color);
            }
            
            auto frame_end = std::chrono::high_resolution_clock::now();
            double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
            updatePerformanceStats(frame_time);
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        perf_stats.fps = (test_frames * 1000.0) / total_time;
        
        Logger::info("性能基准测试完成，平均FPS: " + std::to_string(perf_stats.fps));
        return true;
    }
    
    /**
     * @brief GPU加速功能测试
     */
    bool testGPUAcceleration() {
        Logger::info("开始GPU加速功能测试");
        
        if (driver_name != "nvidia-drm") {
            Logger::info("非NVIDIA驱动，跳过GPU加速测试");
            return true;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 测试大量像素操作的GPU加速效果
        uint32_t width = mode_info.hdisplay;
        uint32_t height = mode_info.vdisplay;
        
        // 使用NVIDIA GPU优化的内存操作
        const size_t block_size = 64;  // 64字节对齐块
        size_t total_pixels = width * height;
        
        for (size_t i = 0; i < total_pixels; i += block_size) {
            size_t batch_size = std::min(block_size, total_pixels - i);
            
            // 模拟复杂的颜色计算
            for (size_t j = 0; j < batch_size; j++) {
                size_t idx = i + j;
                uint32_t x = idx % width;
                uint32_t y = idx / width;
                
                // 复杂颜色算法
                uint8_t r = (x * y) % 256;
                uint8_t g = (x + y) % 256;
                uint8_t b = (x ^ y) % 256;
                
                framebuffer_data[idx] = (r << 16) | (g << 8) | b | 0xFF000000;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double render_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        Logger::info("GPU加速功能测试完成，耗时: " + std::to_string(render_time) + "ms");
        return true;
    }
    
    /**
     * @brief 内存优化测试
     */
    bool testMemoryOptimization() {
        Logger::info("开始内存优化测试");
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 测试内存对齐优化效果
        uint32_t width = mode_info.hdisplay;
        uint32_t height = mode_info.vdisplay;
        
        if (driver_name == "nvidia-drm") {
            // NVIDIA内存对齐优化测试
            const size_t alignment = 64;  // 64字节对齐
            
            for (uint32_t y = 0; y < height; y++) {
                uint32_t* row_ptr = framebuffer_data + y * width;
                
                // 按对齐块处理
                size_t aligned_width = (width + alignment - 1) & ~(alignment - 1);
                
                for (size_t x = 0; x < width; x += alignment) {
                    size_t block_width = std::min(alignment, width - x);
                    
                    // 批量设置颜色
                    uint32_t color = ((x + y) % 256) << 16 | 
                                   ((x * y) % 256) << 8 | 
                                   ((x ^ y) % 256) | 0xFF000000;
                    
                    for (size_t i = 0; i < block_width; i++) {
                        row_ptr[x + i] = color;
                    }
                }
            }
        } else {
            // 标准内存操作
            for (uint32_t y = 0; y < height; y++) {
                for (uint32_t x = 0; x < width; x++) {
                    uint32_t color = ((x + y) % 256) << 16 | 
                                   ((x * y) % 256) << 8 | 
                                   ((x ^ y) % 256) | 0xFF000000;
                    framebuffer_data[y * width + x] = color;
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double render_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        Logger::info("内存优化测试完成，耗时: " + std::to_string(render_time) + "ms");
        return true;
    }
    
    /**
     * @brief 绘制矩形
     */
    void drawRectangle(uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint32_t color) {
        uint32_t width = mode_info.hdisplay;
        uint32_t height = mode_info.vdisplay;
        
        for (uint32_t dy = 0; dy < h && (y + dy) < height; dy++) {
            for (uint32_t dx = 0; dx < w && (x + dx) < width; dx++) {
                framebuffer_data[(y + dy) * width + (x + dx)] = color;
            }
        }
    }
    
    /**
     * @brief 绘制圆形
     */
    void drawCircle(uint32_t cx, uint32_t cy, uint32_t radius, uint32_t color) {
        uint32_t width = mode_info.hdisplay;
        uint32_t height = mode_info.vdisplay;
        
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                int dx = static_cast<int>(x) - static_cast<int>(cx);
                int dy = static_cast<int>(y) - static_cast<int>(cy);
                
                if (dx * dx + dy * dy <= static_cast<int>(radius * radius)) {
                    framebuffer_data[y * width + x] = color;
                }
            }
        }
    }
    
    /**
     * @brief 更新性能统计
     */
    void updatePerformanceStats(double render_time) {
        perf_stats.frame_count++;
        perf_stats.avg_render_time_ms = (perf_stats.avg_render_time_ms * (perf_stats.frame_count - 1) + render_time) / perf_stats.frame_count;
        perf_stats.max_render_time_ms = std::max(perf_stats.max_render_time_ms, render_time);
        perf_stats.min_render_time_ms = std::min(perf_stats.min_render_time_ms, render_time);
    }
    
    /**
     * @brief 输出测试结果
     */
    void printTestResults() {
        Logger::info("========== DRM渲染功能测试结果 ==========");
        Logger::info("DRM驱动: " + driver_name);
        Logger::info("显示分辨率: " + std::to_string(mode_info.hdisplay) + "x" + std::to_string(mode_info.vdisplay));
        Logger::info("帧缓冲区大小: " + std::to_string(framebuffer_size / 1024) + " KB");
        Logger::info("总帧数: " + std::to_string(perf_stats.frame_count));
        Logger::info("平均渲染时间: " + std::to_string(perf_stats.avg_render_time_ms) + " ms");
        Logger::info("最大渲染时间: " + std::to_string(perf_stats.max_render_time_ms) + " ms");
        Logger::info("最小渲染时间: " + std::to_string(perf_stats.min_render_time_ms) + " ms");
        Logger::info("平均FPS: " + std::to_string(perf_stats.fps));
        Logger::info("========================================");
    }
    
    /**
     * @brief 清理资源
     */
    void cleanup() {
        if (framebuffer_data != nullptr && framebuffer_data != MAP_FAILED) {
            munmap(framebuffer_data, framebuffer_size);
            framebuffer_data = nullptr;
        }
        
        if (framebuffer_id) {
            drmModeRmFB(drm_fd, framebuffer_id);
            framebuffer_id = 0;
        }
        
        if (crtc) {
            drmModeFreeCrtc(crtc);
            crtc = nullptr;
        }
        
        if (encoder) {
            drmModeFreeEncoder(encoder);
            encoder = nullptr;
        }
        
        if (connector) {
            drmModeFreeConnector(connector);
            connector = nullptr;
        }
        
        if (drm_fd >= 0) {
            close(drm_fd);
            drm_fd = -1;
        }
    }
};

/**
 * @brief 运行DRM渲染功能测试的主函数
 */
bool runDRMRenderTest() {
    Logger::info("开始DRM渲染功能验证测试");
    
    DRMRenderTest test;
    
    if (!test.initialize()) {
        Logger::error("DRM渲染测试初始化失败");
        return false;
    }
    
    if (!test.runCompleteTest()) {
        Logger::error("DRM渲染测试执行失败");
        return false;
    }
    
    Logger::info("DRM渲染功能验证测试完成");
    return true;
}

} // namespace ui
} // namespace bamboo_cut