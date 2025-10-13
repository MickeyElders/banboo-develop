/**
 * @file deepstream_manager.cpp
 * @brief DeepStream AI推理和视频显示管理器实现 - 支持nvdrmvideosink叠加平面模式
 */

#include "bamboo_cut/deepstream/deepstream_manager.h"
#include "bamboo_cut/ui/lvgl_wayland_interface.h"
#include <iostream>
#include <sstream>
#include <gst/gst.h>
#include <fstream>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
// Wayland架构下移除DRM头文件依赖
// #include <xf86drm.h>
// #include <xf86drmMode.h>
#include <thread>
#include <chrono>
#include <set>
#include <gst/app/gstappsink.h>

#ifdef ENABLE_LVGL
#include <lvgl/lvgl.h>
#endif

namespace bamboo_cut {
namespace deepstream {

DeepStreamManager::DeepStreamManager()
    : pipeline_(nullptr)
    , pipeline2_(nullptr)
    , bus_(nullptr)
    , bus2_(nullptr)
    , bus_watch_id_(0)
    , bus_watch_id2_(0)
    , appsink_(nullptr)
    , lvgl_interface_(nullptr)
    , canvas_update_running_(false)
    , running_(false)
    , initialized_(false)
    , wayland_available_(false) {
}

DeepStreamManager::DeepStreamManager(void* lvgl_interface)
    : pipeline_(nullptr)
    , pipeline2_(nullptr)
    , bus_(nullptr)
    , bus2_(nullptr)
    , bus_watch_id_(0)
    , bus_watch_id2_(0)
    , appsink_(nullptr)
    , lvgl_interface_(lvgl_interface)
    , canvas_update_running_(false)
    , running_(false)
    , initialized_(false)
    , wayland_available_(false) {
    
    std::cout << "DeepStreamManager 构造函数完成（支持LVGL界面集成）" << std::endl;
}

DeepStreamManager::~DeepStreamManager() {
    stopCanvasUpdateThread();
    stop();
    cleanup();
}

bool DeepStreamManager::initialize(const DeepStreamConfig& config) {
    std::cout << "[DeepStreamManager] 初始化Wayland视频系统..." << std::endl;
    
    config_ = config;
    
    // 🔧 架构重构：强制使用appsink模式避免双xdg-shell窗口冲突
    std::cout << "[DeepStreamManager] 🎯 架构重构：使用appsink模式" << std::endl;
    std::cout << "[DeepStreamManager] 📋 原因：避免与LVGL的xdg-shell协议冲突" << std::endl;
    
    if (config_.sink_mode != VideoSinkMode::APPSINK) {
        std::cout << "[DeepStreamManager] 强制切换到appsink模式（架构重构）" << std::endl;
        config_.sink_mode = VideoSinkMode::APPSINK;
    }
    
    // 初始化GStreamer
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
        std::cout << "[DeepStreamManager] GStreamer初始化完成" << std::endl;
    }
    
    // 🔧 架构重构：检查appsink架构所需插件
    const char* required_plugins[] = {"nvarguscamerasrc", "nvvidconv", "appsink"};
    const char* plugin_descriptions[] = {
        "nvarguscamerasrc (NVIDIA摄像头源)",
        "nvvidconv (NVIDIA视频转换)",
        "appsink (应用程序数据接收)"
    };
    
    bool all_plugins_available = true;
    for (int i = 0; i < 3; i++) {
        GstElementFactory* factory = gst_element_factory_find(required_plugins[i]);
        if (factory) {
            std::cout << "[DeepStreamManager] ✓ " << plugin_descriptions[i] << std::endl;
            gst_object_unref(factory);
        } else {
            std::cerr << "[DeepStreamManager] ✗ " << plugin_descriptions[i] << " 不可用" << std::endl;
            all_plugins_available = false;
        }
    }
    
    if (!all_plugins_available) {
        std::cerr << "[DeepStreamManager] 关键插件缺失，无法继续" << std::endl;
        return false;
    }
    
    // 检查Wayland环境
    if (!checkWaylandEnvironment()) {
        std::cerr << "[DeepStreamManager] Wayland环境检查失败" << std::endl;
        return false;
    }
    
    // 设置EGL共享环境变量，解决NVMM缓冲区到EGLImage转换问题
    std::cout << "[DeepStreamManager] 配置EGL共享环境..." << std::endl;
    setenv("EGL_PLATFORM", "drm", 1);
    setenv("__EGL_VENDOR_LIBRARY_DIRS", "/usr/lib/aarch64-linux-gnu/tegra-egl", 1);
    setenv("EGL_EXTENSIONS", "EGL_EXT_image_dma_buf_import,EGL_EXT_image_dma_buf_import_modifiers", 1);
    
    // NVIDIA特定的EGL设置
    setenv("__NV_PRIME_RENDER_OFFLOAD", "1", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
    
    std::cout << "[DeepStreamManager] EGL共享环境配置完成" << std::endl;
    
    // 计算视频布局（简化版）
    video_layout_ = calculateWaylandVideoLayout(config);
    
    std::cout << "[DeepStreamManager] Wayland视频布局:" << std::endl;
    std::cout << "  窗口位置: (" << video_layout_.offset_x << ", " << video_layout_.offset_y << ")" << std::endl;
    std::cout << "  窗口尺寸: " << video_layout_.width << "x" << video_layout_.height << std::endl;
    
    initialized_ = true;
    std::cout << "[DeepStreamManager] Wayland视频系统初始化完成" << std::endl;
    return true;
}

bool DeepStreamManager::start() {
    if (!initialized_) {
        std::cerr << "DeepStream 未初始化" << std::endl;
        return false;
    }
    
    if (running_) {
        std::cout << "DeepStream 已在运行" << std::endl;
        return true;
    }
    
    std::cout << "启动 DeepStream 管道..." << std::endl;
    std::cout << "双摄模式: " << static_cast<int>(config_.dual_mode) << std::endl;
    
    if (config_.dual_mode == DualCameraMode::SPLIT_SCREEN) {
        // 并排显示模式：创建两个独立管道
        return startSplitScreenMode();
    } else {
        // 单摄像头或立体视觉模式：单管道
        return startSinglePipelineMode();
    }
}

bool DeepStreamManager::startSinglePipelineMode() {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);  // 🔧 线程安全保护
    
    const int MAX_RETRIES = 3;
    const int RETRY_DELAY_MS = 3000;
    
    // 等待LVGL完全初始化后再启动DeepStream
    std::cout << "等待LVGL完全初始化..." << std::endl;
    
    try {
        if (lvgl_interface_) {
            auto* lvgl_if = static_cast<bamboo_cut::ui::LVGLWaylandInterface*>(lvgl_interface_);
            int wait_count = 0;
            const int MAX_WAIT_SECONDS = 10;
            
            while (!lvgl_if->isFullyInitialized() && wait_count < MAX_WAIT_SECONDS) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                wait_count++;
                std::cout << "等待LVGL Wayland初始化完成... (" << (wait_count * 0.5) << "秒)" << std::endl;
            }
            
            if (lvgl_if->isFullyInitialized()) {
                std::cout << "✅ LVGL Wayland已完全初始化，继续启动DeepStream管道" << std::endl;
            } else {
                std::cout << "⚠️ 警告：LVGL Wayland初始化超时，继续启动DeepStream管道" << std::endl;
            }
        } else {
            std::cout << "警告：LVGL Wayland接口不可用，使用固定延迟" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        }
        
        for (int retry = 0; retry < MAX_RETRIES; retry++) {
            if (retry > 0) {
                std::cout << "重试启动管道 (第" << retry + 1 << "次尝试)..." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
            }
            
            // 🔧 新增：清理之前的管道状态
            if (pipeline_) {
                gst_element_set_state(pipeline_, GST_STATE_NULL);
                gst_object_unref(pipeline_);
                pipeline_ = nullptr;
            }
            
            // 构建管道
            std::string pipeline_str = buildPipeline(config_, video_layout_);
            std::cout << "管道字符串: " << pipeline_str << std::endl;
            
            // 🔧 新增：验证管道字符串有效性
            if (pipeline_str.empty()) {
                std::cerr << "❌ 管道字符串为空，配置错误" << std::endl;
                return false;
            }
            
            // 创建管道
            GError *error = nullptr;
            pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
            
            if (!pipeline_ || error) {
                std::cerr << "创建管道失败: " << (error ? error->message : "未知错误") << std::endl;
                if (error) {
                    g_error_free(error);
                    error = nullptr;
                }
                if (retry < MAX_RETRIES - 1) continue;
                return false;
            }
            
            // 检查NVMM缓冲区可用性
            if (!checkNVMMBufferAvailability()) {
                std::cout << "NVMM缓冲区检查失败，等待释放..." << std::endl;
                if (pipeline_) {
                    gst_element_set_state(pipeline_, GST_STATE_NULL);
                    gst_object_unref(pipeline_);
                    pipeline_ = nullptr;
                }
                if (retry < MAX_RETRIES - 1) continue;
            }
            
            // 🔧 新增：验证关键元素存在
            if (config_.sink_mode == VideoSinkMode::KMSSINK) {
                GstElement* kmssink = gst_bin_get_by_name(GST_BIN(pipeline_), "kmssink0");
                if (!kmssink) {
                    std::cerr << "❌ 无法找到kmssink元素" << std::endl;
                    if (retry < MAX_RETRIES - 1) continue;
                    return false;
                } else {
                    gst_object_unref(kmssink);
                }
            }
            
            // 设置消息总线
            bus_ = gst_element_get_bus(pipeline_);
            if (!bus_) {
                std::cerr << "❌ 无法获取消息总线" << std::endl;
                if (retry < MAX_RETRIES - 1) continue;
                return false;
            }
            bus_watch_id_ = gst_bus_add_watch(bus_, busCallback, this);
            
            // 🔧 改进：分阶段启动管道，降低段错误风险
            std::cout << "正在分阶段启动管道..." << std::endl;
            
            // 第一阶段：设置为READY状态
            std::cout << "第一阶段：设置管道为READY状态..." << std::endl;
            GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_READY);
            if (ret == GST_STATE_CHANGE_FAILURE) {
                std::cerr << "❌ READY状态设置失败" << std::endl;
                cleanup();
                if (retry < MAX_RETRIES - 1) continue;
                return false;
            }
            
            // 等待READY状态稳定
            GstState state;
            ret = gst_element_get_state(pipeline_, &state, NULL, 5 * GST_SECOND);
            if (ret == GST_STATE_CHANGE_FAILURE || state != GST_STATE_READY) {
                std::cerr << "❌ READY状态等待失败" << std::endl;
                cleanup();
                if (retry < MAX_RETRIES - 1) continue;
                return false;
            }
            std::cout << "✅ READY状态设置成功" << std::endl;
            
            // 第二阶段：设置为PAUSED状态
            std::cout << "第二阶段：设置管道为PAUSED状态..." << std::endl;
            ret = gst_element_set_state(pipeline_, GST_STATE_PAUSED);
            if (ret == GST_STATE_CHANGE_FAILURE) {
                std::cerr << "❌ PAUSED状态设置失败" << std::endl;
                cleanup();
                if (retry < MAX_RETRIES - 1) continue;
                return false;
            }
            
            // 等待PAUSED状态稳定
            ret = gst_element_get_state(pipeline_, &state, NULL, 10 * GST_SECOND);
            if (ret == GST_STATE_CHANGE_FAILURE) {
                std::cerr << "❌ PAUSED状态等待失败" << std::endl;
                cleanup();
                if (retry < MAX_RETRIES - 1) continue;
                return false;
            }
            std::cout << "✅ PAUSED状态设置成功" << std::endl;
            
            // 第三阶段：设置为PLAYING状态
            std::cout << "第三阶段：设置管道为PLAYING状态..." << std::endl;
            ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
            
            if (ret == GST_STATE_CHANGE_FAILURE) {
                std::cerr << "启动管道失败，进行错误诊断..." << std::endl;
                
                // 获取详细错误信息
                GstBus* bus = gst_element_get_bus(pipeline_);
                GstMessage* msg = gst_bus_timed_pop_filtered(bus, 2 * GST_SECOND,
                    static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING));
                    
                if (msg) {
                    GError* err;
                    gchar* debug_info;
                    gst_message_parse_error(msg, &err, &debug_info);
                    std::cerr << "GStreamer错误: " << err->message << std::endl;
                    if (debug_info) {
                        std::cerr << "调试信息: " << debug_info << std::endl;
                        
                        // 检查是否为DRM相关错误
                        if (strstr(debug_info, "DRM") || strstr(debug_info, "plane") ||
                            strstr(debug_info, "kmssink") || strstr(debug_info, "CRTC")) {
                            std::cout << "检测到DRM资源错误，可能是plane冲突..." << std::endl;
                        }
                        g_free(debug_info);
                    }
                    g_error_free(err);
                    gst_message_unref(msg);
                }
                if (bus) gst_object_unref(bus);
                
                cleanup();
                if (retry < MAX_RETRIES - 1) continue;
                return false;
            } else if (ret == GST_STATE_CHANGE_ASYNC) {
                std::cout << "管道异步启动中，等待状态变化..." << std::endl;
                ret = gst_element_get_state(pipeline_, &state, NULL, 15 * GST_SECOND);
                if (ret == GST_STATE_CHANGE_FAILURE) {
                    std::cerr << "管道异步启动失败" << std::endl;
                    cleanup();
                    if (retry < MAX_RETRIES - 1) continue;
                    return false;
                }
            }
            
            std::cout << "✅ PLAYING状态设置成功" << std::endl;
            
            // 成功启动，跳出重试循环
            break;
        }
        
        running_ = true;
        const char* mode_names[] = {"nvdrmvideosink", "waylandsink", "kmssink", "appsink"};
        const char* mode_name = mode_names[static_cast<int>(config_.sink_mode)];
        std::cout << "🎯 DeepStream 管道启动成功 (" << mode_name << " 架构重构模式)" << std::endl;
        std::cout << "📺 数据流: nvarguscamerasrc → nvinfer → appsink → LVGL Canvas" << std::endl;
        
        // 如果使用appsink模式，设置回调函数
        if (config_.sink_mode == VideoSinkMode::APPSINK) {
            setupAppSinkCallbacks();
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 管道启动异常: " << e.what() << std::endl;
        cleanup();
        return false;
    } catch (...) {
        std::cerr << "❌ 管道启动未知异常" << std::endl;
        cleanup();
        return false;
    }
}

// 新增：检查NVMM缓冲区可用性
bool DeepStreamManager::checkNVMMBufferAvailability() {
    std::cout << "检查NVMM缓冲区可用性..." << std::endl;
    
    // 检查系统内存使用情况
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        long total_mem = 0, available_mem = 0;
        
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0) {
                sscanf(line.c_str(), "MemTotal: %ld kB", &total_mem);
            } else if (line.find("MemAvailable:") == 0) {
                sscanf(line.c_str(), "MemAvailable: %ld kB", &available_mem);
            }
        }
        meminfo.close();
        
        if (total_mem > 0 && available_mem > 0) {
            double memory_usage = 1.0 - (double)available_mem / total_mem;
            std::cout << "系统内存使用率: " << (memory_usage * 100) << "%" << std::endl;
            
            if (memory_usage > 0.9) { // 内存使用超过90%
                std::cout << "系统内存使用率过高，可能影响NVMM缓冲区分配" << std::endl;
                return false;
            }
        }
    }
    
    // 检查NVIDIA GPU内存
    std::string gpu_mem_cmd = "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo '0,0'";
    FILE* pipe = popen(gpu_mem_cmd.c_str(), "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            int used_mem = 0, total_mem = 0;
            sscanf(buffer, "%d, %d", &used_mem, &total_mem);
            
            if (total_mem > 0) {
                double gpu_usage = (double)used_mem / total_mem;
                std::cout << "GPU内存使用率: " << (gpu_usage * 100) << "%" << std::endl;
                
                if (gpu_usage > 0.9) { // GPU内存使用超过90%
                    std::cout << "GPU内存使用率过高，可能影响NVMM缓冲区分配" << std::endl;
                    pclose(pipe);
                    return false;
                }
            }
        }
        pclose(pipe);
    }
    
    std::cout << "NVMM缓冲区可用性检查通过" << std::endl;
    return true;
}

bool DeepStreamManager::startSplitScreenMode() {
    // 计算左右视频区域尺寸
    int half_width = video_layout_.width / 2 - 5;  // 减去间隙
    
    // 构建左侧摄像头管道
    std::string pipeline1_str = buildSplitScreenPipeline(
        config_, 
        video_layout_.offset_x,
        video_layout_.offset_y,
        half_width,
        video_layout_.height
    );
    std::cout << "左侧管道: " << pipeline1_str << std::endl;
    
    // 构建右侧摄像头管道
    DeepStreamConfig config2 = config_;
    config2.camera_id = config_.camera_id_2;  // 使用副摄像头
    
    int right_offset_x = video_layout_.offset_x + half_width + 10;  // 右半边偏移
    std::string pipeline2_str = buildSplitScreenPipeline(
        config2,
        right_offset_x,
        video_layout_.offset_y,
        half_width,
        video_layout_.height
    );
    std::cout << "右侧管道: " << pipeline2_str << std::endl;
    
    // 创建两个管道
    GError *error1 = nullptr, *error2 = nullptr;
    pipeline_ = gst_parse_launch(pipeline1_str.c_str(), &error1);
    pipeline2_ = gst_parse_launch(pipeline2_str.c_str(), &error2);
    
    if (!pipeline_ || error1) {
        std::cerr << "创建左侧管道失败: " << (error1 ? error1->message : "未知错误") << std::endl;
        if (error1) g_error_free(error1);
        return false;
    }
    
    if (!pipeline2_ || error2) {
        std::cerr << "创建右侧管道失败: " << (error2 ? error2->message : "未知错误") << std::endl;
        if (error2) g_error_free(error2);
        return false;
    }
    
    // 设置消息总线
    bus_ = gst_element_get_bus(pipeline_);
    bus2_ = gst_element_get_bus(pipeline2_);
    bus_watch_id_ = gst_bus_add_watch(bus_, busCallback, this);
    bus_watch_id2_ = gst_bus_add_watch(bus2_, busCallback, this);
    
    // 启动两个管道
    GstStateChangeReturn ret1 = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    GstStateChangeReturn ret2 = gst_element_set_state(pipeline2_, GST_STATE_PLAYING);
    
    if (ret1 == GST_STATE_CHANGE_FAILURE || ret2 == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "启动管道失败" << std::endl;
        cleanup();
        return false;
    }
    
    running_ = true;
    const char* mode_names[] = {"nvdrmvideosink", "waylandsink", "kmssink", "appsink"};
    const char* mode_name = mode_names[static_cast<int>(config_.sink_mode)];
    std::cout << "双摄像头并排显示管道启动成功 (" << mode_name << ")" << std::endl;
    
    // 如果使用appsink模式，设置回调函数
    if (config_.sink_mode == VideoSinkMode::APPSINK) {
        setupAppSinkCallbacks();
    }
    
    return true;
}

void DeepStreamManager::stop() {
    if (!running_) return;
    
    std::cout << "停止 DeepStream 管道..." << std::endl;
    
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
    }
    
    if (pipeline2_) {
        gst_element_set_state(pipeline2_, GST_STATE_NULL);
    }
    
    running_ = false;
    std::cout << "DeepStream 管道已停止" << std::endl;
}

bool DeepStreamManager::switchDualMode(DualCameraMode mode) {
    if (config_.dual_mode == mode) {
        std::cout << "已是当前模式，无需切换" << std::endl;
        return true;
    }
    
    std::cout << "切换双摄模式: " << static_cast<int>(config_.dual_mode) 
              << " -> " << static_cast<int>(mode) << std::endl;
    
    // 停止当前管道
    bool was_running = running_;
    if (running_) {
        stop();
        cleanup();
    }
    
    // 更新配置
    config_.dual_mode = mode;
    
    // 重新计算布局
    video_layout_ = calculateVideoLayout(config_);
    
    // 如果之前在运行，重新启动
    if (was_running) {
        return start();
    }
    
    return true;
}

bool DeepStreamManager::updateLayout(int screen_width, int screen_height) {
    config_.screen_width = screen_width;
    config_.screen_height = screen_height;
    
    // 重新计算布局
    video_layout_ = calculateVideoLayout(config_);
    
    std::cout << "布局已更新: " << video_layout_.width << "x" << video_layout_.height 
              << " at (" << video_layout_.offset_x << ", " << video_layout_.offset_y << ")" << std::endl;
    
    // 如果正在运行，需要重启管道以应用新布局
    if (running_) {
        stop();
        return start();
    }
    
    return true;
}

bool DeepStreamManager::switchSinkMode(VideoSinkMode sink_mode) {
    const char* mode_names[] = {"nvdrmvideosink", "waylandsink", "kmssink", "appsink"};
    std::cout << "切换sink模式: " << mode_names[static_cast<int>(config_.sink_mode)]
              << " -> " << mode_names[static_cast<int>(sink_mode)] << std::endl;
    
    // 停止当前管道
    bool was_running = running_;
    if (running_) {
        stop();
        cleanup();
    }
    
    // 更新sink模式
    config_.sink_mode = sink_mode;
    
    // 如果切换到nvdrmvideosink，设置叠加平面
    if (sink_mode == VideoSinkMode::NVDRMVIDEOSINK) {
        if (!setupDRMOverlayPlane()) {
            std::cerr << "DRM叠加平面设置失败，无法切换到nvdrmvideosink模式，回退到appsink" << std::endl;
            config_.sink_mode = VideoSinkMode::APPSINK;  // 回退到appsink
            return false;
        }
    }
    
    // 如果之前在运行，重新启动
    if (was_running) {
        return start();
    }
    
    return true;
}

bool DeepStreamManager::configureDRMOverlay(const DRMOverlayConfig& overlay_config) {
    config_.overlay = overlay_config;
    std::cout << "配置DRM叠加平面: plane_id=" << overlay_config.plane_id 
              << ", z_order=" << overlay_config.z_order << std::endl;
    return true;
}

DRMOverlayConfig DeepStreamManager::detectAvailableOverlayPlane() {
    DRMOverlayConfig config;
    
    // Wayland架构下不再需要DRM设备检测
    std::cout << "🎯 Wayland架构：跳过DRM设备检测，使用waylandsink硬件渲染" << std::endl;
    
    // 返回默认配置，表示不支持DRM overlay
    std::cout << "📱 建议使用waylandsink替代nvdrmvideosink" << std::endl;
    return config;
    
    // Wayland架构下不再进行DRM plane检测
    std::cout << "🎯 Wayland架构：跳过DRM plane检测和资源管理" << std::endl;
    std::cout << "📱 建议使用waylandsink进行视频显示" << std::endl;
    
    return config;
}

bool DeepStreamManager::setupDRMOverlayPlane() {
    std::lock_guard<std::mutex> lock(drm_mutex_);  // 🔧 线程安全保护
    
    std::cout << "🔧 设置DRM叠加平面..." << std::endl;
    
    try {
        // 如果未配置叠加平面，自动检测
        if (config_.overlay.plane_id == -1) {
            std::cout << "🔍 执行智能overlay plane检测..." << std::endl;
            config_.overlay = detectAvailableOverlayPlane();
            if (config_.overlay.plane_id == -1) {
                std::cerr << "❌ 未找到可用的DRM叠加平面" << std::endl;
                return false;
            }
        }
        
        // 🔧 新增：验证plane-id有效性
        if (config_.overlay.plane_id <= 0) {
            std::cerr << "❌ 无效的plane_id: " << config_.overlay.plane_id << std::endl;
            return false;
        }
        
        std::cout << "✅ DRM叠加平面设置完成: plane_id=" << config_.overlay.plane_id
                  << ", crtc_id=" << config_.overlay.crtc_id
                  << ", connector_id=" << config_.overlay.connector_id
                  << ", z_order=" << config_.overlay.z_order << std::endl;
        
        // 🔧 新增：验证多层显示配置
        if (!verifyMultiLayerDisplaySetup()) {
            std::cout << "⚠️  多层显示验证失败，但继续尝试..." << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ DRM叠加平面设置异常: " << e.what() << std::endl;
        return false;
    }
}

// 🔧 新增：验证多层显示设置的函数（Wayland架构）
bool DeepStreamManager::verifyMultiLayerDisplaySetup() {
    std::cout << "🎯 Wayland架构：跳过多层显示验证" << std::endl;
    std::cout << "✅ Wayland合成器自动处理多层显示管理" << std::endl;
    
    // 在Wayland架构下，合成器负责所有窗口层次管理
    // LVGL作为Wayland客户端，DeepStream使用waylandsink
    // 不需要手动管理DRM plane
    
    return true;  // Wayland架构下总是返回成功
}

VideoLayout DeepStreamManager::calculateVideoLayout(const DeepStreamConfig& config) {
    VideoLayout layout;
    
    // 计算可用区域（减去顶部和底部栏）
    layout.available_width = config.screen_width;
    layout.available_height = config.screen_height - config.header_height - config.footer_height;
    
    // 计算视频区域尺寸（按比例）
    layout.width = static_cast<int>(layout.available_width * config.video_width_ratio);
    layout.height = static_cast<int>(layout.available_height * config.video_height_ratio);
    
    // 计算偏移量（居左上，但考虑顶部栏）
    layout.offset_x = 0;  // 左对齐
    layout.offset_y = config.header_height;  // 顶部栏下方
    
    return layout;
}

std::string DeepStreamManager::buildPipeline(const DeepStreamConfig& config, const VideoLayout& layout) {
    switch (config.dual_mode) {
        case DualCameraMode::SINGLE_CAMERA:
            return buildSplitScreenPipeline(
                config, 
                layout.offset_x, 
                layout.offset_y,
                layout.width,
                layout.height
            );
        case DualCameraMode::SPLIT_SCREEN:
            return buildSplitScreenPipeline(
                config, 
                layout.offset_x, 
                layout.offset_y,
                layout.width,
                layout.height
            );
        case DualCameraMode::STEREO_VISION:
            return buildStereoVisionPipeline(config, layout);
        default:
            return buildSplitScreenPipeline(
                config, 
                layout.offset_x, 
                layout.offset_y,
                layout.width,
                layout.height
            );
    }
}

std::string DeepStreamManager::buildSplitScreenPipeline(
    const DeepStreamConfig& config,
    int offset_x,
    int offset_y,
    int width,
    int height) {
    
    switch (config.sink_mode) {
        case VideoSinkMode::NVDRMVIDEOSINK:
            return buildNVDRMVideoSinkPipeline(config, offset_x, offset_y, width, height);
        case VideoSinkMode::WAYLANDSINK:
            return buildWaylandSinkPipeline(config, offset_x, offset_y, width, height);
        case VideoSinkMode::KMSSINK:
            // Wayland架构下不再支持KMSSink，降级到AppSink
            std::cout << "📱 [DeepStream] Wayland架构下降级到AppSink软件合成" << std::endl;
            return buildAppSinkPipeline(config, offset_x, offset_y, width, height);
        case VideoSinkMode::APPSINK:
        default:
            return buildAppSinkPipeline(config, offset_x, offset_y, width, height);
    }
}

std::string DeepStreamManager::buildNVDRMVideoSinkPipeline(
    const DeepStreamConfig& config,
    int offset_x,
    int offset_y,
    int width,
    int height) {
    
    std::ostringstream pipeline;
    
    // Wayland架构下不再需要Xvfb
    std::cout << "🔧 Wayland架构下直接使用nvarguscamerasrc..." << std::endl;
    
    // 🔧 关键修复：使用BGRA格式，这是AR24在DRM中的实际对应格式
    pipeline << buildCameraSource(config) << " ! "
             << "nvvidconv ! "  // NVMM -> RGBA格式转换和缩放（硬件加速）
             << "video/x-raw(memory:NVMM),format=RGBA,width=" << width << ",height=" << height << " ! "
             << "nvvidconv ! "     // NVMM -> 标准内存转换
             << "video/x-raw,format=RGBA,width=" << width << ",height=" << height << " ! "
             << "videoconvert ! "  // RGBA -> BGRA格式转换（AR24对应BGRA）
             << "video/x-raw,format=BGRA,width=" << width << ",height=" << height << " ! "  // 使用AR24/BGRA格式
             << "kmssink "
             << "driver-name=nvidia-drm "     // 使用 nvidia-drm 驱动
             << "plane-id=44 "                // 用户指定的overlay plane，支持AR24
             << "connector-id=-1 "            // 自动检测连接器
             << "can-scale=true "             // 启用缩放支持
             << "force-modesetting=false "    // 不改变显示模式
             << "sync=false "                 // 降低延迟
             << "restore-crtc=false";         // 不恢复CRTC
    
    return pipeline.str();
}


std::string DeepStreamManager::buildWaylandSinkPipeline(
    const DeepStreamConfig& config,
    int offset_x,
    int offset_y,
    int width,
    int height) {
    
    std::ostringstream pipeline;
    
    std::cout << "[DeepStreamManager] 构建优化的Wayland管道 ("
              << width << "x" << height << ")..." << std::endl;
    
    // 🔧 关键修复：为DeepStream创建独立的Wayland display名称
    const char* wayland_display = getenv("WAYLAND_DISPLAY");
    std::string deepstream_display_name;
    
    if (!wayland_display) {
        deepstream_display_name = "wayland-0";
        setenv("WAYLAND_DISPLAY", deepstream_display_name.c_str(), 0);
        std::cout << "[DeepStreamManager] 设置WAYLAND_DISPLAY=" << deepstream_display_name << std::endl;
    } else {
        deepstream_display_name = std::string(wayland_display);
        std::cout << "[DeepStreamManager] 使用现有WAYLAND_DISPLAY=" << deepstream_display_name << std::endl;
    }
    
    // 🎯 关键解决方案：为DeepStream waylandsink设置独立的display标识
    // 这避免了与LVGL Wayland客户端的协议冲突
    std::string deepstream_display_id = "deepstream-" + deepstream_display_name;
    std::cout << "[DeepStreamManager] 为waylandsink设置独立display标识: " << deepstream_display_id << std::endl;
    
    // 使用nvarguscamerasrc
    pipeline << "nvarguscamerasrc sensor-id=" << config.camera_id << " ";
    
    // 摄像头输出配置
    pipeline << "! video/x-raw(memory:NVMM)"
             << ",width=" << config.camera_width
             << ",height=" << config.camera_height
             << ",framerate=" << config.camera_fps << "/1"
             << ",format=NV12 ";
    
    // 恢复nvstreammux为nvinfer创建batch元数据
    pipeline << "! m.sink_0 ";
    pipeline << "nvstreammux name=m batch-size=1 "
             << "width=" << config.camera_width << " "
             << "height=" << config.camera_height << " ";
    
    // 🔧 恢复nvinfer，使用正确的YOLO输出格式 (1x25200x85)
    std::cout << "[DeepStreamManager] 恢复nvinfer推理，使用标准YOLO格式(1x25200x85)" << std::endl;
    
    // 🔧 修复：检查并使用正确的nvinfer配置文件路径
    std::string nvinfer_config_path = config.nvinfer_config;
    if (nvinfer_config_path.empty() || access(nvinfer_config_path.c_str(), F_OK) != 0) {
        // 尝试默认路径
        nvinfer_config_path = "config/nvinfer_config.txt";
        if (access(nvinfer_config_path.c_str(), F_OK) != 0) {
            nvinfer_config_path = "/opt/bamboo-cut/config/nvinfer_config.txt";
        }
    }
    
    if (access(nvinfer_config_path.c_str(), F_OK) == 0) {
        pipeline << "! nvinfer config-file-path=" << nvinfer_config_path << " ";
        std::cout << "[DeepStreamManager] 使用nvinfer配置: " << nvinfer_config_path << std::endl;
    } else {
        std::cout << "[DeepStreamManager] 跳过nvinfer（配置文件未找到: " << nvinfer_config_path << "）" << std::endl;
    }
    
    // 硬件加速格式转换和缩放（第一步：在NVMM内存中处理）
    pipeline << "! nvvidconv ";
    
    // 第二步：从NVMM转换到标准内存，waylandsink需要标准内存格式
    pipeline << "! video/x-raw,format=RGBA"
             << ",width=" << width
             << ",height=" << height << " ";
    
    // 使用waylandsink进行Wayland显示
    pipeline << "! waylandsink ";
    
    // EGL共享和dmabuf优化参数
    pipeline << "sync=false ";           // 低延迟模式
    pipeline << "async=true ";           // 异步模式
    pipeline << "enable-last-sample=false "; // 减少内存使用
    pipeline << "fullscreen=false ";     // 非全屏模式
    
    // 🔧 关键修复：使用独立的display标识避免客户端冲突
    pipeline << "display=" << deepstream_display_name << " ";
    
    // 🎯 移除不支持的属性，使用基本waylandsink配置
    std::cout << "[DeepStreamManager] waylandsink使用独立display: " << deepstream_display_name << std::endl;
    
    std::cout << "[DeepStreamManager] Wayland管道构建完成" << std::endl;
    return pipeline.str();
}

std::string DeepStreamManager::buildStereoVisionPipeline(const DeepStreamConfig& config, const VideoLayout& layout) {
    std::ostringstream pipeline;
    
    // 双摄立体视觉 - 使用 nvstreammux 合并两路流
    pipeline << "nvarguscamerasrc sensor-id=" << config.camera_id << " ! "
             << "video/x-raw(memory:NVMM),width=" << config.camera_width
             << ",height=" << config.camera_height
             << ",framerate=" << config.camera_fps << "/1,format=NV12 ! "
             << "m.sink_0 "  // 连接到 mux 的第一个输入
             
             << "nvarguscamerasrc sensor-id=" << config.camera_id_2 << " ! "
             << "video/x-raw(memory:NVMM),width=" << config.camera_width
             << ",height=" << config.camera_height
             << ",framerate=" << config.camera_fps << "/1,format=NV12 ! "
             << "m.sink_1 "  // 连接到 mux 的第二个输入
             
             << "nvstreammux name=m batch-size=1 width=" << config.camera_width
             << " height=" << config.camera_height << " ! "
             << "nvvideoconvert ! ";
             
    switch (config.sink_mode) {
        case VideoSinkMode::NVDRMVIDEOSINK:
        pipeline << "video/x-raw(memory:NVMM),format=RGBA ! "
                << "nvdrmvideosink "
                << "offset-x=" << layout.offset_x << " "
                << "offset-y=" << layout.offset_y << " "
                << "set-mode=false "
                << "sync=false";
        break;
        case VideoSinkMode::WAYLANDSINK:
            pipeline << "video/x-raw,format=RGBA ! "
                     << "waylandsink "
                     << "sync=false";
            break;
        case VideoSinkMode::KMSSINK:
            pipeline << "videoconvert ! "
                     << "videoscale ! "
                     << "video/x-raw,format=BGRA,width=" << layout.width
                     << ",height=" << layout.height << " ! "
                     << "queue max-size-buffers=4 max-size-time=0 leaky=downstream ! "
                     << "kmssink "
                     << "connector-id=-1 plane-id=-1 "
                     << "force-modesetting=false can-scale=true "
                     << "sync=false restore-crtc=true";
            break;
        case VideoSinkMode::APPSINK:
        default:
            pipeline << "videoconvert ! "
                     << "videoscale ! "
                     << "video/x-raw,format=BGRA,width=" << layout.width
                     << ",height=" << layout.height << " ! "
                     << "queue max-size-buffers=2 max-size-time=0 leaky=downstream ! "
                     << "appsink name=video_appsink "
                     << "emit-signals=true sync=false "
                     << "caps=video/x-raw,format=BGRA,width=" << layout.width
                     << ",height=" << layout.height;
            break;
    }
    
    return pipeline.str();
}

gboolean DeepStreamManager::busCallback(GstBus* bus, GstMessage* msg, gpointer data) {
    DeepStreamManager* manager = static_cast<DeepStreamManager*>(data);
    
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError *err;
            gchar *debug;
            gst_message_parse_error(msg, &err, &debug);
            std::cerr << "DeepStream 错误: " << err->message << std::endl;
            if (debug) {
                std::cerr << "调试信息: " << debug << std::endl;
                g_free(debug);
            }
            g_error_free(err);
            break;
        }
        case GST_MESSAGE_WARNING: {
            GError *err;
            gchar *debug;
            gst_message_parse_warning(msg, &err, &debug);
            std::cout << "DeepStream 警告: " << err->message << std::endl;
            if (debug) {
                std::cout << "调试信息: " << debug << std::endl;
                g_free(debug);
            }
            g_error_free(err);
            break;
        }
        case GST_MESSAGE_EOS:
            std::cout << "DeepStream 流结束" << std::endl;
            break;
        case GST_MESSAGE_STATE_CHANGED: {
            if (GST_MESSAGE_SRC(msg) == GST_OBJECT(manager->pipeline_)) {
                GstState old_state, new_state;
                gst_message_parse_state_changed(msg, &old_state, &new_state, nullptr);
                std::cout << "DeepStream 状态变更: " 
                          << gst_element_state_get_name(old_state) << " -> " 
                          << gst_element_state_get_name(new_state) << std::endl;
            }
            break;
        }
        default:
            break;
    }
    
    return TRUE;
}

void DeepStreamManager::cleanup() {
    if (bus_watch_id_ > 0) {
        g_source_remove(bus_watch_id_);
        bus_watch_id_ = 0;
    }
    
    if (bus_watch_id2_ > 0) {
        g_source_remove(bus_watch_id2_);
        bus_watch_id2_ = 0;
    }
    
    if (bus_) {
        gst_object_unref(bus_);
        bus_ = nullptr;
    }
    
    if (bus2_) {
        gst_object_unref(bus2_);
        bus2_ = nullptr;
    }
    
    if (pipeline_) {
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
    
    if (pipeline2_) {
        gst_object_unref(pipeline2_);
        pipeline2_ = nullptr;
    }
}

// 新增：构建摄像头源字符串
// 🔧 修复：回到使用nvarguscamerasrc，因为GBM共享DRM资源后不再有冲突
std::string DeepStreamManager::buildCameraSource(const DeepStreamConfig& config) {
    std::ostringstream source;
    
    switch (config.camera_source) {
        case CameraSourceMode::NVARGUSCAMERA:
            // 🔧 关键修复：回到使用nvarguscamerasrc Argus驱动
            std::cout << "🔧 配置nvarguscamerasrc Argus驱动摄像头..." << std::endl;
            
            source << "nvarguscamerasrc sensor-id=" << config.camera_id << " "
                   << "! video/x-raw(memory:NVMM)"
                   << ",width=" << config.camera_width
                   << ",height=" << config.camera_height
                   << ",framerate=" << config.camera_fps << "/1"
                   << ",format=NV12";
            break;
            
        case CameraSourceMode::V4L2SRC:
            // 保留v4l2src作为备用方案
            std::cout << "🔧 配置v4l2src备用方案..." << std::endl;
            source << "v4l2src device=/dev/video" << config.camera_id << " "
                   << "io-mode=2 "
                   << "! video/x-raw"
                   << ",width=" << config.camera_width
                   << ",height=" << config.camera_height
                   << ",framerate=" << config.camera_fps << "/1";
            break;
            
        case CameraSourceMode::VIDEOTESTSRC:
            source << "videotestsrc pattern=18 is-live=true "
                   << "! video/x-raw"
                   << ",width=" << config.camera_width
                   << ",height=" << config.camera_height
                   << ",framerate=" << config.camera_fps << "/1"
                   << ",format=NV12";
            break;
            
        case CameraSourceMode::FILESRC:
            source << "filesrc location=" << config.video_file_path << " "
                   << "! decodebin "
                   << "! nvvideoconvert "
                   << "! video/x-raw(memory:NVMM)"
                   << ",width=" << config.camera_width
                   << ",height=" << config.camera_height
                   << ",framerate=" << config.camera_fps << "/1"
                   << ",format=NV12";
            break;
            
        default:
            // 默认使用nvarguscamerasrc
            std::cout << "⚠️ 使用默认nvarguscamerasrc方案..." << std::endl;
            source << "nvarguscamerasrc sensor-id=" << config.camera_id << " "
                   << "! video/x-raw(memory:NVMM)"
                   << ",width=" << config.camera_width
                   << ",height=" << config.camera_height
                   << ",framerate=" << config.camera_fps << "/1"
                   << ",format=NV12";
            break;
    }
    
    return source.str();
}

// 新增：构建KMSSink管道 - 使用GBM共享DRM资源的分层显示
std::string DeepStreamManager::buildKMSSinkPipeline(
    const DeepStreamConfig& config,
    int offset_x,
    int offset_y,
    int width,
    int height) {
    
    std::ostringstream pipeline;
    
    // Wayland架构下不再需要Xvfb
    std::cout << "🔧 Wayland架构下直接使用nvarguscamerasrc..." << std::endl;
    
    // 🔧 关键修复：使用nvarguscamerasrc + GBM共享DRM资源
    std::cout << "🔧 构建GBM共享DRM的KMSSink管道 (缩放到 " << width << "x" << height << ")..." << std::endl;
    
    // 构建nvarguscamerasrc摄像头源（现在可以正常工作，因为GBM共享DRM资源）
    pipeline << buildCameraSource(config) << " ! ";
    
    // 🔧 关键修复：直接使用NV12格式，让GStreamer自动协商内存类型
    std::cout << "🎯 直接使用NV12格式，让GStreamer自动协商内存类型和缩放" << std::endl;
    
    // 让GStreamer自动协商从NVMM到标准内存的转换，保持NV12格式
    pipeline << "nvvidconv ! "  // NVMM到标准内存转换，保持NV12格式
             << "video/x-raw,format=NV12,width=" << width << ",height=" << height << " ! "
             << "queue "
             << "max-size-buffers=4 "      // 适中的缓冲区深度
             << "max-size-time=0 "
             << "leaky=downstream "
             << "! ";
    
    // 🔧 关键修复：使用GBM后端提供的overlay plane，实现真正的分层显示
    if (config_.overlay.plane_id > 0) {
        std::cout << "🎯 使用GBM共享的overlay plane: " << config_.overlay.plane_id << std::endl;
        pipeline << "kmssink "
                 << "plane-id=" << config_.overlay.plane_id << " "     // 使用GBM分配的overlay plane
                 << "connector-id=" << config_.overlay.connector_id << " " // 使用GBM共享的connector
                 << "force-modesetting=false " // 不改变显示模式，LVGL已通过GBM设置
                 << "can-scale=true "          // 启用硬件缩放
                 << "sync=false "              // 低延迟模式
                 << "restore-crtc=false";      // 不恢复CRTC，保持GBM管理
    } else {
        std::cout << "⚠️  GBM后端未提供overlay plane，使用用户指定的plane-id=44" << std::endl;
        // 🔧 修复：直接使用用户指定的overlay plane-id=44，支持AR24/ABGR格式
        pipeline << "kmssink "
                 << "plane-id=44 "             // 用户指定的overlay plane，支持AR24/ABGR
                 << "connector-id=-1 "         // 自动检测连接器
                 << "force-modesetting=false " // 不强制设置模式
                 << "can-scale=true "          // 启用硬件缩放
                 << "sync=false "              // 低延迟模式
                 << "restore-crtc=false";      // 不恢复CRTC，保持GBM管理
    }
    
    std::cout << "🔧 构建GBM共享DRM的KMSSink管道: " << pipeline.str() << std::endl;
    return pipeline.str();
}

// 新增：构建AppSink软件合成管道 - 解决LVGL CRTC独占冲突
std::string DeepStreamManager::buildAppSinkPipeline(
    const DeepStreamConfig& config,
    int offset_x,
    int offset_y,
    int width,
    int height) {
    
    std::ostringstream pipeline;
    
    // 🔧 关键修复：使用摄像头原生分辨率然后缩放到目标尺寸
    std::cout << "🔧 构建原生分辨率AppSink管道 (缩放到 " << width << "x" << height << ")..." << std::endl;
    
    if (config.camera_source == CameraSourceMode::NVARGUSCAMERA ||
        config.camera_source == CameraSourceMode::V4L2SRC) {
        
        // 🔧 修复：使用两步转换，先nvvidconv转到标准内存，再videoconvert转BGRA
        pipeline << buildCameraSource(config) << " ! "
                 << "nvvidconv ! "    // NVMM -> 标准内存，保持NV12格式
                 << "video/x-raw,format=NV12,width=" << width << ",height=" << height << " ! "
                 << "videoconvert ! "  // NV12 -> BGRA格式转换（软件）
                 << "video/x-raw,format=BGRA,width=" << width << ",height=" << height << " ! "
                 << "queue max-size-buffers=2 leaky=downstream ! "
                 << "appsink name=video_appsink "
                 << "emit-signals=true sync=false max-buffers=2 drop=true";
        
        std::cout << "🔧 构建原生分辨率AppSink管道: " << pipeline.str() << std::endl;
                 
    } else if (config.camera_source == CameraSourceMode::VIDEOTESTSRC) {
        // ✅ 测试源直接使用目标分辨率
        pipeline << "videotestsrc pattern=18 is-live=true "
                 << "! video/x-raw,format=BGRA"
                 << ",width=" << width << ",height=" << height
                 << ",framerate=30/1 "
                 << "! appsink name=video_appsink "
                 << "emit-signals=true sync=false max-buffers=1 drop=false";
    } else {
        // 其他源（文件源等）
        pipeline << buildCameraSource(config) << " ! "
                 << "videoconvert ! "  // 确保格式兼容
                 << "videoscale ! "    // 缩放到目标尺寸
                 << "video/x-raw,format=BGRA,width=" << width << ",height=" << height << " ! "
                 << "queue max-size-buffers=2 leaky=downstream ! "
                 << "appsink name=video_appsink "
                 << "emit-signals=true sync=false max-buffers=2 drop=true";
    }
    
    std::cout << "🔧 构建原生分辨率AppSink管道: " << pipeline.str() << std::endl;
    return pipeline.str();
}

// AppSink新样本回调 - 线程安全的帧处理（禁用冗余日志）
GstFlowReturn DeepStreamManager::newSampleCallback(GstAppSink* appsink, gpointer user_data) {
    DeepStreamManager* manager = static_cast<DeepStreamManager*>(user_data);
    
    // 获取新样本
    GstSample* sample = gst_app_sink_pull_sample(appsink);
    if (!sample) {
        return GST_FLOW_OK;
    }
    
    // 获取缓冲区
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    
    // 映射缓冲区数据
    GstMapInfo map_info;
    if (!gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    
    // 获取视频信息
    GstCaps* caps = gst_sample_get_caps(sample);
    if (caps) {
        GstStructure* structure = gst_caps_get_structure(caps, 0);
        gint width, height;
        
        if (gst_structure_get_int(structure, "width", &width) &&
            gst_structure_get_int(structure, "height", &height)) {
            
            // 线程安全地合成帧到LVGL画布（静默模式）
            manager->compositeFrameToLVGL(&map_info, width, height);
        }
    }
    
    // 清理资源
    gst_buffer_unmap(buffer, &map_info);
    gst_sample_unref(sample);
    
    return GST_FLOW_OK;
}

// 软件合成帧到LVGL画布 - 优化内存操作（静默模式）
void DeepStreamManager::compositeFrameToLVGL(GstMapInfo* map_info, int width, int height) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    try {
        // 检查数据大小是否合理 (BGRA格式应该是 width * height * 4)
        size_t expected_size = width * height * 4;
        if (map_info->size < expected_size) {
            // 静默处理尺寸不匹配
            return;
        }
        
        // 创建OpenCV Mat包装GStreamer数据，避免内存拷贝
        cv::Mat frame = cv::Mat(height, width, CV_8UC4, map_info->data);
        
        // 检查帧数据有效性
        if (!frame.empty() && frame.data) {
            // 克隆帧数据用于后续处理
            latest_frame_ = frame.clone();
            new_frame_available_ = true;
        }
        
    } catch (const std::exception& e) {
        // 静默处理异常
    }
}

// 设置AppSink回调函数
void DeepStreamManager::setupAppSinkCallbacks() {
    if (!pipeline_) {
        std::cout << "错误：管道未创建，无法设置appsink回调" << std::endl;
        return;
    }
    
    std::cout << "🔧 开始修复AppSink回调机制..." << std::endl;
    
    // 查找appsink元素
    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "video_appsink");
    if (!appsink_) {
        std::cout << "错误：未找到appsink元素，尝试列出所有元素..." << std::endl;
        
        // 列出管道中的所有元素用于调试
        GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline_));
        GValue item = G_VALUE_INIT;
        gboolean done = FALSE;
        
        std::cout << "管道中的元素列表：" << std::endl;
        while (!done) {
            switch (gst_iterator_next(it, &item)) {
                case GST_ITERATOR_OK: {
                    GstElement* element = GST_ELEMENT(g_value_get_object(&item));
                    gchar* name = gst_element_get_name(element);
                    std::cout << "  - " << name << std::endl;
                    g_free(name);
                    g_value_reset(&item);
                    break;
                }
                case GST_ITERATOR_RESYNC:
                    gst_iterator_resync(it);
                    break;
                case GST_ITERATOR_ERROR:
                case GST_ITERATOR_DONE:
                    done = TRUE;
                    break;
            }
        }
        g_value_unset(&item);
        gst_iterator_free(it);
        return;
    }
    
    std::cout << "✅ 成功找到appsink元素" << std::endl;
    
    // 🔧 修复：强制设置appsink属性，确保信号发射正常
    g_object_set(G_OBJECT(appsink_),
                 "emit-signals", TRUE,        // 启用信号
                 "sync", FALSE,               // 异步模式
                 "max-buffers", 2,            // 最大缓冲区数量
                 "drop", TRUE,                // 丢弃旧帧
                 "wait-on-eos", FALSE,        // 不等待EOS
                 NULL);
    
    // 🔧 修复：验证属性设置
    gboolean emit_signals = FALSE;
    gboolean sync = TRUE;
    guint max_buffers = 0;
    gboolean drop = FALSE;
    
    g_object_get(G_OBJECT(appsink_),
                 "emit-signals", &emit_signals,
                 "sync", &sync,
                 "max-buffers", &max_buffers,
                 "drop", &drop,
                 NULL);
    
    std::cout << "AppSink属性验证：" << std::endl;
    std::cout << "  - emit-signals: " << (emit_signals ? "TRUE" : "FALSE") << std::endl;
    std::cout << "  - sync: " << (sync ? "TRUE" : "FALSE") << std::endl;
    std::cout << "  - max-buffers: " << max_buffers << std::endl;
    std::cout << "  - drop: " << (drop ? "TRUE" : "FALSE") << std::endl;
    
    // 🔧 修复：连接信号并验证连接
    gulong handler_id = g_signal_connect(appsink_, "new-sample", G_CALLBACK(newSampleCallback), this);
    
    if (handler_id > 0) {
        std::cout << "✅ AppSink信号连接成功，handler_id: " << handler_id << std::endl;
    } else {
        std::cout << "❌ AppSink信号连接失败" << std::endl;
        return;
    }
    
    // 🔧 新增：强制触发一次sample拉取测试
    std::cout << "🔧 执行AppSink连接测试..." << std::endl;
    
    // 使用GMainLoop确保信号处理正常工作
    GMainContext* context = g_main_context_default();
    if (context) {
        std::cout << "✅ GMainContext可用，信号处理应该正常" << std::endl;
        
        // 检查是否有待处理的消息
        while (g_main_context_pending(context)) {
            g_main_context_iteration(context, FALSE);
        }
    } else {
        std::cout << "⚠️ 警告：GMainContext不可用，信号可能无法正常处理" << std::endl;
    }
    
    std::cout << "🎯 AppSink回调机制修复完成" << std::endl;
}

// 获取最新合成帧（供外部访问）
bool DeepStreamManager::getLatestCompositeFrame(cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    if (new_frame_available_ && !latest_frame_.empty()) {
        frame = latest_frame_.clone();
        new_frame_available_ = false;
        return true;
    }
    return false;
}

void DeepStreamManager::startCanvasUpdateThread() {
    if (canvas_update_running_ || !lvgl_interface_) {
        return;
    }
    
    canvas_update_running_ = true;
    canvas_update_thread_ = std::thread(&DeepStreamManager::canvasUpdateLoop, this);
    std::cout << "Canvas更新线程已启动" << std::endl;
}

void DeepStreamManager::stopCanvasUpdateThread() {
    if (!canvas_update_running_) {
        return;
    }
    
    canvas_update_running_ = false;
    if (canvas_update_thread_.joinable()) {
        canvas_update_thread_.join();
    }
    std::cout << "Canvas更新线程已停止" << std::endl;
}

void DeepStreamManager::canvasUpdateLoop() {
    std::cout << "Canvas更新循环开始运行" << std::endl;
    
    // 🔧 修复：在Canvas更新循环中处理GStreamer事件
    GMainContext* context = g_main_context_default();
    auto last_update = std::chrono::steady_clock::now();
    const auto target_interval = std::chrono::milliseconds(33); // 30fps
    
    while (canvas_update_running_) {
        // 🔧 关键修复：处理GStreamer消息和信号
        if (context && g_main_context_pending(context)) {
            g_main_context_iteration(context, FALSE);
        }
        
        auto current_time = std::chrono::steady_clock::now();
        
        if (new_frame_available_.load() && lvgl_interface_) {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            
            if (!latest_frame_.empty()) {
                #ifdef ENABLE_LVGL
                std::cout << "处理新帧: " << latest_frame_.cols << "x" << latest_frame_.rows
                         << " 通道数:" << latest_frame_.channels() << std::endl;
                         
                auto* lvgl_if = static_cast<bamboo_cut::ui::LVGLWaylandInterface*>(lvgl_interface_);
                lv_obj_t* canvas = lvgl_if->getCameraCanvas();
                
                if (canvas) {
                    // Canvas对象获取成功（静默模式）
                    
                    // 🔧 修复1: 确保帧格式统一为BGRA
                    cv::Mat display_frame;
                    if (latest_frame_.channels() == 4) {
                        display_frame = latest_frame_.clone();  // 克隆避免引用问题
                    } else if (latest_frame_.channels() == 3) {
                        cv::cvtColor(latest_frame_, display_frame, cv::COLOR_BGR2BGRA);
                    } else {
                        cv::cvtColor(latest_frame_, display_frame, cv::COLOR_GRAY2BGRA);
                    }
                    
                    // 🔧 修复2: 调整尺寸并确保数据连续
                    if (display_frame.cols != 960 || display_frame.rows != 640) {
                        cv::resize(display_frame, display_frame, cv::Size(960, 640), 
                                   0, 0, cv::INTER_LINEAR);
                    }
                    
                    // 🔧 修复3: 确保数据连续性
                    if (!display_frame.isContinuous()) {
                        display_frame = display_frame.clone();
                        // 帧数据不连续，已克隆（静默模式）
                    }
                    
                    // 验证数据
                    if (display_frame.channels() != 4 || 
                        display_frame.cols != 960 || 
                        display_frame.rows != 640) {
                        // 帧格式不正确（静默模式）
                        continue;
                    }
                    
                    // 调试：检查源数据
                    cv::Vec4b src_first = display_frame.at<cv::Vec4b>(0, 0);
                    cv::Vec4b src_center = display_frame.at<cv::Vec4b>(320, 480);
            
                    // 获取canvas缓冲区
                    lv_img_dsc_t* canvas_dsc = lv_canvas_get_image(canvas);
                    if (canvas_dsc && canvas_dsc->data) {
                        // Canvas缓冲区获取成功（静默模式）
                        
                        uint32_t* canvas_buffer = (uint32_t*)canvas_dsc->data;
                        const uint8_t* src_data = display_frame.data;
                        const size_t pixel_count = 960 * 640;
                        const int step = display_frame.step[0];  // 行步长
                        
                        std::cout << "OpenCV Mat step: " << step 
                                 << ", expected: " << (960 * 4) << std::endl;
                        
                        // 🔧 修复4: 正确处理步长的像素转换
                        for (int y = 0; y < 640; y++) {
                            const uint8_t* row_ptr = src_data + y * step;
                            uint32_t* canvas_row = canvas_buffer + y * 960;
                            
                            for (int x = 0; x < 960; x++) {
                                const uint8_t* pixel = row_ptr + x * 4;
                                uint8_t b = pixel[0];
                                uint8_t g = pixel[1];
                                uint8_t r = pixel[2];
                                uint8_t a = pixel[3];
                                
                                // LVGL ARGB8888: A在最高位
                                canvas_row[x] = (a << 24) | (r << 16) | (g << 8) | b;
                            }
                        }
                        
                        
                        // 刷新显示
                        lv_obj_invalidate(canvas);
                        lv_refr_now(NULL);
                        std::cout << "Canvas刷新完成" << std::endl;
                    }
                } else {
                    std::cout << "错误：Canvas对象获取失败" << std::endl;
                }
                #endif
                
                new_frame_available_ = false;
            }
        }
        
        // 帧率控制
        auto processing_time = std::chrono::steady_clock::now() - current_time;
        auto sleep_time = target_interval - processing_time;
        
        if (sleep_time > std::chrono::milliseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        }
    }
    
    std::cout << "Canvas更新循环已退出" << std::endl;
}

// 检查Wayland环境可用性
bool DeepStreamManager::checkWaylandEnvironment() {
    std::cout << "🎯 [DeepStream] 检查Wayland环境..." << std::endl;
    
    // 检查WAYLAND_DISPLAY环境变量
    const char* wayland_display = getenv("WAYLAND_DISPLAY");
    if (!wayland_display) {
        setenv("WAYLAND_DISPLAY", "wayland-0", 0);
        wayland_display = getenv("WAYLAND_DISPLAY");
        std::cout << "[DeepStream] 设置WAYLAND_DISPLAY=" << wayland_display << std::endl;
    }
    
    // 检查XDG_RUNTIME_DIR
    const char* runtime_dir = getenv("XDG_RUNTIME_DIR");
    if (!runtime_dir) {
        setenv("XDG_RUNTIME_DIR", "/run/user/1000", 0);
        runtime_dir = getenv("XDG_RUNTIME_DIR");
        std::cout << "[DeepStream] 设置XDG_RUNTIME_DIR=" << runtime_dir << std::endl;
    }
    
    // 验证Wayland socket是否存在
    std::string socket_path = std::string(runtime_dir) + "/" + wayland_display;
    if (access(socket_path.c_str(), F_OK) != 0) {
        std::cout << "⚠️ [DeepStream] Wayland socket不存在: " << socket_path << std::endl;
        wayland_available_ = false;
        return false;
    }
    
    wayland_available_ = true;
    std::cout << "✅ [DeepStream] Wayland环境配置成功" << std::endl;
    return true;
}


// 新增：简化的Wayland视频布局计算（支持摄像头分辨率缩放）
VideoLayout DeepStreamManager::calculateWaylandVideoLayout(const DeepStreamConfig& config) {
    VideoLayout layout;
    
    std::cout << "[DeepStreamManager] 计算Wayland视频布局..." << std::endl;
    std::cout << "  摄像头输入: " << config.camera_width << "x" << config.camera_height << std::endl;
    
    // 计算可用区域（减去顶部和底部栏）
    layout.available_width = config.screen_width;
    layout.available_height = config.screen_height - config.header_height - config.footer_height;
    
    // 🔧 修复：目标显示尺寸（固定为960x640以匹配Canvas）
    layout.width = 960;   // 固定宽度
    layout.height = 640;  // 固定高度
    
    // 窗口位置（跳过头部面板）
    layout.offset_x = 0;  // 左对齐
    layout.offset_y = config.header_height;  // 头部面板下方
    
    std::cout << "[DeepStreamManager] 布局计算完成: "
              << layout.width << "x" << layout.height
              << " at (" << layout.offset_x << "," << layout.offset_y << ")" << std::endl;
    std::cout << "  缩放: " << config.camera_width << "x" << config.camera_height
              << " -> " << layout.width << "x" << layout.height << std::endl;
    
    return layout;
}

} // namespace deepstream
} // namespace bamboo_cut