/**
 * @file deepstream_manager.cpp
 * @brief DeepStream AI推理和视频显示管理器实现 - 支持nvdrmvideosink叠加平面模式
 */

#include "bamboo_cut/deepstream/deepstream_manager.h"
#include "bamboo_cut/ui/lvgl_wayland_interface.h"
#include <iostream>
#include <sstream>
#include <gst/gst.h>
#include <gst/video/videooverlay.h>  // ?? GstVideoOverlay 接口
#include <fstream>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include <algorithm>
// Wayland架构下移除DRM头文件依赖
// #include <xf86drm.h>
// #include <xf86drmMode.h>
#include <thread>
#include <chrono>
#include <set>
#include <vector>
#include <gst/app/gstappsink.h>

// ?? 新增：Wayland头文件包含
#include <wayland-client.h>

#ifdef ENABLE_LVGL
#include <lvgl/lvgl.h>
#endif

namespace fs = std::filesystem;

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
    , wayland_available_(false)
    , video_surface_(nullptr)
    , video_subsurface_(nullptr) {
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
    , wayland_available_(false)
    , video_surface_(nullptr)
    , video_subsurface_(nullptr) {
    
    std::cout << "DeepStreamManager 构造函数完成（支持LVGL界面集成）" << std::endl;
}

DeepStreamManager::~DeepStreamManager() {
    stopCanvasUpdateThread();
    stop();
    cleanup();
}

bool DeepStreamManager::initializeWithSubsurface(
    void* wl_display,
    void* wl_surface,
    int width,
    int height) {
    
    std::cout << "?? [DeepStream] 初始化Wayland Subsurface模式..." << std::endl;
    
    // 保存 subsurface 指针
    video_surface_ = wl_surface;
    
    // 配置基础参数
    config_.screen_width = width;
    config_.screen_height = height;
    
    // 初始化GStreamer
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }
    
    // 计算布局
    video_layout_ = calculateWaylandVideoLayout(config_);
    
    initialized_ = true;
    std::cout << "? [DeepStream] Subsurface初始化完成" << std::endl;
    return true;
}

bool DeepStreamManager::initializeWithSubsurface(
    void* parent_display,
    void* parent_compositor,
    void* parent_subcompositor,
    void* parent_surface,
    const SubsurfaceConfig& config) {
    
    std::cout << "?? [DeepStream] 初始化Wayland Subsurface模式（完整版：自动创建subsurface）..." << std::endl;
    std::cout << "?? Subsurface配置: offset(" << config.offset_x << ", " << config.offset_y 
              << ") size(" << config.width << "x" << config.height << ") "
              << (config.use_sync_mode ? "同步模式" : "异步模式") << std::endl;
    
    // ?? 修复：正确的类型转换
    auto* wl_display = static_cast<struct wl_display*>(parent_display);
    auto* wl_compositor = static_cast<struct wl_compositor*>(parent_compositor);
    auto* wl_subcompositor = static_cast<struct wl_subcompositor*>(parent_subcompositor);
    auto* wl_parent_surface = static_cast<struct wl_surface*>(parent_surface);
    
    // 记录父 wl_surface（LVGL 主 surface），后续可以将视频直接绑定到该 surface 调试
    parent_wl_display_ = parent_display;
    parent_wl_surface_ = parent_surface;

    
    // ?? 新增：检查父display健康状态
    if (wl_display) {
        int parent_error_code = wl_display_get_error(wl_display);
        if (parent_error_code != 0) {
            std::cerr << "? [DeepStream] 父Wayland display已损坏，错误码: " 
                      << parent_error_code << std::endl;
            std::cerr << "?? [DeepStream] 降级到AppSink模式" << std::endl;
            
            // 创建AppSink配置
            config_.sink_mode = VideoSinkMode::APPSINK;
            config_.screen_width = config.width;
            config_.screen_height = config.height;
            
            return initialize(config_);  // 使用AppSink模式初始化
        }
    }
    
    // 验证参数
    if (!wl_display || !wl_compositor || !wl_subcompositor || !wl_parent_surface) {
        std::cerr << "? [DeepStream] 无效的Wayland父窗口对象" << std::endl;
        return false;
    }
    
    subsurface_config_ = config;
    
    // ?? 关键步骤1：创建视频表面
    auto* wl_surface = wl_compositor_create_surface(wl_compositor);
    video_surface_ = static_cast<void*>(wl_surface);
    if (!video_surface_) {
        std::cerr << "? [DeepStream] 创建视频surface失败" << std::endl;
        return false;
    }
    std::cout << "? [DeepStream] 创建视频surface" << std::endl;
    
    // ?? 关键步骤2：创建subsurface并附加到父表面
    auto* wl_subsurface = wl_subcompositor_get_subsurface(
        wl_subcompositor, wl_surface, wl_parent_surface);
    video_subsurface_ = static_cast<void*>(wl_subsurface);
    
    if (!video_subsurface_) {
        std::cerr << "? [DeepStream] 创建subsurface失败" << std::endl;
        wl_surface_destroy(wl_surface);
        video_surface_ = nullptr;
        return false;
    }
    std::cout << "? [DeepStream] 创建subsurface并附加到父窗口" << std::endl;
    
    // ?? 关键步骤3：设置subsurface位置
    wl_subsurface_set_position(wl_subsurface, config.offset_x, config.offset_y);
    std::cout << "?? [DeepStream] Subsurface位置: ("
              << config.offset_x << ", " << config.offset_y << ")" << std::endl;
    
    // ?? 关键步骤4：设置同步模式
    if (config.use_sync_mode) {
        wl_subsurface_set_sync(wl_subsurface);
        std::cout << "?? [DeepStream] 使用同步模式（与父窗口同步刷新）" << std::endl;
    } else {
        wl_subsurface_set_desync(wl_subsurface);
        std::cout << "? [DeepStream] 使用异步模式（独立刷新）" << std::endl;
    }
    
    // === 关键步骤5: Z-order 设置 ===
    // 为确保视频始终位于 LVGL UI 之上，显式将 subsurface 放到父 surface 之上。
    wl_subsurface_place_above(wl_subsurface, wl_parent_surface);
    std::cout << "[DeepStream] Subsurface Z-order: above parent surface (camera panel overlay)" << std::endl;
    
    
    // ?? 关键步骤6：commit subsurface 和 parent surface
    // ??  注意：Wayland subsurface 机制
    // - subsurface 相对于 parent surface 的位置/Z-order 在 parent commit 时生效
    // - subsurface 自己的 buffer 在 subsurface commit 时生效
    // - 在 desync 模式下，subsurface 可以独立 commit，不需要与 parent 同步
    wl_surface_commit(wl_surface);
    std::cout << "? [DeepStream] Subsurface 已 commit（空 commit，等待 waylandsink attach buffer）" << std::endl;
    
    wl_surface_commit(wl_parent_surface);
    std::cout << "? [DeepStream] 父 surface 已 commit（应用 subsurface 位置和 Z-order）" << std::endl;
    
    wl_display_flush(wl_display);
    std::cout << "? [DeepStream] Display flush 完成" << std::endl;
    
    std::cout << "\n?? [架构诊断] Waylandsink 预期行为：" << std::endl;
    std::cout << "  1?? waylandsink 接收 subsurface 作为 window_handle" << std::endl;
    std::cout << "  2?? waylandsink attach video buffer 到 subsurface" << std::endl;
    std::cout << "  3?? waylandsink commit subsurface（显示视频帧）" << std::endl;
    std::cout << "  4?? compositor 混合: 父 surface（LVGL UI，camera area 透明）+ subsurface（视频）" << std::endl;
    std::cout << "  ? 结果：UI + 视频同时可见\n" << std::endl;
    
    // 可选: 通过环境变量启用 Wayland 测试图案（默认使用真实摄像头）
    if (const char* env = std::getenv("BAMBOO_WAYLAND_TEST_PATTERN")) {
        config_.camera_source = CameraSourceMode::VIDEOTESTSRC;
        config_.test_pattern = std::atoi(env);
        if (config_.test_pattern < 0) config_.test_pattern = 0;

        // 使用 Subsurface 尺寸作为测试图案尺寸
        config_.camera_width = config.width;
        config_.camera_height = config.height;

        std::cout << "[DeepStreamManager] 使用 videotestsrc 测试模式, pattern="
                  << config_.test_pattern << ", size="
                  << config_.camera_width << "x" << config_.camera_height
                  << std::endl;
    }

    // 将配置更新到 initialize() 调用
    config_.sink_mode = VideoSinkMode::WAYLANDSINK;
    config_.screen_width = config.width;
    config_.screen_height = config.height;

    
    
    
    if (!initialize(config_)) {
        std::cerr << "? [DeepStream] DeepStream配置初始化失败" << std::endl;
        return false;
    }
    
    std::cout << "? [DeepStream] Wayland Subsurface初始化完成" << std::endl;
    return true;
}

bool DeepStreamManager::initialize(const DeepStreamConfig& config) {
    std::cout << "[DeepStreamManager] 初始化Wayland视频系统..." << std::endl;
    
    config_ = config;
    inference_available_ = false;
    resolved_nvinfer_config_.clear();
    
    if (config_.enable_inference) {
        resolved_nvinfer_config_ = resolveFilePath(config_.nvinfer_config);
        if (!resolved_nvinfer_config_.empty()) {
            bool assets_ok = logInferenceAssets(resolved_nvinfer_config_);
            if (assets_ok) {
                config_.nvinfer_config = resolved_nvinfer_config_;
                inference_available_ = true;
                std::cout << "[DeepStreamManager] ✅ 推理资产完整，启用 nvinfer: " << resolved_nvinfer_config_ << std::endl;
            } else {
                inference_available_ = false;
                config_.enable_inference = false;
                std::cout << "[DeepStreamManager] ⚠️  推理资产缺失，已自动禁用 nvinfer，继续仅显示视频" << std::endl;
            }
        } else {
            std::cout << "[DeepStreamManager] ⚠️  未找到 nvinfer 配置文件: " << config_.nvinfer_config << "，将禁用推理" << std::endl;
            config_.enable_inference = false;
            inference_available_ = false;
        }
    } else {
        std::cout << "[DeepStreamManager] ⚠️  推理被显式禁用，将跳过 nvinfer 初始化" << std::endl;
    }
    
    // ?? 架构重构：使用Wayland Subsurface模式替代appsink
    std::cout << "[DeepStreamManager] ?? 架构重构：迁移到Wayland Subsurface模式" << std::endl;
    std::cout << "[DeepStreamManager] ?? 目标：零拷贝GPU硬件合成，提升性能" << std::endl;
    
    // 检查是否有可用的Subsurface配置
    if (video_subsurface_) {
        std::cout << "[DeepStreamManager] 检测到Subsurface配置，使用waylandsink模式" << std::endl;
    } else {
        std::cout << "[DeepStreamManager] 未配置Subsurface，回退到appsink模式" << std::endl;
        config_.sink_mode = VideoSinkMode::APPSINK;
    }
    
    // 初始化GStreamer
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
        std::cout << "[DeepStreamManager] GStreamer初始化完成" << std::endl;
    }
    
    // ?? 架构重构：检查appsink架构所需插件
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
            std::cout << "[DeepStreamManager] ? " << plugin_descriptions[i] << std::endl;
            gst_object_unref(factory);
        } else {
            std::cerr << "[DeepStreamManager] ? " << plugin_descriptions[i] << " 不可用" << std::endl;
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
    std::lock_guard<std::mutex> lock(pipeline_mutex_);  // ?? 线程安全保护
    
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
                std::cout << "? LVGL Wayland已完全初始化，继续启动DeepStream管道" << std::endl;
            } else {
                std::cout << "?? 警告：LVGL Wayland初始化超时，继续启动DeepStream管道" << std::endl;
            }
        } else {
            std::cout << "警告：LVGL Wayland接口不可用，使用固定延迟" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        }
        
        for (int retry = 0; retry < MAX_RETRIES; retry++) {
            if (retry > 0) {
                std::cout << "重试启动管道 (第" << retry + 1 << "次尝试)..." << std::endl;
                // ?? 增加重试延迟，确保摄像头资源完全释放
                // nvargus-daemon 需要更多时间清理资源
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS * (retry + 1)));
                std::cout << "等待摄像头资源释放..." << std::endl;
            }
            
            // ?? 新增：清理之前的管道状态
            if (pipeline_) {
                gst_element_set_state(pipeline_, GST_STATE_NULL);
                gst_object_unref(pipeline_);
                pipeline_ = nullptr;
            }
            
            // 构建管道
            std::string pipeline_str = buildPipeline(config_, video_layout_);
            std::cout << "管道字符串: " << pipeline_str << std::endl;
            
            // ?? 新增：验证管道字符串有效性
            if (pipeline_str.empty()) {
                std::cerr << "? 管道字符串为空，配置错误" << std::endl;
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
            
            // ?? 新增：验证关键元素存在
            if (config_.sink_mode == VideoSinkMode::KMSSINK) {
                GstElement* kmssink = gst_bin_get_by_name(GST_BIN(pipeline_), "kmssink0");
                if (!kmssink) {
                    std::cerr << "? 无法找到kmssink元素" << std::endl;
                    if (retry < MAX_RETRIES - 1) continue;
                    return false;
                } else {
                    gst_object_unref(kmssink);
                }
            }
            
            // 设置消息总线
            bus_ = gst_element_get_bus(pipeline_);
            if (!bus_) {
                std::cerr << "? 无法获取消息总线" << std::endl;
                if (retry < MAX_RETRIES - 1) continue;
                return false;
            }
            bus_watch_id_ = gst_bus_add_watch(bus_, busCallback, this);
            
            // ?? 修复：通过 GstVideoOverlay 接口传递 subsurface 给 waylandsink
            // waylandsink 实现了 GstVideoOverlay 接口，支持外部窗口句柄
            if (video_surface_) {
                // 设置同步消息处理器，处理 Wayland display context 和 window handle
                gst_bus_set_sync_handler(bus_, 
                    [](GstBus* bus, GstMessage* message, gpointer user_data) -> GstBusSyncReply {
                        DeepStreamManager* self = static_cast<DeepStreamManager*>(user_data);
                        
                        // 处理 Wayland display context 请求
                        if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_NEED_CONTEXT) {
                            const gchar* context_type;
                            gst_message_parse_context_type(message, &context_type);
                            
                            if (g_strcmp0(context_type, "GstWaylandDisplayHandleContextType") == 0) {
                                // 创建 Wayland display context
                                GstContext* context = gst_context_new("GstWaylandDisplayHandleContextType", TRUE);
                                GstStructure* structure = gst_context_writable_structure(context);
                                
                                // 设置 Wayland display（waylandsink 需要）
                                gst_structure_set(structure, 
                                    "display", G_TYPE_POINTER, self->parent_wl_display_,
                                    NULL);
                                
                                gst_element_set_context(GST_ELEMENT(GST_MESSAGE_SRC(message)), context);
                                gst_context_unref(context);
                                
                                std::cout << "? [DeepStream] Wayland display context 已传递" << std::endl;
                                return GST_BUS_DROP;
                            }
                        }
                        
                        // ?? 关键：处理 prepare-window-handle 消息（GstVideoOverlay）
                        if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ELEMENT) {
                            const GstStructure* structure = gst_message_get_structure(message);
                            
                            if (gst_structure_has_name(structure, "prepare-window-handle")) {
                                // waylandsink 请求窗口句柄，传递我们的 subsurface
                                GstElement* sink = GST_ELEMENT(GST_MESSAGE_SRC(message));
                                
                                if (GST_IS_VIDEO_OVERLAY(sink)) {
                                    // 将窗口句柄绑定到视频 subsurface（位置由 subsurface 决定）
                                    void* target_surface = self->video_surface_ ? self->video_surface_ : self->parent_wl_surface_;
                                    gst_video_overlay_set_window_handle(
                                        GST_VIDEO_OVERLAY(sink),
                                        reinterpret_cast<guintptr>(target_surface)
                                    );

                                    // 在 subsurface 内部全幅渲染
                                    gst_video_overlay_set_render_rectangle(
                                        GST_VIDEO_OVERLAY(sink),
                                        0,
                                        0,
                                        self->subsurface_config_.width,
                                        self->subsurface_config_.height
                                    );

                                    std::cout << "[DeepStream] window handle=subsurface, render rect: (0, 0) "
                                              << self->subsurface_config_.width << "x"
                                              << self->subsurface_config_.height << std::endl;

                                    return GST_BUS_DROP;
                                }

                            }
                        }
                        
                        return GST_BUS_PASS;
                    }, 
                    this, 
                    NULL);
                
                std::cout << "? [DeepStream] 已设置 Wayland 显示和窗口句柄传递机制" << std::endl;
            }
            
            // ?? 改进：分阶段启动管道，降低段错误风险
            std::cout << "正在分阶段启动管道..." << std::endl;
            
            // 第一阶段：设置为READY状态
            std::cout << "第一阶段：设置管道为READY状态..." << std::endl;
            GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_READY);
            if (ret == GST_STATE_CHANGE_FAILURE) {
                std::cerr << "? READY状态设置失败" << std::endl;
                cleanup();
                if (retry < MAX_RETRIES - 1) continue;
                return false;
            }
            
            // 等待READY状态稳定
            GstState state;
            ret = gst_element_get_state(pipeline_, &state, NULL, 5 * GST_SECOND);
            if (ret == GST_STATE_CHANGE_FAILURE || state != GST_STATE_READY) {
                std::cerr << "? READY状态等待失败" << std::endl;
                cleanup();
                if (retry < MAX_RETRIES - 1) continue;
                return false;
            }
            std::cout << "? READY状态设置成功" << std::endl;
            
            // 第二阶段：设置为PAUSED状态
            std::cout << "第二阶段：设置管道为PAUSED状态..." << std::endl;
            ret = gst_element_set_state(pipeline_, GST_STATE_PAUSED);
            if (ret == GST_STATE_CHANGE_FAILURE) {
                std::cerr << "? PAUSED状态设置失败" << std::endl;
                cleanup();
                if (retry < MAX_RETRIES - 1) continue;
                return false;
            }
            
            // 等待PAUSED状态稳定
            ret = gst_element_get_state(pipeline_, &state, NULL, 10 * GST_SECOND);
            if (ret == GST_STATE_CHANGE_FAILURE) {
                std::cerr << "? PAUSED状态等待失败" << std::endl;
                cleanup();
                if (retry < MAX_RETRIES - 1) continue;
                return false;
            }
            std::cout << "? PAUSED状态设置成功" << std::endl;
            
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
                    std::cerr << "? 管道异步启动失败，进行错误诊断..." << std::endl;
                    
                    // 获取详细错误信息
                    GstBus* bus = gst_element_get_bus(pipeline_);
                    GstMessage* msg = gst_bus_timed_pop_filtered(bus, 2 * GST_SECOND,
                        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING));
                        
                    if (msg) {
                        GError* err;
                        gchar* debug_info;
                        
                        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                            gst_message_parse_error(msg, &err, &debug_info);
                            std::cerr << "?? GStreamer错误: " << err->message << std::endl;
                        } else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_WARNING) {
                            gst_message_parse_warning(msg, &err, &debug_info);
                            std::cerr << "?? GStreamer警告: " << err->message << std::endl;
                        }
                        
                        if (debug_info) {
                            std::cerr << "?? 调试信息: " << debug_info << std::endl;
                            
                            // 检查常见错误模式
                            if (strstr(debug_info, "Wayland") || strstr(debug_info, "wl_")) {
                                std::cout << "?? 检测到Wayland相关错误，可能是surface未就绪..." << std::endl;
                            } else if (strstr(debug_info, "NVMM") || strstr(debug_info, "nvarguscamerasrc")) {
                                std::cout << "?? 检测到NVMM/摄像头错误，可能是资源冲突..." << std::endl;
                            } else if (strstr(debug_info, "waylandsink")) {
                                std::cout << "?? 检测到waylandsink错误，可能是display连接问题..." << std::endl;
                            }
                            g_free(debug_info);
                        }
                        g_error_free(err);
                        gst_message_unref(msg);
                    } else {
                        std::cerr << "?? 无法从bus获取错误消息（可能超时）" << std::endl;
                    }
                    if (bus) gst_object_unref(bus);
                    
                    cleanup();
                    if (retry < MAX_RETRIES - 1) continue;
                    return false;
                }
            }
            
            std::cout << "? PLAYING状态设置成功" << std::endl;
            
            // 成功启动，跳出重试循环
            break;
        }
        
        running_ = true;
        const char* mode_names[] = {"nvdrmvideosink", "waylandsink", "kmssink", "appsink"};
        const char* mode_name = mode_names[static_cast<int>(config_.sink_mode)];
        std::cout << "?? DeepStream 管道启动成功 (" << mode_name << " Subsurface架构)" << std::endl;
        std::cout << "?? 数据流: nvarguscamerasrc → nvinfer → waylandsink → subsurface → GPU合成" << std::endl;
        
        // ? AppSink回调已移除 - 使用Subsurface硬件合成
        // if (config_.sink_mode == VideoSinkMode::APPSINK) {
        //     setupAppSinkCallbacks();
        // }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "? 管道启动异常: " << e.what() << std::endl;
        cleanup();
        return false;
    } catch (...) {
        std::cerr << "? 管道启动未知异常" << std::endl;
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
    
    // ? AppSink回调已移除 - 使用Subsurface硬件合成
    // if (config_.sink_mode == VideoSinkMode::APPSINK) {
    //     setupAppSinkCallbacks();
    // }
    
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
    
    // ?? 新增：完全停止时才清理 subsurface
    cleanupSubsurface();
    
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
    std::cout << "?? Wayland架构：跳过DRM设备检测，使用waylandsink硬件渲染" << std::endl;
    
    // 返回默认配置，表示不支持DRM overlay
    std::cout << "?? 建议使用waylandsink替代nvdrmvideosink" << std::endl;
    return config;
    
    // Wayland架构下不再进行DRM plane检测
    std::cout << "?? Wayland架构：跳过DRM plane检测和资源管理" << std::endl;
    std::cout << "?? 建议使用waylandsink进行视频显示" << std::endl;
    
    return config;
}

bool DeepStreamManager::setupDRMOverlayPlane() {
    std::lock_guard<std::mutex> lock(drm_mutex_);  // ?? 线程安全保护
    
    std::cout << "?? 设置DRM叠加平面..." << std::endl;
    
    try {
        // 如果未配置叠加平面，自动检测
        if (config_.overlay.plane_id == -1) {
            std::cout << "?? 执行智能overlay plane检测..." << std::endl;
            config_.overlay = detectAvailableOverlayPlane();
            if (config_.overlay.plane_id == -1) {
                std::cerr << "? 未找到可用的DRM叠加平面" << std::endl;
                return false;
            }
        }
        
        // ?? 新增：验证plane-id有效性
        if (config_.overlay.plane_id <= 0) {
            std::cerr << "? 无效的plane_id: " << config_.overlay.plane_id << std::endl;
            return false;
        }
        
        std::cout << "? DRM叠加平面设置完成: plane_id=" << config_.overlay.plane_id
                  << ", crtc_id=" << config_.overlay.crtc_id
                  << ", connector_id=" << config_.overlay.connector_id
                  << ", z_order=" << config_.overlay.z_order << std::endl;
        
        // ?? 新增：验证多层显示配置
        if (!verifyMultiLayerDisplaySetup()) {
            std::cout << "??  多层显示验证失败，但继续尝试..." << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "? DRM叠加平面设置异常: " << e.what() << std::endl;
        return false;
    }
}

// ?? 新增：验证多层显示设置的函数（Wayland架构）
bool DeepStreamManager::verifyMultiLayerDisplaySetup() {
    std::cout << "?? Wayland架构：跳过多层显示验证" << std::endl;
    std::cout << "? Wayland合成器自动处理多层显示管理" << std::endl;
    
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
    
    // 计算视频区域尺寸（暂时固定为摄像头原始分辨率，避免奇数尺寸缩放导致的 mem copy 失败）
    layout.width = config.camera_width;
    layout.height = config.camera_height;

    // 对齐到偶数，避免下游硬件拷贝在奇数尺寸上失败
    if (layout.width % 2 != 0) layout.width -= 1;
    if (layout.height % 2 != 0) layout.height -= 1;
    
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
            std::cout << "?? [DeepStream] Wayland架构下降级到AppSink软件合成" << std::endl;
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
    std::cout << "?? Wayland架构下直接使用nvarguscamerasrc..." << std::endl;
    
    // ?? 关键修复：使用BGRA格式，这是AR24在DRM中的实际对应格式
    pipeline << buildCameraSource(config) << " ! "
             << "nvvidconv ! "  // NVMM -> RGBA格式转换和缩放（硬件加速）
             << "video/x-raw(memory:NVMM),format=RGBA,width=" << width << ",height=" << height << " ! "
             << "nvvidconv ! "     // NVMM -> 标准内存转换
             << "video/x-raw(memory:NVMM),format=RGBA,width=" << width << ",height=" << height << " ! "
             << "videoconvert ! "  // RGBA -> BGRA格式转换（AR24对应BGRA）
             << "video/x-raw(memory:NVMM),format=BGRA,width=" << width << ",height=" << height << " ! "  // 使用AR24/BGRA格式
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
    
    pipeline << buildCameraSource(config) << " ! "
             << "queue max-size-buffers=6 max-size-time=0 leaky=downstream ! "
             << buildInferenceChain(config, width, height, 1, true)
             << "waylandsink name=video_sink sync=false async=true ";
    
    return pipeline.str();
}

std::string DeepStreamManager::buildStereoVisionPipeline(const DeepStreamConfig& config, const VideoLayout& layout) {
    std::ostringstream pipeline;
    const int batch_size = (config.dual_mode == DualCameraMode::STEREO_VISION) ? 2 : 1;
    const bool is_wayland = (config.sink_mode == VideoSinkMode::WAYLANDSINK);
    const int mux_width = (config.enable_inference && inference_available_) ? std::max(config.infer_width, 1) : layout.width;
    const int mux_height = (config.enable_inference && inference_available_) ? std::max(config.infer_height, 1) : layout.height;
    
    pipeline << "nvarguscamerasrc sensor-id=" << config.camera_id << " ! "
             << "video/x-raw(memory:NVMM),width=" << config.camera_width
             << ",height=" << config.camera_height
             << ",framerate=" << config.camera_fps << "/1,format=NV12 ! "
             << "queue max-size-buffers=6 max-size-time=0 leaky=downstream ! "
             << "m.sink_0 "
             << "nvarguscamerasrc sensor-id=" << config.camera_id_2 << " ! "
             << "video/x-raw(memory:NVMM),width=" << config.camera_width
             << ",height=" << config.camera_height
             << ",framerate=" << config.camera_fps << "/1,format=NV12 ! "
             << "queue max-size-buffers=6 max-size-time=0 leaky=downstream ! "
             << "m.sink_1 "
             << "nvstreammux name=m batch-size=" << batch_size
             << " width=" << mux_width
             << " height=" << mux_height
             << " live-source=1 batched-push-timeout=40000 ! "
             << buildInferenceChain(config, layout.width, layout.height, batch_size, is_wayland);
    
    switch (config.sink_mode) {
        case VideoSinkMode::NVDRMVIDEOSINK:
            pipeline << "nvdrmvideosink "
                     << "offset-x=" << layout.offset_x << " "
                     << "offset-y=" << layout.offset_y << " "
                     << "set-mode=false "
                     << "sync=false";
            break;
        case VideoSinkMode::WAYLANDSINK:
            pipeline << "waylandsink name=video_sink sync=false";
            break;
        case VideoSinkMode::KMSSINK:
            pipeline << "kmssink "
                     << "connector-id=-1 plane-id=-1 "
                     << "force-modesetting=false can-scale=true "
                     << "sync=false restore-crtc=true";
            break;
        case VideoSinkMode::APPSINK:
        default:
            pipeline << "appsink name=video_appsink emit-signals=true sync=false "
                     << "caps=video/x-raw(memory:NVMM),format=BGRA,width=" << layout.width
                     << ",height=" << layout.height;
            break;
    }
    
    return pipeline.str();
}

std::string DeepStreamManager::buildInferenceChain(
    const DeepStreamConfig& config,
    int display_width,
    int display_height,
    int batch_size,
    bool wayland_sink_mode) {
    
    std::ostringstream chain;
    const bool enable_infer = config.enable_inference && inference_available_;
    const int safe_display_width = display_width > 0 ? display_width : config.camera_width;
    const int safe_display_height = display_height > 0 ? display_height : config.camera_height;
    // 显示/推理尺寸对齐到偶数，避免 nvvideoconvert 在非对齐尺寸上 mem copy 失败
    const int aligned_display_width = std::max(2, safe_display_width - (safe_display_width % 2));
    const int aligned_display_height = std::max(2, safe_display_height - (safe_display_height % 2));
    const int safe_infer_width = enable_infer ? std::max(config.infer_width, 1) : aligned_display_width;
    const int safe_infer_height = enable_infer ? std::max(config.infer_height, 1) : aligned_display_height;
    const char* final_format = "BGRx"; // waylandsink 接受的系统内存格式
    
    if (enable_infer) {
        chain << "nvvideoconvert ! "
              << "video/x-raw(memory:NVMM),format=NV12,width=" << safe_infer_width
              << ",height=" << safe_infer_height << " ! "
              << "nvinfer name=primary_gie config-file-path=" << config.nvinfer_config
              << " batch-size=" << std::max(batch_size, 1)
              << " unique-id=1 process-mode=1 interval=0 ! "
              << "nvdsosd name=osd display-text=1 process-mode=0 gpu-id=0 ! ";
    }
    
    // 最终输出给 waylandsink：使用 CPU 内存平面，减少兼容性问题
    chain << "nvvideoconvert ! "
          << "video/x-raw,format=" << final_format
          << ",width=" << aligned_display_width
          << ",height=" << aligned_display_height << " ! ";
    
    return chain.str();
}

std::string DeepStreamManager::resolveFilePath(const std::string& path) const {
    if (path.empty()) {
        return {};
    }
    
    fs::path candidate(path);
    std::vector<fs::path> search_paths;
    
    if (candidate.is_absolute()) {
        search_paths.push_back(candidate);
    } else {
        search_paths.push_back(candidate);
        search_paths.push_back(fs::path("config") / candidate);
        search_paths.push_back(fs::path("/opt/bamboo-cut") / candidate);
        search_paths.push_back(fs::path("/opt/bamboo-cut/config") / candidate.filename());
    }
    
    for (const auto& option : search_paths) {
        if (option.empty()) continue;
        std::error_code ec;
        fs::path normalized = option;
        if (fs::exists(normalized, ec)) {
            fs::path resolved = fs::weakly_canonical(normalized, ec);
            if (!ec) {
                return resolved.string();
            }
            return normalized.string();
        }
    }
    
    return {};
}

bool DeepStreamManager::logInferenceAssets(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cout << "[DeepStreamManager] ⚠️  无法读取 nvinfer 配置: " << config_path << std::endl;
        return false;
    }
    
    auto trim_value = [](std::string value) -> std::string {
        const std::string whitespace = " \t\"";
        const auto start = value.find_first_not_of(whitespace);
        if (start == std::string::npos) return {};
        const auto end = value.find_last_not_of(whitespace);
        return value.substr(start, end - start + 1);
    };
    
    std::string onnx_path;
    std::string engine_path;
    std::string line;
    
    while (std::getline(file, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        
        std::string key = line.substr(0, pos);
        std::string value = trim_value(line.substr(pos + 1));
        
        if (key.find("onnx-file") != std::string::npos) {
            onnx_path = value;
        } else if (key.find("model-engine-file") != std::string::npos) {
            engine_path = value;
        }
    }
    
    bool ok = true;

    if (!onnx_path.empty()) {
        auto resolved = resolveFilePath(onnx_path);
        if (!resolved.empty()) {
            std::cout << "[DeepStreamManager] ✅ ONNX 模型: " << resolved << std::endl;
        } else {
            std::cout << "[DeepStreamManager] ⚠️  无法解析 ONNX 路径: " << onnx_path << std::endl;
            ok = false;
        }
    } else {
        std::cout << "[DeepStreamManager] ⚠️  nvinfer 配置中缺少 onnx-file 项" << std::endl;
        ok = false;
    }
    
    if (!engine_path.empty()) {
        auto resolved = resolveFilePath(engine_path);
        if (!resolved.empty()) {
            std::cout << "[DeepStreamManager] ✅ TensorRT 引擎: " << resolved << std::endl;
        } else {
            std::cout << "[DeepStreamManager] ⚠️  无法解析引擎路径: " << engine_path << std::endl;
            ok = false;
        }
    } else {
        std::cout << "[DeepStreamManager] ⚠️  nvinfer 配置中缺少 model-engine-file 项" << std::endl;
        ok = false;
    }

    return ok;
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
    // ?? 关键修复：先将管道设置为 NULL 状态，确保摄像头资源完全释放
    if (pipeline_) {
        std::cout << "?? [DeepStream] 正在停止管道并释放摄像头资源..." << std::endl;
        
        // 先设置为 NULL 状态，等待资源释放
        GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_NULL);
        if (ret == GST_STATE_CHANGE_ASYNC) {
            // 等待状态变化完成（最多3秒）
            GstState state;
            ret = gst_element_get_state(pipeline_, &state, NULL, 3 * GST_SECOND);
            if (ret == GST_STATE_CHANGE_FAILURE) {
                std::cerr << "?? [DeepStream] 管道停止失败，但继续清理" << std::endl;
            } else {
                std::cout << "? [DeepStream] 管道已停止至 NULL 状态" << std::endl;
            }
        }
        
        // 额外等待，确保 nvarguscamerasrc 完全释放摄像头
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    if (pipeline2_) {
        std::cout << "?? [DeepStream] 正在停止管道2..." << std::endl;
        gst_element_set_state(pipeline2_, GST_STATE_NULL);
        // 等待一小段时间确保资源释放
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
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
        std::cout << "? [DeepStream] 管道已释放" << std::endl;
    }
    
    if (pipeline2_) {
        gst_object_unref(pipeline2_);
        pipeline2_ = nullptr;
    }
    
    // ?? 修复：不在 cleanup() 中销毁 subsurface
    // subsurface 的生命周期由父窗口管理，重试时需要保留
    // 只有在完全停止时才销毁
    // 注意：清理动作已移至 cleanupSubsurface()
}

// ?? 新增：专门清理 subsurface 资源（仅在完全停止时调用）
void DeepStreamManager::cleanupSubsurface() {
    if (video_subsurface_) {
        auto* wl_subsurface = static_cast<struct wl_subsurface*>(video_subsurface_);
        wl_subsurface_destroy(wl_subsurface);
        video_subsurface_ = nullptr;
        std::cout << "? [DeepStream] 已清理video_subsurface_" << std::endl;
    }
    
    if (video_surface_) {
        auto* wl_surface = static_cast<struct wl_surface*>(video_surface_);
        wl_surface_destroy(wl_surface);
        video_surface_ = nullptr;
        std::cout << "? [DeepStream] 已清理video_surface_" << std::endl;
    }
}

// 新增：构建摄像头源字符串
// ?? 修复：回到使用nvarguscamerasrc，因为GBM共享DRM资源后不再有冲突
std::string DeepStreamManager::buildCameraSource(const DeepStreamConfig& config) {
    std::ostringstream source;
    
    switch (config.camera_source) {
        case CameraSourceMode::NVARGUSCAMERA:
            // ?? 关键修复：回到使用nvarguscamerasrc Argus驱动
            std::cout << "?? 配置nvarguscamerasrc Argus驱动摄像头..." << std::endl;
            
            source << "nvarguscamerasrc sensor-id=" << config.camera_id << " "
                   << "! video/x-raw(memory:NVMM)"
                   << ",width=" << config.camera_width
                   << ",height=" << config.camera_height
                   << ",framerate=" << config.camera_fps << "/1"
                   << ",format=NV12";
            break;
            
        case CameraSourceMode::V4L2SRC:
            // 保留v4l2src作为备用方案
            std::cout << "?? 配置v4l2src备用方案..." << std::endl;
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
            std::cout << "?? 使用默认nvarguscamerasrc方案..." << std::endl;
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
    std::cout << "?? Wayland架构下直接使用nvarguscamerasrc..." << std::endl;
    
    // ?? 关键修复：使用nvarguscamerasrc + GBM共享DRM资源
    std::cout << "?? 构建GBM共享DRM的KMSSink管道 (缩放到 " << width << "x" << height << ")..." << std::endl;
    
    // 构建nvarguscamerasrc摄像头源（现在可以正常工作，因为GBM共享DRM资源）
    pipeline << buildCameraSource(config) << " ! ";
    
    // ?? 关键修复：直接使用NV12格式，让GStreamer自动协商内存类型
    std::cout << "?? 直接使用NV12格式，让GStreamer自动协商内存类型和缩放" << std::endl;
    
    // 让GStreamer自动协商从NVMM到标准内存的转换，保持NV12格式
    pipeline << "nvvidconv ! "  // NVMM到标准内存转换，保持NV12格式
             << "video/x-raw(memory:NVMM),format=NV12,width=" << width << ",height=" << height << " ! "
             << "queue "
             << "max-size-buffers=4 "      // 适中的缓冲区深度
             << "max-size-time=0 "
             << "leaky=downstream "
             << "! ";
    
    // ?? 关键修复：使用GBM后端提供的overlay plane，实现真正的分层显示
    if (config_.overlay.plane_id > 0) {
        std::cout << "?? 使用GBM共享的overlay plane: " << config_.overlay.plane_id << std::endl;
        pipeline << "kmssink "
                 << "plane-id=" << config_.overlay.plane_id << " "     // 使用GBM分配的overlay plane
                 << "connector-id=" << config_.overlay.connector_id << " " // 使用GBM共享的connector
                 << "force-modesetting=false " // 不改变显示模式，LVGL已通过GBM设置
                 << "can-scale=true "          // 启用硬件缩放
                 << "sync=false "              // 低延迟模式
                 << "restore-crtc=false";      // 不恢复CRTC，保持GBM管理
    } else {
        std::cout << "??  GBM后端未提供overlay plane，使用用户指定的plane-id=44" << std::endl;
        // ?? 修复：直接使用用户指定的overlay plane-id=44，支持AR24/ABGR格式
        pipeline << "kmssink "
                 << "plane-id=44 "             // 用户指定的overlay plane，支持AR24/ABGR
                 << "connector-id=-1 "         // 自动检测连接器
                 << "force-modesetting=false " // 不强制设置模式
                 << "can-scale=true "          // 启用硬件缩放
                 << "sync=false "              // 低延迟模式
                 << "restore-crtc=false";      // 不恢复CRTC，保持GBM管理
    }
    
    std::cout << "?? 构建GBM共享DRM的KMSSink管道: " << pipeline.str() << std::endl;
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
    
    // ?? 关键修复：使用摄像头原生分辨率然后缩放到目标尺寸
    std::cout << "?? 构建原生分辨率AppSink管道 (缩放到 " << width << "x" << height << ")..." << std::endl;
    
    if (config.camera_source == CameraSourceMode::NVARGUSCAMERA ||
        config.camera_source == CameraSourceMode::V4L2SRC) {
        
        // ?? 修复：使用两步转换，先nvvidconv转到标准内存，再videoconvert转BGRA
        pipeline << buildCameraSource(config) << " ! "
                 << "nvvidconv ! "    // NVMM -> 标准内存，保持NV12格式
                 << "video/x-raw(memory:NVMM),format=NV12,width=" << width << ",height=" << height << " ! "
                 << "videoconvert ! "  // NV12 -> BGRA格式转换（软件）
                 << "video/x-raw(memory:NVMM),format=BGRA,width=" << width << ",height=" << height << " ! "
                 << "queue max-size-buffers=2 leaky=downstream ! "
                 << "appsink name=video_appsink "
                 << "emit-signals=true sync=false max-buffers=2 drop=true";
        
        std::cout << "?? 构建原生分辨率AppSink管道: " << pipeline.str() << std::endl;
                 
    } else if (config.camera_source == CameraSourceMode::VIDEOTESTSRC) {
        // ? 测试源直接使用目标分辨率
        pipeline << "videotestsrc pattern=18 is-live=true "
                 << "! video/x-raw(memory:NVMM),format=BGRA"
                 << ",width=" << width << ",height=" << height
                 << ",framerate=30/1 "
                 << "! appsink name=video_appsink "
                 << "emit-signals=true sync=false max-buffers=1 drop=false";
    } else {
        // 其他源（文件源等）
        pipeline << buildCameraSource(config) << " ! "
                 << "videoconvert ! "  // 确保格式兼容
                 << "videoscale ! "    // 缩放到目标尺寸
                 << "video/x-raw(memory:NVMM),format=BGRA,width=" << width << ",height=" << height << " ! "
                 << "queue max-size-buffers=2 leaky=downstream ! "
                 << "appsink name=video_appsink "
                 << "emit-signals=true sync=false max-buffers=2 drop=true";
    }
    
    std::cout << "?? 构建原生分辨率AppSink管道: " << pipeline.str() << std::endl;
    return pipeline.str();
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

// ? Canvas更新线程已移除 - 使用 Wayland 合成器 GPU 合成
void DeepStreamManager::startCanvasUpdateThread() {
    // Wayland 合成器自动在 GPU 中合成视频和 UI，不需要手动 Canvas 更新线程
    std::cout << "?? [DeepStream] Canvas更新线程已被 Wayland Subsurface GPU 合成替代" << std::endl;
}

void DeepStreamManager::stopCanvasUpdateThread() {
    // Canvas更新线程已被GPU硬件合成替代
    std::cout << "?? [DeepStream] 停止Canvas更新线程（GPU合成模式下为空操作）" << std::endl;
}

void DeepStreamManager::canvasUpdateLoop() {
    std::cout << "Canvas更新循环开始运行" << std::endl;
    
    // ?? 修复：在Canvas更新循环中处理GStreamer事件
    GMainContext* context = g_main_context_default();
    auto last_update = std::chrono::steady_clock::now();
    const auto target_interval = std::chrono::milliseconds(33); // 30fps
    
    while (canvas_update_running_) {
        // ?? 关键修复：处理GStreamer消息和信号
        if (context && g_main_context_pending(context)) {
            g_main_context_iteration(context, FALSE);
        }
        
        auto current_time = std::chrono::steady_clock::now();
        
        if (new_frame_available_.load() && lvgl_interface_) {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            
            if (!latest_frame_.empty()) {
                #ifdef ENABLE_LVGL
                         
                auto* lvgl_if = static_cast<bamboo_cut::ui::LVGLWaylandInterface*>(lvgl_interface_);
                lv_obj_t* canvas = lvgl_if->getCameraCanvas();
                
                if (canvas) {
                    // Canvas对象获取成功（静默模式）
                    
                    // ?? 修复1: 确保帧格式统一为BGRA
                    cv::Mat display_frame;
                    if (latest_frame_.channels() == 4) {
                        display_frame = latest_frame_.clone();  // 克隆避免引用问题
                    } else if (latest_frame_.channels() == 3) {
                        cv::cvtColor(latest_frame_, display_frame, cv::COLOR_BGR2BGRA);
                    } else {
                        cv::cvtColor(latest_frame_, display_frame, cv::COLOR_GRAY2BGRA);
                    }
                    
                    // ?? 修复2: 调整尺寸并确保数据连续
                    if (display_frame.cols != 960 || display_frame.rows != 640) {
                        cv::resize(display_frame, display_frame, cv::Size(960, 640), 
                                   0, 0, cv::INTER_LINEAR);
                    }
                    
                    // ?? 修复3: 确保数据连续性
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
                        
                        
                        // ?? 修复4: 正确处理步长的像素转换
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
    std::cout << "?? [DeepStream] 检查Wayland环境..." << std::endl;
    
    // 检查WAYLAND_DISPLAY环境变量
    const char* wayland_display = getenv("WAYLAND_DISPLAY");
    if (!wayland_display) {
        setenv("WAYLAND_DISPLAY", "wayland-0", 0);
        wayland_display = getenv("WAYLAND_DISPLAY");
        std::cout << "[DeepStream] 设置WAYLAND_DISPLAY=" << wayland_display << std::endl;
    }
    
    // 检查 XDG_RUNTIME_DIR：优先使用已有值，否则在 Jetson + nvweston 场景下使用 /run/nvidia-wayland
    const char* runtime_dir = getenv("XDG_RUNTIME_DIR");
    if (!runtime_dir || runtime_dir[0] != '/') {
        setenv("XDG_RUNTIME_DIR", "/run/nvidia-wayland", 0);
        runtime_dir = getenv("XDG_RUNTIME_DIR");
        std::cout << "[DeepStream] 设置XDG_RUNTIME_DIR=" << runtime_dir << std::endl;
    }
    
    // 验证Wayland socket是否存在
    std::string socket_path = std::string(runtime_dir) + "/" + wayland_display;
    if (access(socket_path.c_str(), F_OK) != 0) {
        std::cout << "?? [DeepStream] Wayland socket不存在: " << socket_path << std::endl;
        wayland_available_ = false;
        return false;
    }
    
    wayland_available_ = true;
    std::cout << "? [DeepStream] Wayland环境配置成功" << std::endl;
    return true;
}


// 新增：简化的Wayland视频布局计算（支持摄像头分辨率缩放）
VideoLayout DeepStreamManager::calculateWaylandVideoLayout(const DeepStreamConfig& config) {
    VideoLayout layout;
    
    std::cout << "[DeepStreamManager] 计算Wayland视频布局..." << std::endl;
    std::cout << "  摄像头输入: " << config.camera_width << "x" << config.camera_height << std::endl;
    
    // ?? 关键修复：使用 subsurface 的实际尺寸（从 initializeWithSubsurface 传入）
    // config.screen_width 和 config.screen_height 已经是 subsurface 的实际尺寸
    layout.width = config.screen_width;
    layout.height = config.screen_height;
    
    // 计算可用区域（用于显示）
    layout.available_width = config.screen_width;
    layout.available_height = config.screen_height;
    
    // 窗口位置（使用 subsurface 的偏移量，而不是固定的 header_height）
    layout.offset_x = 0;  // 相对于 subsurface 自身
    layout.offset_y = 0;  // 相对于 subsurface 自身
    
    std::cout << "[DeepStreamManager] 布局计算完成: "
              << layout.width << "x" << layout.height
              << " at (" << layout.offset_x << "," << layout.offset_y << ")" << std::endl;
    std::cout << "  缩放: " << config.camera_width << "x" << config.camera_height
              << " -> " << layout.width << "x" << layout.height << std::endl;
    
    return layout;
}

} // namespace deepstream
} // namespace bamboo_cut


