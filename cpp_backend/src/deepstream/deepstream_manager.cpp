/**
 * @file deepstream_manager.cpp
 * @brief DeepStream AI推理和视频显示管理器实现 - 支持nvdrmvideosink叠加平面模式
 */

#include "bamboo_cut/deepstream/deepstream_manager.h"
#include "bamboo_cut/ui/lvgl_interface.h"
#include <iostream>
#include <sstream>
#include <gst/gst.h>
#include <fstream>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <thread>
#include <chrono>
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
    , initialized_(false) {
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
    , initialized_(false) {
    
    std::cout << "DeepStreamManager 构造函数完成（支持LVGL界面集成）" << std::endl;
}

DeepStreamManager::~DeepStreamManager() {
    stopCanvasUpdateThread();
    stop();
    cleanup();
}

bool DeepStreamManager::initialize(const DeepStreamConfig& config) {
    std::cout << "初始化 DeepStream 系统..." << std::endl;
    
    config_ = config;
    
    // 初始化 GStreamer
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
        std::cout << "GStreamer 初始化完成" << std::endl;
    }
    
    // 检查视频输出sink可用性
    const char* sink_names[] = {"appsink", "kmssink", "nvdrmvideosink", "waylandsink"};
    const char* sink_descriptions[] = {
        "appsink (软件合成到LVGL画布，推荐)",
        "kmssink (KMS多层渲染模式)",
        "nvdrmvideosink (DRM叠加平面模式)",
        "waylandsink (Wayland合成器模式)"
    };
    
    bool found_sink = false;
    for (int i = 0; i < 4; i++) {
        GstElementFactory *factory = gst_element_factory_find(sink_names[i]);
        if (factory) {
            std::cout << "✓ 可用: " << sink_descriptions[i] << std::endl;
            gst_object_unref(factory);
            found_sink = true;
        }
    }
    
    if (!found_sink) {
        std::cerr << "警告: 未找到合适的视频sink" << std::endl;
    }
    
    // 设置DRM叠加平面
    if (config_.sink_mode == VideoSinkMode::NVDRMVIDEOSINK) {
        if (!setupDRMOverlayPlane()) {
            std::cout << "DRM叠加平面设置失败，将回退到appsink软件合成模式" << std::endl;
            config_.sink_mode = VideoSinkMode::APPSINK;
        }
    }
    
    // 计算视频布局
    video_layout_ = calculateVideoLayout(config);
    
    std::cout << "视频布局计算完成:" << std::endl;
    std::cout << "  偏移: (" << video_layout_.offset_x << ", " << video_layout_.offset_y << ")" << std::endl;
    std::cout << "  尺寸: " << video_layout_.width << "x" << video_layout_.height << std::endl;
    std::cout << "  可用区域: " << video_layout_.available_width << "x" << video_layout_.available_height << std::endl;
    
    initialized_ = true;
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
    const int MAX_RETRIES = 3;
    const int RETRY_DELAY_MS = 1000;
    
    for (int retry = 0; retry < MAX_RETRIES; retry++) {
        if (retry > 0) {
            std::cout << "重试启动管道 (第" << retry + 1 << "次尝试)..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
        }
        
        // 构建管道
        std::string pipeline_str = buildPipeline(config_, video_layout_);
        std::cout << "管道字符串: " << pipeline_str << std::endl;
        
        // 创建管道
        GError *error = nullptr;
        pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
        
        if (!pipeline_ || error) {
            std::cerr << "创建管道失败: " << (error ? error->message : "未知错误") << std::endl;
            if (error) g_error_free(error);
            if (retry < MAX_RETRIES - 1) continue;
            return false;
        }
        
        // 检查NVMM缓冲区可用性
        if (!checkNVMMBufferAvailability()) {
            std::cout << "NVMM缓冲区检查失败，等待释放..." << std::endl;
            if (pipeline_) {
                gst_object_unref(pipeline_);
                pipeline_ = nullptr;
            }
            if (retry < MAX_RETRIES - 1) continue;
        }
        
        // 设置消息总线
        bus_ = gst_element_get_bus(pipeline_);
        bus_watch_id_ = gst_bus_add_watch(bus_, busCallback, this);
        
        // 启动管道 - 添加详细错误诊断和重试机制
        std::cout << "正在设置管道状态为PLAYING..." << std::endl;
        GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "启动管道失败，进行错误诊断..." << std::endl;
            
            // 获取详细错误信息
            GstBus* bus = gst_element_get_bus(pipeline_);
            GstMessage* msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
                static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING));
                
            if (msg) {
                GError* err;
                gchar* debug_info;
                gst_message_parse_error(msg, &err, &debug_info);
                std::cerr << "GStreamer错误: " << err->message << std::endl;
                if (debug_info) {
                    std::cerr << "调试信息: " << debug_info << std::endl;
                    
                    // 检查是否为NVMM相关错误
                    if (strstr(debug_info, "NvBuffer") || strstr(debug_info, "NVMM")) {
                        std::cout << "检测到NVMM缓冲区错误，准备重试..." << std::endl;
                    }
                    g_free(debug_info);
                }
                g_error_free(err);
                gst_message_unref(msg);
            }
            gst_object_unref(bus);
            
            cleanup();
            if (retry < MAX_RETRIES - 1) continue;
            return false;
        } else if (ret == GST_STATE_CHANGE_ASYNC) {
            std::cout << "管道异步启动中，等待状态变化..." << std::endl;
            // 增加超时时间，给NVMM缓冲区分配更多时间
            GstState state;
            ret = gst_element_get_state(pipeline_, &state, NULL, 10 * GST_SECOND);
            if (ret == GST_STATE_CHANGE_FAILURE) {
                std::cerr << "管道异步启动失败" << std::endl;
                cleanup();
                if (retry < MAX_RETRIES - 1) continue;
                return false;
            }
        }
        
        // 成功启动，跳出重试循环
        break;
    }
    
    running_ = true;
    const char* mode_names[] = {"nvdrmvideosink", "waylandsink", "kmssink", "appsink"};
    const char* mode_name = mode_names[static_cast<int>(config_.sink_mode)];
    std::cout << "DeepStream 管道启动成功 (" << mode_name << "，与LVGL协同工作)" << std::endl;
    
    // 如果使用appsink模式，设置回调函数
    if (config_.sink_mode == VideoSinkMode::APPSINK) {
        setupAppSinkCallbacks();
    }
    
    return true;
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
    
    // 尝试打开nvidia-drm设备，按优先级顺序
    int drm_fd = -1;
    const char* drm_devices[] = {
        "/dev/dri/card1",
    };
    
    for (int i = 0; i < 5; i++) {
        drm_fd = open(drm_devices[i], O_RDWR);
        if (drm_fd >= 0) {
            // 检查是否是nvidia-drm设备
            drmVersionPtr version = drmGetVersion(drm_fd);
            if (version) {
                std::cout << "检查DRM设备 " << drm_devices[i] << ": 驱动=" << version->name << std::endl;
                if (strcmp(version->name, "nvidia-drm") == 0) {
                    std::cout << "找到nvidia-drm设备: " << drm_devices[i] << std::endl;
                    drmFreeVersion(version);
                    break;
                }
                drmFreeVersion(version);
            }
            close(drm_fd);
            drm_fd = -1;
        }
    }
    
    if (drm_fd < 0) {
        std::cerr << "无法找到可用的nvidia-drm设备" << std::endl;
        return config;
    }
    
    std::cout << "开始检测DRM叠加平面..." << std::endl;
    
    // 获取DRM资源
    drmModeRes* resources = drmModeGetResources(drm_fd);
    if (!resources) {
        std::cerr << "无法获取DRM资源" << std::endl;
        close(drm_fd);
        return config;
    }
    
    std::cout << "找到 " << resources->count_crtcs << " 个CRTC, "
              << resources->count_connectors << " 个连接器" << std::endl;
    
    // 查找活跃的CRTC
    uint32_t active_crtc_id = 0;
    int active_crtc_index = -1;
    
    for (int i = 0; i < resources->count_crtcs; i++) {
        drmModeCrtc* crtc = drmModeGetCrtc(drm_fd, resources->crtcs[i]);
        if (crtc && crtc->mode_valid) {
            active_crtc_id = resources->crtcs[i];
            active_crtc_index = i;
            std::cout << "找到活跃CRTC: " << active_crtc_id << " (索引: " << i << ")" << std::endl;
            drmModeFreeCrtc(crtc);
            break;
        }
        if (crtc) drmModeFreeCrtc(crtc);
    }
    
    // 如果没有找到活跃CRTC，使用第一个CRTC
    if (active_crtc_id == 0 && resources->count_crtcs > 0) {
        active_crtc_id = resources->crtcs[0];
        active_crtc_index = 0;
        std::cout << "使用第一个CRTC: " << active_crtc_id << " (索引: " << active_crtc_index << ")" << std::endl;
    }
    
    // 查找可用的叠加平面
    drmModePlaneRes* plane_resources = drmModeGetPlaneResources(drm_fd);
    if (plane_resources) {
        std::cout << "找到 " << plane_resources->count_planes << " 个平面，开始检查..." << std::endl;
        
        for (uint32_t i = 0; i < plane_resources->count_planes; i++) {
            uint32_t plane_id = plane_resources->planes[i];
            drmModePlane* plane = drmModeGetPlane(drm_fd, plane_id);
            
            if (plane) {
                std::cout << "检查平面 " << plane_id << ": possible_crtcs=0x"
                          << std::hex << plane->possible_crtcs << std::dec
                          << ", crtc_id=" << plane->crtc_id
                          << ", fb_id=" << plane->fb_id;
                
                // 首先检查平面是否未被占用
                bool is_free = (plane->crtc_id == 0 && plane->fb_id == 0);
                if (!is_free) {
                    std::cout << " [已占用]" << std::endl;
                    drmModeFreePlane(plane);
                    continue;
                }
                
                // 检查possible_crtcs位掩码是否与活跃CRTC匹配
                // possible_crtcs是位掩码，每一位对应一个CRTC索引
                if (active_crtc_index >= 0 && (plane->possible_crtcs & (1 << active_crtc_index))) {
                    
                    // 检查平面类型，NVIDIA DRM中Overlay类型值可能为0
                    drmModeObjectProperties* props = drmModeObjectGetProperties(drm_fd, plane_id, DRM_MODE_OBJECT_PLANE);
                    bool is_overlay = false;
                    uint64_t plane_type = 0;
                    
                    if (props) {
                        for (uint32_t j = 0; j < props->count_props; j++) {
                            drmModePropertyRes* prop = drmModeGetProperty(drm_fd, props->props[j]);
                            if (prop && strcmp(prop->name, "type") == 0) {
                                plane_type = props->prop_values[j];
                                // NVIDIA DRM中: 0=Overlay, 1=Primary, 2=Cursor (与标准不同)
                                if (plane_type == 0) {  // NVIDIA的Overlay类型值
                                    is_overlay = true;
                                    std::cout << " [OVERLAY(NVIDIA)]";
                                } else if (plane_type == 1) {
                                    std::cout << " [PRIMARY]";
                                } else if (plane_type == 2) {
                                    std::cout << " [CURSOR]";
                                } else {
                                    std::cout << " [TYPE=" << plane_type << "]";
                                }
                                drmModeFreeProperty(prop);
                                break;
                            }
                            if (prop) drmModeFreeProperty(prop);
                        }
                        drmModeFreeObjectProperties(props);
                    }
                    
                    // 如果找不到type属性或检测为Overlay类型，则尝试使用该平面
                    if (is_overlay || (props == nullptr)) {
                        // 找到可用的叠加平面
                        config.plane_id = plane_id;
                        config.crtc_id = active_crtc_id;
                        
                        // 查找连接到此CRTC的连接器
                        for (int j = 0; j < resources->count_connectors; j++) {
                            drmModeConnector* connector = drmModeGetConnector(drm_fd, resources->connectors[j]);
                            if (connector && connector->connection == DRM_MODE_CONNECTED) {
                                config.connector_id = resources->connectors[j];
                                drmModeFreeConnector(connector);
                                break;
                            }
                            if (connector) drmModeFreeConnector(connector);
                        }
                        
                        std::cout << " -> 选中!" << std::endl;
                        std::cout << "检测到可用DRM叠加平面: plane_id=" << config.plane_id
                                  << ", crtc_id=" << config.crtc_id
                                  << ", connector_id=" << config.connector_id << std::endl;
                        
                        drmModeFreePlane(plane);
                        break;
                    } else {
                        std::cout << " [非叠加平面]" << std::endl;
                    }
                } else {
                    std::cout << " [CRTC不匹配]" << std::endl;
                }
                
                drmModeFreePlane(plane);
            }
        }
        drmModeFreePlaneResources(plane_resources);
    }
    
    drmModeFreeResources(resources);
    close(drm_fd);
    
    if (config.plane_id == -1) {
        std::cerr << "未找到可用的DRM叠加平面" << std::endl;
    }
    
    return config;
}

bool DeepStreamManager::setupDRMOverlayPlane() {
    std::cout << "设置DRM叠加平面..." << std::endl;
    
    // 如果未配置叠加平面，自动检测
    if (config_.overlay.plane_id == -1) {
        config_.overlay = detectAvailableOverlayPlane();
        if (config_.overlay.plane_id == -1) {
            std::cerr << "未找到可用的DRM叠加平面" << std::endl;
            return false;
        }
    }
    
    std::cout << "DRM叠加平面设置完成: plane_id=" << config_.overlay.plane_id
              << ", z_order=" << config_.overlay.z_order << std::endl;
    return true;
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
            return buildKMSSinkPipeline(config, offset_x, offset_y, width, height);
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
    
    // 先用测试源验证 kmssink
    pipeline << "videotestsrc pattern=smpte ! "
             << "video/x-raw,width=" << width
             << ",height=" << height
             << ",framerate=" << config.camera_fps << "/1 ! "
             << "videoconvert ! "
             << "video/x-raw,format=BGRx ! "  // kmssink 支持的格式
             << "kmssink "
             << "driver-name=nvidia-drm "     // 使用 nvidia-drm 驱动
             << "plane-id=44 "                // overlay plane
             << "connector-id=63 "            // 从检测中获取的 connector
             << "can-scale=true "             // 启用缩放支持
             << "force-modesetting=false "    // 不改变显示模式
             << "sync=false";                 // 降低延迟
    
    return pipeline.str();
}


std::string DeepStreamManager::buildWaylandSinkPipeline(
    const DeepStreamConfig& config,
    int offset_x,
    int offset_y,
    int width,
    int height) {
    
    std::ostringstream pipeline;
    
    // 使用 waylandsink（Wayland合成器模式）
    pipeline << "nvarguscamerasrc sensor-id=" << config.camera_id << " ! "
             << "video/x-raw(memory:NVMM),width=" << config.camera_width
             << ",height=" << config.camera_height
             << ",framerate=" << config.camera_fps << "/1,format=NV12 ! "
             << "nvvideoconvert ! "
             << "video/x-raw,format=RGBA ! "
             << "waylandsink "
             << "sync=false";  // 降低延迟
    
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
// 修复：移除不支持的属性
std::string DeepStreamManager::buildCameraSource(const DeepStreamConfig& config) {
    std::ostringstream source;
    
    switch (config.camera_source) {
        case CameraSourceMode::NVARGUSCAMERA:
            // 真实摄像头源 - JetPack 6兼容配置
            source << "nvarguscamerasrc sensor-id=" << config.camera_id << " "
                   << "wbmode=0 "                    // 自动白平衡
                   << "aelock=false "                // 自动曝光不锁定
                   << "awblock=false "               // 自动白平衡不锁定
                   << "! video/x-raw(memory:NVMM)"
                   << ",width=" << config.camera_width
                   << ",height=" << config.camera_height
                   << ",framerate=" << config.camera_fps << "/1"
                   << ",format=NV12";
            break;
            
        case CameraSourceMode::VIDEOTESTSRC:
            // 虚拟测试源 - 使用稳定的彩色条纹图案
            source << "videotestsrc pattern=0 "     // 使用彩色条纹图案而非smpte
                   << "is-live=true "
                   << "do-timestamp=true "
                   << "! video/x-raw"
                   << ",width=" << config.camera_width
                   << ",height=" << config.camera_height
                   << ",framerate=" << config.camera_fps << "/1"
                   << ",format=I420";
            break;
            
        case CameraSourceMode::FILESRC:
            // 文件源
            source << "filesrc location=" << config.video_file_path << " "
                   << "! decodebin "
                   << "! nvvidconv "                 // 使用硬件加速转换器
                   << "! video/x-raw"
                   << ",width=" << config.camera_width
                   << ",height=" << config.camera_height
                   << ",framerate=" << config.camera_fps << "/1"
                   << ",format=I420";
            break;
            
        default:
            // 默认使用真实摄像头，如果失败回退到测试源
            std::cout << "默认尝试使用真实摄像头源..." << std::endl;
            source << "nvarguscamerasrc sensor-id=" << config.camera_id << " "
                   << "wbmode=0 "
                   << "! video/x-raw(memory:NVMM)"
                   << ",width=" << config.camera_width
                   << ",height=" << config.camera_height
                   << ",framerate=" << config.camera_fps << "/1"
                   << ",format=NV12";
            break;
    }
    
    return source.str();
}

// 新增：构建KMSSink管道 - 解决多层显示冲突
std::string DeepStreamManager::buildKMSSinkPipeline(
    const DeepStreamConfig& config,
    int offset_x,
    int offset_y,
    int width,
    int height) {
    
    std::ostringstream pipeline;
    
    // 构建摄像头源
    pipeline << buildCameraSource(config) << " ! ";
    
    // 使用JetPack 6的硬件加速转换器（正确的分步骤转换）
    if (config.camera_source == CameraSourceMode::NVARGUSCAMERA) {
        // 对于真实摄像头（NVMM格式），分两步转换：
        // 第一步：NV12 → RGBA（保持在NVMM硬件加速内存中）
        pipeline << "nvvidconv ! "
                 << "video/x-raw(memory:NVMM),format=RGBA ! ";
        
        // 第二步：RGBA(NVMM) → BGRA（转到系统内存）+ 尺寸调整
        pipeline << "nvvidconv ! "
                 << "video/x-raw,format=BGRA ! "
                 << "videoscale ! "
                 << "video/x-raw,format=BGRA,width=" << width << ",height=" << height << " ! ";
    } else {
        // 对于测试源（普通内存），使用软件转换
        pipeline << "videoconvert ! "
                 << "videoscale ! "
                 << "video/x-raw,format=BGRA,width=" << width << ",height=" << height << " ! ";
    }
    
    pipeline << "queue "
             << "max-size-buffers=4 "      // 适中的缓冲区深度
             << "max-size-time=0 "
             << "leaky=downstream "
             << "! ";
    
    // 使用kmssink - 更好的多层渲染兼容性
    pipeline << "kmssink "
             << "connector-id=-1 "         // 自动检测连接器
             << "plane-id=-1 "             // 自动检测平面
             << "force-modesetting=false " // 不强制设置模式
             << "can-scale=true "          // 启用硬件缩放
             << "sync=false "              // 低延迟模式
             << "restore-crtc=true";       // 退出时恢复CRTC状态
    
    std::cout << "构建KMSSink管道 (多层渲染兼容): " << pipeline.str() << std::endl;
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
    
    if (config.camera_source == CameraSourceMode::NVARGUSCAMERA) {
        // ✅ 使用相机原生分辨率，避免NVMM缓冲区问题
        pipeline << "nvarguscamerasrc sensor-id=" << config.camera_id << " "
                 << "sensor-mode=2 "  // 1920x1080@30fps
                 << "! video/x-raw(memory:NVMM)"
                 << ",width=1920,height=1080"
                 << ",framerate=30/1,format=NV12 "
                 << "! nvvidconv "  // 硬件缩放 + 格式转换
                 << "! video/x-raw"
                 << ",width=" << width << ",height=" << height
                 << ",format=RGBA "
                 << "! videoconvert "  // RGBA → BGRA
                 << "! video/x-raw,format=BGRA ";
    } else {
        pipeline << buildCameraSource(config) << " ! "
                 << "videoconvert ! "
                 << "video/x-raw,format=BGRA ! "
                 << "videoscale ! "
                 << "video/x-raw,format=BGRA"
                 << ",width=" << width << ",height=" << height << " ";
    }
    
    pipeline << "! queue "
             << "max-size-buffers=2 "
             << "max-size-time=0 "
             << "leaky=downstream "
             << "! appsink name=video_appsink "
             << "emit-signals=true "
             << "sync=false "
             << "max-buffers=2 "
             << "drop=true";
    
    std::cout << "构建AppSink软件合成管道: " << pipeline.str() << std::endl;
    return pipeline.str();
}

// AppSink新样本回调 - 线程安全的帧处理
GstFlowReturn DeepStreamManager::newSampleCallback(GstAppSink* appsink, gpointer user_data) {
    DeepStreamManager* manager = static_cast<DeepStreamManager*>(user_data);
    
    std::cout << "newSampleCallback被调用" << std::endl;
    
    // 获取新样本
    GstSample* sample = gst_app_sink_pull_sample(appsink);
    if (!sample) {
        std::cout << "错误：无法获取sample" << std::endl;
        return GST_FLOW_OK;
    }
    
    std::cout << "成功获取sample" << std::endl;
    
    // 获取缓冲区
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        std::cout << "错误：无法获取buffer" << std::endl;
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    
    std::cout << "成功获取buffer" << std::endl;
    
    // 映射缓冲区数据
    GstMapInfo map_info;
    if (!gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
        std::cout << "错误：无法映射buffer数据" << std::endl;
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    
    std::cout << "成功映射buffer，数据大小: " << map_info.size << " 字节" << std::endl;
    
    // 获取视频信息
    GstCaps* caps = gst_sample_get_caps(sample);
    if (caps) {
        gchar* caps_str = gst_caps_to_string(caps);
        std::cout << "Caps信息: " << caps_str << std::endl;
        g_free(caps_str);
        
        GstStructure* structure = gst_caps_get_structure(caps, 0);
        gint width, height;
        const gchar* format;
        
        format = gst_structure_get_string(structure, "format");
        if (format) {
            std::cout << "视频格式: " << format << std::endl;
        }
        
        if (gst_structure_get_int(structure, "width", &width) &&
            gst_structure_get_int(structure, "height", &height)) {
            
            std::cout << "视频尺寸: " << width << "x" << height << std::endl;
            
            // 线程安全地合成帧到LVGL画布
            manager->compositeFrameToLVGL(&map_info, width, height);
        } else {
            std::cout << "错误：无法获取视频尺寸信息" << std::endl;
        }
    } else {
        std::cout << "错误：无法获取caps信息" << std::endl;
    }
    
    // 清理资源
    gst_buffer_unmap(buffer, &map_info);
    gst_sample_unref(sample);
    
    return GST_FLOW_OK;
}

// 软件合成帧到LVGL画布 - 优化内存操作
void DeepStreamManager::compositeFrameToLVGL(GstMapInfo* map_info, int width, int height) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    std::cout << "compositeFrameToLVGL被调用，尺寸: " << width << "x" << height
              << "，数据大小: " << map_info->size << " 字节" << std::endl;
    
    try {
        // 检查数据大小是否合理 (BGRA格式应该是 width * height * 4)
        size_t expected_size = width * height * 4;
        if (map_info->size < expected_size) {
            std::cout << "警告：数据大小不匹配，期望: " << expected_size
                     << "，实际: " << map_info->size << std::endl;
        }
        
        // 创建OpenCV Mat包装GStreamer数据，避免内存拷贝
        cv::Mat frame;
        
        // 统一使用BGRA格式，确保兼容性
        frame = cv::Mat(height, width, CV_8UC4, map_info->data);
        
        std::cout << "创建OpenCV Mat: " << frame.cols << "x" << frame.rows
                 << "，通道数: " << frame.channels()
                 << "，数据指针: " << (void*)frame.data << std::endl;
        
        // 检查帧数据有效性
        if (!frame.empty() && frame.data) {
            // 检查第一个像素的值，确保数据不是全黑
            if (frame.channels() == 4) {
                cv::Vec4b first_pixel = frame.at<cv::Vec4b>(0, 0);
                cv::Vec4b center_pixel = frame.at<cv::Vec4b>(height/2, width/2);
                
                std::cout << "第一个像素BGRA值: ["
                         << (int)first_pixel[0] << ", " << (int)first_pixel[1]
                         << ", " << (int)first_pixel[2] << ", " << (int)first_pixel[3] << "]" << std::endl;
                         
                std::cout << "中心像素BGRA值: ["
                         << (int)center_pixel[0] << ", " << (int)center_pixel[1]
                         << ", " << (int)center_pixel[2] << ", " << (int)center_pixel[3] << "]" << std::endl;
            }
            
            // 克隆帧数据用于后续处理
            latest_frame_ = frame.clone();
            new_frame_available_ = true;
            
            std::cout << "帧数据已更新，设置new_frame_available标志" << std::endl;
        } else {
            std::cout << "错误：帧数据为空或无效" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "合成帧到LVGL时发生异常: " << e.what() << std::endl;
    }
}

// 设置AppSink回调函数
void DeepStreamManager::setupAppSinkCallbacks() {
    if (!pipeline_) {
        std::cout << "错误：管道未创建，无法设置appsink回调" << std::endl;
        return;
    }
    
    // 查找appsink元素
    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "video_appsink");
    if (!appsink_) {
        std::cout << "错误：未找到appsink元素" << std::endl;
        return;
    }
    
    // 设置appsink属性
    g_object_set(G_OBJECT(appsink_),
                 "emit-signals", TRUE,    // 启用信号
                 "sync", FALSE,           // 异步模式
                 "max-buffers", 2,        // 最大缓冲区数量
                 "drop", TRUE,            // 丢弃旧帧
                 NULL);
    
    // 连接新样本信号
    g_signal_connect(appsink_, "new-sample", G_CALLBACK(newSampleCallback), this);
    
    std::cout << "AppSink回调函数设置完成" << std::endl;
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
    
    auto last_update = std::chrono::steady_clock::now();
    const auto target_interval = std::chrono::milliseconds(33); // 30fps
    
    while (canvas_update_running_) {
        auto current_time = std::chrono::steady_clock::now();
        
        if (new_frame_available_.load() && lvgl_interface_) {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            
            if (!latest_frame_.empty()) {
                #ifdef ENABLE_LVGL
                std::cout << "处理新帧: " << latest_frame_.cols << "x" << latest_frame_.rows
                         << " 通道数:" << latest_frame_.channels() << std::endl;
                         
                auto* lvgl_if = static_cast<bamboo_cut::ui::LVGLInterface*>(lvgl_interface_);
                lv_obj_t* canvas = lvgl_if->getCameraCanvas();
                
                if (canvas) {
                    std::cout << "Canvas对象获取成功" << std::endl;
                    
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
                        std::cout << "帧数据不连续，已克隆" << std::endl;
                    }
                    
                    // 验证数据
                    if (display_frame.channels() != 4 || 
                        display_frame.cols != 960 || 
                        display_frame.rows != 640) {
                        std::cout << "错误：帧格式不正确" << std::endl;
                        continue;
                    }
                    
                    // 调试：检查源数据
                    cv::Vec4b src_first = display_frame.at<cv::Vec4b>(0, 0);
                    cv::Vec4b src_center = display_frame.at<cv::Vec4b>(320, 480);
                    std::cout << "源数据 - 第一个像素BGRA: [" 
                             << (int)src_first[0] << "," << (int)src_first[1] 
                             << "," << (int)src_first[2] << "," << (int)src_first[3] << "]" << std::endl;
                    std::cout << "源数据 - 中心像素BGRA: [" 
                             << (int)src_center[0] << "," << (int)src_center[1] 
                             << "," << (int)src_center[2] << "," << (int)src_center[3] << "]" << std::endl;
                    
                    // 获取canvas缓冲区
                    lv_img_dsc_t* canvas_dsc = lv_canvas_get_image(canvas);
                    if (canvas_dsc && canvas_dsc->data) {
                        std::cout << "Canvas缓冲区获取成功" << std::endl;
                        
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
                        
                        std::cout << "像素转换完成" << std::endl;
                        
                        // 验证转换结果
                        uint32_t dst_first = canvas_buffer[0];
                        uint32_t dst_center = canvas_buffer[320 * 960 + 480];
                        
                        std::cout << "目标数据 - 第一个像素ARGB: 0x" << std::hex << dst_first << std::dec;
                        std::cout << " [A=" << ((dst_first >> 24) & 0xFF)
                                 << ",R=" << ((dst_first >> 16) & 0xFF)
                                 << ",G=" << ((dst_first >> 8) & 0xFF)
                                 << ",B=" << (dst_first & 0xFF) << "]" << std::endl;
                                 
                        std::cout << "目标数据 - 中心像素ARGB: 0x" << std::hex << dst_center << std::dec;
                        std::cout << " [A=" << ((dst_center >> 24) & 0xFF)
                                 << ",R=" << ((dst_center >> 16) & 0xFF)
                                 << ",G=" << ((dst_center >> 8) & 0xFF)
                                 << ",B=" << (dst_center & 0xFF) << "]" << std::endl;
                        
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

} // namespace deepstream
} // namespace bamboo_cut