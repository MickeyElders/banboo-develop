/**
 * @file deepstream_manager.cpp
 * @brief DeepStream AI推理和视频显示管理器实现 - 支持nvdrmvideosink叠加平面模式
 */

#include "bamboo_cut/deepstream/deepstream_manager.h"
#include <iostream>
#include <sstream>
#include <gst/gst.h>
#include <fstream>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

namespace bamboo_cut {
namespace deepstream {

DeepStreamManager::DeepStreamManager()
    : pipeline_(nullptr)
    , pipeline2_(nullptr)
    , bus_(nullptr)
    , bus2_(nullptr)
    , bus_watch_id_(0)
    , bus_watch_id2_(0)
    , running_(false)
    , initialized_(false) {
}

DeepStreamManager::~DeepStreamManager() {
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
    const char* sink_names[] = {"nvdrmvideosink", "nv3dsink", "nveglglessink"};
    const char* sink_descriptions[] = {
        "nvdrmvideosink (DRM叠加平面模式，独立显示层)",
        "nv3dsink (窗口模式，与LVGL兼容)",
        "nveglglessink (EGL窗口模式)"
    };
    
    bool found_sink = false;
    for (int i = 0; i < 3; i++) {
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
            std::cout << "DRM叠加平面设置失败，将回退到nv3dsink模式" << std::endl;
            config_.sink_mode = VideoSinkMode::NV3DSINK;
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
    // 构建管道
    std::string pipeline_str = buildPipeline(config_, video_layout_);
    std::cout << "管道字符串: " << pipeline_str << std::endl;
    
    // 创建管道
    GError *error = nullptr;
    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    
    if (!pipeline_ || error) {
        std::cerr << "创建管道失败: " << (error ? error->message : "未知错误") << std::endl;
        if (error) g_error_free(error);
        return false;
    }
    
    // 设置消息总线
    bus_ = gst_element_get_bus(pipeline_);
    bus_watch_id_ = gst_bus_add_watch(bus_, busCallback, this);
    
    // 启动管道 - 添加详细错误诊断
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
                g_free(debug_info);
            }
            g_error_free(err);
            gst_message_unref(msg);
        }
        gst_object_unref(bus);
        
        cleanup();
        return false;
    } else if (ret == GST_STATE_CHANGE_ASYNC) {
        std::cout << "管道异步启动中，等待状态变化..." << std::endl;
        // 等待异步状态变化完成
        GstState state;
        ret = gst_element_get_state(pipeline_, &state, NULL, GST_CLOCK_TIME_NONE);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "管道异步启动失败" << std::endl;
            cleanup();
            return false;
        }
    }
    
    running_ = true;
    const char* mode_name = config_.sink_mode == VideoSinkMode::NVDRMVIDEOSINK ? 
        "DRM叠加平面模式" : "窗口模式";
    std::cout << "DeepStream 管道启动成功 (" << mode_name << "，与LVGL分离显示)" << std::endl;
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
    const char* mode_name = config_.sink_mode == VideoSinkMode::NVDRMVIDEOSINK ? 
        "DRM叠加平面模式" : "窗口模式";
    std::cout << "双摄像头并排显示管道启动成功 (" << mode_name << ")" << std::endl;
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
    const char* mode_names[] = {"nv3dsink", "nvdrmvideosink", "waylandsink"};
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
            std::cerr << "DRM叠加平面设置失败，无法切换到nvdrmvideosink模式" << std::endl;
            config_.sink_mode = VideoSinkMode::NV3DSINK;  // 回退
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
        "/dev/dri/card0",
        "/dev/dri/card1",
        "/dev/dri/card2",
        "/dev/dri/renderD128",
        "/dev/dri/renderD129"
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
        case VideoSinkMode::NV3DSINK:
        default:
            return buildNV3DSinkPipeline(config, offset_x, offset_y, width, height);
    }
}

std::string DeepStreamManager::buildNVDRMVideoSinkPipeline(
    const DeepStreamConfig& config,
    int offset_x,
    int offset_y,
    int width,
    int height) {
    
    std::ostringstream pipeline;
    
    // 使用 nvdrmvideosink（DRM叠加平面模式，独立于LVGL显示层）
    pipeline << "nvarguscamerasrc sensor-id=" << config.camera_id << " ! "
             << "video/x-raw(memory:NVMM),width=" << config.camera_width
             << ",height=" << config.camera_height
             << ",framerate=" << config.camera_fps << "/1,format=NV12 ! "
             << "nvvideoconvert ! "
             << "video/x-raw(memory:NVMM),format=RGBA ! "
             << "nvdrmvideosink ";
             
    // 使用正确的nvdrmvideosink属性
    if (config.overlay.plane_id != -1) {
        pipeline << "plane-id=" << config.overlay.plane_id << " ";
    }
    
    if (config.overlay.connector_id != -1) {
        pipeline << "conn-id=" << config.overlay.connector_id << " ";
    }
    
    if (config.overlay.crtc_id != -1) {
        pipeline << "crtc-id=" << config.overlay.crtc_id << " ";
    }
    
    // 添加位置参数
    pipeline << "offset-x=" << offset_x << " "
             << "offset-y=" << offset_y << " ";
    
    // 其他设置（nvdrmvideosink不支持zorder属性，层级由DRM硬件平面类型决定）
    pipeline << "set-mode=false "
             << "show-preroll-frame=false "
             << "sync=false";
             
    if (config.overlay.enable_scaling) {
        pipeline << " enable-last-sample=false";
    }
    
    return pipeline.str();
}

std::string DeepStreamManager::buildNV3DSinkPipeline(
    const DeepStreamConfig& config,
    int offset_x,
    int offset_y,
    int width,
    int height) {
    
    std::ostringstream pipeline;
    
    // 使用 nv3dsink 窗口模式（与LVGL共存）
    pipeline << "nvarguscamerasrc sensor-id=" << config.camera_id << " ! "
             << "video/x-raw(memory:NVMM),width=" << config.camera_width
             << ",height=" << config.camera_height
             << ",framerate=" << config.camera_fps << "/1,format=NV12 ! "
             << "nvvideoconvert ! "
             << "video/x-raw(memory:NVMM),format=RGBA ! "
             << "nv3dsink "
             << "window-x=" << offset_x << " "
             << "window-y=" << offset_y << " "
             << "window-width=" << width << " "
             << "window-height=" << height << " "
             << "sync=false";  // 降低延迟
    
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
                     << "nvdrmvideosink ";
            // 使用正确的nvdrmvideosink属性
            if (config.overlay.plane_id != -1) {
                pipeline << "plane-id=" << config.overlay.plane_id << " ";
            }
            // 添加位置参数
            pipeline << "offset-x=" << layout.offset_x << " "
                    << "offset-y=" << layout.offset_y << " ";
            // nvdrmvideosink不支持zorder，使用conn-id和crtc-id代替
            if (config.overlay.connector_id != -1) {
                pipeline << "conn-id=" << config.overlay.connector_id << " ";
            }
            if (config.overlay.crtc_id != -1) {
                pipeline << "crtc-id=" << config.overlay.crtc_id << " ";
            }
            pipeline << "set-mode=false "
            break;
        case VideoSinkMode::WAYLANDSINK:
            pipeline << "video/x-raw,format=RGBA ! "
                     << "waylandsink "
                     << "sync=false";
            break;
        case VideoSinkMode::NV3DSINK:
        default:
            pipeline << "video/x-raw(memory:NVMM),format=RGBA ! "
                     << "nv3dsink "
                     << "window-x=" << layout.offset_x << " "
                     << "window-y=" << layout.offset_y << " "
                     << "window-width=" << layout.width << " "
                     << "window-height=" << layout.height << " "
                     << "sync=false";
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

} // namespace deepstream
} // namespace bamboo_cut