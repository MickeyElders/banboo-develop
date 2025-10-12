/**
 * @file deepstream_wayland_manager.cpp
 * @brief DeepStream Wayland适配管理器 - 专门为Wayland合成器架构设计
 */

#include "bamboo_cut/deepstream/deepstream_manager.h"
#include "bamboo_cut/ui/lvgl_wayland_interface.h"
#include <iostream>
#include <sstream>
#include <gst/gst.h>
#include <thread>
#include <chrono>

namespace bamboo_cut {
namespace deepstream {

/**
 * @brief Wayland专用的DeepStream管理器
 * 移除DRM复杂性，专注于waylandsink模式
 */
class DeepStreamWaylandManager {
public:
    DeepStreamWaylandManager() 
        : pipeline_(nullptr)
        , bus_(nullptr)
        , bus_watch_id_(0)
        , running_(false)
        , initialized_(false) {
    }
    
    ~DeepStreamWaylandManager() {
        stop();
        cleanup();
    }

    bool initialize(const WaylandSinkConfig& config) {
        std::cout << "[DeepStreamWayland] 初始化Wayland视频系统..." << std::endl;
        config_ = config;
        
        // 初始化GStreamer
        if (!gst_is_initialized()) {
            gst_init(nullptr, nullptr);
            std::cout << "[DeepStreamWayland] GStreamer初始化完成" << std::endl;
        }
        
        // 检查waylandsink可用性
        GstElementFactory* factory = gst_element_factory_find("waylandsink");
        if (!factory) {
            std::cerr << "[DeepStreamWayland] waylandsink插件不可用" << std::endl;
            return false;
        }
        gst_object_unref(factory);
        std::cout << "[DeepStreamWayland] waylandsink插件可用" << std::endl;
        
        // 检查Wayland环境
        if (!checkWaylandEnvironment()) {
            std::cerr << "[DeepStreamWayland] Wayland环境检查失败" << std::endl;
            return false;
        }
        
        initialized_ = true;
        return true;
    }

    bool start() {
        if (!initialized_) {
            std::cerr << "[DeepStreamWayland] 系统未初始化" << std::endl;
            return false;
        }
        
        if (running_) {
            std::cout << "[DeepStreamWayland] 已在运行" << std::endl;
            return true;
        }
        
        std::cout << "[DeepStreamWayland] 启动Wayland视频管道..." << std::endl;
        
        // 构建Wayland管道
        std::string pipeline_str = buildWaylandPipeline();
        std::cout << "[DeepStreamWayland] 管道: " << pipeline_str << std::endl;
        
        // 创建管道
        GError* error = nullptr;
        pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
        
        if (!pipeline_ || error) {
            std::cerr << "[DeepStreamWayland] 管道创建失败: " 
                      << (error ? error->message : "未知错误") << std::endl;
            if (error) g_error_free(error);
            return false;
        }
        
        // 设置消息总线
        bus_ = gst_element_get_bus(pipeline_);
        bus_watch_id_ = gst_bus_add_watch(bus_, busCallback, this);
        
        // 启动管道
        GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "[DeepStreamWayland] 管道启动失败" << std::endl;
            cleanup();
            return false;
        } else if (ret == GST_STATE_CHANGE_ASYNC) {
            std::cout << "[DeepStreamWayland] 管道异步启动中..." << std::endl;
            GstState state;
            ret = gst_element_get_state(pipeline_, &state, NULL, 10 * GST_SECOND);
            if (ret == GST_STATE_CHANGE_FAILURE) {
                std::cerr << "[DeepStreamWayland] 管道异步启动失败" << std::endl;
                cleanup();
                return false;
            }
        }
        
        running_ = true;
        std::cout << "[DeepStreamWayland] Wayland视频管道启动成功" << std::endl;
        return true;
    }

    void stop() {
        if (!running_) return;
        
        std::cout << "[DeepStreamWayland] 停止Wayland视频管道..." << std::endl;
        
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
        }
        
        running_ = false;
        std::cout << "[DeepStreamWayland] Wayland视频管道已停止" << std::endl;
    }

    bool isRunning() const { return running_; }

private:
    bool checkWaylandEnvironment() {
        // 检查WAYLAND_DISPLAY环境变量
        const char* wayland_display = getenv("WAYLAND_DISPLAY");
        if (!wayland_display) {
            setenv("WAYLAND_DISPLAY", config_.wayland_display.c_str(), 0);
            wayland_display = getenv("WAYLAND_DISPLAY");
            std::cout << "[DeepStreamWayland] 设置WAYLAND_DISPLAY=" << wayland_display << std::endl;
        } else {
            std::cout << "[DeepStreamWayland] 检测到WAYLAND_DISPLAY=" << wayland_display << std::endl;
        }
        
        // 检查XDG_RUNTIME_DIR
        const char* runtime_dir = getenv("XDG_RUNTIME_DIR");
        if (!runtime_dir) {
            setenv("XDG_RUNTIME_DIR", "/run/user/1000", 0);
            runtime_dir = getenv("XDG_RUNTIME_DIR");
            std::cout << "[DeepStreamWayland] 设置XDG_RUNTIME_DIR=" << runtime_dir << std::endl;
        } else {
            std::cout << "[DeepStreamWayland] 检测到XDG_RUNTIME_DIR=" << runtime_dir << std::endl;
        }
        
        return true;
    }

    std::string buildWaylandPipeline() {
        std::ostringstream pipeline;
        
        std::cout << "[DeepStreamWayland] 构建优化的Wayland管道..." << std::endl;
        
        // 摄像头源配置
        pipeline << "nvarguscamerasrc sensor-id=" << config_.camera_id << " ";
        
        // 摄像头参数
        pipeline << "! video/x-raw(memory:NVMM)"
                 << ",width=" << config_.camera_width
                 << ",height=" << config_.camera_height  
                 << ",framerate=" << config_.framerate << "/1"
                 << ",format=NV12 ";
        
        // 可选：AI推理插件
        if (config_.enable_ai_inference && !config_.nvinfer_config.empty()) {
            pipeline << "! nvinfer config-file-path=" << config_.nvinfer_config << " ";
        }
        
        // 格式转换和缩放
        pipeline << "! nvvidconv ";
        
        // 输出格式配置
        pipeline << "! video/x-raw"
                 << ",format=" << config_.output_format
                 << ",width=" << config_.output_width
                 << ",height=" << config_.output_height << " ";
        
        // Wayland sink配置
        pipeline << "! waylandsink";
        
        // Wayland sink参数
        if (!config_.fullscreen) {
            pipeline << " render-rectangle=\"<"
                     << config_.window_x << "," << config_.window_y << ","
                     << config_.window_width << "," << config_.window_height << ">\"";
        }
        
        // 性能优化参数
        pipeline << " sync=" << (config_.sync ? "true" : "false");
        pipeline << " async=" << (config_.async ? "true" : "false");
        
        if (!config_.wayland_display.empty()) {
            pipeline << " display=" << config_.wayland_display;
        }
        
        return pipeline.str();
    }

    static gboolean busCallback(GstBus* bus, GstMessage* msg, gpointer data) {
        DeepStreamWaylandManager* manager = static_cast<DeepStreamWaylandManager*>(data);
        
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR: {
                GError* err;
                gchar* debug;
                gst_message_parse_error(msg, &err, &debug);
                std::cerr << "[DeepStreamWayland] 错误: " << err->message << std::endl;
                if (debug) {
                    std::cerr << "[DeepStreamWayland] 调试信息: " << debug << std::endl;
                    g_free(debug);
                }
                g_error_free(err);
                break;
            }
            case GST_MESSAGE_WARNING: {
                GError* err;
                gchar* debug;
                gst_message_parse_warning(msg, &err, &debug);
                std::cout << "[DeepStreamWayland] 警告: " << err->message << std::endl;
                if (debug) {
                    std::cout << "[DeepStreamWayland] 调试信息: " << debug << std::endl;
                    g_free(debug);
                }
                g_error_free(err);
                break;
            }
            case GST_MESSAGE_EOS:
                std::cout << "[DeepStreamWayland] 流结束" << std::endl;
                break;
            case GST_MESSAGE_STATE_CHANGED: {
                if (GST_MESSAGE_SRC(msg) == GST_OBJECT(manager->pipeline_)) {
                    GstState old_state, new_state;
                    gst_message_parse_state_changed(msg, &old_state, &new_state, nullptr);
                    std::cout << "[DeepStreamWayland] 状态变更: " 
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

    void cleanup() {
        if (bus_watch_id_ > 0) {
            g_source_remove(bus_watch_id_);
            bus_watch_id_ = 0;
        }
        
        if (bus_) {
            gst_object_unref(bus_);
            bus_ = nullptr;
        }
        
        if (pipeline_) {
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }

private:
    struct WaylandSinkConfig {
        // 摄像头配置
        int camera_id = 0;
        int camera_width = 1920;
        int camera_height = 1080;
        int framerate = 30;
        
        // 输出配置
        std::string output_format = "RGBA";
        int output_width = 960;
        int output_height = 640;
        
        // Wayland窗口配置
        bool fullscreen = false;
        int window_x = 0;
        int window_y = 80;           // 跳过头部面板
        int window_width = 960;
        int window_height = 640;
        
        // 性能配置
        bool sync = false;           // 低延迟模式
        bool async = true;
        
        // Wayland环境
        std::string wayland_display = "wayland-0";
        
        // AI推理（可选）
        bool enable_ai_inference = false;
        std::string nvinfer_config;
    } config_;

    GstElement* pipeline_;
    GstBus* bus_;
    guint bus_watch_id_;
    
    std::atomic<bool> running_;
    std::atomic<bool> initialized_;
};

} // namespace deepstream
} // namespace bamboo_cut