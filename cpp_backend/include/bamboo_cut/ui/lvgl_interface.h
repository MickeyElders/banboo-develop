/**
 * @file lvgl_interface.h
 * @brief C++ LVGL一体化系统界面管理器
 * 工业级竹子识别系统LVGL界面实现
 */

#pragma once

#ifdef ENABLE_LVGL
#include <lvgl/lvgl.h>
#else
// LVGL未启用时的占位符定义
typedef void* lv_obj_t;
typedef void* lv_event_t;
typedef void* lv_disp_drv_t;
typedef void* lv_indev_drv_t;
typedef void* lv_disp_t;
typedef void* lv_indev_t;
typedef void* lv_area_t;
typedef void* lv_color_t;
typedef void* lv_draw_buf_t;
typedef void* lv_indev_data_t;
#endif
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include "bamboo_cut/core/data_bridge.h"
#include "bamboo_cut/utils/jetson_monitor.h"

// DRM头文件包含
#ifdef ENABLE_DRM
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm/drm.h>
#include <drm/drm_mode.h>
#endif

namespace bamboo_cut {
namespace ui {

// === 全局样式声明 ===
#ifdef ENABLE_LVGL
extern lv_style_t style_card;
extern lv_style_t style_text_title;
extern lv_style_t style_btn_primary;
extern lv_style_t style_btn_success;
extern lv_style_t style_btn_warning;
extern lv_style_t style_btn_danger;
extern lv_style_t style_btn_pressed;
#else
extern char style_card[64];
extern char style_text_title[64];
extern char style_btn_primary[64];
extern char style_btn_success[64];
extern char style_btn_warning[64];
extern char style_btn_danger[64];
extern char style_btn_pressed[64];
#endif

/**
 * @brief LVGL配置参数
 */
struct LVGLConfig {
    int screen_width;           // 屏幕宽度
    int screen_height;          // 屏幕高度
    int refresh_rate;           // 刷新率(Hz)
    std::string touch_device;   // 触摸设备路径
    std::string display_device; // 显示设备路径
    bool enable_touch;          // 启用触摸
    
    LVGLConfig()
        : screen_width(1280)
        , screen_height(800) 
        , refresh_rate(60)
        , touch_device("/dev/input/event0")
        , display_device("/dev/fb0")
        , enable_touch(true) {}
};

/**
 * @brief 工作流程步骤定义
 */
enum class WorkflowStep {
    FEED_DETECTION = 1,     // 进料检测
    VISION_RECOGNITION = 2, // 视觉识别
    COORDINATE_TRANSFER = 3,// 坐标传输
    CUT_PREPARE = 4,        // 切割准备
    EXECUTE_CUT = 5         // 执行切割
};

/**
 * @brief LVGL界面管理器
 * 负责整个工业界面的创建、更新和事件处理
 */
class LVGLInterface {
public:
    explicit LVGLInterface(std::shared_ptr<core::DataBridge> data_bridge);
    ~LVGLInterface();

    /**
     * @brief 初始化LVGL界面系统
     */
    bool initialize(const LVGLConfig& config);

    /**
     * @brief 启动界面线程
     */
    bool start();

    /**
     * @brief 停止界面线程
     */
    void stop();

    /**
     * @brief 检查界面是否在运行
     */
    bool isRunning() const { return running_.load(); }

    /**
     * @brief 设置全屏模式
     */
    void setFullscreen(bool fullscreen);

private:
    /**
     * @brief 界面线程主循环
     */
    void uiLoop();

    /**
     * @brief 创建主界面
     */
    void createMainInterface();

    /**
     * @brief 创建头部面板
     */
    lv_obj_t* createHeaderPanel();

    /**
     * @brief 创建摄像头显示区域
     */
    lv_obj_t* createCameraPanel(lv_obj_t* parent = nullptr);

    /**
     * @brief 创建控制面板
     */
    lv_obj_t* createControlPanel(lv_obj_t* parent = nullptr);

    /**
     * @brief 创建状态面板
     */
    lv_obj_t* createStatusPanel();

    /**
     * @brief 创建底部控制栏
     */
    lv_obj_t* createFooterPanel();

    // === 新增拆分的面板创建函数 ===
    
    /**
     * @brief 创建Jetson监控区域
     */
    void createJetsonMonitoringSection(lv_obj_t* parent);

    /**
     * @brief 创建AI模型监控区域
     */
    void createAIModelSection(lv_obj_t* parent);

    /**
     * @brief 创建Modbus通信区域
     */
    void createModbusSection(lv_obj_t* parent);

    /**
     * @brief 创建版本信息区域
     */
    void createVersionSection(lv_obj_t* parent);

    // === 数据更新逻辑函数 ===

    /**
     * @brief 更新界面数据
     */
    void updateInterface();

    /**
     * @brief 更新摄像头画面
     */
    void updateCameraView();
    
    /**
     * @brief 更新Canvas显示摄像头帧数据
     */
    void updateCanvasFrame(const cv::Mat& frame);

    /**
     * @brief 更新系统状态信息
     */
    void updateSystemStats();

    /**
     * @brief 更新AI模型监控数据
     */
    void updateAIModelStats();

    /**
     * @brief 更新竹子检测状态数据
     */
    void updateBambooDetectionStats();

    /**
     * @brief 更新摄像头状态数据
     */
    void updateCameraStats();

    /**
     * @brief 更新Modbus寄存器显示
     */
    void updateModbusDisplay();

    /**
     * @brief 更新工作流程状态
     */
    void updateWorkflowStatus();

    /**
     * @brief 绘制检测结果标注
     */
    void drawDetectionResults(const core::DetectionResult& result);

    // 事件处理器
    static void onStartButtonClicked(lv_event_t* e);
    static void onStopButtonClicked(lv_event_t* e);
    static void onPauseButtonClicked(lv_event_t* e);
    static void onEmergencyButtonClicked(lv_event_t* e);
    static void onBladeSelectionChanged(lv_event_t* e);
    static void onSettingsButtonClicked(lv_event_t* e);
    
    // 双摄切换按钮事件处理器
    static void onSingleCameraButtonClicked(lv_event_t* e);
    static void onSplitScreenButtonClicked(lv_event_t* e);
    static void onStereoVisionButtonClicked(lv_event_t* e);

    /**
     * @brief 显示消息对话框
     */
    void showMessageDialog(const std::string& title, const std::string& message);

    /**
     * @brief 初始化LVGL样式主题
     */
    void initializeTheme();

    /**
     * @brief 初始化显示驱动
     */
    bool initializeDisplay();

    /**
     * @brief 初始化输入设备
     */
    bool initializeInput();

    /**
     * @brief 自动检测DRM显示器分辨率
     */
    bool detectDisplayResolution(int& width, int& height);

    // === DRM显示驱动相关函数 ===

    /**
     * @brief 设置LVGL显示缓冲区
     */
    bool setupLVGLDisplayBuffer(uint32_t buf_size);

    /**
     * @brief 创建LVGL显示驱动
     */
    bool createLVGLDisplay(uint32_t buf_size);

    // === 辅助更新函数 ===
    
    /**
     * @brief 更新Modbus寄存器状态
     */
    void updateModbusRegisters(const core::ModbusRegisters& modbus_registers);

    /**
     * @brief 更新竹子检测状态数据（带参数版本）
     */
    void updateBambooDetectionStats(const core::BambooDetection& bamboo_detection);

    /**
     * @brief 更新温度监控数据
     */
    void updateTemperatureStats(const utils::SystemStats& stats);

    /**
     * @brief 更新电源监控数据
     */
    void updatePowerStats(const utils::SystemStats& stats);

    /**
     * @brief 更新扩展系统统计数据
     */
    void updateSystemExtendedStats(const utils::SystemStats& stats);

    /**
     * @brief 更新单个摄像头状态
     */
    void updateSingleCameraStats(int camera_id, const core::CameraStatus& camera_info,
                                lv_obj_t* status_label, lv_obj_t* fps_label, lv_obj_t* resolution_label,
                                lv_obj_t* exposure_label, lv_obj_t* lighting_label);

    /**
     * @brief 更新模拟统计数据
     */
    void updateSimulatedStats();

    /**
     * @brief 更新指标标签
     */
    void updateMetricLabels();

    /**
     * @brief 查找可用的DRM设备
     */
    int findDRMDevice();

    /**
     * @brief 打开DRM设备并配置显示
     */
    bool openDRMDevice(int fd);

    /**
     * @brief 配置DRM连接器
     */
    bool configureDRMConnector(int fd);

    /**
     * @brief 创建DRM帧缓冲区
     */
    bool createDRMFramebuffer(int fd);

    /**
     * @brief 设置DRM显示模式
     */
    bool setDRMMode(int fd);

    /**
     * @brief 复制像素数据到DRM缓冲区
     */
    void copyPixelsToDRM(const lv_area_t* area, const uint8_t* px_map);

private:
    std::shared_ptr<core::DataBridge> data_bridge_;
    std::shared_ptr<utils::JetsonMonitor> jetson_monitor_;
    LVGLConfig config_;
    
    std::thread ui_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    
    // LVGL对象引用
    lv_obj_t* main_screen_;
    lv_obj_t* header_panel_;
    lv_obj_t* camera_panel_;
    lv_obj_t* camera_canvas_;
    lv_obj_t* control_panel_;
    lv_obj_t* status_panel_;
    lv_obj_t* footer_panel_;
    
    // 界面组件
    struct {
        lv_obj_t* system_title;
        lv_obj_t* heartbeat_label;
        lv_obj_t* response_label;
        std::vector<lv_obj_t*> workflow_buttons;
    } header_widgets_;
    
    struct {
        lv_obj_t* info_label;
        lv_obj_t* coord_value;
        lv_obj_t* quality_value;
        lv_obj_t* blade_value;
        
        // 双摄切换控件
        lv_obj_t* mode_label;       // 模式标签
        lv_obj_t* single_btn;       // 单摄按钮
        lv_obj_t* split_btn;        // 并排按钮
        lv_obj_t* stereo_btn;       // 立体按钮
        lv_obj_t* status_value;     // 状态显示
        
        // Canvas缓冲区
#ifdef ENABLE_LVGL
        lv_color_t* canvas_buffer;
#else
        void* canvas_buffer;
#endif
    } camera_widgets_;
    
    // Canvas常量定义
    static const int CANVAS_WIDTH = 640;
    static const int CANVAS_HEIGHT = 360;
    
    struct {
        // Jetson系统监控组件
        lv_obj_t* cpu_bar;
        lv_obj_t* cpu_label;
        lv_obj_t* gpu_bar;
        lv_obj_t* gpu_label;
        lv_obj_t* mem_bar;
        lv_obj_t* mem_label;
        
        // 温度监控组件
        lv_obj_t* cpu_temp_label;
        lv_obj_t* gpu_temp_label;
        lv_obj_t* soc_temp_label;
        lv_obj_t* thermal_temp_label;
        
        // 电源监控组件
        lv_obj_t* power_in_label;
        lv_obj_t* power_cpu_gpu_label;
        lv_obj_t* power_soc_label;
        
        // 其他系统信息组件
        lv_obj_t* emc_freq_label;
        lv_obj_t* vic_usage_label;
        lv_obj_t* fan_speed_label;
        lv_obj_t* swap_usage_label;
        
        // AI模型监控组件
        lv_obj_t* ai_model_version_label;      // 模型版本
        lv_obj_t* ai_inference_time_label;     // 推理时间
        lv_obj_t* ai_confidence_threshold_label; // 置信阈值
        lv_obj_t* ai_detection_accuracy_label;  // 检测精度
        lv_obj_t* ai_total_detections_label;    // 总检测数
        lv_obj_t* ai_daily_detections_label;    // 今日检测数
        
        // 当前竹子检测状态组件
        lv_obj_t* bamboo_diameter_label;       // 竹子直径
        lv_obj_t* bamboo_length_label;         // 竹子长度
        lv_obj_t* bamboo_cut_positions_label;  // 预切位置
        lv_obj_t* bamboo_confidence_label;     // 检测置信度
        lv_obj_t* bamboo_detection_time_label; // 检测耗时
        
        // 摄像头状态组件
        lv_obj_t* camera1_status_label;        // 摄像头-1在线状态
        lv_obj_t* camera1_fps_label;           // 摄像头-1帧率
        lv_obj_t* camera1_resolution_label;    // 摄像头-1分辨率
        lv_obj_t* camera1_exposure_label;      // 摄像头-1曝光模式
        lv_obj_t* camera1_lighting_label;      // 摄像头-1光照评分
        lv_obj_t* camera2_status_label;        // 摄像头-2在线状态
        lv_obj_t* camera2_fps_label;           // 摄像头-2帧率
        lv_obj_t* camera2_resolution_label;    // 摄像头-2分辨率
        lv_obj_t* camera2_exposure_label;      // 摄像头-2曝光模式
        lv_obj_t* camera2_lighting_label;      // 摄像头-2光照评分
        
        // Modbus通信统计组件
        lv_obj_t* modbus_connection_label;      // 连接状态
        lv_obj_t* modbus_address_label;         // 连接地址
        lv_obj_t* modbus_latency_label;         // 通讯延迟
        lv_obj_t* modbus_last_success_label;    // 最后通讯时间
        lv_obj_t* modbus_error_count_label;     // 错误计数
        lv_obj_t* modbus_message_count_label;   // 消息计数
        lv_obj_t* modbus_packets_label;
        lv_obj_t* modbus_errors_label;
        lv_obj_t* modbus_heartbeat_label;
        
        // Modbus寄存器状态组件
        lv_obj_t* modbus_system_status_label;      // 40001: 系统状态
        lv_obj_t* modbus_plc_command_label;        // 40002: PLC命令
        lv_obj_t* modbus_coord_ready_label;        // 40003: 坐标就绪
        lv_obj_t* modbus_x_coordinate_label;       // 40004-05: X坐标
        lv_obj_t* modbus_cut_quality_label;        // 40006: 切割质量
        lv_obj_t* modbus_blade_number_label;       // 40009: 刀片编号
        lv_obj_t* modbus_health_status_label;      // 40010: 健康状态
        
        // 原有组件
        lv_obj_t* modbus_table;
        std::vector<lv_obj_t*> blade_buttons;
        
        // 版本信息控件（移到这里）
        lv_obj_t* system_version_label;     // 系统版本
        lv_obj_t* lvgl_version_label;       // LVGL版本
        lv_obj_t* build_time_label;         // 编译时间
        lv_obj_t* git_commit_label;         // Git提交
        lv_obj_t* jetpack_version_label;    // JetPack版本
        lv_obj_t* cuda_version_label;       // CUDA版本
    } control_widgets_;
    
    struct {
        // 12项系统指标标签
        std::vector<lv_obj_t*> metric_labels;
        
        // 版本信息标签
        lv_obj_t* jetpack_version;
        lv_obj_t* cuda_version;
        lv_obj_t* tensorrt_version;
        lv_obj_t* opencv_version;
        lv_obj_t* ubuntu_version;
        
        // 版本信息控件（系统版本部分）
        lv_obj_t* system_version_label;     // 系统版本
        lv_obj_t* lvgl_version_label;       // LVGL版本
        lv_obj_t* build_time_label;         // 编译时间
        lv_obj_t* git_commit_label;         // Git提交
        lv_obj_t* jetpack_version_label;    // JetPack版本
        lv_obj_t* cuda_version_label;       // CUDA版本
    } status_widgets_;
    
    struct {
        lv_obj_t* start_btn;
        lv_obj_t* pause_btn;
        lv_obj_t* stop_btn;
        lv_obj_t* emergency_btn;
        lv_obj_t* power_btn;
        lv_obj_t* process_label;
        lv_obj_t* stats_label;
    } footer_widgets_;
    
    // LVGL驱动 (v9 API)
    lv_display_t* display_;
    lv_indev_t* input_device_;
    
    // 显示缓冲区
#ifdef ENABLE_LVGL
    static lv_color_t* disp_buf1_;
    static lv_color_t* disp_buf2_;
    static lv_draw_buf_t draw_buf_;
#else
    static void* disp_buf1_;
    static void* disp_buf2_;
    static char draw_buf_[64]; // 占位符缓冲区
#endif

    // === DRM设备相关成员变量 ===
    static int drm_fd_;                    // DRM设备文件描述符
    static uint32_t drm_crtc_id_;         // CRTC ID
    static uint32_t drm_connector_id_;    // 连接器 ID
    static uint32_t drm_fb_id_;           // 帧缓冲区 ID
    static void* drm_map_;                // 内存映射指针
    static uint32_t drm_handle_;          // DRM对象句柄
    static uint32_t drm_stride_;          // 行字节数
    static uint32_t drm_size_;            // 缓冲区大小
    
    // 性能监控
    std::chrono::high_resolution_clock::time_point last_update_time_;
    int frame_count_;
    float ui_fps_;
    
    // 当前状态
    WorkflowStep current_step_;
    bool system_running_;
    bool emergency_stop_;
    int selected_blade_;
    
    // 颜色主题
    lv_color_t color_primary_;
    lv_color_t color_secondary_;
    lv_color_t color_success_;
    lv_color_t color_warning_;
    lv_color_t color_error_;
    lv_color_t color_background_;
    lv_color_t color_surface_;

    // === LVGL样式对象 ===
#ifdef ENABLE_LVGL
    static lv_style_t style_card;
    static lv_style_t style_text_title;
#else
    static char style_card[64];     // 占位符
    static char style_text_title[64]; // 占位符
#endif
    
    // Grid布局已完全移除，使用安全的Flex布局替代
};

/**
 * @brief LVGL显示刷新回调 (v9 API)
 */
void display_flush_cb(lv_display_t* disp, const lv_area_t* area, uint8_t* px_map);

/**
 * @brief LVGL输入设备读取回调 (v9 API)
 */
void input_read_cb(lv_indev_t* indev, lv_indev_data_t* data);

// === DRM辅助函数声明 ===

/**
 * @brief 初始化DRM设备
 */
bool initializeDRMDevice(int& drm_fd, uint32_t& fb_id, drmModeCrtc*& crtc,
                        drmModeConnector*& connector, uint32_t*& framebuffer,
                        uint32_t& fb_handle, int& init_attempt_count,
                        uint32_t& drm_width, uint32_t& drm_height,
                        uint32_t& stride, uint32_t& buffer_size);

/**
 * @brief 复制像素数据
 */
void copyPixelData(const lv_area_t* area, const uint8_t* px_map, uint32_t* framebuffer,
                  uint32_t drm_width, uint32_t drm_height, uint32_t stride, uint32_t buffer_size);

/**
 * @brief 设置DRM显示
 */
bool setupDRMDisplay(int drm_fd, uint32_t& fb_id, drmModeCrtc*& crtc,
                    drmModeConnector*& connector, uint32_t*& framebuffer,
                    uint32_t& fb_handle, uint32_t& drm_width, uint32_t& drm_height,
                    uint32_t& stride, uint32_t& buffer_size);

/**
 * @brief 清理DRM资源
 */
void cleanupDRMResources(int drm_fd, uint32_t fb_id, drmModeCrtc* crtc,
                        drmModeConnector* connector, uint32_t* framebuffer,
                        uint32_t fb_handle, uint32_t buffer_size);

/**
 * @brief 查找合适的CRTC
 */
bool findSuitableCRTC(int drm_fd, drmModeRes* resources, drmModeConnector* connector,
                     drmModeCrtc*& crtc);

/**
 * @brief 创建帧缓冲区
 */
bool createFramebuffer(int drm_fd, uint32_t drm_width, uint32_t drm_height,
                      uint32_t& fb_id, uint32_t& fb_handle, uint32_t*& framebuffer,
                      uint32_t& stride, uint32_t& buffer_size);

/**
 * @brief 设置CRTC模式
 */
bool setCRTCMode(int drm_fd, drmModeCrtc* crtc, uint32_t fb_id,
                drmModeConnector* connector, drmModeModeInfo* mode);

} // namespace ui
} // namespace bamboo_cut