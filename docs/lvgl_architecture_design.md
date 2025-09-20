# LVGL架构设计文档

## 系统架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                 LVGL GUI 应用层                              │
├─────────────────────────────────────────────────────────────┤
│  主界面     │  视频显示   │  控制面板   │  状态栏   │  设置页面 │
│  MainApp    │  VideoView  │  ControlPanel│  StatusBar│  Settings│
├─────────────────────────────────────────────────────────────┤
│                 LVGL 核心框架                               │
├─────────────────────────────────────────────────────────────┤
│  显示驱动   │  输入驱动   │  绘制引擎   │  动画引擎  │  主题引擎 │
│  Display    │  Input      │  Render     │  Animation │  Theme   │
├─────────────────────────────────────────────────────────────┤
│                 C++ 业务逻辑层                              │
├─────────────────────────────────────────────────────────────┤
│  摄像头管理器│  AI检测引擎  │  视频处理器  │  事件管理器│  配置管理器│
│  CameraManager│ AIDetector  │ VideoProcessor│EventManager│ConfigManager│
├─────────────────────────────────────────────────────────────┤
│                 硬件抽象层                                  │
├─────────────────────────────────────────────────────────────┤
│  V4L2 Camera │  TensorRT   │  Framebuffer │  Touch Input│  GPIO/串口│
│  Driver      │  Engine     │  Driver      │  evdev      │  Control  │
├─────────────────────────────────────────────────────────────┤
│                 Linux 内核层                                │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件设计

### 1. 显示系统架构

#### 1.1 帧缓冲配置
```c
// 直接操作framebuffer，无需X11
#define DISPLAY_WIDTH   1920
#define DISPLAY_HEIGHT  1080
#define DISPLAY_BPP     32
#define FB_DEVICE      "/dev/fb0"

// 双缓冲配置
typedef struct {
    uint32_t *front_buffer;
    uint32_t *back_buffer;
    int width;
    int height;
    int stride;
    bool vsync_enabled;
} display_buffer_t;
```

#### 1.2 LVGL显示驱动
```c
// 显示刷新回调
void display_flush_cb(lv_disp_drv_t *disp_drv, const lv_area_t *area, lv_color_t *color_p) {
    // DMA硬件加速拷贝到framebuffer
    dma_copy_to_framebuffer(area, color_p);
    lv_disp_flush_ready(disp_drv);
}

// GPU硬件加速支持
void gpu_blend_cb(lv_disp_drv_t *disp_drv, lv_color_t *dest, const lv_color_t *src, 
                  uint32_t length, lv_opa_t opa) {
    // 使用Jetson GPU进行混合操作
    jetson_gpu_blend(dest, src, length, opa);
}
```

### 2. 输入系统架构

#### 2.1 触摸输入驱动
```c
// evdev触摸设备配置
#define TOUCH_DEVICE "/dev/input/event0"

typedef struct {
    int fd;
    int x;
    int y;
    bool pressed;
    bool gesture_enabled;
} touch_device_t;

// 触摸事件处理
void touch_read_cb(lv_indev_drv_t *indev_drv, lv_indev_data_t *data) {
    struct input_event ev;
    read(touch_fd, &ev, sizeof(ev));
    
    // 转换坐标系并传递给LVGL
    data->point.x = scale_touch_x(ev.value);
    data->point.y = scale_touch_y(ev.value);
    data->state = ev.type == EV_KEY ? LV_INDEV_STATE_PR : LV_INDEV_STATE_REL;
}
```

### 3. 摄像头管道架构

#### 3.1 V4L2摄像头驱动
```cpp
class V4L2CameraDriver {
public:
    struct CameraConfig {
        std::string device_path = "/dev/video0";
        int width = 1920;
        int height = 1080;
        int fps = 30;
        uint32_t pixel_format = V4L2_PIX_FMT_YUYV;
        int buffer_count = 4;
    };
    
    bool initialize(const CameraConfig& config);
    bool start_streaming();
    bool get_frame(cv::Mat& frame);
    void stop_streaming();
    
private:
    int fd_;
    std::vector<void*> buffers_;
    size_t buffer_size_;
};
```

#### 3.2 CUDA图像处理管道
```cpp
class CUDAImageProcessor {
public:
    // YUV到RGB转换 (GPU加速)
    void yuv_to_rgb_gpu(const uint8_t* yuv_data, uint8_t* rgb_data, 
                        int width, int height);
    
    // 图像缩放 (GPU加速)
    void resize_gpu(const cv::Mat& src, cv::Mat& dst, cv::Size size);
    
    // 图像预处理 (为AI推理准备)
    void preprocess_for_inference(const cv::Mat& input, float* output,
                                  int model_width, int model_height);
};
```

### 4. AI推理架构

#### 4.1 TensorRT引擎封装
```cpp
class TensorRTInference {
public:
    struct ModelConfig {
        std::string engine_path;
        int max_batch_size = 1;
        bool use_int8 = true;
        bool use_dla = true;  // 深度学习加速器
    };
    
    bool load_engine(const ModelConfig& config);
    std::vector<Detection> infer(const cv::Mat& image);
    
private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    
    void* gpu_input_buffer_;
    void* gpu_output_buffer_;
    cudaStream_t stream_;
};
```

#### 4.2 检测结果处理
```cpp
struct BambooDetection {
    cv::Rect bbox;
    float confidence;
    cv::Point2f cut_point;    // 切割点坐标
    float angle;              // 切割角度
    bool is_valid;
};

class DetectionProcessor {
public:
    std::vector<BambooDetection> process_detections(
        const std::vector<Detection>& raw_detections,
        const cv::Mat& original_image);
        
    // 运动检测逻辑
    bool detect_motion(const cv::Mat& current_frame, 
                      const cv::Mat& background);
};
```

### 5. LVGL GUI组件

#### 5.1 主界面布局
```c
// 主屏幕对象
lv_obj_t *main_screen;
lv_obj_t *video_container;
lv_obj_t *control_panel;
lv_obj_t *status_bar;

void create_main_ui() {
    main_screen = lv_obj_create(NULL);
    lv_scr_load(main_screen);
    
    // 视频显示区域 (70%宽度)
    video_container = lv_obj_create(main_screen);
    lv_obj_set_size(video_container, 
                   LV_HOR_RES * 0.7, LV_VER_RES);
    lv_obj_align(video_container, LV_ALIGN_LEFT_MID, 0, 0);
    
    // 控制面板 (30%宽度)
    control_panel = lv_obj_create(main_screen);
    lv_obj_set_size(control_panel, 
                   LV_HOR_RES * 0.3, LV_VER_RES);
    lv_obj_align(control_panel, LV_ALIGN_RIGHT_MID, 0, 0);
}
```

#### 5.2 视频显示组件
```c
lv_obj_t *video_canvas;
lv_obj_t *detection_overlay;

void create_video_display() {
    // 创建视频画布
    video_canvas = lv_canvas_create(video_container);
    lv_canvas_set_buffer(video_canvas, video_buffer, 
                        VIDEO_WIDTH, VIDEO_HEIGHT, LV_IMG_CF_TRUE_COLOR);
    
    // 检测结果叠加层
    detection_overlay = lv_obj_create(video_container);
    lv_obj_set_style_bg_opa(detection_overlay, LV_OPA_TRANSP, 0);
}

// 更新视频帧
void update_video_frame(const cv::Mat& frame) {
    // 转换OpenCV Mat到LVGL格式
    convert_mat_to_lvgl_buffer(frame, video_buffer);
    lv_canvas_invalidate(video_canvas);
}
```

#### 5.3 控制面板组件
```c
lv_obj_t *start_btn;
lv_obj_t *stop_btn;
lv_obj_t *emergency_btn;
lv_obj_t *settings_btn;

void create_control_panel() {
    // 启动按钮
    start_btn = lv_btn_create(control_panel);
    lv_obj_add_event_cb(start_btn, start_operation_cb, LV_EVENT_CLICKED, NULL);
    
    // 停止按钮
    stop_btn = lv_btn_create(control_panel);
    lv_obj_add_event_cb(stop_btn, stop_operation_cb, LV_EVENT_CLICKED, NULL);
    
    // 紧急停止按钮
    emergency_btn = lv_btn_create(control_panel);
    lv_obj_set_style_bg_color(emergency_btn, lv_color_hex(0xFF0000), 0);
    lv_obj_add_event_cb(emergency_btn, emergency_stop_cb, LV_EVENT_CLICKED, NULL);
}
```

### 6. 事件管理系统

#### 6.1 事件调度器
```cpp
class EventManager {
public:
    enum EventType {
        CAMERA_FRAME_READY,
        DETECTION_COMPLETE,
        TOUCH_INPUT,
        SYSTEM_ERROR,
        CONFIG_CHANGED
    };
    
    struct Event {
        EventType type;
        void* data;
        uint64_t timestamp;
    };
    
    void post_event(const Event& event);
    void process_events();
    void register_handler(EventType type, std::function<void(const Event&)> handler);
    
private:
    std::queue<Event> event_queue_;
    std::mutex queue_mutex_;
    std::map<EventType, std::vector<std::function<void(const Event&)>>> handlers_;
};
```

### 7. 配置管理

#### 7.1 配置结构
```cpp
struct SystemConfig {
    struct Camera {
        std::string device = "/dev/video0";
        int width = 1920;
        int height = 1080;
        int fps = 30;
    } camera;
    
    struct AI {
        std::string model_path = "/opt/bamboo/models/yolov8n.engine";
        float confidence_threshold = 0.7f;
        float nms_threshold = 0.4f;
        bool use_int8 = true;
    } ai;
    
    struct Display {
        std::string framebuffer = "/dev/fb0";
        bool vsync = true;
        int brightness = 80;
    } display;
    
    struct Touch {
        std::string device = "/dev/input/event0";
        bool calibration_enabled = true;
        int sensitivity = 10;
    } touch;
};
```

## 性能优化策略

### 1. 内存管理
- **零拷贝设计**: 摄像头数据直接传递给AI推理
- **内存池管理**: 预分配固定大小的内存块
- **GPU内存管理**: CUDA统一内存管理

### 2. 多线程架构
```cpp
// 主要线程分工
Thread 1: LVGL GUI渲染 (主线程)
Thread 2: 摄像头数据采集
Thread 3: AI推理处理
Thread 4: 检测结果后处理
Thread 5: 系统监控和日志
```

### 3. 硬件加速
- **DMA传输**: 减少CPU负载
- **GPU渲染**: LVGL GPU加速
- **TensorRT优化**: INT8量化 + DLA加速

## 文件结构设计

```
lvgl_frontend/
├── CMakeLists.txt
├── src/
│   ├── main.cpp                    # 主程序入口
│   ├── app/
│   │   ├── main_app.cpp           # 主应用逻辑
│   │   ├── event_manager.cpp      # 事件管理器
│   │   └── config_manager.cpp     # 配置管理器
│   ├── display/
│   │   ├── framebuffer_driver.cpp # 帧缓冲驱动
│   │   ├── lvgl_display.cpp       # LVGL显示驱动
│   │   └── gpu_accelerated.cpp    # GPU加速渲染
│   ├── input/
│   │   ├── touch_driver.cpp       # 触摸驱动
│   │   └── input_calibration.cpp  # 输入校准
│   ├── camera/
│   │   ├── v4l2_camera.cpp        # V4L2摄像头驱动
│   │   ├── cuda_processor.cpp     # CUDA图像处理
│   │   └── camera_manager.cpp     # 摄像头管理器
│   ├── ai/
│   │   ├── tensorrt_engine.cpp    # TensorRT推理引擎
│   │   ├── yolo_detector.cpp      # YOLO检测器
│   │   └── detection_processor.cpp # 检测结果处理
│   └── gui/
│       ├── video_view.cpp         # 视频显示组件
│       ├── control_panel.cpp      # 控制面板
│       ├── status_bar.cpp         # 状态栏
│       └── settings_page.cpp      # 设置页面
├── include/
│   └── [对应的头文件]
├── resources/
│   ├── fonts/                     # 字体文件
│   ├── images/                    # 图片资源
│   └── config/
│       └── default_config.json    # 默认配置
└── scripts/
    ├── build.sh                   # 构建脚本
    └── deploy.sh                  # 部署脚本
```

## 迁移优势

### 相比QT的优势：
1. **更轻量**: 内存占用减少60%+
2. **更简单**: 无需复杂的QML渲染引擎
3. **更直接**: 直接操作framebuffer，减少抽象层
4. **更稳定**: 减少依赖，提高系统稳定性
5. **更高效**: 专为嵌入式优化的渲染引擎

### 性能提升预期：
- **启动时间**: 从8秒降低到2秒
- **内存使用**: 从200MB降低到80MB
- **渲染延迟**: 从50ms降低到16ms
- **触摸响应**: 从100ms降低到30ms