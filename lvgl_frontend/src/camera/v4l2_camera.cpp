#include "camera/v4l2_camera.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <errno.h>
#include <time.h>

// 获取当前时间戳（毫秒）
static uint64_t get_timestamp_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// YUYV转RGB转换函数
static void yuyv_to_rgb(const uint8_t* yuyv, uint8_t* rgb, int width, int height) {
    for (int i = 0; i < width * height / 2; i++) {
        int y1 = yuyv[i * 4 + 0];
        int u  = yuyv[i * 4 + 1];
        int y2 = yuyv[i * 4 + 2];
        int v  = yuyv[i * 4 + 3];
        
        // 第一个像素
        int c1 = y1 - 16;
        int d1 = u - 128;
        int e1 = v - 128;
        
        int r1 = (298 * c1 + 409 * e1 + 128) >> 8;
        int g1 = (298 * c1 - 100 * d1 - 208 * e1 + 128) >> 8;
        int b1 = (298 * c1 + 516 * d1 + 128) >> 8;
        
        rgb[i * 6 + 0] = r1 < 0 ? 0 : (r1 > 255 ? 255 : r1);
        rgb[i * 6 + 1] = g1 < 0 ? 0 : (g1 > 255 ? 255 : g1);
        rgb[i * 6 + 2] = b1 < 0 ? 0 : (b1 > 255 ? 255 : b1);
        
        // 第二个像素
        int c2 = y2 - 16;
        int r2 = (298 * c2 + 409 * e1 + 128) >> 8;
        int g2 = (298 * c2 - 100 * d1 - 208 * e1 + 128) >> 8;
        int b2 = (298 * c2 + 516 * d1 + 128) >> 8;
        
        rgb[i * 6 + 3] = r2 < 0 ? 0 : (r2 > 255 ? 255 : r2);
        rgb[i * 6 + 4] = g2 < 0 ? 0 : (g2 > 255 ? 255 : g2);
        rgb[i * 6 + 5] = b2 < 0 ? 0 : (b2 > 255 ? 255 : b2);
    }
}

v4l2_camera_t* v4l2_camera_create(const char* device_path) {
    if (!device_path) {
        printf("错误：设备路径为空\n");
        return nullptr;
    }
    
    v4l2_camera_t* camera = (v4l2_camera_t*)calloc(1, sizeof(v4l2_camera_t));
    if (!camera) {
        printf("错误：分配摄像头内存失败\n");
        return nullptr;
    }
    
    // 初始化基本参数
    strncpy(camera->device_path, device_path, sizeof(camera->device_path) - 1);
    camera->fd = -1;
    camera->state = V4L2_STATE_CLOSED;
    camera->buffers = nullptr;
    camera->buffer_count = 0;
    
    // 初始化默认控制参数
    camera->brightness = 128;
    camera->contrast = 128;
    camera->saturation = 128;
    camera->hue = 0;
    camera->exposure = -1; // 自动曝光
    camera->gain = -1;     // 自动增益
    camera->auto_exposure = true;
    camera->auto_white_balance = true;
    
    // 初始化统计信息
    camera->frame_count = 0;
    camera->dropped_frames = 0;
    camera->fps = 0.0;
    camera->last_frame_time = 0;
    
    // 初始化线程同步
    pthread_mutex_init(&camera->mutex, nullptr);
    camera->thread_running = false;
    
    printf("创建V4L2摄像头: %s\n", device_path);
    return camera;
}

void v4l2_camera_destroy(v4l2_camera_t* camera) {
    if (!camera) return;
    
    printf("销毁V4L2摄像头\n");
    
    // 停止流式传输
    if (camera->state == V4L2_STATE_STREAMING) {
        v4l2_camera_stop_streaming(camera);
    }
    
    // 释放缓冲区
    v4l2_camera_free_buffers(camera);
    
    // 关闭设备
    v4l2_camera_close(camera);
    
    // 销毁互斥锁
    pthread_mutex_destroy(&camera->mutex);
    
    free(camera);
}

bool v4l2_camera_open(v4l2_camera_t* camera) {
    if (!camera) return false;
    
    pthread_mutex_lock(&camera->mutex);
    
    if (camera->state != V4L2_STATE_CLOSED) {
        printf("警告：摄像头已经打开\n");
        pthread_mutex_unlock(&camera->mutex);
        return true;
    }
    
    // 打开设备文件
    camera->fd = open(camera->device_path, O_RDWR | O_NONBLOCK);
    if (camera->fd < 0) {
        printf("错误：无法打开设备 %s: %s\n", camera->device_path, strerror(errno));
        pthread_mutex_unlock(&camera->mutex);
        return false;
    }
    
    // 查询设备能力
    if (ioctl(camera->fd, VIDIOC_QUERYCAP, &camera->cap) < 0) {
        printf("错误：查询设备能力失败: %s\n", strerror(errno));
        close(camera->fd);
        camera->fd = -1;
        pthread_mutex_unlock(&camera->mutex);
        return false;
    }
    
    // 检查设备能力
    if (!(camera->cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        printf("错误：设备不支持视频捕获\n");
        close(camera->fd);
        camera->fd = -1;
        pthread_mutex_unlock(&camera->mutex);
        return false;
    }
    
    if (!(camera->cap.capabilities & V4L2_CAP_STREAMING)) {
        printf("错误：设备不支持流式传输\n");
        close(camera->fd);
        camera->fd = -1;
        pthread_mutex_unlock(&camera->mutex);
        return false;
    }
    
    camera->state = V4L2_STATE_OPENED;
    printf("成功打开摄像头设备: %s\n", (char*)camera->cap.card);
    
    pthread_mutex_unlock(&camera->mutex);
    return true;
}

void v4l2_camera_close(v4l2_camera_t* camera) {
    if (!camera) return;
    
    pthread_mutex_lock(&camera->mutex);
    
    if (camera->fd >= 0) {
        close(camera->fd);
        camera->fd = -1;
    }
    
    camera->state = V4L2_STATE_CLOSED;
    printf("关闭摄像头设备\n");
    
    pthread_mutex_unlock(&camera->mutex);
}

bool v4l2_camera_set_format(v4l2_camera_t* camera, int width, int height, uint32_t pixelformat) {
    if (!camera || camera->fd < 0) return false;
    
    pthread_mutex_lock(&camera->mutex);
    
    memset(&camera->format, 0, sizeof(camera->format));
    camera->format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    camera->format.fmt.pix.width = width;
    camera->format.fmt.pix.height = height;
    camera->format.fmt.pix.pixelformat = pixelformat;
    camera->format.fmt.pix.field = V4L2_FIELD_INTERLACED;
    
    if (ioctl(camera->fd, VIDIOC_S_FMT, &camera->format) < 0) {
        printf("错误：设置格式失败: %s\n", strerror(errno));
        pthread_mutex_unlock(&camera->mutex);
        return false;
    }
    
    camera->state = V4L2_STATE_CONFIGURED;
    printf("设置格式: %dx%d, 格式: %s\n", 
           camera->format.fmt.pix.width, 
           camera->format.fmt.pix.height,
           v4l2_pixelformat_to_string(camera->format.fmt.pix.pixelformat));
    
    pthread_mutex_unlock(&camera->mutex);
    return true;
}

bool v4l2_camera_set_fps(v4l2_camera_t* camera, int fps) {
    if (!camera || camera->fd < 0) return false;
    
    pthread_mutex_lock(&camera->mutex);
    
    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = fps;
    
    if (ioctl(camera->fd, VIDIOC_S_PARM, &parm) < 0) {
        printf("警告：设置帧率失败: %s\n", strerror(errno));
        pthread_mutex_unlock(&camera->mutex);
        return false;
    }
    
    printf("设置帧率: %d fps\n", fps);
    
    pthread_mutex_unlock(&camera->mutex);
    return true;
}

bool v4l2_camera_allocate_buffers(v4l2_camera_t* camera, int buffer_count) {
    if (!camera || camera->fd < 0) return false;
    
    pthread_mutex_lock(&camera->mutex);
    
    // 请求缓冲区
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = buffer_count;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    
    if (ioctl(camera->fd, VIDIOC_REQBUFS, &req) < 0) {
        printf("错误：请求缓冲区失败: %s\n", strerror(errno));
        pthread_mutex_unlock(&camera->mutex);
        return false;
    }
    
    if (req.count < 2) {
        printf("错误：缓冲区数量不足\n");
        pthread_mutex_unlock(&camera->mutex);
        return false;
    }
    
    // 分配缓冲区数组
    camera->buffers = (v4l2_buffer_t*)calloc(req.count, sizeof(v4l2_buffer_t));
    if (!camera->buffers) {
        printf("错误：分配缓冲区数组失败\n");
        pthread_mutex_unlock(&camera->mutex);
        return false;
    }
    
    camera->buffer_count = req.count;
    
    // 映射每个缓冲区
    for (int i = 0; i < camera->buffer_count; i++) {
        memset(&camera->buffers[i].buffer, 0, sizeof(camera->buffers[i].buffer));
        camera->buffers[i].buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        camera->buffers[i].buffer.memory = V4L2_MEMORY_MMAP;
        camera->buffers[i].buffer.index = i;
        
        if (ioctl(camera->fd, VIDIOC_QUERYBUF, &camera->buffers[i].buffer) < 0) {
            printf("错误：查询缓冲区失败: %s\n", strerror(errno));
            v4l2_camera_free_buffers(camera);
            pthread_mutex_unlock(&camera->mutex);
            return false;
        }
        
        camera->buffers[i].length = camera->buffers[i].buffer.length;
        camera->buffers[i].start = mmap(nullptr, camera->buffers[i].buffer.length,
                                       PROT_READ | PROT_WRITE, MAP_SHARED,
                                       camera->fd, camera->buffers[i].buffer.m.offset);
        
        if (camera->buffers[i].start == MAP_FAILED) {
            printf("错误：映射缓冲区失败: %s\n", strerror(errno));
            v4l2_camera_free_buffers(camera);
            pthread_mutex_unlock(&camera->mutex);
            return false;
        }
    }
    
    printf("成功分配 %d 个缓冲区\n", camera->buffer_count);
    
    pthread_mutex_unlock(&camera->mutex);
    return true;
}

void v4l2_camera_free_buffers(v4l2_camera_t* camera) {
    if (!camera || !camera->buffers) return;
    
    pthread_mutex_lock(&camera->mutex);
    
    // 取消映射所有缓冲区
    for (int i = 0; i < camera->buffer_count; i++) {
        if (camera->buffers[i].start != MAP_FAILED) {
            munmap(camera->buffers[i].start, camera->buffers[i].length);
        }
    }
    
    free(camera->buffers);
    camera->buffers = nullptr;
    camera->buffer_count = 0;
    
    printf("释放缓冲区\n");
    
    pthread_mutex_unlock(&camera->mutex);
}

bool v4l2_camera_start_streaming(v4l2_camera_t* camera) {
    if (!camera || camera->fd < 0 || !camera->buffers) return false;
    
    pthread_mutex_lock(&camera->mutex);
    
    if (camera->state == V4L2_STATE_STREAMING) {
        printf("警告：摄像头已经在流式传输\n");
        pthread_mutex_unlock(&camera->mutex);
        return true;
    }
    
    // 将所有缓冲区加入队列
    for (int i = 0; i < camera->buffer_count; i++) {
        if (ioctl(camera->fd, VIDIOC_QBUF, &camera->buffers[i].buffer) < 0) {
            printf("错误：缓冲区入队失败: %s\n", strerror(errno));
            pthread_mutex_unlock(&camera->mutex);
            return false;
        }
    }
    
    // 开始流式传输
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(camera->fd, VIDIOC_STREAMON, &type) < 0) {
        printf("错误：开始流式传输失败: %s\n", strerror(errno));
        pthread_mutex_unlock(&camera->mutex);
        return false;
    }
    
    camera->state = V4L2_STATE_STREAMING;
    camera->frame_count = 0;
    camera->dropped_frames = 0;
    camera->last_frame_time = get_timestamp_ms();
    
    printf("开始流式传输\n");
    
    pthread_mutex_unlock(&camera->mutex);
    return true;
}

bool v4l2_camera_stop_streaming(v4l2_camera_t* camera) {
    if (!camera || camera->fd < 0) return false;
    
    pthread_mutex_lock(&camera->mutex);
    
    if (camera->state != V4L2_STATE_STREAMING) {
        pthread_mutex_unlock(&camera->mutex);
        return true;
    }
    
    // 停止流式传输
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(camera->fd, VIDIOC_STREAMOFF, &type) < 0) {
        printf("错误：停止流式传输失败: %s\n", strerror(errno));
        pthread_mutex_unlock(&camera->mutex);
        return false;
    }
    
    camera->state = V4L2_STATE_CONFIGURED;
    printf("停止流式传输\n");
    
    pthread_mutex_unlock(&camera->mutex);
    return true;
}

bool v4l2_camera_get_frame(v4l2_camera_t* camera, cv::Mat& frame, int timeout_ms) {
    if (!camera || camera->fd < 0 || camera->state != V4L2_STATE_STREAMING) {
        return false;
    }
    
    // 等待帧数据
    if (timeout_ms > 0) {
        fd_set fds;
        struct timeval tv;
        
        FD_ZERO(&fds);
        FD_SET(camera->fd, &fds);
        
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        
        int ret = select(camera->fd + 1, &fds, nullptr, nullptr, &tv);
        if (ret <= 0) {
            if (ret == 0) {
                printf("警告：等待帧超时\n");
            } else {
                printf("错误：select失败: %s\n", strerror(errno));
            }
            return false;
        }
    }
    
    pthread_mutex_lock(&camera->mutex);
    
    // 出队一个缓冲区
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    
    if (ioctl(camera->fd, VIDIOC_DQBUF, &buf) < 0) {
        if (errno != EAGAIN) {
            printf("错误：缓冲区出队失败: %s\n", strerror(errno));
        }
        pthread_mutex_unlock(&camera->mutex);
        return false;
    }
    
    // 更新统计信息
    camera->frame_count++;
    uint64_t current_time = get_timestamp_ms();
    if (camera->last_frame_time > 0) {
        uint64_t frame_interval = current_time - camera->last_frame_time;
        if (frame_interval > 0) {
            camera->fps = 1000.0 / frame_interval;
        }
    }
    camera->last_frame_time = current_time;
    
    // 转换图像格式
    if (camera->format.fmt.pix.pixelformat == V4L2_PIX_FMT_YUYV) {
        // YUYV转RGB
        int width = camera->format.fmt.pix.width;
        int height = camera->format.fmt.pix.height;
        
        frame = cv::Mat(height, width, CV_8UC3);
        yuyv_to_rgb((uint8_t*)camera->buffers[buf.index].start, 
                   frame.data, width, height);
    } else {
        // 其他格式直接复制（可能需要进一步处理）
        frame = cv::Mat(camera->format.fmt.pix.height, 
                       camera->format.fmt.pix.width, 
                       CV_8UC3, 
                       camera->buffers[buf.index].start).clone();
    }
    
    // 重新入队缓冲区
    if (ioctl(camera->fd, VIDIOC_QBUF, &buf) < 0) {
        printf("错误：缓冲区重新入队失败: %s\n", strerror(errno));
        pthread_mutex_unlock(&camera->mutex);
        return false;
    }
    
    pthread_mutex_unlock(&camera->mutex);
    return true;
}

bool v4l2_camera_set_control(v4l2_camera_t* camera, uint32_t control_id, int value) {
    if (!camera || camera->fd < 0) return false;
    
    struct v4l2_control ctrl;
    ctrl.id = control_id;
    ctrl.value = value;
    
    if (ioctl(camera->fd, VIDIOC_S_CTRL, &ctrl) < 0) {
        printf("警告：设置控制参数失败: %s\n", strerror(errno));
        return false;
    }
    
    return true;
}

bool v4l2_camera_get_control(v4l2_camera_t* camera, uint32_t control_id, int* value) {
    if (!camera || camera->fd < 0 || !value) return false;
    
    struct v4l2_control ctrl;
    ctrl.id = control_id;
    
    if (ioctl(camera->fd, VIDIOC_G_CTRL, &ctrl) < 0) {
        printf("警告：获取控制参数失败: %s\n", strerror(errno));
        return false;
    }
    
    *value = ctrl.value;
    return true;
}

bool v4l2_camera_set_brightness(v4l2_camera_t* camera, int brightness) {
    if (v4l2_camera_set_control(camera, V4L2_CID_BRIGHTNESS, brightness)) {
        camera->brightness = brightness;
        return true;
    }
    return false;
}

bool v4l2_camera_set_contrast(v4l2_camera_t* camera, int contrast) {
    if (v4l2_camera_set_control(camera, V4L2_CID_CONTRAST, contrast)) {
        camera->contrast = contrast;
        return true;
    }
    return false;
}

bool v4l2_camera_set_exposure(v4l2_camera_t* camera, int exposure) {
    if (exposure < 0) {
        // 启用自动曝光
        if (v4l2_camera_set_control(camera, V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_AUTO)) {
            camera->auto_exposure = true;
            camera->exposure = -1;
            return true;
        }
    } else {
        // 手动曝光
        if (v4l2_camera_set_control(camera, V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_MANUAL) &&
            v4l2_camera_set_control(camera, V4L2_CID_EXPOSURE_ABSOLUTE, exposure)) {
            camera->auto_exposure = false;
            camera->exposure = exposure;
            return true;
        }
    }
    return false;
}

bool v4l2_camera_set_gain(v4l2_camera_t* camera, int gain) {
    if (gain < 0) {
        // 启用自动增益
        if (v4l2_camera_set_control(camera, V4L2_CID_AUTOGAIN, 1)) {
            camera->gain = -1;
            return true;
        }
    } else {
        // 手动增益
        if (v4l2_camera_set_control(camera, V4L2_CID_AUTOGAIN, 0) &&
            v4l2_camera_set_control(camera, V4L2_CID_GAIN, gain)) {
            camera->gain = gain;
            return true;
        }
    }
    return false;
}

bool v4l2_camera_set_auto_exposure(v4l2_camera_t* camera, bool enable) {
    return v4l2_camera_set_exposure(camera, enable ? -1 : camera->exposure);
}

bool v4l2_camera_set_auto_white_balance(v4l2_camera_t* camera, bool enable) {
    if (v4l2_camera_set_control(camera, V4L2_CID_AUTO_WHITE_BALANCE, enable ? 1 : 0)) {
        camera->auto_white_balance = enable;
        return true;
    }
    return false;
}

bool v4l2_camera_is_streaming(v4l2_camera_t* camera) {
    return camera && camera->state == V4L2_STATE_STREAMING;
}

double v4l2_camera_get_current_fps(v4l2_camera_t* camera) {
    return camera ? camera->fps : 0.0;
}

int v4l2_enumerate_devices(char devices[][64], int max_devices) {
    int device_count = 0;
    
    for (int i = 0; i < 10 && device_count < max_devices; i++) {
        char device_path[64];
        snprintf(device_path, sizeof(device_path), "/dev/video%d", i);
        
        int fd = open(device_path, O_RDWR);
        if (fd >= 0) {
            struct v4l2_capability cap;
            if (ioctl(fd, VIDIOC_QUERYCAP, &cap) >= 0 &&
                (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
                strcpy(devices[device_count], device_path);
                device_count++;
            }
            close(fd);
        }
    }
    
    return device_count;
}

bool v4l2_check_format_support(const char* device_path, int width, int height, uint32_t pixelformat) {
    int fd = open(device_path, O_RDWR);
    if (fd < 0) return false;
    
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = pixelformat;
    
    bool supported = (ioctl(fd, VIDIOC_TRY_FMT, &fmt) >= 0);
    close(fd);
    
    return supported;
}

const char* v4l2_pixelformat_to_string(uint32_t pixelformat) {
    static char str[5];
    str[0] = (pixelformat >> 0) & 0xff;
    str[1] = (pixelformat >> 8) & 0xff;
    str[2] = (pixelformat >> 16) & 0xff;
    str[3] = (pixelformat >> 24) & 0xff;
    str[4] = '\0';
    return str;
}

void v4l2_camera_print_info(v4l2_camera_t* camera) {
    if (!camera) return;
    
    printf("摄像头信息:\n");
    printf("  设备路径: %s\n", camera->device_path);
    printf("  驱动: %s\n", camera->cap.driver);
    printf("  设备名: %s\n", camera->cap.card);
    printf("  状态: %d\n", camera->state);
    printf("  格式: %dx%d, %s\n", 
           camera->format.fmt.pix.width,
           camera->format.fmt.pix.height,
           v4l2_pixelformat_to_string(camera->format.fmt.pix.pixelformat));
    printf("  缓冲区数量: %d\n", camera->buffer_count);
    printf("  帧计数: %lu\n", camera->frame_count);
    printf("  当前帧率: %.2f fps\n", camera->fps);
}