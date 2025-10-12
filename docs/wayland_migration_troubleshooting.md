# Bamboo Recognition System - Wayland迁移问题排查指南

**版本:** 1.0.0  
**日期:** 2024-12-12  
**作者:** Bamboo Development Team

---

## 📋 目录

1. [Weston启动问题](#weston启动问题)
2. [LVGL Wayland连接问题](#lvgl-wayland连接问题)
3. [waylandsink显示问题](#waylandsink显示问题)
4. [nvarguscamerasrc初始化问题](#nvarguscamerasrc初始化问题)
5. [触摸输入问题](#触摸输入问题)
6. [性能问题](#性能问题)
7. [内存泄漏问题](#内存泄漏问题)
8. [编译和构建问题](#编译和构建问题)

---

## 🔧 Weston启动问题

### 症状描述
- Weston启动失败
- 黑屏或无显示输出
- "Failed to create compositor" 错误
- DRM设备权限错误

### 可能原因
1. **DRM设备权限不足**
2. **显卡驱动未正确安装**
3. **Weston配置文件错误**
4. **用户权限不足**
5. **与其他显示服务冲突**

### 诊断命令
```bash
# 检查DRM设备权限
ls -la /dev/dri/

# 检查Weston状态
systemctl status weston

# 检查Weston日志
journalctl -u weston -f

# 手动启动Weston查看详细错误
weston --log=/tmp/weston.log

# 检查显卡驱动
lsmod | grep drm
nvidia-smi  # Jetson平台
```

### 解决方案

#### 1. 修复DRM设备权限
```bash
# 添加用户到video和render组
sudo usermod -a -G video,render $USER

# 重新登录或重启
sudo reboot

# 或者临时修改权限
sudo chmod 666 /dev/dri/*
```

#### 2. 配置Weston正确权限
```bash
# 编辑Weston服务文件
sudo systemctl edit weston

# 添加以下内容
[Service]
SupplementaryGroups=video render input
```

#### 3. 检查和修复Weston配置
```bash
# 检查配置文件语法
weston --help

# 使用最小配置测试
cat > /tmp/minimal_weston.ini << EOF
[core]
backend=drm-backend.so

[shell]
background-image=/usr/share/backgrounds/warty-final-ubuntu.png
background-type=scale-crop

[output]
name=HDMI-A-1
mode=1920x1200@60
EOF

# 使用最小配置启动
weston --config=/tmp/minimal_weston.ini
```

#### 4. Jetson特定修复
```bash
# 确保NVIDIA DRM启用
sudo modprobe nvidia-drm modeset=1

# 检查tegra-drm
sudo modprobe tegra-drm

# 添加到启动配置
echo "nvidia-drm" | sudo tee -a /etc/modules
echo "options nvidia-drm modeset=1" | sudo tee /etc/modprobe.d/nvidia-drm.conf
```

### 验证方法
```bash
# 检查Weston进程
pgrep weston

# 检查Wayland socket
ls -la /run/user/$(id -u)/wayland-*

# 测试连接
wayland-info
```

---

## 🖼️ LVGL Wayland连接问题

### 症状描述
- LVGL应用无法连接到Wayland
- "Failed to connect to Wayland display" 错误
- 应用启动但无窗口显示
- lv_drivers Wayland后端初始化失败

### 可能原因
1. **WAYLAND_DISPLAY环境变量未设置**
2. **lv_drivers配置错误**
3. **lv_drv_conf.h设置不正确**
4. **Wayland权限问题**
5. **库版本不兼容**

### 诊断命令
```bash
# 检查Wayland环境变量
echo $WAYLAND_DISPLAY
echo $XDG_RUNTIME_DIR

# 检查Wayland socket
ls -la $XDG_RUNTIME_DIR/wayland-*

# 测试基本Wayland连接
wayland-info

# 检查LVGL和lv_drivers编译
ldd build/bamboo_integrated | grep -i wayland
```

### 解决方案

#### 1. 设置正确的环境变量
```bash
# 设置Wayland显示
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/run/user/$(id -u)

# 添加到启动脚本
echo "export WAYLAND_DISPLAY=wayland-0" >> ~/.bashrc
echo "export XDG_RUNTIME_DIR=/run/user/\$(id -u)" >> ~/.bashrc
```

#### 2. 检查lv_drv_conf.h配置
```c
// 确保这些设置正确
#define USE_WAYLAND       1
#define WAYLAND_HOR_RES   1920
#define WAYLAND_VER_RES   1200
#define USE_WAYLAND_POINTER  1
#define USE_WAYLAND_KEYBOARD 1
#define USE_WAYLAND_TOUCH    1
```

#### 3. 验证lv_drivers编译
```bash
# 重新编译lv_drivers
cd third_party/lv_drivers
make clean
make

# 检查编译输出
find . -name "*.o" | grep wayland
```

#### 4. 权限修复
```bash
# 确保socket可访问
chmod 755 $XDG_RUNTIME_DIR
chmod 666 $XDG_RUNTIME_DIR/wayland-*
```

### 验证方法
```bash
# 编译并运行简单测试
cat > /tmp/lvgl_wayland_test.c << 'EOF'
#include "lvgl/lvgl.h"
#include "lv_drivers/wayland/wayland.h"

int main() {
    lv_init();
    wayland_init();
    printf("LVGL Wayland initialization successful\n");
    return 0;
}
EOF

gcc -o /tmp/lvgl_wayland_test /tmp/lvgl_wayland_test.c -llvgl
/tmp/lvgl_wayland_test
```

---

## 📺 waylandsink显示问题

### 症状描述
- GStreamer waylandsink找不到display
- 视频窗口显示在错误位置
- "No Wayland display found" 错误
- 视频帧率低或卡顿

### 可能原因
1. **WAYLAND_DISPLAY未正确设置**
2. **waylandsink插件未安装**
3. **GStreamer权限问题**
4. **视频窗口配置错误**
5. **硬件解码问题**

### 诊断命令
```bash
# 检查waylandsink插件
gst-inspect-1.0 waylandsink

# 测试简单管道
gst-launch-1.0 videotestsrc ! waylandsink

# 检查GStreamer插件
gst-inspect-1.0 | grep wayland

# 检查视频设备
ls /dev/video*

# Jetson特定检查
gst-inspect-1.0 nvarguscamerasrc
```

### 解决方案

#### 1. 安装和配置waylandsink
```bash
# 安装GStreamer Wayland插件
sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-plugins-good

# Jetson特定安装
sudo apt install gstreamer1.0-plugins-tegra
```

#### 2. 修复环境变量
```bash
# DeepStream启动前设置
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/run/user/$(id -u)

# 在代码中设置
setenv("WAYLAND_DISPLAY", "wayland-0", 1);
setenv("XDG_RUNTIME_DIR", g_get_user_runtime_dir(), 1);
```

#### 3. 配置视频窗口参数
```cpp
// 在DeepStreamManager中正确配置
gst_structure_set(props,
    "display", G_TYPE_STRING, "wayland-0",
    "fullscreen", G_TYPE_BOOLEAN, FALSE,
    "x", G_TYPE_INT, config.window_x,
    "y", G_TYPE_INT, config.window_y,
    "width", G_TYPE_INT, config.window_width,
    "height", G_TYPE_INT, config.window_height,
    "sync", G_TYPE_BOOLEAN, FALSE,
    NULL);
```

#### 4. Jetson优化管道
```bash
# 使用硬件加速管道
gst-launch-1.0 nvarguscamerasrc ! \
    'video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1' ! \
    nvvidconv ! \
    'video/x-raw, format=BGRx' ! \
    nvvidconv ! \
    waylandsink sync=false
```

### 验证方法
```bash
# 测试基本waylandsink
timeout 10s gst-launch-1.0 videotestsrc pattern=ball ! \
    video/x-raw,width=640,height=480 ! waylandsink

# 测试摄像头管道
timeout 10s gst-launch-1.0 nvarguscamerasrc num-buffers=300 ! \
    'video/x-raw(memory:NVMM)' ! nvvidconv ! waylandsink
```

---

## 📷 nvarguscamerasrc初始化问题

### 症状描述
- "Failed to create camera source" 错误
- EGL初始化失败
- 摄像头设备未找到
- Permission denied访问/dev/video*

### 可能原因
1. **摄像头权限不足**
2. **Argus服务未运行**
3. **EGL环境未正确配置**
4. **摄像头硬件问题**
5. **驱动版本不兼容**

### 诊断命令
```bash
# 检查摄像头设备
ls -la /dev/video*

# 检查Argus服务
systemctl status nvargus-daemon

# 测试摄像头
gst-launch-1.0 nvarguscamerasrc num-buffers=10 ! fakesink

# 检查V4L2设备
v4l2-ctl --list-devices

# 检查用户组
groups $USER
```

### 解决方案

#### 1. 修复摄像头权限
```bash
# 添加用户到video组
sudo usermod -a -G video $USER

# 重启或重新登录
sudo reboot

# 或临时修复权限
sudo chmod 666 /dev/video*
```

#### 2. 启动Argus服务
```bash
# 启动nvargus daemon
sudo systemctl start nvargus-daemon
sudo systemctl enable nvargus-daemon

# 检查状态
sudo systemctl status nvargus-daemon
```

#### 3. 配置EGL环境
```bash
# 设置EGL环境变量
export EGL_PLATFORM=wayland
export WAYLAND_DISPLAY=wayland-0

# 在应用启动前设置
setenv("EGL_PLATFORM", "wayland", 1);
```

#### 4. 硬件检查
```bash
# 检查摄像头连接
dmesg | grep -i camera

# 检查I2C总线
sudo i2cdetect -y 9  # 通常是9或10

# 检查设备树
cat /proc/device-tree/model
```

### 验证方法
```bash
# 简单摄像头测试
nvgstcapture-1.0 --mode=1 --automate --capture-auto

# GStreamer测试
gst-launch-1.0 nvarguscamerasrc ! \
    'video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1' ! \
    nvvidconv ! fakesink
```

---

## 👆 触摸输入问题

### 症状描述
- 触摸无响应
- 触摸坐标偏移
- 多点触控不工作
- libinput错误

### 可能原因
1. **输入设备权限问题**
2. **libinput配置错误**
3. **触摸屏校准问题**
4. **事件设备映射错误**

### 诊断命令
```bash
# 检查输入设备
ls -la /dev/input/

# 检查触摸设备
cat /proc/bus/input/devices | grep -A 5 Touch

# 测试触摸事件
sudo evtest /dev/input/event0

# 检查libinput
libinput list-devices
```

### 解决方案

#### 1. 修复输入设备权限
```bash
# 添加用户到input组
sudo usermod -a -G input $USER

# 创建udev规则
sudo tee /etc/udev/rules.d/99-input.rules << EOF
SUBSYSTEM=="input", GROUP="input", MODE="0664"
KERNEL=="event*", GROUP="input", MODE="0664"
EOF

# 重新加载udev规则
sudo udevadm control --reload-rules
sudo udevadm trigger
```

#### 2. 配置libinput
```bash
# 创建libinput配置
sudo mkdir -p /etc/X11/xorg.conf.d/
sudo tee /etc/X11/xorg.conf.d/40-libinput.conf << EOF
Section "InputClass"
    Identifier "libinput touchscreen catchall"
    MatchIsTouchscreen "on"
    MatchDevicePath "/dev/input/event*"
    Driver "libinput"
EndSection
EOF
```

#### 3. 触摸屏校准
```bash
# 安装校准工具
sudo apt install xinput-calibrator

# 校准触摸屏
xinput_calibrator

# 或使用libinput校准
libinput measure touchpad-size /dev/input/event0
```

### 验证方法
```bash
# 测试触摸事件
sudo evtest /dev/input/event0
# 触摸屏幕应该看到事件输出

# 检查LVGL触摸配置
# 在lv_drv_conf.h中确保:
# #define USE_WAYLAND_TOUCH 1
```

---

## ⚡ 性能问题

### 症状描述
- 帧率低于预期
- UI响应迟缓
- CPU/GPU使用率过高
- 内存使用持续增长

### 可能原因
1. **硬件加速未启用**
2. **缓冲配置不当**
3. **算法效率问题**
4. **内存碎片化**

### 诊断命令
```bash
# 系统性能监控
top -p $(pgrep bamboo_integrated)
htop

# GPU监控 (Jetson)
tegrastats
nvidia-smi

# 内存分析
valgrind --tool=massif ./bamboo_integrated

# 性能基准测试
./scripts/performance_benchmark.sh
```

### 解决方案

#### 1. 启用硬件加速
```cpp
// 在LVGL配置中启用GPU加速
#define LV_USE_GPU_STM32_DMA2D 1
#define LV_USE_GPU_NXP_PXP 1
#define LV_USE_GPU_NXP_VG_LITE 1

// Wayland特定优化
#define WAYLAND_USE_DMABUF 1
#define WAYLAND_BUFFER_COUNT 3
```

#### 2. 优化缓冲配置
```cpp
// 在DeepStreamManager中配置
g_object_set(waylandsink,
    "sync", FALSE,
    "max-lateness", 1000000,  // 1ms
    "qos", TRUE,
    NULL);
```

#### 3. CPU调度优化
```bash
# 设置高优先级
sudo chrt -f -p 80 $(pgrep bamboo_integrated)

# CPU亲和性设置
sudo taskset -cp 0-3 $(pgrep bamboo_integrated)
```

### 验证方法
```bash
# 运行性能测试
./scripts/performance_benchmark.sh

# 检查帧率
gst-launch-1.0 videotestsrc ! fpsdisplaysink video-sink=waylandsink
```

---

## 🧠 内存泄漏问题

### 症状描述
- 内存使用持续增长
- 长时间运行后崩溃
- OOM killer杀死进程

### 诊断命令
```bash
# 内存泄漏检测
valgrind --leak-check=full ./bamboo_integrated

# 内存使用监控
watch -n 1 "ps aux | grep bamboo_integrated | grep -v grep"

# 详细内存分析
valgrind --tool=memcheck --track-origins=yes ./bamboo_integrated
```

### 解决方案

#### 1. 修复常见泄漏
```cpp
// 确保正确释放LVGL对象
lv_obj_del(obj);

// 释放GStreamer资源
gst_object_unref(pipeline);
gst_deinit();

// 释放OpenCV内存
frame.release();
```

#### 2. 实现内存监控
```cpp
class MemoryMonitor {
private:
    std::chrono::steady_clock::time_point last_check_;
    
public:
    void checkMemoryUsage() {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_check_).count() > 60) {
            // 每分钟检查一次内存使用
            struct rusage usage;
            getrusage(RUSAGE_SELF, &usage);
            LOG_INFO("Memory usage: %ld KB", usage.ru_maxrss);
            last_check_ = now;
        }
    }
};
```

### 验证方法
```bash
# 长时间运行测试
timeout 1800s ./bamboo_integrated &  # 30分钟
watch -n 10 "ps aux | grep bamboo_integrated | grep -v grep"
```

---

## 🔨 编译和构建问题

### 症状描述
- CMake配置失败
- 链接错误
- 找不到Wayland库
- 版本冲突

### 诊断命令
```bash
# 检查依赖库
pkg-config --list-all | grep wayland
pkg-config --cflags --libs wayland-client

# 检查编译环境
gcc --version
cmake --version

# 详细编译输出
make VERBOSE=1
```

### 解决方案

#### 1. 安装缺失依赖
```bash
# 运行环境准备脚本
./scripts/install_wayland_deps.sh

# 手动安装
sudo apt update
sudo apt install libwayland-dev libwayland-egl1-mesa-dev wayland-protocols
```

#### 2. 清理和重新构建
```bash
# 完全清理
rm -rf build/
rm -rf third_party/lv_drivers/

# 重新设置lv_drivers
./scripts/setup_lv_drivers.sh

# 重新构建
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 验证方法
```bash
# 测试编译
./scripts/test_wayland_migration.sh
```

---

## 📞 获取帮助

如果上述解决方案无法解决您的问题，请：

1. **运行完整诊断**
   ```bash
   ./scripts/test_wayland_migration.sh > diagnosis.log 2>&1
   ```

2. **收集系统信息**
   ```bash
   ./scripts/performance_benchmark.sh
   ```

3. **提供以下信息**
   - 系统版本 (`lsb_release -a`)
   - 硬件信息 (`lscpu`, `nvidia-smi`)
   - 错误日志
   - 重现步骤

4. **联系开发团队**
   - 邮箱: support@bamboo-recognition.com
   - GitHub Issues: [项目仓库]
   - 技术支持热线: [电话号码]

---

## 📋 快速检查清单

在联系支持前，请确认已完成以下检查：

- [ ] Weston正在运行 (`pgrep weston`)
- [ ] WAYLAND_DISPLAY已设置 (`echo $WAYLAND_DISPLAY`)
- [ ] 用户在必要组中 (`groups $USER`)
- [ ] 权限正确设置 (`ls -la /dev/dri/`)
- [ ] 依赖库已安装 (`pkg-config --exists wayland-client`)
- [ ] 编译成功完成 (`test -f build/bamboo_integrated`)
- [ ] 配置文件正确 (`test -f lv_drv_conf.h`)

**记住：Wayland架构比DRM直接访问更稳定，但需要正确的配置和权限设置。**