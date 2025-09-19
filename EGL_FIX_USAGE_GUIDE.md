# EGL 显示初始化问题修复指南

## 问题描述

根据日志分析，智能切竹机在Jetson设备上运行时遇到以下EGL显示初始化失败问题：

```
Could not initialize egl display
qt.qpa.eglfs.kms: Event reader thread: entering event loop
Could not initialize egl display
```

## 根本原因分析

1. **NVIDIA库路径未正确配置** - 缺少 `LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:$LD_LIBRARY_PATH`
2. **X11桌面服务冲突** - 需要禁用X11桌面服务，切换到控制台模式
3. **DRM设备权限问题** - GPU设备权限不正确
4. **KMS配置不完整** - HDMI显示器配置需要优化

## 修复解决方案

### 自动修复（推荐）

我们已经创建了一个综合修复脚本 `fix_egl_display_issues.sh`，它将自动处理所有问题。

#### 使用步骤：

1. **添加执行权限**
   ```bash
   chmod +x fix_egl_display_issues.sh
   ```

2. **以root权限运行修复脚本**
   ```bash
   sudo ./fix_egl_display_issues.sh
   ```

3. **检查修复结果**
   ```bash
   # 查看服务状态
   sudo systemctl status bamboo-cut-jetpack
   
   # 查看实时日志
   sudo journalctl -u bamboo-cut-jetpack -f
   ```

### 修复脚本功能说明

该脚本将自动执行以下修复操作：

#### 1. 配置NVIDIA Tegra库路径
- 检测系统中的Tegra库目录
- 创建环境配置脚本设置正确的 `LD_LIBRARY_PATH`
- 配置EGL和OpenGL相关环境变量

#### 2. 禁用X11桌面服务
- 停止和禁用所有X11相关服务（gdm3, lightdm等）
- 设置系统默认启动到控制台模式
- 避免X11与EGLFS的冲突

#### 3. 优化KMS显示配置
- 更新KMS配置文件，添加HDMI显示器支持
- 设置正确的分辨率和显示参数
- 启用硬件光标和缓冲区配置

#### 4. 修复DRM设备权限
- 设置正确的GPU设备权限（/dev/dri/card0等）
- 创建udev规则确保权限持久化
- 重新加载udev规则

#### 5. 创建增强版启动脚本
- 包含完整的EGL环境设置
- 添加详细的设备检测和诊断
- 健壮的错误处理和重试机制

#### 6. 更新systemd服务配置
- 设置正确的环境变量
- 使用root权限运行（避免权限问题）
- 配置重启策略和错误处理

## 手动修复方法（备用）

如果自动修复脚本无法运行，可以手动执行以下步骤：

### 1. 配置NVIDIA库路径

```bash
# 创建环境配置文件
sudo tee /opt/bamboo-cut/nvidia_tegra_env.sh > /dev/null << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:${LD_LIBRARY_PATH}"
export EGL_PLATFORM=drm
export GBM_BACKEND=nvidia-drm
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export LIBGL_ALWAYS_SOFTWARE=0
export MESA_LOADER_DRIVER_OVERRIDE=""
unset MESA_GL_VERSION_OVERRIDE
unset MESA_GLSL_VERSION_OVERRIDE
EOF

sudo chmod +x /opt/bamboo-cut/nvidia_tegra_env.sh
```

### 2. 禁用X11桌面服务

```bash
# 禁用显示管理器
sudo systemctl disable gdm3 lightdm xdm sddm display-manager 2>/dev/null || true

# 设置默认启动到控制台模式
sudo systemctl set-default multi-user.target
```

### 3. 修复DRM设备权限

```bash
# 设置设备权限
sudo chmod 666 /dev/dri/card* /dev/dri/renderD* 2>/dev/null || true

# 创建udev规则
sudo tee /etc/udev/rules.d/99-drm-permissions.rules > /dev/null << 'EOF'
SUBSYSTEM=="drm", KERNEL=="card*", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="renderD*", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 4. 更新KMS配置

```bash
sudo tee /opt/bamboo-cut/config/kms.conf > /dev/null << 'EOF'
{
  "device": "/dev/dri/card0",
  "hwcursor": false,
  "pbuffers": true,
  "separateScreens": false,
  "outputs": [
    {
      "name": "HDMI1",
      "mode": "1920x1080",
      "physicalSizeMM": [510, 287],
      "off": false,
      "primary": true
    }
  ]
}
EOF
```

### 5. 更新systemd服务

```bash
sudo tee /etc/systemd/system/bamboo-cut-jetpack.service > /dev/null << 'EOF'
[Unit]
Description=智能切竹机系统 (JetPack SDK) - EGL修复版
After=network.target
StartLimitIntervalSec=300

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bamboo-cut
ExecStart=/opt/bamboo-cut/start_bamboo_cut_jetpack_fixed.sh
Restart=on-failure
RestartSec=30
StartLimitBurst=3
Environment=QT_QPA_PLATFORM=eglfs
Environment=QT_QPA_EGLFS_INTEGRATION=eglfs_kms
Environment=QT_QPA_EGLFS_KMS_CONFIG=/opt/bamboo-cut/config/kms.conf
Environment=LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl
Environment=EGL_PLATFORM=drm
Environment=GBM_BACKEND=nvidia-drm

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable bamboo-cut-jetpack
sudo systemctl restart bamboo-cut-jetpack
```

## 验证修复结果

### 1. 检查服务状态
```bash
sudo systemctl status bamboo-cut-jetpack
```

### 2. 查看实时日志
```bash
sudo journalctl -u bamboo-cut-jetpack -f
```

### 3. 检查关键环境
```bash
# 检查DRM设备
ls -la /dev/dri/

# 检查Tegra库
ls -la /usr/lib/aarch64-linux-gnu/tegra*/

# 检查系统启动目标
systemctl get-default
```

## 预期结果

修复成功后，应该看到：

1. **EGL初始化成功** - 不再出现 "Could not initialize egl display" 错误
2. **Qt前端正常启动** - EGLFS平台成功初始化
3. **HDMI显示正常** - 1920x1080分辨率输出
4. **服务稳定运行** - systemd服务持续运行不重启

## 故障排除

如果修复后仍有问题：

1. **检查Jetson设备型号**
   ```bash
   cat /proc/device-tree/model
   ```

2. **检查JetPack版本**
   ```bash
   cat /etc/nv_tegra_release
   ```

3. **检查NVIDIA驱动**
   ```bash
   nvidia-smi  # 如果可用
   ```

4. **检查内核模块**
   ```bash
   lsmod | grep nvidia
   ```

5. **查看详细日志**
   ```bash
   journalctl -u bamboo-cut-jetpack --no-pager
   ```

## 注意事项

1. **备份重要配置** - 在运行修复脚本前，建议备份重要配置文件
2. **重启要求** - 某些更改可能需要重启系统才能完全生效
3. **权限要求** - 所有修复操作都需要root权限
4. **设备兼容性** - 此修复方案专为Jetson设备设计

## 支持

如果遇到问题，请提供：
- Jetson设备型号和JetPack版本
- 完整的错误日志
- 系统配置信息