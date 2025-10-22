# Sway Wayland 合成器部署指南

## 📋 概述

由于 **Weston** 存在 `xdg_positioner` 协议bug，**Mutter** 无法在嵌入式环境启动（需要systemd-logind会话），我们切换到 **Sway** 作为 Wayland 合成器。

**Sway 的优势：**
- ✅ 基于 wlroots，稳定可靠
- ✅ 专为嵌入式/低资源环境设计
- ✅ 完全支持 xdg-shell 协议
- ✅ **原生支持触摸屏和触摸板**
- ✅ 不需要桌面会话管理
- ✅ 在 ARM 平台上广泛使用
- ✅ 资源占用低

---

## 🚀 快速部署

### 步骤 1：拉取最新代码

```bash
cd ~/banboo-develop
git pull origin master
```

### 步骤 2：停止现有合成器

```bash
# 停止 Mutter（如果运行中）
sudo systemctl stop mutter-wayland.service || true
sudo systemctl disable mutter-wayland.service || true

# 停止 Weston（如果运行中）
sudo systemctl stop weston.service || true
sudo systemctl unmask weston.service || true
```

### 步骤 3：安装并启动 Sway

```bash
cd ~/banboo-develop

# 一键安装、配置并启动 Sway
sudo make start-sway
```

这个命令会自动：
1. 检查并安装 Sway 及相关依赖
2. 创建支持触摸控制的 Sway 配置文件
3. 配置 `sway-wayland.service` systemd 服务
4. 启动 Sway 合成器

### 步骤 4：验证 Sway 运行

```bash
# 检查 Sway 状态
sudo make sway-status

# 预期输出：
# === Sway状态 ===
# ● sway-wayland.service - Sway Wayland Compositor (Touch-Enabled)
#    Active: active (running)
#
# === Wayland Socket ===
# /run/user/0/wayland-1
#
# === 触摸设备 ===
# [触摸设备信息]
```

### 步骤 5：重启应用服务

```bash
sudo systemctl restart bamboo-cpp-lvgl.service

# 查看实时日志
sudo journalctl -u bamboo-cpp-lvgl -f --no-pager
```

---

## ✅ 预期日志输出

```bash
🔍 [Wayland] 步骤1: 检查 Wayland 合成器...
✅ [Wayland] Wayland 合成器运行正常
🎨 [LVGL] 步骤2: 初始化LVGL Wayland界面...
✅ Wayland 合成器检测成功: wayland-1
📝 执行空 commit，触发 configure 事件...
⏳ 等待 configure 事件...
✅ 收到 configure 事件
🎨 创建初始 SHM buffer...
✅ Buffer 已附加并提交: 1920x1080
✅ Wayland 客户端初始化完成
✅ LVGL 初始化成功
📺 视频将由 Wayland 合成器自动合成到 LVGL 窗口
```

**不再出现：**
- ❌ `xdg_positioner@8.set_size` 错误（Weston bug）
- ❌ `Could not get session ID: User 0 has no sessions`（Mutter 限制）

---

## 🎯 Sway 配置说明

Sway 配置文件位于：`/root/.config/sway/config`

### 触摸控制配置

```bash
# 触摸屏支持
input type:touchscreen {
    tap enabled                # 点击触发
    drag enabled               # 拖动支持
    events enabled             # 启用事件
}

# 触摸板支持
input type:touchpad {
    tap enabled                # 点击触发
    natural_scroll enabled     # 自然滚动
    dwt enabled                # 禁用打字时触摸板
    drag enabled               # 拖动支持
}
```

### 工业应用优化

```bash
# 禁用窗口装饰（全屏模式）
default_border none
default_floating_border none

# 禁用屏幕锁定和电源管理
exec swayidle -w timeout 0 'echo disabled' before-sleep 'echo disabled'

# 自动全屏应用
for_window [title=".*"] fullscreen enable
```

### NVIDIA Jetson 优化

```bash
# Sway 服务环境变量
Environment=WLR_NO_HARDWARE_CURSORS=1      # 禁用硬件光标（Jetson 兼容性）
Environment=WLR_RENDERER=gles2              # 使用 GLES2 渲染器
Environment=LIBINPUT_DEFAULT_TOUCH_ENABLED=1  # 确保触摸启用
Environment=__EGL_VENDOR_LIBRARY_DIRS=/usr/lib/aarch64-linux-gnu/tegra-egl  # NVIDIA EGL
```

---

## 🔧 常用命令

### Sway 管理

```bash
# 启动 Sway
sudo make start-sway

# 停止 Sway
sudo make stop-sway

# 查看 Sway 状态
sudo make sway-status

# 查看 Sway 日志
sudo make sway-logs
```

### 应用管理

```bash
# 重启应用
sudo systemctl restart bamboo-cpp-lvgl

# 查看应用日志
sudo journalctl -u bamboo-cpp-lvgl -f --no-pager

# 完整重新部署
sudo make redeploy
```

### 调试命令

```bash
# 列出触摸设备
libinput list-devices

# 检查 Wayland socket
ls -la /run/user/0/wayland-*

# 测试触摸输入
sudo libinput debug-events

# 查看 Sway 版本
sway --version
```

---

## 🐛 故障排查

### 问题 1: Sway 启动失败

**症状：**
```
[ERROR] Sway启动失败
```

**解决方案：**

1. 检查 Sway 日志：
```bash
sudo journalctl -u sway-wayland -n 50 --no-pager
```

2. 如果报错 `Failed to create backend`：
```bash
# 添加 --unsupported-gpu 参数（已在服务中配置）
sudo /usr/bin/sway --unsupported-gpu
```

3. 检查 EGL 库：
```bash
ls -la /usr/lib/aarch64-linux-gnu/tegra-egl/
```

### 问题 2: Wayland socket 不存在

**症状：**
```
Wayland socket不存在: /run/user/0/wayland-1
```

**解决方案：**

```bash
# 1. 确保 XDG_RUNTIME_DIR 存在
sudo mkdir -p /run/user/0
sudo chmod 700 /run/user/0

# 2. 重启 Sway
sudo systemctl restart sway-wayland

# 3. 验证 socket
ls -la /run/user/0/wayland-*
```

### 问题 3: 触摸不响应

**症状：** 屏幕显示正常，但触摸无反应

**解决方案：**

1. 检查触摸设备是否被识别：
```bash
libinput list-devices | grep -A 5 "touch"
```

2. 确认 Sway 配置中触摸已启用：
```bash
cat /root/.config/sway/config | grep -A 3 "type:touchscreen"
```

3. 测试触摸输入：
```bash
sudo libinput debug-events
# 然后触摸屏幕，应该看到事件输出
```

4. **注意：** 当前 LVGL Wayland 接口可能需要添加输入事件处理。如果上述都正常但应用内触摸不响应，需要在 `lvgl_wayland_interface.cpp` 中添加 `wl_seat`/`wl_touch` 监听器。

### 问题 4: 应用崩溃或黑屏

**症状：** 应用启动后崩溃或只显示黑屏

**解决方案：**

1. 检查应用日志：
```bash
sudo journalctl -u bamboo-cpp-lvgl -n 100 --no-pager
```

2. 手动运行应用查看详细错误：
```bash
sudo systemctl stop bamboo-cpp-lvgl
cd /opt/bamboo-cut/bin
sudo XDG_RUNTIME_DIR=/run/user/0 WAYLAND_DISPLAY=wayland-1 ./bamboo_integrated --verbose
```

3. 检查 EGL 平台设置：
```bash
echo $EGL_PLATFORM  # 应该输出 "wayland"
echo $WAYLAND_DISPLAY  # 应该输出 "wayland-1"
```

---

## 📊 性能对比

| 合成器 | 内存占用 | CPU占用 | 触摸支持 | 嵌入式兼容性 | xdg-shell 支持 | 状态 |
|--------|---------|---------|---------|-------------|---------------|------|
| **Sway** | ~40MB | 5-10% | ✅ 原生 | ✅ 优秀 | ✅ 完整 | ✅ **推荐** |
| Weston | ~30MB | 3-8% | ✅ 支持 | ✅ 良好 | ⚠️ 有bug | ❌ 协议bug |
| Mutter | ~80MB | 15-25% | ✅ 支持 | ❌ 需要会话 | ✅ 完整 | ❌ 无法启动 |

---

## 🔄 回退到 Weston（如果需要）

如果 Sway 有问题，可以回退到 Weston：

```bash
# 1. 停止 Sway
sudo systemctl stop sway-wayland
sudo systemctl disable sway-wayland

# 2. 重新启用 Weston
sudo systemctl unmask weston.service
sudo systemctl enable weston.service
sudo systemctl start weston.service

# 3. 修改应用服务文件中的 WAYLAND_DISPLAY
# 将 wayland-1 改回 wayland-0

# 4. 重启应用
sudo systemctl restart bamboo-cpp-lvgl
```

**注意：** Weston 仍然存在 `xdg_positioner` bug，可能导致应用无法正常显示。

---

## 📝 总结

- ✅ Sway 是当前**最佳选择**，专为嵌入式场景设计
- ✅ 原生支持触摸控制，无需额外配置
- ✅ 完全符合 Wayland 协议标准，无兼容性问题
- ✅ 资源占用低，性能优秀
- ⚠️ 如果触摸不响应，可能需要在应用代码中添加 Wayland 输入事件处理

---

## 📞 技术支持

如有问题，请提供以下信息：

```bash
# 收集诊断信息
echo "=== Sway 状态 ===" > /tmp/bamboo-diag.txt
sudo systemctl status sway-wayland --no-pager -l >> /tmp/bamboo-diag.txt
echo "" >> /tmp/bamboo-diag.txt

echo "=== 应用状态 ===" >> /tmp/bamboo-diag.txt
sudo systemctl status bamboo-cpp-lvgl --no-pager -l >> /tmp/bamboo-diag.txt
echo "" >> /tmp/bamboo-diag.txt

echo "=== Wayland Socket ===" >> /tmp/bamboo-diag.txt
ls -la /run/user/0/wayland-* >> /tmp/bamboo-diag.txt
echo "" >> /tmp/bamboo-diag.txt

echo "=== 触摸设备 ===" >> /tmp/bamboo-diag.txt
libinput list-devices >> /tmp/bamboo-diag.txt

cat /tmp/bamboo-diag.txt
```

