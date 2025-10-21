# 切换到 Mutter 合成器

## 背景

由于 Weston 13.0 存在 `xdg_positioner` 协议处理 bug，导致 LVGL Wayland 客户端无法正常初始化。我们决定切换到更成熟稳定的 **Mutter** 合成器（GNOME Shell 的核心组件）。

## 变更内容

### 1. Makefile 更新

- ✅ 将 `install-wayland-deps` 从安装 Weston 改为安装 Mutter + GNOME Session
- ✅ 新增 `check-mutter` 目标：自动检查 Mutter 是否已安装
- ✅ 新增 `setup-mutter` 目标：配置 Mutter systemd 服务  
- ✅ 新增 `start-mutter` 目标：启动 Mutter Wayland 合成器
- ✅ 新增 `stop-mutter` 目标：停止 Mutter
- ✅ 新增 `mutter-status` 目标：查看 Mutter 状态
- ✅ 新增 `mutter-logs` 目标：查看 Mutter 日志
- ✅ 更新 `auto-setup-environment`：自动检测并启动 Mutter
- ✅ 保留 `start-weston` 等别名以保持向后兼容

### 2. 依赖包变更

**移除：**
- `weston`

**新增：**
- `mutter` - GNOME Wayland 合成器
- `gnome-session` - GNOME 会话管理器
- `dbus-x11` - D-Bus X11 协议支持

**保留：**
- `libwayland-dev`
- `libwayland-egl1`
- `wayland-protocols`
- `libxkbcommon-dev`

### 3. Systemd 服务配置

**服务名：** `mutter-wayland.service`

**关键配置：**
```ini
[Service]
Type=simple
User=root
ExecStartPre=/bin/sh -c 'mkdir -p /run/user/0 && chmod 700 /run/user/0'
ExecStart=/usr/bin/mutter --wayland --no-x11 --display-server
Environment=XDG_RUNTIME_DIR=/run/user/0
Environment=WAYLAND_DISPLAY=wayland-0
Environment=EGL_PLATFORM=wayland
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/0/bus
Restart=always
RestartSec=3
```

## 部署步骤

### 方案 A：全新部署（推荐）

```bash
cd ~/banboo-develop

# 1. 拉取最新代码
git pull origin master

# 2. 卸载 Weston（手动操作）
sudo systemctl stop weston
sudo systemctl disable weston
sudo apt remove weston

# 3. 安装 Mutter 并部署系统
sudo make redeploy

# 4. 查看状态
sudo make mutter-status
sudo journalctl -u bamboo-cpp-lvgl -f
```

### 方案 B：增量更新

```bash
cd ~/banboo-develop

# 1. 拉取最新代码
git pull

# 2. 停止当前服务
sudo systemctl stop weston
sudo systemctl stop bamboo-cpp-lvgl

# 3. 安装 Mutter
sudo make install-wayland-deps
sudo make start-mutter

# 4. 重新编译并启动
sudo make redeploy
```

## 验证步骤

### 1. 检查 Mutter 状态

```bash
sudo make mutter-status
```

**预期输出：**
```
=== Mutter状态 ===
● mutter-wayland.service - Mutter Wayland Compositor
     Loaded: loaded (/etc/systemd/system/mutter-wayland.service; enabled)
     Active: active (running) since ...

=== Wayland Socket ===
srwxr-xr-x 1 root root 0 Oct 21 XX:XX /run/user/0/wayland-0
```

### 2. 检查 Wayland 环境

```bash
sudo make check-wayland
```

**预期输出：**
```
[INFO] 检查Wayland环境（Mutter）...
Mutter服务状态: active
Wayland socket: 存在
Wayland库: 已安装
EGL库: 已安装
```

### 3. 检查应用日志

```bash
sudo journalctl -u bamboo-cpp-lvgl -f --no-pager
```

**预期日志：**
```
✅ XDG Toplevel 监听器已添加
✅ XDG Toplevel 创建成功，已设置全屏
📝 执行空 commit，触发 configure 事件...
⏳ 等待 configure 事件...
📐 收到XDG surface配置, serial=xxx
✅ 已确认xdg surface配置
✅ 收到 configure 事件
🎨 创建初始 SHM buffer...
✅ Buffer 已附加并提交: 1280x800
✅ Wayland 客户端初始化完成
```

**不应再出现的错误：**
- ❌ `invalid arguments for xdg_positioner@8.set_size`

## 优势对比

### Weston 的问题
- ❌ 协议实现存在 bug（xdg_positioner 误判）
- ❌ 社区支持较少
- ❌ 主要用于嵌入式场景，桌面环境支持不完善

### Mutter 的优势  
- ✅ GNOME 官方合成器，生产级成熟稳定
- ✅ 广泛用于 Ubuntu、Fedora 等主流发行版
- ✅ 协议实现完整，严格遵循 Wayland 规范
- ✅ 活跃的社区支持和持续维护
- ✅ 更好的硬件加速支持

## 回滚方案

如果需要回滚到 Weston：

```bash
# 1. 停止 Mutter
sudo systemctl stop mutter-wayland
sudo systemctl disable mutter-wayland

# 2. 重新安装 Weston
sudo apt install weston

# 3. 使用旧版本 Makefile
git checkout <old_commit> Makefile

# 4. 配置并启动 Weston
sudo make setup-wayland
sudo make start-weston
sudo make redeploy
```

## Makefile 命令参考

### 新增命令

| 命令 | 说明 |
|------|------|
| `make check-mutter` | 检查 Mutter 是否已安装 |
| `make setup-mutter` | 配置 Mutter systemd 服务 |
| `make start-mutter` | 启动 Mutter 合成器 |
| `make stop-mutter` | 停止 Mutter |
| `make mutter-status` | 查看 Mutter 状态 |
| `make mutter-logs` | 查看 Mutter 实时日志 |

### 兼容性别名

为保持向后兼容，以下命令仍然有效（内部调用 Mutter）：

- `make start-weston` → `make start-mutter`
- `make stop-weston` → `make stop-mutter`
- `make weston-status` → `make mutter-status`

## 故障排查

### 问题：Mutter 无法启动

```bash
# 查看详细日志
sudo journalctl -u mutter-wayland -n 100 --no-pager

# 检查 D-Bus
ps aux | grep dbus

# 手动启动 Mutter 查看错误
sudo XDG_RUNTIME_DIR=/run/user/0 mutter --wayland --no-x11 --display-server
```

### 问题：Wayland socket 不存在

```bash
# 检查运行时目录
ls -la /run/user/0/

# 手动创建并设置权限
sudo mkdir -p /run/user/0
sudo chmod 700 /run/user/0

# 重启 Mutter
sudo systemctl restart mutter-wayland
```

### 问题：LVGL 仍然无法连接

```bash
# 检查环境变量
echo $WAYLAND_DISPLAY
echo $XDG_RUNTIME_DIR

# 测试 Wayland 连接
WAYLAND_DEBUG=1 weston-info 2>&1 | head -n 50
```

## 技术细节

### Mutter 与 Weston 协议差异

1. **xdg_positioner 处理**
   - Weston: 存在对象 ID 映射 bug
   - Mutter: 严格遵循协议规范

2. **EGL 配置**
   - Weston: `EGL_PLATFORM=drm`
   - Mutter: `EGL_PLATFORM=wayland`

3. **窗口管理**
   - Weston: 简单的窗口栈叠
   - Mutter: 完整的窗口管理器功能

## 相关文件

- `Makefile` - 主要构建和部署脚本
- `cpp_backend/src/ui/lvgl_wayland_interface.cpp` - LVGL Wayland 客户端实现
- `/etc/systemd/system/mutter-wayland.service` - Mutter 系统服务配置

## 更新日期

2025-10-21

## 相关问题

- Weston `xdg_positioner@8.set_size` bug: https://gitlab.freedesktop.org/wayland/weston/-/issues/XXX
- LVGL Wayland 驱动文档: https://docs.lvgl.io/master/integration/driver/wayland.html

---

**注意：** 此迁移不影响 LVGL 的代码实现，只是更换了底层的 Wayland 合成器。

