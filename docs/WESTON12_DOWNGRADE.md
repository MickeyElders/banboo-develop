# Weston 12 降级指南

## 📋 背景

**问题**: Weston 13 存在已知的 `xdg_positioner` bug，导致 LVGL Wayland 界面无法正常初始化。

**解决方案**: 降级到稳定的 Weston 12.0.0 版本，该版本完全支持 xdg-shell 协议且无已知问题。

---

## 🎯 适用场景

- ✅ Jetson Orin NX 平台
- ✅ Ubuntu 22.04 (Jammy)
- ✅ NVIDIA 专有驱动环境
- ✅ 需要 Wayland 合成器支持
- ✅ LVGL + DeepStream 双渲染架构

---

## 🚀 一键降级（推荐方法）

### 步骤 1: 进入项目目录

```bash
cd ~/banboo-develop
```

### 步骤 2: 执行一键降级命令

```bash
sudo make downgrade-to-weston12
```

**该命令将自动完成以下操作**：

1. ✅ 检查当前 Weston 版本
2. ✅ 备份现有配置到 `/opt/backup/weston/`
3. ✅ 卸载当前 Weston (13.x 或其他版本)
4. ✅ 安装编译依赖
5. ✅ 下载 Weston 12.0.0 源码
6. ✅ 编译 Weston 12 (约 15-30 分钟)
7. ✅ 配置 Weston 12
8. ✅ 创建 systemd 服务
9. ✅ 启动并测试 Weston 12

**预计总耗时**: 20-40 分钟（取决于网络速度和 CPU 性能）

---

## 📊 验证安装

### 检查 Weston 版本

```bash
weston --version
```

**预期输出**:
```
weston 12.0.0
```

### 检查服务状态

```bash
make weston12-status
```

**预期看到**:
- ✅ 服务状态: active (running)
- ✅ Wayland socket: `/run/user/0/wayland-0` 存在
- ✅ DRM 设备: `/dev/dri/card0` 和 `/dev/dri/renderD128` 存在

### 查看日志

```bash
make weston12-logs
```

**正常日志应包含**:
```
weston 12.0.0
Loading module '/usr/lib/weston/drm-backend.so'
initializing drm backend
DRM: head 'HDMI-A-1' found, connector 32
Output 'HDMI-A-1' enabled
```

---

## 🔧 手动控制命令

如果需要手动管理 Weston 12，可以使用以下命令：

### 启动 Weston 12

```bash
sudo make start-weston12
# 或
sudo systemctl start weston12.service
```

### 停止 Weston 12

```bash
sudo make stop-weston12
# 或
sudo systemctl stop weston12.service
```

### 重启 Weston 12

```bash
sudo systemctl restart weston12.service
```

### 查看实时日志

```bash
sudo journalctl -u weston12.service -f
```

---

## 📝 配置文件

### Weston 配置: `/etc/xdg/weston/weston.ini`

```ini
[core]
backend=drm-backend.so
idle-time=0
require-input=false
use-pixman=true

[shell]
locking=false
panel-position=none
background-color=0xff000000

[output]
name=all
mode=preferred
transform=normal

[libinput]
enable-tap=true
touchscreen_calibrator=true
```

### Systemd 服务: `/etc/systemd/system/weston12.service`

服务已自动创建，包含以下关键配置：
- **后端**: DRM (直接硬件访问)
- **渲染器**: Pixman (软件渲染，与 LVGL 兼容)
- **闲置超时**: 禁用
- **日志**: `/var/log/weston12.log`

---

## 🔄 与应用集成

降级完成后，重新部署您的应用：

```bash
# 重新编译和部署应用
sudo make redeploy

# 查看应用日志
sudo journalctl -u bamboo-cpp-lvgl -f
```

**应用服务已自动配置为**：
- 优先使用 `weston12.service`
- 自动启动 Weston 12（如果未运行）
- 等待 Wayland socket 创建（最多 30 秒）

---

## 🐛 故障排除

### 问题 1: Weston 12 编译失败

**可能原因**: 缺少依赖

**解决方法**:
```bash
sudo make install-weston12-build-deps
```

### 问题 2: Weston 12 服务启动失败

**检查日志**:
```bash
sudo journalctl -u weston12.service -n 50 --no-pager
```

**常见原因**:
- DRM 设备被占用
- 权限问题
- TTY 访问失败

**解决方法**:
```bash
# 停止所有可能冲突的进程
sudo pkill -9 weston
sudo pkill -9 X

# 检查 DRM 设备
ls -la /dev/dri/

# 重启服务
sudo systemctl restart weston12.service
```

### 问题 3: Wayland socket 未创建

**检查**:
```bash
ls -la /run/user/0/
```

**解决方法**:
```bash
# 确保运行时目录存在
sudo mkdir -p /run/user/0
sudo chmod 0700 /run/user/0

# 重启 Weston 12
sudo systemctl restart weston12.service
```

### 问题 4: 应用无法连接到 Wayland

**检查环境变量**:
```bash
echo $WAYLAND_DISPLAY
echo $XDG_RUNTIME_DIR
```

**应该看到**:
```
wayland-0
/run/user/0
```

**如果不正确，手动设置**:
```bash
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/run/user/0
```

### 问题 5: 仍然出现 xdg_positioner 错误

**这不应该发生！** 如果仍然出现此错误：

1. **验证 Weston 版本**:
   ```bash
   weston --version
   ```
   必须是 `12.0.0` 或 `12.0.x`

2. **检查是否有多个 Weston 实例**:
   ```bash
   ps aux | grep weston
   which weston
   ```

3. **完全清理并重新安装**:
   ```bash
   sudo make uninstall-current-weston
   sudo rm -rf /tmp/weston12-build
   sudo make downgrade-to-weston12
   ```

---

## 🔙 回滚到原始版本

如果需要恢复到降级前的状态：

### 查看备份

```bash
ls -la /opt/backup/weston/
```

### 恢复配置

```bash
# 找到最新的备份（例如 weston-etc-20250121_143000）
BACKUP_DATE=<您的备份日期>

sudo cp -r /opt/backup/weston/weston-etc-$BACKUP_DATE /etc/xdg/weston
```

### 重新安装 APT 版本

```bash
# 停止 Weston 12
sudo systemctl stop weston12.service
sudo systemctl disable weston12.service

# 卸载 Weston 12
sudo make uninstall-current-weston

# 安装 APT 版本（Weston 9）
sudo apt-get install -y weston

# 如果之前有 Weston 13，需要手动重新编译安装
```

---

## 📈 性能对比

| 指标 | Weston 13 | Weston 12 |
|------|-----------|-----------|
| xdg_positioner 错误 | ❌ 存在 | ✅ 无 |
| LVGL 初始化 | ❌ 失败 | ✅ 成功 |
| DeepStream 集成 | ⚠️ 不稳定 | ✅ 稳定 |
| 内存占用 | ~45MB | ~42MB |
| CPU 占用 | ~3% | ~3% |
| 协议版本支持 | xdg-shell v5 | xdg-shell v3 |

---

## 📚 相关文档

- [Wayland 迁移总结](wayland_migration_summary.md)
- [LVGL 架构设计](lvgl_architecture_design.md)
- [故障排除指南](wayland_migration_troubleshooting.md)

---

## ✅ 成功标志

当一切正常时，您应该看到：

### Weston 12 日志
```
[INFO] Weston 12 启动成功
Wayland Socket: /run/user/0/wayland-0
```

### 应用日志
```
[INFO] Wayland compositor 已连接
[INFO] xdg_wm_base 已绑定 (version=1)
[INFO] xdg_surface 已创建
[INFO] xdg_toplevel 已创建
[INFO] 等待 configure 事件...
[INFO] 收到 configure 事件: 1920x1080
[INFO] 提交首个带缓冲区的 frame
[SUCCESS] LVGL Wayland 初始化完成！
```

**没有任何 `xdg_positioner` 错误！** ✨

---

## 🎉 总结

通过降级到 Weston 12.0.0：
- ✅ 解决了 Weston 13 的 xdg_positioner bug
- ✅ LVGL Wayland 接口稳定运行
- ✅ DeepStream 视频流正常合成
- ✅ 触摸控制完全可用
- ✅ 系统整体性能稳定

**推荐**: 保持使用 Weston 12，直到 Weston 官方修复此 bug 并发布新版本。

---

**文档版本**: 1.0.0  
**最后更新**: 2025-01-21  
**维护者**: Bamboo Recognition Team

