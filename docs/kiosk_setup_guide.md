# 智能切竹机 Kiosk模式设置指南

## 概述

本指南详细说明如何配置智能切竹机系统为Kiosk模式，实现开机自动登录并直接进入触摸控制界面，同时覆盖系统默认开机动画为专业的工业界面。

## 系统要求

### 硬件要求
- **处理器**: ARM64 (如Jetson Nano) 或 x86_64
- **内存**: 最低 2GB RAM，推荐 4GB+
- **存储**: 最低 16GB，推荐 32GB+ SSD
- **显示**: 支持触摸的显示器，推荐分辨率 1024x768 或更高
- **网络**: 以太网连接（用于Modbus通信）

### 软件要求
- **操作系统**: Ubuntu 20.04+ / Debian 11+ / Fedora 35+ / Arch Linux
- **桌面环境**: GNOME 40+
- **Python**: 3.8+
- **显示服务器**: X11 或 Wayland

## 安装步骤

### 1. 系统准备

#### 1.1 更新系统
```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# Fedora
sudo dnf update -y

# Arch Linux
sudo pacman -Syu
```

#### 1.2 安装基础依赖
```bash
# Ubuntu/Debian
sudo apt install -y git python3 python3-pip python3-venv build-essential

# Fedora
sudo dnf install -y git python3 python3-pip python3-virtualenv gcc gcc-c++

# Arch Linux
sudo pacman -S git python python-pip python-virtualenv base-devel
```

### 2. 项目部署

#### 2.1 克隆项目
```bash
cd /opt
sudo git clone <项目仓库URL> bamboo-cut-system
sudo chown -R $USER:$USER bamboo-cut-system
cd bamboo-cut-system
```

#### 2.2 安装Python依赖
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2.3 验证系统功能
```bash
# 测试GUI依赖
python3 -c "import gi; gi.require_version('Gtk', '4.0'); from gi.repository import Gtk, Adw; print('GTK4/Adwaita 可用')"

# 测试设备连接
python3 test/test_hardware.py

# 测试视觉系统
python3 test/test_vision_ai.py
```

### 3. Kiosk模式配置

#### 3.1 运行自动配置脚本
```bash
cd /opt/bamboo-cut-system
sudo ./scripts/create_kiosk_startup.sh
```

**配置过程包括**：
1. 自动检测系统信息
2. 安装必要的GNOME和GTK4组件
3. 可选创建专用Kiosk用户
4. 配置GDM自动登录
5. 创建应用自启动配置
6. 优化GNOME设置（禁用屏保、通知等）
7. 创建系统启动脚本
8. 可选创建systemd服务

#### 3.2 手动配置（高级用户）

**创建自启动目录**：
```bash
mkdir -p ~/.config/autostart
```

**创建自启动桌面文件**：
```bash
cat > ~/.config/autostart/bamboo-cutter-touch.desktop << EOF
[Desktop Entry]
Type=Application
Name=智能切竹机控制系统
Exec=/usr/bin/python3 /opt/bamboo-cut-system/src/gui/touch_interface.py
Terminal=false
StartupNotify=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=5
Hidden=false
EOF
```

**配置GNOME设置**：
```bash
# 禁用屏幕保护
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false
gsettings set org.gnome.desktop.screensaver idle-activation-enabled false

# 禁用电源管理
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'

# 禁用通知
gsettings set org.gnome.desktop.notifications show-banners false
gsettings set org.gnome.desktop.notifications show-in-lock-screen false
```

### 4. 自定义启动动画

#### 4.1 运行启动动画配置脚本
```bash
sudo ./scripts/setup_custom_splash.sh
```

**配置内容**：
1. 创建工业风格Plymouth主题
2. 设置品牌化启动画面
3. 隐藏系统启动信息
4. 优化启动速度
5. 配置GRUB启动参数

#### 4.2 主题定制

**自定义Logo**：
将您的Logo图片放置在 `assets/logo.png`，推荐尺寸 200x80 像素。

**自定义颜色**：
编辑 `/usr/share/plymouth/themes/bamboo-cutter/bamboo-cutter.script`，修改颜色定义：
```javascript
bg_color = "#2c3e50";        // 背景色
accent_color = "#27ae60";    // 强调色
text_color = "#ecf0f1";      // 文本色
progress_color = "#3498db";  // 进度条色
```

### 5. 系统优化

#### 5.1 性能优化

**禁用不必要的服务**：
```bash
sudo systemctl disable bluetooth.service
sudo systemctl disable cups.service
sudo systemctl disable ModemManager.service
```

**内存优化**（对于低内存设备）：
```bash
# 添加到 /etc/sysctl.conf
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
echo "vm.vfs_cache_pressure=50" | sudo tee -a /etc/sysctl.conf
```

#### 5.2 触摸优化

**校准触摸屏**：
```bash
# 安装触摸校准工具
sudo apt install xinput-calibrator  # Ubuntu/Debian
sudo dnf install xinput_calibrator  # Fedora

# 运行校准
xinput_calibrator
```

**禁用鼠标光标**（纯触摸模式）：
```bash
sudo apt install unclutter
# 添加到启动脚本：unclutter -idle 1 -root &
```

### 6. 安全配置

#### 6.1 用户权限

**创建专用用户**：
```bash
sudo useradd -m -s /bin/bash bamboocutter
sudo usermod -aG dialout,gpio bamboocutter  # 添加硬件访问权限
```

**设置文件权限**：
```bash
sudo chown -R bamboocutter:bamboocutter /opt/bamboo-cut-system
sudo chmod +x /opt/bamboo-cut-system/src/gui/touch_interface.py
```

#### 6.2 网络安全

**配置防火墙**：
```bash
sudo ufw enable
sudo ufw allow 502/tcp  # Modbus TCP端口
sudo ufw deny ssh      # 禁用SSH（生产环境）
```

## 验证和测试

### 1. 功能测试

#### 1.1 界面测试
```bash
# 手动启动触摸界面
cd /opt/bamboo-cut-system
python3 src/gui/touch_interface.py
```

#### 1.2 自启动测试
```bash
# 检查自启动配置
ls -la ~/.config/autostart/
cat ~/.config/autostart/bamboo-cutter-touch.desktop
```

#### 1.3 主题测试
```bash
# 预览Plymouth主题
sudo plymouth show-splash &
sleep 5
sudo plymouth hide-splash
```

### 2. 性能测试

#### 2.1 启动时间测试
```bash
# 分析启动时间
systemd-analyze
systemd-analyze blame
systemd-analyze critical-chain
```

#### 2.2 内存使用测试
```bash
# 监控内存使用
free -h
ps aux --sort=-%mem | head -10
```

## 故障排除

### 1. 常见问题

#### 1.1 界面无法启动

**症状**: 开机后没有出现触摸界面

**解决方案**:
1. 检查自启动配置：
   ```bash
   ls -la ~/.config/autostart/
   ```

2. 检查Python依赖：
   ```bash
   python3 -c "import gi; gi.require_version('Gtk', '4.0')"
   ```

3. 查看错误日志：
   ```bash
   tail -f /tmp/bamboo_gui.log
   journalctl -u gdm -f
   ```

#### 1.2 自动登录失败

**症状**: 系统启动到登录界面

**解决方案**:
1. 检查GDM配置：
   ```bash
   sudo cat /etc/gdm3/custom.conf
   # 或
   sudo cat /etc/gdm/custom.conf
   ```

2. 确认配置正确：
   ```
   [daemon]
   AutomaticLoginEnable=True
   AutomaticLogin=用户名
   ```

#### 1.3 Plymouth主题不显示

**症状**: 启动时仍显示默认启动画面

**解决方案**:
1. 检查主题是否安装：
   ```bash
   sudo plymouth-set-default-theme --list
   ```

2. 重新生成initramfs：
   ```bash
   # Ubuntu/Debian
   sudo update-initramfs -u
   
   # Fedora
   sudo dracut --force
   
   # Arch
   sudo mkinitcpio -P
   ```

### 2. 调试工具

#### 2.1 日志查看
```bash
# 系统日志
journalctl -f

# GNOME日志
journalctl -u gdm -f

# 应用日志
tail -f /tmp/bamboo_gui.log

# 启动日志
dmesg | tail -50
```

#### 2.2 进程监控
```bash
# 监控GTK应用
ps aux | grep python3
ps aux | grep touch_interface

# 监控系统资源
top
htop
iotop
```

### 3. 恢复操作

#### 3.1 恢复默认设置

**禁用自动登录**：
```bash
sudo sed -i 's/AutomaticLoginEnable=True/AutomaticLoginEnable=False/' /etc/gdm3/custom.conf
```

**删除自启动配置**：
```bash
rm ~/.config/autostart/bamboo-cutter-touch.desktop
```

**恢复默认Plymouth主题**：
```bash
sudo plymouth-set-default-theme ubuntu-logo
sudo update-initramfs -u
```

#### 3.2 使用卸载脚本
```bash
# 运行自动卸载脚本
./scripts/uninstall_kiosk.sh
./scripts/uninstall_splash.sh
```

## 维护指南

### 1. 日常维护

#### 1.1 系统更新
```bash
# 定期更新系统（建议每月）
sudo apt update && sudo apt upgrade -y

# 更新Python依赖
cd /opt/bamboo-cut-system
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

#### 1.2 日志清理
```bash
# 清理系统日志
sudo journalctl --vacuum-time=7d

# 清理应用日志
sudo find /tmp -name "bamboo_*.log" -mtime +7 -delete
```

#### 1.3 配置备份
```bash
# 备份重要配置
sudo tar -czf /backup/kiosk-config-$(date +%Y%m%d).tar.gz \
    /etc/gdm3/custom.conf \
    /etc/gdm/custom.conf \
    ~/.config/autostart/ \
    /usr/share/plymouth/themes/bamboo-cutter/
```

### 2. 性能监控

#### 2.1 设置监控脚本
```bash
#!/bin/bash
# /opt/bamboo-cut-system/scripts/monitor_system.sh

# 记录系统状态
echo "=== $(date) ===" >> /var/log/bamboo-monitor.log
free -h >> /var/log/bamboo-monitor.log
df -h >> /var/log/bamboo-monitor.log
uptime >> /var/log/bamboo-monitor.log

# 检查关键进程
if ! pgrep -f "touch_interface.py" > /dev/null; then
    echo "WARNING: Touch interface not running" >> /var/log/bamboo-monitor.log
fi
```

#### 2.2 定期任务
```bash
# 添加到crontab
crontab -e

# 每5分钟检查系统状态
*/5 * * * * /opt/bamboo-cut-system/scripts/monitor_system.sh

# 每天重启应用（可选）
0 3 * * * sudo systemctl restart bamboo-cutter.service
```

## 高级配置

### 1. 多屏幕支持

**配置主显示器**：
```bash
# 设置主显示器
xrandr --output eDP-1 --primary
xrandr --output HDMI-1 --off
```

**保存显示配置**：
```bash
# 添加到启动脚本
echo "xrandr --output eDP-1 --primary" >> ~/.profile
```

### 2. 网络配置

**设置静态IP**：
```bash
# 编辑网络配置
sudo nano /etc/netplan/01-netcfg.yaml

# 配置内容
network:
  version: 2
  ethernets:
    eth0:
      addresses: [192.168.1.100/24]
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]

# 应用配置
sudo netplan apply
```

### 3. 远程管理

**配置SSH访问**（开发环境）：
```bash
# 安装SSH服务器
sudo apt install openssh-server

# 配置密钥认证
ssh-keygen -t rsa -b 4096
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# 禁用密码登录
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart ssh
```

**配置VNC访问**（可选）：
```bash
# 安装VNC服务器
sudo apt install x11vnc

# 设置VNC密码
x11vnc -storepasswd

# 创建VNC服务
sudo nano /etc/systemd/system/x11vnc.service
```

## 总结

通过本指南，您已经成功配置了智能切竹机的Kiosk模式，实现了：

1. **自动登录**: 开机无需人工干预
2. **触摸界面**: 专业的工业控制界面
3. **自定义启动**: 品牌化的启动动画
4. **系统优化**: 针对工业应用的性能调优
5. **安全配置**: 适合生产环境的安全设置

系统现在将在开机后自动进入智能切竹机控制界面，为操作人员提供直观、高效的设备控制体验。

**重要提醒**：
- 在生产环境部署前，请充分测试所有功能
- 定期备份重要配置和数据
- 建立适当的维护计划
- 确保网络安全配置符合工厂要求 