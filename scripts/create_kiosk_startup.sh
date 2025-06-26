#!/bin/bash
# -*- coding: utf-8 -*-
"""
智能切竹机 - Kiosk模式自启动配置脚本
实现开机直接进入触摸控制界面，覆盖系统开机动画

功能：
1. 创建系统级自启动配置
2. 配置无人值守登录
3. 禁用不必要的系统服务
4. 优化启动性能
5. 设置全屏Kiosk模式
"""

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查运行权限
check_privileges() {
    if [ "$EUID" -ne 0 ]; then
        log_error "请使用root权限运行此脚本"
        echo "使用方法: sudo $0"
        exit 1
    fi
}

# 检测系统信息
detect_system() {
    log_step "检测系统信息..."
    
    # 检测发行版
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
        log_info "检测到系统: $PRETTY_NAME"
    else
        log_error "无法检测系统版本"
        exit 1
    fi
    
    # 检测桌面环境
    if command -v gnome-shell &> /dev/null; then
        DESKTOP_ENV="gnome"
        log_info "检测到桌面环境: GNOME"
    else
        log_error "当前系统不支持GNOME桌面环境"
        exit 1
    fi
    
    # 检测用户
    if [ -z "$SUDO_USER" ]; then
        log_error "无法检测目标用户，请使用sudo运行"
        exit 1
    fi
    
    TARGET_USER=$SUDO_USER
    USER_HOME="/home/$TARGET_USER"
    log_info "目标用户: $TARGET_USER"
    log_info "用户主目录: $USER_HOME"
}

# 安装必要的软件包
install_dependencies() {
    log_step "安装必要的软件包..."
    
    case $DISTRO in
        "ubuntu"|"debian")
            apt update
            apt install -y \
                gnome-session \
                gnome-shell \
                gdm3 \
                python3-gi \
                python3-gi-cairo \
                gir1.2-gtk-4.0 \
                gir1.2-adw-1 \
                xdotool \
                unclutter \
                xserver-xorg \
                plymouth-themes
            ;;
        "fedora"|"centos"|"rhel")
            dnf install -y \
                gnome-session \
                gnome-shell \
                gdm \
                python3-gobject \
                gtk4-devel \
                libadwaita-devel \
                xdotool \
                unclutter \
                xorg-x11-server-Xorg \
                plymouth-themes
            ;;
        "arch"|"manjaro")
            pacman -S --noconfirm \
                gnome-session \
                gnome-shell \
                gdm \
                python-gobject \
                gtk4 \
                libadwaita \
                xdotool \
                unclutter \
                xorg-server \
                plymouth
            ;;
        *)
            log_warning "未知的发行版，请手动安装依赖"
            ;;
    esac
    
    log_info "软件包安装完成"
}

# 创建专用用户（可选）
create_kiosk_user() {
    local create_user_choice
    read -p "是否创建专用的Kiosk用户？(y/N): " create_user_choice
    
    if [[ $create_user_choice =~ ^[Yy]$ ]]; then
        KIOSK_USER="bamboocutter"
        KIOSK_HOME="/home/$KIOSK_USER"
        
        if id "$KIOSK_USER" &>/dev/null; then
            log_info "用户 $KIOSK_USER 已存在"
        else
            log_step "创建Kiosk用户: $KIOSK_USER"
            useradd -m -s /bin/bash $KIOSK_USER
            log_info "Kiosk用户创建完成"
        fi
        
        TARGET_USER=$KIOSK_USER
        USER_HOME=$KIOSK_HOME
    fi
}

# 配置自动登录
configure_autologin() {
    log_step "配置自动登录..."
    
    # 配置GDM自动登录
    local gdm_config="/etc/gdm3/custom.conf"
    local gdm_dir="/etc/gdm3"
    
    # 某些发行版使用不同的路径
    if [ ! -d "$gdm_dir" ]; then
        gdm_config="/etc/gdm/custom.conf"
        gdm_dir="/etc/gdm"
    fi
    
    if [ ! -f "$gdm_config" ]; then
        log_warning "GDM配置文件不存在，创建默认配置"
        mkdir -p "$gdm_dir"
        cat > "$gdm_config" << EOF
[daemon]
AutomaticLoginEnable=True
AutomaticLogin=$TARGET_USER

[security]

[xdmcp]

[chooser]

[debug]
EOF
    else
        # 备份原配置
        cp "$gdm_config" "${gdm_config}.backup.$(date +%Y%m%d_%H%M%S)"
        
        # 修改配置
        sed -i '/^\[daemon\]/,/^\[/ {
            /^AutomaticLoginEnable=/c\AutomaticLoginEnable=True
            /^AutomaticLogin=/c\AutomaticLogin='$TARGET_USER'
            /^\[daemon\]$/a\AutomaticLoginEnable=True\nAutomaticLogin='$TARGET_USER'
        }' "$gdm_config"
        
        # 如果没有[daemon]节，添加它
        if ! grep -q "^\[daemon\]" "$gdm_config"; then
            echo -e "\n[daemon]\nAutomaticLoginEnable=True\nAutomaticLogin=$TARGET_USER" >> "$gdm_config"
        fi
    fi
    
    log_info "自动登录配置完成"
}

# 创建应用启动桌面文件
create_desktop_file() {
    log_step "创建应用桌面文件..."
    
    local app_desktop="/usr/share/applications/bamboo-cutter-touch.desktop"
    local project_path=$(pwd)
    
    cat > "$app_desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=智能切竹机控制系统
Name[en]=Smart Bamboo Cutter Control System
Comment=工业级智能切竹机触摸控制界面
Comment[en]=Industrial Smart Bamboo Cutter Touch Control Interface
Exec=/usr/bin/python3 $project_path/src/gui/touch_interface.py
Icon=$project_path/assets/icon.png
Terminal=false
StartupNotify=false
Categories=Industrial;Engineering;
Keywords=bamboo;cutter;industrial;control;
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=3
StartupWMClass=bamboo-cutter-touch
EOF
    
    chmod 644 "$app_desktop"
    log_info "桌面文件创建完成: $app_desktop"
}

# 配置用户自启动
configure_user_autostart() {
    log_step "配置用户自启动..."
    
    local autostart_dir="$USER_HOME/.config/autostart"
    local app_autostart="$autostart_dir/bamboo-cutter-touch.desktop"
    local project_path=$(pwd)
    
    # 创建自启动目录
    sudo -u $TARGET_USER mkdir -p "$autostart_dir"
    
    # 创建自启动桌面文件
    cat > "$app_autostart" << EOF
[Desktop Entry]
Type=Application
Encoding=UTF-8
Name=智能切竹机控制系统
Comment=自动启动智能切竹机触摸控制界面
Exec=/usr/bin/python3 $project_path/src/gui/touch_interface.py
Terminal=false
StartupNotify=false
Categories=Application;
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=5
Hidden=false
EOF
    
    chown $TARGET_USER:$TARGET_USER "$app_autostart"
    chmod 644 "$app_autostart"
    
    log_info "用户自启动配置完成"
}

# 配置GNOME设置
configure_gnome_settings() {
    log_step "配置GNOME设置..."
    
    # 使用sudo -u运行gsettings命令
    sudo -u $TARGET_USER dbus-launch gsettings set org.gnome.desktop.session idle-delay 0
    sudo -u $TARGET_USER dbus-launch gsettings set org.gnome.desktop.screensaver lock-enabled false
    sudo -u $TARGET_USER dbus-launch gsettings set org.gnome.desktop.screensaver idle-activation-enabled false
    sudo -u $TARGET_USER dbus-launch gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
    sudo -u $TARGET_USER dbus-launch gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'
    
    # 禁用通知
    sudo -u $TARGET_USER dbus-launch gsettings set org.gnome.desktop.notifications show-banners false
    sudo -u $TARGET_USER dbus-launch gsettings set org.gnome.desktop.notifications show-in-lock-screen false
    
    # 隐藏桌面图标（如果不需要）
    sudo -u $TARGET_USER dbus-launch gsettings set org.gnome.shell.extensions.desktop-icons show-home false
    sudo -u $TARGET_USER dbus-launch gsettings set org.gnome.shell.extensions.desktop-icons show-trash false
    
    log_info "GNOME设置配置完成"
}

# 创建启动脚本
create_startup_script() {
    log_step "创建专用启动脚本..."
    
    local startup_script="/usr/local/bin/bamboo-cutter-startup.sh"
    local project_path=$(pwd)
    
    cat > "$startup_script" << 'EOF'
#!/bin/bash
# 智能切竹机自启动脚本

# 等待桌面环境完全加载
sleep 8

# 设置环境变量
export DISPLAY=:0
export GDK_BACKEND=wayland,x11
export XDG_RUNTIME_DIR="/run/user/$(id -u)"

# 隐藏鼠标光标（适合纯触摸操作）
unclutter -idle 1 -root &

# 禁用屏幕保护
xset s off
xset -dpms
xset s noblank

# 启动应用
cd PROJECT_PATH_PLACEHOLDER
/usr/bin/python3 src/gui/touch_interface.py

# 如果应用崩溃，尝试重启
while [ $? -ne 0 ]; do
    echo "应用异常退出，5秒后重启..."
    sleep 5
    /usr/bin/python3 src/gui/touch_interface.py
done
EOF
    
    # 替换项目路径
    sed -i "s|PROJECT_PATH_PLACEHOLDER|$project_path|g" "$startup_script"
    
    chmod +x "$startup_script"
    log_info "启动脚本创建完成: $startup_script"
    
    # 创建对应的desktop文件
    local startup_desktop="$USER_HOME/.config/autostart/bamboo-startup.desktop"
    cat > "$startup_desktop" << EOF
[Desktop Entry]
Type=Application
Name=Bamboo Cutter Startup
Exec=$startup_script
Terminal=false
StartupNotify=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=1
Hidden=false
EOF
    
    chown $TARGET_USER:$TARGET_USER "$startup_desktop"
    chmod 644 "$startup_desktop"
}

# 配置系统服务（可选）
create_systemd_service() {
    local create_service_choice
    read -p "是否创建systemd服务以确保系统级启动？(y/N): " create_service_choice
    
    if [[ $create_service_choice =~ ^[Yy]$ ]]; then
        log_step "创建systemd服务..."
        
        local service_file="/etc/systemd/system/bamboo-cutter.service"
        local project_path=$(pwd)
        
        cat > "$service_file" << EOF
[Unit]
Description=Smart Bamboo Cutter Touch Interface
After=graphical-session.target
Wants=graphical-session.target

[Service]
Type=simple
User=$TARGET_USER
Group=$TARGET_USER
Environment=DISPLAY=:0
Environment=GDK_BACKEND=wayland,x11
Environment=XDG_RUNTIME_DIR=/run/user/$(id -u $TARGET_USER)
WorkingDirectory=$project_path
ExecStart=/usr/bin/python3 $project_path/src/gui/touch_interface.py
Restart=always
RestartSec=5

[Install]
WantedBy=graphical.target
EOF
        
        systemctl daemon-reload
        systemctl enable bamboo-cutter.service
        
        log_info "systemd服务创建并启用完成"
    fi
}

# 优化系统启动
optimize_boot() {
    log_step "优化系统启动..."
    
    # 禁用不必要的服务
    local services_to_disable=(
        "bluetooth.service"
        "cups.service"
        "ModemManager.service"
        "whoopsie.service"
        "apport.service"
    )
    
    for service in "${services_to_disable[@]}"; do
        if systemctl is-enabled "$service" &>/dev/null; then
            systemctl disable "$service"
            log_info "已禁用服务: $service"
        fi
    done
    
    # 配置Plymouth主题（隐藏启动画面或使用自定义主题）
    if command -v plymouth-set-default-theme &> /dev/null; then
        # 设置无主题或简单主题
        plymouth-set-default-theme text
        update-initramfs -u 2>/dev/null || dracut --force 2>/dev/null || true
        log_info "Plymouth主题已设置为文本模式"
    fi
    
    log_info "系统启动优化完成"
}

# 创建图标文件
create_icon() {
    log_step "创建应用图标..."
    
    local assets_dir="assets"
    local icon_file="$assets_dir/icon.png"
    
    mkdir -p "$assets_dir"
    
    # 如果没有图标，创建一个简单的默认图标（使用ImageMagick）
    if [ ! -f "$icon_file" ] && command -v convert &> /dev/null; then
        convert -size 128x128 xc:transparent \
            -fill '#27ae60' -draw 'roundrectangle 10,10 118,118 15,15' \
            -fill white -pointsize 24 -gravity center \
            -annotate +0+0 '竹' \
            "$icon_file"
        log_info "默认图标创建完成"
    fi
}

# 设置权限
set_permissions() {
    log_step "设置文件权限..."
    
    # 确保项目文件权限正确
    chown -R $TARGET_USER:$TARGET_USER "$(pwd)"
    
    # 设置Python文件可执行权限
    chmod +x src/gui/touch_interface.py
    
    log_info "权限设置完成"
}

# 创建卸载脚本
create_uninstall_script() {
    log_step "创建卸载脚本..."
    
    local uninstall_script="scripts/uninstall_kiosk.sh"
    
    cat > "$uninstall_script" << EOF
#!/bin/bash
# 智能切竹机Kiosk模式卸载脚本

echo "开始卸载Kiosk模式配置..."

# 禁用自动登录
sudo sed -i '/^AutomaticLoginEnable=/c\AutomaticLoginEnable=False' /etc/gdm3/custom.conf 2>/dev/null || true
sudo sed -i '/^AutomaticLoginEnable=/c\AutomaticLoginEnable=False' /etc/gdm/custom.conf 2>/dev/null || true

# 删除自启动文件
rm -f /home/$TARGET_USER/.config/autostart/bamboo-cutter-touch.desktop
rm -f /home/$TARGET_USER/.config/autostart/bamboo-startup.desktop
rm -f /usr/share/applications/bamboo-cutter-touch.desktop

# 删除启动脚本
rm -f /usr/local/bin/bamboo-cutter-startup.sh

# 停止并删除systemd服务
sudo systemctl stop bamboo-cutter.service 2>/dev/null || true
sudo systemctl disable bamboo-cutter.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/bamboo-cutter.service
sudo systemctl daemon-reload

echo "Kiosk模式配置已卸载"
EOF
    
    chmod +x "$uninstall_script"
    log_info "卸载脚本创建完成: $uninstall_script"
}

# 测试配置
test_configuration() {
    log_step "测试配置..."
    
    # 检查必要文件是否存在
    local required_files=(
        "src/gui/touch_interface.py"
        "/usr/local/bin/bamboo-cutter-startup.sh"
        "$USER_HOME/.config/autostart/bamboo-startup.desktop"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "必要文件不存在: $file"
            return 1
        fi
    done
    
    # 检查Python依赖
    if ! sudo -u $TARGET_USER python3 -c "import gi; gi.require_version('Gtk', '4.0'); from gi.repository import Gtk, Adw" 2>/dev/null; then
        log_error "Python GTK4/Adwaita依赖检查失败"
        return 1
    fi
    
    log_info "配置测试通过"
}

# 主函数
main() {
    echo "========================================"
    echo "智能切竹机 Kiosk模式配置脚本"
    echo "========================================"
    
    check_privileges
    detect_system
    
    echo
    echo "即将配置以下功能："
    echo "- 自动登录到指定用户"
    echo "- 开机自动启动触摸界面"
    echo "- 全屏Kiosk模式"
    echo "- 禁用屏幕保护和锁屏"
    echo "- 优化系统启动性能"
    echo
    
    read -p "是否继续？(y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        log_info "配置已取消"
        exit 0
    fi
    
    install_dependencies
    create_kiosk_user
    configure_autologin
    create_icon
    create_desktop_file
    configure_user_autostart
    configure_gnome_settings
    create_startup_script
    create_systemd_service
    optimize_boot
    set_permissions
    create_uninstall_script
    
    if test_configuration; then
        echo
        log_info "======================================"
        log_info "Kiosk模式配置完成！"
        log_info "======================================"
        echo
        echo "配置详情："
        echo "- 目标用户: $TARGET_USER"
        echo "- 自动登录: 已启用"
        echo "- 触摸界面: 开机自启动"
        echo "- 启动脚本: /usr/local/bin/bamboo-cutter-startup.sh"
        echo "- 卸载脚本: scripts/uninstall_kiosk.sh"
        echo
        echo "重启系统后将自动进入智能切竹机控制界面"
        echo
        read -p "是否立即重启系统？(y/N): " reboot_confirm
        if [[ $reboot_confirm =~ ^[Yy]$ ]]; then
            log_info "系统即将重启..."
            sleep 3
            reboot
        fi
    else
        log_error "配置验证失败，请检查错误信息"
        exit 1
    fi
}

# 脚本入口
main "$@" 