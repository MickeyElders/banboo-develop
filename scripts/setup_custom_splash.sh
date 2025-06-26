#!/bin/bash
# -*- coding: utf-8 -*-
"""
智能切竹机 - 自定义启动动画配置脚本
替换系统默认Plymouth启动画面为专用工业界面

功能：
1. 创建自定义Plymouth主题
2. 设置工业风格启动动画
3. 配置快速启动优化
4. 隐藏内核启动信息
5. 创建品牌化启动体验
"""

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# 检查权限
check_privileges() {
    if [ "$EUID" -ne 0 ]; then
        log_error "请使用root权限运行此脚本"
        echo "使用方法: sudo $0"
        exit 1
    fi
}

# 检测系统
detect_system() {
    log_step "检测系统信息..."
    
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        log_info "检测到系统: $PRETTY_NAME"
    else
        log_error "无法检测系统版本"
        exit 1
    fi
    
    # 检查Plymouth支持
    if ! command -v plymouth &> /dev/null; then
        log_warning "Plymouth未安装，将尝试安装"
        install_plymouth
    else
        log_info "Plymouth已安装"
    fi
}

# 安装Plymouth
install_plymouth() {
    log_step "安装Plymouth..."
    
    case $DISTRO in
        "ubuntu"|"debian")
            apt update
            apt install -y plymouth plymouth-themes
            ;;
        "fedora"|"centos"|"rhel")
            dnf install -y plymouth plymouth-scripts plymouth-themes
            ;;
        "arch"|"manjaro")
            pacman -S --noconfirm plymouth
            ;;
        *)
            log_error "不支持的发行版，请手动安装Plymouth"
            exit 1
            ;;
    esac
    
    log_info "Plymouth安装完成"
}

# 创建自定义主题目录
create_theme_directory() {
    log_step "创建自定义主题目录..."
    
    THEME_NAME="bamboo-cutter"
    THEME_DIR="/usr/share/plymouth/themes/$THEME_NAME"
    
    mkdir -p "$THEME_DIR"
    log_info "主题目录创建: $THEME_DIR"
}

# 创建主题配置文件
create_theme_config() {
    log_step "创建主题配置文件..."
    
    cat > "$THEME_DIR/$THEME_NAME.plymouth" << EOF
[Plymouth Theme]
Name=智能切竹机启动主题
Description=Smart Bamboo Cutter Boot Theme - Industrial Style
ModuleName=script

[script]
ImageDir=$THEME_DIR
ScriptFile=$THEME_DIR/$THEME_NAME.script
EOF
    
    log_info "主题配置文件创建完成"
}

# 创建启动脚本
create_theme_script() {
    log_step "创建主题脚本..."
    
    cat > "$THEME_DIR/$THEME_NAME.script" << 'EOF'
# 智能切竹机Plymouth启动脚本

# 窗口和屏幕设置
screen_width = Window.GetWidth();
screen_height = Window.GetHeight();

# 颜色定义
bg_color = "#2c3e50";
accent_color = "#27ae60";
text_color = "#ecf0f1";
progress_color = "#3498db";

# 创建背景
background.image = Image.Color(bg_color);
background.sprite = Sprite(background.image);
background.sprite.SetPosition(0, 0, -100);

# 加载或创建logo
logo_image = Image("logo.png");
if (logo_image == NULL) {
    # 如果没有logo图片，创建文本logo
    logo_text = Image.Text("智能切竹机", 48, 1, 1, 1);
    logo.sprite = Sprite(logo_text);
} else {
    logo.sprite = Sprite(logo_image);
}

# 设置logo位置（居中偏上）
logo_x = (screen_width - logo.sprite.GetWidth()) / 2;
logo_y = screen_height * 0.3;
logo.sprite.SetPosition(logo_x, logo_y, 1);

# 创建系统信息文本
system_text = Image.Text("工业级智能切竹系统", 24, 0.9, 0.9, 0.9);
system_sprite = Sprite(system_text);
system_x = (screen_width - system_sprite.GetWidth()) / 2;
system_y = logo_y + logo.sprite.GetHeight() + 20;
system_sprite.SetPosition(system_x, system_y, 1);

# 创建版本信息
version_text = Image.Text("Version 1.0 - Industrial Edition", 16, 0.7, 0.7, 0.7);
version_sprite = Sprite(version_text);
version_x = (screen_width - version_sprite.GetWidth()) / 2;
version_y = system_y + system_sprite.GetHeight() + 10;
version_sprite.SetPosition(version_x, version_y, 1);

# 进度条设置
progress_bar_width = screen_width * 0.6;
progress_bar_height = 8;
progress_bar_x = (screen_width - progress_bar_width) / 2;
progress_bar_y = screen_height * 0.8;

# 进度条背景
progress_bg = Image.Color("#34495e", progress_bar_width, progress_bar_height);
progress_bg_sprite = Sprite(progress_bg);
progress_bg_sprite.SetPosition(progress_bar_x, progress_bar_y, 1);

# 进度条前景
progress_fg = Image.Color(progress_color, 1, progress_bar_height);
progress_fg_sprite = Sprite(progress_fg);
progress_fg_sprite.SetPosition(progress_bar_x, progress_bar_y, 2);

# 状态文本
status_text = Image.Text("正在启动系统...", 18, 0.9, 0.9, 0.9);
status_sprite = Sprite(status_text);
status_x = (screen_width - status_sprite.GetWidth()) / 2;
status_y = progress_bar_y + progress_bar_height + 20;
status_sprite.SetPosition(status_x, status_y, 1);

# 进度更新函数
fun progress_callback(duration, progress) {
    # 更新进度条
    new_width = progress_bar_width * progress;
    if (new_width > 0) {
        progress_fg.image = Image.Color(progress_color, new_width, progress_bar_height);
        progress_fg_sprite.SetImage(progress_fg.image);
    }
    
    # 更新状态文本
    if (progress < 0.3) {
        status_text.image = Image.Text("正在加载内核模块...", 18, 0.9, 0.9, 0.9);
    } else if (progress < 0.6) {
        status_text.image = Image.Text("正在初始化硬件...", 18, 0.9, 0.9, 0.9);
    } else if (progress < 0.9) {
        status_text.image = Image.Text("正在启动服务...", 18, 0.9, 0.9, 0.9);
    } else {
        status_text.image = Image.Text("即将进入控制界面...", 18, 0.9, 0.9, 0.9);
    }
    status_sprite.SetImage(status_text.image);
}

Plymouth.SetBootProgressFunction(progress_callback);

# 消息显示函数
message_sprite = Sprite();
message_sprite.SetPosition(20, screen_height - 40, 1);

fun message_callback(text) {
    # 只显示重要消息，隐藏技术细节
    if (text.SubString(0, 5) == "ERROR" || 
        text.SubString(0, 7) == "WARNING" ||
        text.SubString(0, 4) == "FAIL") {
        message_image = Image.Text(text, 14, 1, 0.3, 0.3);
        message_sprite.SetImage(message_image);
    }
}

Plymouth.SetMessageFunction(message_callback);

# 密码提示函数（如果需要）
fun question_callback(prompt, entry) {
    question_image = Image.Text(prompt, 16, 1, 1, 1);
    question_sprite = Sprite(question_image);
    question_sprite.SetPosition(20, 20, 2);
}

Plymouth.SetQuestionFunction(question_callback);

# 显示密码提示函数
fun display_password_callback(prompt, bullets) {
    prompt_image = Image.Text(prompt, 16, 1, 1, 1);
    prompt_sprite = Sprite(prompt_image);
    prompt_sprite.SetPosition(20, 20, 2);
    
    # 显示密码点
    bullets_image = Image.Text(bullets, 16, 1, 1, 1);
    bullets_sprite = Sprite(bullets_image);
    bullets_sprite.SetPosition(20, 40, 2);
}

Plymouth.SetDisplayPasswordFunction(display_password_callback);

# 退出时的清理
fun quit_callback() {
    # 淡出效果
    for (i = 0; i < 50; i++) {
        opacity = 1.0 - (i / 50.0);
        logo.sprite.SetOpacity(opacity);
        system_sprite.SetOpacity(opacity);
        version_sprite.SetOpacity(opacity);
        progress_bg_sprite.SetOpacity(opacity);
        progress_fg_sprite.SetOpacity(opacity);
        status_sprite.SetOpacity(opacity);
        Plymouth.Delay(20);
    }
}

Plymouth.SetQuitFunction(quit_callback);
EOF
    
    log_info "主题脚本创建完成"
}

# 创建或下载图片资源
create_theme_assets() {
    log_step "创建主题资源..."
    
    # 创建简单的logo图片（如果有ImageMagick）
    if command -v convert &> /dev/null; then
        # 创建logo
        convert -size 200x80 xc:transparent \
            -fill '#27ae60' -pointsize 36 -gravity center \
            -annotate +0-10 '智能切竹机' \
            -fill '#3498db' -pointsize 16 -gravity center \
            -annotate +0+15 'Smart Bamboo Cutter' \
            "$THEME_DIR/logo.png"
        
        log_info "Logo图片创建完成"
    else
        log_warning "ImageMagick未安装，跳过图片创建"
    fi
    
    # 创建背景图片（可选）
    if command -v convert &> /dev/null; then
        convert -size 1920x1080 \
            -gradient "#2c3e50-#34495e" \
            "$THEME_DIR/background.png"
        
        log_info "背景图片创建完成"
    fi
}

# 应用主题
apply_theme() {
    log_step "应用自定义主题..."
    
    # 设置新主题为默认
    plymouth-set-default-theme $THEME_NAME
    
    # 更新initramfs
    case $DISTRO in
        "ubuntu"|"debian")
            update-initramfs -u
            ;;
        "fedora"|"centos"|"rhel")
            dracut --force
            ;;
        "arch"|"manjaro")
            mkinitcpio -P
            ;;
    esac
    
    log_info "主题应用完成"
}

# 配置快速启动
configure_fast_boot() {
    log_step "配置快速启动优化..."
    
    # 添加内核参数以隐藏启动信息和加速启动
    local grub_config="/etc/default/grub"
    
    if [ -f "$grub_config" ]; then
        # 备份GRUB配置
        cp "$grub_config" "${grub_config}.backup.$(date +%Y%m%d_%H%M%S)"
        
        # 修改GRUB配置
        sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="[^"]*/& quiet splash loglevel=3 rd.systemd.show_status=false rd.udev.log-priority=3 vt.global_cursor_default=0/' "$grub_config"
        
        # 设置启动超时
        sed -i 's/GRUB_TIMEOUT=.*/GRUB_TIMEOUT=1/' "$grub_config"
        
        # 更新GRUB
        case $DISTRO in
            "ubuntu"|"debian")
                update-grub
                ;;
            "fedora"|"centos"|"rhel"|"arch"|"manjaro")
                grub-mkconfig -o /boot/grub/grub.cfg
                ;;
        esac
        
        log_info "GRUB配置优化完成"
    else
        log_warning "未找到GRUB配置文件，跳过启动优化"
    fi
}

# 配置Plymouth设置
configure_plymouth_settings() {
    log_step "配置Plymouth高级设置..."
    
    # 创建Plymouth配置文件
    local plymouth_config="/etc/plymouth/plymouthd.conf"
    
    cat > "$plymouth_config" << EOF
[Daemon]
Theme=$THEME_NAME
ShowDelay=0
DeviceTimeout=8
EOF
    
    log_info "Plymouth配置完成"
}

# 测试主题
test_theme() {
    log_step "测试主题..."
    
    # 检查主题是否正确安装
    if plymouth-set-default-theme --list | grep -q "$THEME_NAME"; then
        log_info "主题安装成功: $THEME_NAME"
        
        # 可选：预览主题（需要在控制台环境下）
        read -p "是否预览启动主题？(需要切换到控制台)(y/N): " preview_choice
        if [[ $preview_choice =~ ^[Yy]$ ]]; then
            plymouth show-splash &
            sleep 5
            plymouth hide-splash
            log_info "主题预览完成"
        fi
    else
        log_error "主题安装失败"
        return 1
    fi
}

# 创建主题卸载脚本
create_uninstall_script() {
    log_step "创建主题卸载脚本..."
    
    local uninstall_script="scripts/uninstall_splash.sh"
    
    cat > "$uninstall_script" << EOF
#!/bin/bash
# 智能切竹机启动主题卸载脚本

echo "开始卸载自定义启动主题..."

# 恢复默认主题
sudo plymouth-set-default-theme ubuntu-logo 2>/dev/null || \\
sudo plymouth-set-default-theme fedora-logo 2>/dev/null || \\
sudo plymouth-set-default-theme arch-logo 2>/dev/null || \\
sudo plymouth-set-default-theme default

# 删除自定义主题
sudo rm -rf /usr/share/plymouth/themes/$THEME_NAME

# 恢复GRUB配置
if [ -f /etc/default/grub.backup.* ]; then
    latest_backup=\$(ls -t /etc/default/grub.backup.* | head -1)
    sudo cp "\$latest_backup" /etc/default/grub
fi

# 更新配置
case \$(lsb_release -si 2>/dev/null || echo "Unknown") in
    "Ubuntu"|"Debian")
        sudo update-grub
        sudo update-initramfs -u
        ;;
    "Fedora"|"CentOS"|"RedHat")
        sudo grub-mkconfig -o /boot/grub/grub.cfg
        sudo dracut --force
        ;;
    "Arch"|"Manjaro")
        sudo grub-mkconfig -o /boot/grub/grub.cfg
        sudo mkinitcpio -P
        ;;
esac

echo "自定义启动主题已卸载"
EOF
    
    chmod +x "$uninstall_script"
    log_info "卸载脚本创建完成: $uninstall_script"
}

# 主函数
main() {
    echo "========================================"
    echo "智能切竹机 自定义启动动画配置"
    echo "========================================"
    
    check_privileges
    detect_system
    
    echo
    echo "即将配置以下功能："
    echo "- 创建工业风格Plymouth主题"
    echo "- 隐藏系统启动信息"
    echo "- 优化启动速度"
    echo "- 自定义品牌化启动体验"
    echo
    
    read -p "是否继续？(y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        log_info "配置已取消"
        exit 0
    fi
    
    create_theme_directory
    create_theme_config
    create_theme_script
    create_theme_assets
    configure_plymouth_settings
    apply_theme
    configure_fast_boot
    create_uninstall_script
    
    if test_theme; then
        echo
        log_info "======================================"
        log_info "自定义启动动画配置完成！"
        log_info "======================================"
        echo
        echo "配置详情："
        echo "- 主题名称: $THEME_NAME"
        echo "- 主题路径: $THEME_DIR"
        echo "- 卸载脚本: scripts/uninstall_splash.sh"
        echo
        echo "重启系统后将看到新的启动动画"
        echo
        read -p "是否立即重启以查看效果？(y/N): " reboot_confirm
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