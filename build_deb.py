#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机系统 DEB 包构建脚本
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

# 项目配置
PROJECT_NAME = "bamboo-cut-intelligent-system"
VERSION = "1.0.0"
ARCHITECTURE = "all"
BUILD_DIR = "build"
DEB_DIR = f"{BUILD_DIR}/{PROJECT_NAME}_{VERSION}_{ARCHITECTURE}"

def run_command(cmd, cwd=None, check=True):
    """执行系统命令"""
    print(f"执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"命令执行失败: {cmd}")
        print(f"错误输出: {result.stderr}")
        sys.exit(1)
    return result

def create_directory_structure():
    """创建 DEB 包目录结构"""
    print("创建 DEB 包目录结构...")
    
    # 清理旧的构建目录
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    
    # 创建目录结构
    directories = [
        f"{DEB_DIR}/DEBIAN",
        f"{DEB_DIR}/opt/bamboo-cut",
        f"{DEB_DIR}/etc/bamboo-cut",
        f"{DEB_DIR}/etc/systemd/system",
        f"{DEB_DIR}/usr/bin",
        f"{DEB_DIR}/usr/share/applications",
        f"{DEB_DIR}/usr/share/pixmaps",
        f"{DEB_DIR}/usr/share/doc/bamboo-cut",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def copy_debian_files():
    """复制 DEBIAN 控制文件"""
    print("复制 DEBIAN 控制文件...")
    
    debian_files = ["control", "postinst", "prerm"]
    for file in debian_files:
        src = f"package/DEBIAN/{file}"
        dst = f"{DEB_DIR}/DEBIAN/{file}"
        if os.path.exists(src):
            shutil.copy2(src, dst)
            # 设置脚本权限
            if file in ["postinst", "prerm"]:
                os.chmod(dst, 0o755)
        else:
            print(f"警告: 文件不存在 {src}")

def copy_application_files():
    """复制应用程序文件"""
    print("复制应用程序文件...")
    
    # 复制源代码
    if os.path.exists("src"):
        shutil.copytree("src", f"{DEB_DIR}/opt/bamboo-cut/src")
    
    # 复制主程序
    main_files = ["main.py", "demo_ai_vision.py", "test_touch_interface.py", "test_model.py"]
    for file in main_files:
        if os.path.exists(file):
            shutil.copy2(file, f"{DEB_DIR}/opt/bamboo-cut/")
            # 同时复制到 /usr/bin
            shutil.copy2(file, f"{DEB_DIR}/usr/bin/")
    
    # 复制配置文件
    if os.path.exists("config/system_config.yaml"):
        shutil.copy2("config/system_config.yaml", f"{DEB_DIR}/etc/bamboo-cut/")
    
    # 复制模型文件
    if os.path.exists("models"):
        shutil.copytree("models", f"{DEB_DIR}/opt/bamboo-cut/models")
    
    # 复制文档
    doc_files = ["README.md", "INSTALL.md"]
    for file in doc_files:
        if os.path.exists(file):
            shutil.copy2(file, f"{DEB_DIR}/usr/share/doc/bamboo-cut/")
    
    # 复制特定文档
    doc_source_files = [
        "docs/kiosk_setup_guide.md",
        "docs/plc_communication_spec.md", 
        "docs/plc_requirements.md"
    ]
    for file in doc_source_files:
        if os.path.exists(file):
            shutil.copy2(file, f"{DEB_DIR}/usr/share/doc/bamboo-cut/")
    
    # 复制 requirements.txt
    if os.path.exists("requirements.txt"):
        shutil.copy2("requirements.txt", f"{DEB_DIR}/usr/share/doc/bamboo-cut/")

def copy_system_files():
    """复制系统配置文件"""
    print("复制系统配置文件...")
    
    # systemd 服务文件
    service_files = ["bamboo-cut.service", "bamboo-cut-kiosk.service"]
    for file in service_files:
        src = f"package/{file}"
        dst = f"{DEB_DIR}/etc/systemd/system/{file}"
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    # 桌面文件
    if os.path.exists("package/bamboo-cut.desktop"):
        shutil.copy2("package/bamboo-cut.desktop", f"{DEB_DIR}/usr/share/applications/")
    
    # 启动脚本
    script_files = ["bamboo-cut", "bamboo-cut-kiosk"]
    for file in script_files:
        src = f"package/{file}"
        dst = f"{DEB_DIR}/usr/bin/{file}"
        if os.path.exists(src):
            shutil.copy2(src, dst)
            os.chmod(dst, 0o755)
    
    # 图标文件（如果存在）
    icon_file = "package/bamboo-cut.png"
    if os.path.exists(icon_file):
        shutil.copy2(icon_file, f"{DEB_DIR}/usr/share/pixmaps/")
    else:
        # 创建临时图标
        create_temporary_icon(f"{DEB_DIR}/usr/share/pixmaps/bamboo-cut.png")

def create_temporary_icon(icon_path):
    """创建临时图标文件"""
    print("创建临时图标文件...")
    
    # 检查是否有 ImageMagick
    result = run_command("which convert", check=False)
    if result.returncode == 0:
        # 使用 ImageMagick 创建简单图标
        cmd = f"""convert -size 128x128 xc:'#4CAF50' \
                 -font DejaVu-Sans-Bold -pointsize 24 \
                 -fill white -gravity center \
                 -annotate +0+0 '竹切' \
                 '{icon_path}'"""
        run_command(cmd, check=False)
    else:
        # 创建占位符文件
        print("ImageMagick 不可用，创建占位符图标")
        with open(icon_path, 'w') as f:
            f.write("# 占位符图标文件\n# 请替换为实际的 PNG 图标\n")

def calculate_installed_size():
    """计算安装后大小"""
    print("计算安装后大小...")
    
    total_size = 0
    for root, dirs, files in os.walk(DEB_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    
    # 转换为 KB
    size_kb = (total_size // 1024) + 1
    return size_kb

def update_control_file():
    """更新 control 文件中的安装大小"""
    print("更新 control 文件...")
    
    control_file = f"{DEB_DIR}/DEBIAN/control"
    size_kb = calculate_installed_size()
    
    if os.path.exists(control_file):
        with open(control_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 更新 Installed-Size
        content = content.replace("Installed-Size: 50000", f"Installed-Size: {size_kb}")
        
        with open(control_file, 'w', encoding='utf-8') as f:
            f.write(content)

def build_deb_package():
    """构建 DEB 包"""
    print("构建 DEB 包...")
    
    # 检查是否有 dpkg-deb
    result = run_command("which dpkg-deb", check=False)
    if result.returncode != 0:
        print("错误: dpkg-deb 不可用。请安装 dpkg-dev 包。")
        print("Ubuntu/Debian: sudo apt install dpkg-dev")
        sys.exit(1)
    
    # 构建包
    deb_file = f"{BUILD_DIR}/{PROJECT_NAME}_{VERSION}_{ARCHITECTURE}.deb"
    cmd = f"dpkg-deb --build {DEB_DIR} {deb_file}"
    run_command(cmd)
    
    print(f"DEB 包构建完成: {deb_file}")
    
    # 显示包信息
    print("\n包信息:")
    run_command(f"dpkg --info {deb_file}")
    
    print(f"\n包文件大小: {os.path.getsize(deb_file) / (1024*1024):.2f} MB")
    
    return deb_file

def main():
    """主函数"""
    print("=" * 50)
    print("智能切竹机系统 DEB 包构建工具")
    print("=" * 50)
    
    # 检查是否在项目根目录
    if not os.path.exists("main.py") or not os.path.exists("src"):
        print("错误: 请在项目根目录运行此脚本")
        sys.exit(1)
    
    try:
        # 创建目录结构
        create_directory_structure()
        
        # 复制文件
        copy_debian_files()
        copy_application_files()
        copy_system_files()
        
        # 更新控制文件
        update_control_file()
        
        # 构建包
        deb_file = build_deb_package()
        
        print("\n" + "=" * 50)
        print("构建完成！")
        print(f"DEB 包位置: {deb_file}")
        print("\n安装命令:")
        print(f"sudo dpkg -i {deb_file}")
        print("\n卸载命令:")
        print(f"sudo dpkg -r {PROJECT_NAME}")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n构建已取消")
        sys.exit(1)
    except Exception as e:
        print(f"构建失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 