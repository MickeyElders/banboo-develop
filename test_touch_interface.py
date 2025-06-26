#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机 - 触摸界面测试启动器
测试阶段独立运行脚本，不依赖PLC和后端系统

使用方法:
python3 test_touch_interface.py
"""

import sys
import os

# 添加项目路径到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def main():
    """测试模式主函数"""
    try:
        # 设置环境变量
        os.environ['GDK_BACKEND'] = 'wayland,x11'
        
        print("智能切竹机触摸界面 - 测试模式")
        print("=============================")
        print("当前运行在测试模式，使用模拟数据")
        print("不需要连接PLC或其他硬件设备")
        print("=============================")
        
        # 导入并启动触摸界面
        from src.gui.touch_interface import BambooTouchInterface
        
        app = BambooTouchInterface()
        exit_code = app.run(sys.argv)
        
        return exit_code
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保安装了以下依赖:")
        print("- python3-gi")
        print("- python3-gi-cairo") 
        print("- gir1.2-gtk-4.0")
        print("- gir1.2-adw-1")
        print()
        print("Ubuntu/Debian安装命令:")
        print("sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0 gir1.2-adw-1")
        return 1
        
    except Exception as e:
        print(f"启动失败: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 