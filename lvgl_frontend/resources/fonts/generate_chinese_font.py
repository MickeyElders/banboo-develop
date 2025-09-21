#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LVGL 中文字体生成脚本
使用 lv_font_conv 工具从 TTF/TTC 字体文件生成 LVGL 可用的 C 文件
"""

import os
import sys
import subprocess

def run_command(cmd):
    """执行命令并返回结果"""
    print(f"执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return False
    print(f"输出: {result.stdout}")
    return True

def generate_chinese_font():
    """生成中文字体"""
    # 字体文件路径
    font_file = "NotoSansCJK-Regular.ttf"
    if not os.path.exists(font_file):
        print(f"错误: 字体文件 {font_file} 不存在")
        return False
    
    # 常用中文字符范围
    chinese_chars = (
        # 基本中文字符
        "0x4E00-0x9FFF,"  # CJK统一汉字
        "0x3400-0x4DBF,"  # CJK扩展A
        "0x20000-0x2A6DF," # CJK扩展B (部分)
        # 标点符号
        "0x3000-0x303F,"  # CJK符号和标点
        "0xFF00-0xFFEF,"  # 全角ASCII、半角片假名、半角韩文字母
        # 基本ASCII
        "0x0020-0x007F,"  # 基本ASCII
        # 数字和基本符号
        "0x0030-0x0039,"  # 数字 0-9
        "0x0041-0x005A,"  # 大写字母 A-Z
        "0x0061-0x007A"   # 小写字母 a-z
    )
    
    # 生成14像素字体
    cmd_14 = (
        f"npx lv_font_conv "
        f"--font {font_file} "
        f"--size 14 "
        f"--bpp 4 "
        f"--range {chinese_chars} "
        f"--format lvgl "
        f"--output lv_font_noto_sans_cjk_14.c "
        f"--force-fast-kern-format"
    )
    
    # 生成16像素字体
    cmd_16 = (
        f"npx lv_font_conv "
        f"--font {font_file} "
        f"--size 16 "
        f"--bpp 4 "
        f"--range {chinese_chars} "
        f"--format lvgl "
        f"--output lv_font_noto_sans_cjk_16.c "
        f"--force-fast-kern-format"
    )
    
    print("开始生成LVGL中文字体文件...")
    
    # 检查 npm 是否可用
    if not run_command("npm --version"):
        print("错误: npm 未安装或不可用")
        return False
    
    # 生成字体文件
    success = True
    success &= run_command(cmd_14)
    success &= run_command(cmd_16)
    
    if success:
        print("字体生成成功!")
        print("生成的文件:")
        print("  - lv_font_noto_sans_cjk_14.c")
        print("  - lv_font_noto_sans_cjk_16.c")
    else:
        print("字体生成失败!")
    
    return success

if __name__ == "__main__":
    # 切换到字体目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = generate_chinese_font()
    sys.exit(0 if success else 1)