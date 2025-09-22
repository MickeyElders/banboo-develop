#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成轻量级中文字体脚本
只包含项目中实际使用的中文字符，大大减少文件大小
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

def generate_minimal_chinese_font():
    """生成轻量级中文字体"""
    # 字体文件路径
    font_file = "NotoSansCJK-Regular.ttf"
    if not os.path.exists(font_file):
        print(f"错误: 字体文件 {font_file} 不存在")
        return False
    
    # 项目中实际使用的中文字符（从GUI代码中提取）
    chinese_chars_text = (
        "AI竹节识别切割系统"
        "Modbus TCP"
        "实时检测画面"
        "导轨范围"
        "精度"
        "X轴导轨"
        "X坐标"
        "切割质量"
        "正常"
        "刀片选择"
        "双刀片"
        "启动系统"
        "暂停"
        "停止"
        "紧急停止"
        "关机"
        "现代化"
        "工作流程"
        "心跳监控"
        "模块状态"
        "性能监控"
        "通信统计"
        "连接时间"
        "数据包"
        "延迟"
        "吞吐量"
        "推理时间"
        "置信度"
        "识别数量"
        "错误率"
        "检测"
        "推理中"
        "竹材检测视野"
        "内存使用"
        "处理器使用率"
        "显卡使用率"
        "千米"
        "毫米"
        "秒"
        "分钟"
        "小时"
        "天"
        "百分比"
        "字节"
        "中文测试"
    )
    
    # 转换为Unicode范围字符串
    unicode_chars = []
    for char in chinese_chars_text:
        unicode_chars.append(f"0x{ord(char):X}")
    
    # 添加基本符号和数字
    basic_ranges = [
        "0x0020-0x007F",  # 基本ASCII
        "0x3000-0x303F",  # CJK符号和标点
        "0xFF00-0xFFEF",  # 全角ASCII
    ]
    
    # 合并字符范围
    all_chars = ",".join(basic_ranges + unicode_chars)
    
    # 生成14像素字体
    cmd_14 = (
        f"npx lv_font_conv "
        f"--font {font_file} "
        f"--size 14 "
        f"--bpp 4 "
        f"--range {all_chars} "
        f"--format lvgl "
        f"--output lv_font_noto_sans_cjk_14_minimal.c "
        f"--force-fast-kern-format"
    )
    
    # 生成16像素字体
    cmd_16 = (
        f"npx lv_font_conv "
        f"--font {font_file} "
        f"--size 16 "
        f"--bpp 4 "
        f"--range {all_chars} "
        f"--format lvgl "
        f"--output lv_font_noto_sans_cjk_16_minimal.c "
        f"--force-fast-kern-format"
    )
    
    print("开始生成轻量级LVGL中文字体文件...")
    print(f"包含字符数量: {len(set(chinese_chars_text))}")
    
    # 检查 npm 是否可用
    if not run_command("npm --version"):
        print("错误: npm 未安装或不可用")
        return False
    
    # 生成字体文件
    success = True
    success &= run_command(cmd_14)
    success &= run_command(cmd_16)
    
    if success:
        print("轻量级字体生成成功!")
        print("生成的文件:")
        print("  - lv_font_noto_sans_cjk_14_minimal.c")
        print("  - lv_font_noto_sans_cjk_16_minimal.c")
        
        # 显示文件大小
        for filename in ["lv_font_noto_sans_cjk_14_minimal.c", "lv_font_noto_sans_cjk_16_minimal.c"]:
            if os.path.exists(filename):
                size_mb = os.path.getsize(filename) / (1024 * 1024)
                print(f"  - {filename}: {size_mb:.2f} MB")
    else:
        print("轻量级字体生成失败!")
    
    return success

if __name__ == "__main__":
    # 切换到字体目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = generate_minimal_chinese_font()
    sys.exit(0 if success else 1)