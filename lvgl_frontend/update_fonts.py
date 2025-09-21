#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量更新所有GUI文件中的字体引用脚本
将 lv_font_montserrat_14 替换为 lv_font_noto_sans_cjk_14
"""

import os
import re

def update_font_references():
    """更新字体引用"""
    gui_files = [
        "src/gui/video_view.cpp",
        "src/gui/control_panel.cpp", 
        "src/gui/settings_page.cpp"
    ]
    
    for file_path in gui_files:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
            
        print(f"更新文件: {file_path}")
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 添加头文件包含（如果不存在）
        if 'resources/fonts/lv_font_noto_sans_cjk.h' not in content:
            # 在第一个include之后添加
            content = re.sub(
                r'(#include "gui/[^"]+\.h")',
                r'\1\n#include "resources/fonts/lv_font_noto_sans_cjk.h"',
                content,
                count=1
            )
        
        # 替换字体引用
        content = content.replace('&lv_font_montserrat_14', '&lv_font_noto_sans_cjk_14')
        content = content.replace('&lv_font_montserrat_16', '&lv_font_noto_sans_cjk_16')
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"完成更新: {file_path}")

if __name__ == "__main__":
    # 切换到LVGL前端目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    update_font_references()
    print("所有字体引用更新完成!")