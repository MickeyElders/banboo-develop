#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# 读取 README.md 作为长描述
def read_readme():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return "智能切竹机控制系统"

# 读取 requirements.txt
def read_requirements():
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except:
        return []

setup(
    name='bamboo-cut-intelligent-system',
    version='1.0.0',
    author='Bamboo Cut Development Team',
    author_email='developer@bamboo-cut.com',
    description='基于AI视觉识别的智能切竹机控制系统',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/bamboo-cut/bamboo-cut-develop',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    # 包含所有必要的数据文件
    package_data={
        '': ['*.yaml', '*.yml', '*.json', '*.md', '*.txt', '*.pt', '*.pth'],
    },
    
    # 安装主程序脚本
    scripts=[
        'main.py',
        'demo_ai_vision.py',
        'test_touch_interface.py',
        'test_model.py',
    ],
    
    # 数据文件安装配置
    data_files=[
        # 配置文件
        ('/etc/bamboo-cut', [
            'config/system_config.yaml',
        ]),
        
        # 模型文件
        ('/opt/bamboo-cut/models', [
            'models/yolov8n_bamboo_best.pt',
            'models/model_info.json',
            'models/README.md',
        ]),
        
        # systemd 服务文件
        ('/etc/systemd/system', [
            'package/bamboo-cut.service',
            'package/bamboo-cut-kiosk.service',
        ]),
        
        # 桌面文件
        ('/usr/share/applications', [
            'package/bamboo-cut.desktop',
        ]),
        
        # 启动脚本
        ('/usr/bin', [
            'package/bamboo-cut',
            'package/bamboo-cut-kiosk',
        ]),
        
        # 文档
        ('/usr/share/doc/bamboo-cut', [
            'README.md',
            'INSTALL.md',
            'docs/kiosk_setup_guide.md',
            'docs/plc_communication_spec.md',
            'docs/plc_requirements.md',
        ]),
        
        # 图标和资源
        ('/usr/share/pixmaps', [
            'package/bamboo-cut.png',
        ]),
    ],
    
    # Python 依赖
    install_requires=read_requirements(),
    
    # Python 版本要求
    python_requires='>=3.8',
    
    # 分类信息
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Manufacturing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Hardware :: Hardware Drivers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: POSIX :: Linux',
        'Environment :: X11 Applications :: GTK',
    ],
    
    # 关键词
    keywords='bamboo cutting machine AI vision YOLO industrial automation PLC',
    
    # 项目链接
    project_urls={
        'Bug Reports': 'https://github.com/bamboo-cut/bamboo-cut-develop/issues',
        'Documentation': 'https://github.com/bamboo-cut/bamboo-cut-develop/docs',
        'Source': 'https://github.com/bamboo-cut/bamboo-cut-develop',
    },
    
    # 入口点
    entry_points={
        'console_scripts': [
            'bamboo-cut=main:main',
            'bamboo-cut-demo=demo_ai_vision:main',
            'bamboo-cut-test=test_touch_interface:main',
        ],
    },
) 