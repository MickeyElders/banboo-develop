#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Bamboo Recognition System - GTK3界面
基于GTK3的工业级竹子识别切割系统界面
"""

import gi
gi.require_version('', '3.0')
from gi.repository import Gtk, Gdk, GLib, cairo, GdkPixbuf
import time
import threading
import random
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw
import sys
import os

class BambooSystemUI:
    def __init__(self):
        """初始化GTK3界面系统"""
        
        # GTK3主题色彩配置
        self.colors = {
            'bg_main': '#1E1E1E',
            'bg_panel': '#2D2D2D', 
            'accent': '#FF6B35',
            'success': '#4CAF50',
            'warning': '#FFC107',
            'error': '#F44336',
            'text_primary': '#FFFFFF',
            'text_secondary': '#B0B0B0',
            'border': '#404040',
            'modbus_blue': '#2196F3',
            'emergency': '#FF1744',
            'power': '#9C27B0',
            'jetson_green': '#76B900'
        }
        
        # 系统状态数据
        self.system_state = {
            'is_running': False,
            'current_step': 1,
            'heartbeat_counter': 12345,
            'x_coordinate': 245.8,
            'cut_quality': 0,
            'blade_selection': 3,
            'plc_command': 0,
            'coordinate_ready': 0
        }
        
        # Jetson Orin NX性能数据
        self.jetson_data = {
            'cpu': {'usage': 45, 'freq': 1500, 'temp': 52},
            'gpu': {'usage': 32, 'freq': 624, 'temp': 49},
            'memory': {'used': 2.1, 'total': 8.0},
            'thermal': {'temp': 45},
            'fan': {'speed': 2150},
            'power': {'draw': 8.2, 'voltage': 5.1},
            'emc': {'freq': 2133},
            'storage': {'used': 45, 'total': 128}
        }
        
        # AI推理模型数据
        self.ai_data = {
            'inference_time': 15.3,
            'detection_fps': 28.5,
            'total_detections': 15432,
            'today_detections': 89,
            'accuracy': 94.2
        }
        
        # 状态映射表
        self.status_maps = {
            'system_status': ['Stopped', 'Running', 'Error', 'Paused', 'Emergency', 'Maintenance'],
            'plc_command': ['None', 'Feed Detection', 'Cut Prepare', 'Cut Complete', 'Start Feed', 'Pause', 'Emergency', 'Resume'],
            'cut_quality': ['Normal', 'Abnormal'],
            'blade_type': ['None', 'Blade 1', 'Blade 2', 'Dual Blade'],
            'health_status': ['Normal', 'Warning', 'Error', 'Critical']
        }
        
        # 工作流程步骤
        self.workflow_steps = [
            {'id': 1, 'name': 'Feed Detection', 'plc_cmd': 1},
            {'id': 2, 'name': 'Vision Recognition', 'plc_cmd': 0},
            {'id': 3, 'name': 'Coordinate Transfer', 'plc_cmd': 0},
            {'id': 4, 'name': 'Cut Prepare', 'plc_cmd': 2},
            {'id': 5, 'name': 'Execute Cut', 'plc_cmd': 3}
        ]
        
        # 界面组件引用
        self.widgets = {}
        self.progress_bars = {}
        self.register_values = {}
        self.workflow_buttons = []
        self.blade_buttons = []
        
        # 控制标志
        self.running = True
        
        # 创建GTK3界面
        self.create_gtk_ui()
        
        # 启动后台更新线程
        self.start_update_threads()
    
    def create_gtk_ui(self):
        """创建GTK3主界面"""
        # 创建主窗口
        self.window = Gtk.Window()
        self.window.set_title("AI Bamboo Recognition System v2.1 - GTK3")
        self.window.set_default_size(1280, 800)
        self.window.set_position(Gtk.WindowPosition.CENTER)
        self.window.connect("destroy", self.on_window_destroy)
        
        # 设置CSS样式
        self.setup_css_styling()
        
        # 创建主容器
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.window.add(main_box)
        
        # 创建头部面板
        header_frame = self.create_header_panel()
        main_box.pack_start(header_frame, False, False, 0)
        
        # 创建中间内容区域
        content_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        content_box.set_margin_left(10)
        content_box.set_margin_right(10)
        content_box.set_margin_top(10)
        content_box.set_margin_bottom(10)
        main_box.pack_start(content_box, True, True, 0)
        
        # 创建摄像头面板
        camera_frame = self.create_camera_panel()
        content_box.pack_start(camera_frame, True, True, 0)
        
        # 创建控制面板
        control_frame = self.create_control_panel()
        content_box.pack_start(control_frame, False, False, 0)
        
        # 创建底部控制栏
        footer_frame = self.create_footer_panel()
        main_box.pack_start(footer_frame, False, False, 0)
        
        # 显示窗口
        self.window.show_all()
    
    def setup_css_styling(self):
        """设置GTK3 CSS样式"""
        css_provider = Gtk.CssProvider()
        css_data = f"""
        .main-window {{
            background-color: {self.colors['bg_main']};
            color: {self.colors['text_primary']};
        }}
        
        .header-panel {{
            background-color: {self.colors['bg_panel']};
            border: 2px solid {self.colors['border']};
            padding: 10px;
        }}
        
        .control-panel {{
            background-color: {self.colors['bg_panel']};
            border: 2px solid {self.colors['border']};
            border-radius: 8px;
            padding: 10px;
        }}
        
        .camera-panel {{
            background-color: {self.colors['bg_panel']};
            border: 2px solid {self.colors['border']};
            border-radius: 8px;
            padding: 10px;
        }}
        
        .section-title {{
            color: {self.colors['accent']};
            font-weight: bold;
            font-size: 14px;
        }}
        
        .value-label {{
            color: {self.colors['text_primary']};
            font-weight: bold;
        }}
        
        .secondary-label {{
            color: {self.colors['text_secondary']};
        }}
        
        .success-button {{
            background-color: {self.colors['success']};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }}
        
        .warning-button {{
            background-color: {self.colors['warning']};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }}
        
        .error-button {{
            background-color: {self.colors['error']};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }}
        
        .emergency-button {{
            background-color: {self.colors['emergency']};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }}
        
        .workflow-button {{
            background-color: {self.colors['bg_main']};
            border: 1px solid {self.colors['border']};
            border-radius: 15px;
            padding: 5px 10px;
        }}
        
        .workflow-active {{
            background-color: {self.colors['accent']};
            color: white;
        }}
        
        .workflow-completed {{
            background-color: {self.colors['success']};
            color: white;
        }}
        """
        
        css_provider.load_from_data(css_data.encode())
        screen = Gdk.Screen.get_default()
        style_context = Gtk.StyleContext()
        style_context.add_provider_for_screen(screen, css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    
    def create_header_panel(self):
        """创建头部面板"""
        frame = Gtk.Frame()
        frame.get_style_context().add_class("header-panel")
        
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        header_box.set_margin_top(10)
        header_box.set_margin_bottom(10)
        header_box.set_margin_left(15)
        header_box.set_margin_right(15)
        frame.add(header_box)
        
        # 系统标题
        title_label = Gtk.Label("AI Bamboo Recognition Cutting System v2.1")
        title_label.get_style_context().add_class("section-title")
        header_box.pack_start(title_label, False, False, 0)
        
        # 工作流程状态指示器
        workflow_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        header_box.set_center_widget(workflow_box)
        
        for step in self.workflow_steps:
            btn = Gtk.Button(label=step['name'][:8])
            btn.get_style_context().add_class("workflow-button")
            btn.set_size_request(70, 30)
            workflow_box.pack_start(btn, False, False, 0)
            self.workflow_buttons.append(btn)
        
        # 心跳监控
        heartbeat_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        header_box.pack_end(heartbeat_box, False, False, 0)
        
        self.widgets['heartbeat_label'] = Gtk.Label(f"Heartbeat: {self.system_state['heartbeat_counter']}")
        self.widgets['heartbeat_label'].get_style_context().add_class("value-label")
        heartbeat_box.pack_start(self.widgets['heartbeat_label'], False, False, 0)
        
        self.widgets['response_label'] = Gtk.Label("Response: 12ms")
        self.widgets['response_label'].get_style_context().add_class("secondary-label")
        heartbeat_box.pack_start(self.widgets['response_label'], False, False, 0)
        
        return frame
    
    def create_camera_panel(self):
        """创建摄像头面板"""
        frame = Gtk.Frame()
        frame.get_style_context().add_class("camera-panel")
        frame.set_size_request(800, 400)
        
        camera_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        camera_box.set_margin_top(10)
        camera_box.set_margin_bottom(10)
        camera_box.set_margin_left(15)
        camera_box.set_margin_right(15)
        frame.add(camera_box)
        
        # 摄像头标题和信息
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        camera_box.pack_start(header_box, False, False, 0)
        
        title_label = Gtk.Label("Real-time Detection View")
        title_label.get_style_context().add_class("section-title")
        header_box.pack_start(title_label, False, False, 0)
        
        info_label = Gtk.Label("Rail Range: 0-1000.0mm | Precision: 0.1mm | FPS: 28.5")
        info_label.get_style_context().add_class("secondary-label")
        header_box.pack_end(info_label, False, False, 0)
        
        # 摄像头画布
        self.widgets['camera_canvas'] = Gtk.DrawingArea()
        self.widgets['camera_canvas'].set_size_request(750, 250)
        self.widgets['camera_canvas'].connect("draw", self.on_camera_draw)
        camera_box.pack_start(self.widgets['camera_canvas'], True, True, 0)
        
        # 坐标显示面板
        coord_frame = Gtk.Frame()
        coord_frame.set_size_request(-1, 60)
        coord_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        coord_box.set_margin_top(10)
        coord_box.set_margin_bottom(10)
        coord_box.set_margin_left(15)
        coord_box.set_margin_right(15)
        coord_frame.add(coord_box)
        
        # X坐标显示
        x_coord_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        coord_box.pack_start(x_coord_box, False, False, 20)
        
        x_label = Gtk.Label("X Coordinate")
        x_label.get_style_context().add_class("secondary-label")
        x_coord_box.pack_start(x_label, False, False, 0)
        
        self.widgets['x_coord_value'] = Gtk.Label("245.8mm")
        self.widgets['x_coord_value'].get_style_context().add_class("value-label")
        x_coord_box.pack_start(self.widgets['x_coord_value'], False, False, 0)
        
        # 切割质量显示
        quality_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        coord_box.pack_start(quality_box, False, False, 20)
        
        quality_label = Gtk.Label("Cut Quality")
        quality_label.get_style_context().add_class("secondary-label")
        quality_box.pack_start(quality_label, False, False, 0)
        
        self.widgets['quality_value'] = Gtk.Label("Normal")
        self.widgets['quality_value'].get_style_context().add_class("value-label")
        quality_box.pack_start(self.widgets['quality_value'], False, False, 0)
        
        # 刀具选择显示
        blade_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        coord_box.pack_end(blade_box, False, False, 20)
        
        blade_label = Gtk.Label("Blade Selection")
        blade_label.get_style_context().add_class("secondary-label")
        blade_box.pack_start(blade_label, False, False, 0)
        
        self.widgets['blade_value'] = Gtk.Label("Dual Blade")
        self.widgets['blade_value'].get_style_context().add_class("value-label")
        blade_box.pack_start(self.widgets['blade_value'], False, False, 0)
        
        camera_box.pack_start(coord_frame, False, False, 0)
        
        return frame
    
    def create_control_panel(self):
        """创建控制面板"""
        frame = Gtk.Frame()
        frame.get_style_context().add_class("control-panel")
        frame.set_size_request(380, 600)
        
        # 创建滚动窗口
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        frame.add(scrolled)
        
        control_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        control_box.set_margin_top(10)
        control_box.set_margin_bottom(10)
        control_box.set_margin_left(10)
        control_box.set_margin_right(10)
        scrolled.add(control_box)
        
        # Modbus寄存器部分
        modbus_section = self.create_modbus_section()
        control_box.pack_start(modbus_section, False, False, 0)
        
        # PLC通信部分
        plc_section = self.create_plc_section()
        control_box.pack_start(plc_section, False, False, 0)
        
        # Jetson系统信息部分
        jetson_section = self.create_jetson_section()
        control_box.pack_start(jetson_section, False, False, 0)
        
        # AI模型状态部分
        ai_section = self.create_ai_section()
        control_box.pack_start(ai_section, False, False, 0)
        
        return frame
    
    def create_modbus_section(self):
        """创建Modbus寄存器部分"""
        frame = Gtk.Frame()
        frame.set_label("Modbus Registers")
        
        grid = Gtk.Grid()
        grid.set_margin_top(10)
        grid.set_margin_bottom(10)
        grid.set_margin_left(10)
        grid.set_margin_right(10)
        grid.set_row_spacing(5)
        grid.set_column_spacing(10)
        frame.add(grid)
        
        registers = [
            ("40001", "System Status", "reg_40001"),
            ("40002", "PLC Command", "reg_40002"),
            ("40003", "Coord Ready", "reg_40003"),
            ("40004", "X Coordinate", "reg_40004"),
            ("40006", "Cut Quality", "reg_40006"),
            ("40007", "Heartbeat", "reg_40007"),
            ("40009", "Blade Number", "reg_40009"),
            ("40010", "Health Status", "reg_40010")
        ]
        
        for i, (addr, desc, key) in enumerate(registers):
            # 地址
            addr_label = Gtk.Label(addr)
            addr_label.get_style_context().add_class("value-label")
            addr_label.set_xalign(0)
            grid.attach(addr_label, 0, i, 1, 1)
            
            # 描述
            desc_label = Gtk.Label(desc)
            desc_label.get_style_context().add_class("secondary-label")
            desc_label.set_xalign(0)
            grid.attach(desc_label, 1, i, 1, 1)
            
            # 值
            value_label = Gtk.Label("0")
            value_label.get_style_context().add_class("value-label")
            value_label.set_xalign(1)
            grid.attach(value_label, 2, i, 1, 1)
            
            self.register_values[key] = value_label
        
        return frame
    
    def create_plc_section(self):
        """创建PLC通信部分"""
        frame = Gtk.Frame()
        frame.set_label("PLC Communication")
        
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_left(10)
        vbox.set_margin_right(10)
        frame.add(vbox)
        
        # PLC状态信息
        status_items = [
            ("Connection:", "Connected"),
            ("PLC Address:", "192.168.1.100"),
            ("Response Time:", "12ms"),
            ("Total Cuts:", "1,247")
        ]
        
        for label_text, value_text in status_items:
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            vbox.pack_start(hbox, False, False, 0)
            
            label = Gtk.Label(label_text)
            label.get_style_context().add_class("secondary-label")
            label.set_xalign(0)
            hbox.pack_start(label, False, False, 0)
            
            value = Gtk.Label(value_text)
            value.get_style_context().add_class("value-label")
            value.set_xalign(1)
            hbox.pack_end(value, False, False, 0)
        
        # 刀具选择按钮
        blade_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        blade_box.set_margin_top(10)
        vbox.pack_start(blade_box, False, False, 0)
        
        blade_names = ["Blade 1", "Blade 2", "Dual Blade"]
        for i, name in enumerate(blade_names):
            btn = Gtk.Button(label=name)
            btn.connect("clicked", self.on_blade_select, i)
            blade_box.pack_start(btn, True, True, 0)
            self.blade_buttons.append(btn)
            
            # 设置双刀为默认选择
            if i == 2:
                btn.get_style_context().add_class("success-button")
        
        return frame
    
    def create_jetson_section(self):
        """创建Jetson系统信息部分"""
        frame = Gtk.Frame()
        frame.set_label("Jetson Orin NX 16GB")
        
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_left(10)
        vbox.set_margin_right(10)
        frame.add(vbox)
        
        # 性能模式标识
        mode_label = Gtk.Label("15W")
        mode_label.get_style_context().add_class("value-label")
        mode_label.set_xalign(1)
        vbox.pack_start(mode_label, False, False, 0)
        
        # 进度条
        progress_items = [
            ("CPU (6-core ARM Cortex-A78AE)", "cpu"),
            ("GPU (1024-core NVIDIA Ampere)", "gpu"),
            ("Memory (LPDDR5)", "memory")
        ]
        
        for title, key in progress_items:
            # 标题和百分比
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            vbox.pack_start(hbox, False, False, 0)
            
            title_label = Gtk.Label(title)
            title_label.get_style_context().add_class("secondary-label")
            title_label.set_xalign(0)
            hbox.pack_start(title_label, False, False, 0)
            
            percent_label = Gtk.Label("45%")
            percent_label.get_style_context().add_class("value-label")
            percent_label.set_xalign(1)
            hbox.pack_end(percent_label, False, False, 0)
            
            # 进度条
            progress = Gtk.ProgressBar()
            progress.set_fraction(0.45)
            vbox.pack_start(progress, False, False, 0)
            
            self.progress_bars[key] = {'bar': progress, 'label': percent_label}
        
        # 系统信息网格
        grid = Gtk.Grid()
        grid.set_row_spacing(5)
        grid.set_column_spacing(15)
        grid.set_margin_top(10)
        vbox.pack_start(grid, False, False, 0)
        
        info_items = [
            ("CPU Freq:", "1.5GHz"), ("GPU Freq:", "624MHz"), ("EMC Freq:", "2133MHz"),
            ("CPU Temp:", "52°C"), ("GPU Temp:", "49°C"), ("Thermal:", "45°C"),
            ("Fan Speed:", "2150RPM"), ("Power:", "8.2W"), ("Voltage:", "5.1V")
        ]
        
        for i, (label_text, value_text) in enumerate(info_items):
            row = i // 3
            col = (i % 3) * 2
            
            label = Gtk.Label(label_text)
            label.get_style_context().add_class("secondary-label")
            label.set_xalign(0)
            grid.attach(label, col, row, 1, 1)
            
            value = Gtk.Label(value_text)
            value.get_style_context().add_class("value-label")
            value.set_xalign(0)
            grid.attach(value, col + 1, row, 1, 1)
        
        # 版本信息
        version_label = Gtk.Label("JetPack 6.0 | CUDA 12.2 | TensorRT 8.6.1\nPython 3.10.12 | PyTorch 2.1.0")
        version_label.get_style_context().add_class("secondary-label")
        version_label.set_line_wrap(True)
        vbox.pack_start(version_label, False, False, 0)
        
        return frame
    
    def create_ai_section(self):
        """创建AI模型状态部分"""
        frame = Gtk.Frame()
        frame.set_label("AI Model Status")
        
        grid = Gtk.Grid()
        grid.set_margin_top(10)
        grid.set_margin_bottom(10)
        grid.set_margin_left(10)
        grid.set_margin_right(10)
        grid.set_row_spacing(5)
        grid.set_column_spacing(10)
        frame.add(grid)
        
        ai_items = [
            ("Model Version:", "YOLOv8n"),
            ("Inference Time:", "15.3ms"),
            ("Confidence:", "0.85"),
            ("Detection Accuracy:", "94.2%"),
            ("Total Detections:", "15,432"),
            ("Today Detections:", "89")
        ]
        
        for i, (label_text, value_text) in enumerate(ai_items):
            label = Gtk.Label(label_text)
            label.get_style_context().add_class("secondary-label")
            label.set_xalign(0)
            grid.attach(label, 0, i, 1, 1)
            
            value = Gtk.Label(value_text)
            value.get_style_context().add_class("value-label")
            value.set_xalign(1)
            grid.attach(value, 1, i, 1, 1)
        
        return frame
    
    def create_footer_panel(self):
        """创建底部控制面板"""
        frame = Gtk.Frame()
        frame.get_style_context().add_class("header-panel")
        
        footer_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        footer_box.set_margin_top(10)
        footer_box.set_margin_bottom(10)
        footer_box.set_margin_left(15)
        footer_box.set_margin_right(15)
        frame.add(footer_box)
        
        # 控制按钮
        control_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        footer_box.pack_start(control_box, False, False, 0)
        
        self.widgets['start_btn'] = Gtk.Button(label="START")
        self.widgets['start_btn'].get_style_context().add_class("success-button")
        self.widgets['start_btn'].connect("clicked", self.on_start_clicked)
        control_box.pack_start(self.widgets['start_btn'], False, False, 0)
        
        self.widgets['pause_btn'] = Gtk.Button(label="PAUSE")
        self.widgets['pause_btn'].get_style_context().add_class("warning-button")
        self.widgets['pause_btn'].connect("clicked", self.on_pause_clicked)
        control_box.pack_start(self.widgets['pause_btn'], False, False, 0)
        
        self.widgets['stop_btn'] = Gtk.Button(label="STOP")
        self.widgets['stop_btn'].get_style_context().add_class("error-button")
        self.widgets['stop_btn'].connect("clicked", self.on_stop_clicked)
        control_box.pack_start(self.widgets['stop_btn'], False, False, 0)
        
        # 状态信息
        status_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        footer_box.set_center_widget(status_box)
        
        self.widgets['process_label'] = Gtk.Label("Current Process: Feed Detection")
        self.widgets['process_label'].get_style_context().add_class("secondary-label")
        status_box.pack_start(self.widgets['process_label'], False, False, 0)
        
        self.widgets['stats_label'] = Gtk.Label("Last Cut: 14:25:33 | Today: 89 cuts | Efficiency: 94.2%")
        self.widgets['stats_label'].get_style_context().add_class("secondary-label")
        status_box.pack_start(self.widgets['stats_label'], False, False, 0)
        
        # 紧急按钮
        emergency_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        footer_box.pack_end(emergency_box, False, False, 0)
        
        self.widgets['emergency_btn'] = Gtk.Button(label="EMERGENCY")
        self.widgets['emergency_btn'].get_style_context().add_class("emergency-button")
        self.widgets['emergency_btn'].connect("clicked", self.on_emergency_clicked)
        emergency_box.pack_start(self.widgets['emergency_btn'], False, False, 0)
        
        self.widgets['power_btn'] = Gtk.Button(label="POWER")
        self.widgets['power_btn'].set_size_request(80, -1)
        self.widgets['power_btn'].connect("clicked", self.on_power_clicked)
        emergency_box.pack_start(self.widgets['power_btn'], False, False, 0)
        
        return frame
    
    def on_camera_draw(self, widget, cr):
        """摄像头画布绘制回调"""
        allocation = widget.get_allocation()
        width = allocation.width
        height = allocation.height
        
        # 绘制黑色背景
        cr.set_source_rgb(0, 0, 0)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        
        # 绘制网格线
        cr.set_source_rgba(0.3, 0.3, 0.3, 0.5)
        cr.set_line_width(1)
        
        # 垂直网格线
        for i in range(1, 10):
            x = width * i / 10
            cr.move_to(x, 0)
            cr.line_to(x, height)
            cr.stroke()
        
        # 水平网格线
        for i in range(1, 6):
            y = height * i / 6
            cr.move_to(0, y)
            cr.line_to(width, y)
            cr.stroke()
        
        # 绘制文本信息
        cr.set_source_rgb(0.7, 0.7, 0.7)
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(14)
        
        text_lines = [
            "Bamboo Detection View",
            "1280 x 720 | YOLOv8 Inference", 
            f"Inference Time: {self.ai_data['inference_time']:.1f}ms"
        ]
        
        for i, line in enumerate(text_lines):
            text_extents = cr.text_extents(line)
            x = (width - text_extents.width) / 2
            y = height / 2 - 20 + i * 20
            cr.move_to(x, y)
            cr.show_text(line)
        
        # 绘制切割位置指示器
        if hasattr(self, 'system_state'):
            position_percent = self.system_state['x_coordinate'] / 1000.0
            indicator_x = width * 0.1 + (width * 0.8) * position_percent
            
            cr.set_source_rgb(1.0, 0.2, 0.2)  # 红色指示线
            cr.set_line_width(3)
            cr.move_to(indicator_x, height - 40)
            cr.line_to(indicator_x, height - 10)
            cr.stroke()
    
    def start_update_threads(self):
        """启动后台更新线程"""
        # 心跳更新线程 (50Hz)
        heartbeat_thread = threading.Thread(target=self.heartbeat_update_loop)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        
        # 系统信息更新线程 (2s间隔)
        system_thread = threading.Thread(target=self.system_update_loop)
        system_thread.daemon = True
        system_thread.start()
        
        # 工作流程模拟线程
        workflow_thread = threading.Thread(target=self.workflow_simulation_loop)
        workflow_thread.daemon = True
        workflow_thread.start()
    
    def heartbeat_update_loop(self):
        """心跳更新循环"""
        while self.running:
            self.system_state['heartbeat_counter'] += 1
            if self.system_state['heartbeat_counter'] > 4294967295:
                self.system_state['heartbeat_counter'] = 0
            
            GLib.idle_add(self.update_heartbeat_ui)
            time.sleep(0.02)  # 50Hz
    
    def system_update_loop(self):
        """系统信息更新循环"""
        while self.running:
            self.update_system_data()
            GLib.idle_add(self.update_system_ui)
            time.sleep(2.0)
    
    def workflow_simulation_loop(self):
        """工作流程模拟循环"""
        while self.running:
            if self.system_state['is_running']:
                self.simulate_workflow_step()
            time.sleep(1.0)
    
    def update_system_data(self):
        """更新系统数据"""
        # 更新Jetson数据
        self.jetson_data['cpu']['usage'] = 40 + random.random() * 20
        self.jetson_data['gpu']['usage'] = 25 + random.random() * 25
        self.jetson_data['cpu']['temp'] = 45 + random.random() * 15
        self.jetson_data['gpu']['temp'] = 40 + random.random() * 20
        
        # 更新AI数据
        self.ai_data['inference_time'] = 12 + random.random() * 8
        self.ai_data['detection_fps'] = 25 + random.random() * 8
    
    def update_heartbeat_ui(self):
        """更新心跳UI"""
        self.widgets['heartbeat_label'].set_text(f"Heartbeat: {self.system_state['heartbeat_counter']}")
        self.widgets['camera_canvas'].queue_draw()  # 重绘摄像头画布
        return False
    
    def update_system_ui(self):
        """更新系统UI"""
        # 更新进度条
        cpu_usage = int(self.jetson_data['cpu']['usage'])
        gpu_usage = int(self.jetson_data['gpu']['usage'])
        memory_usage = int(self.jetson_data['memory']['used'] / self.jetson_data['memory']['total'] * 100)
        
        self.progress_bars['cpu']['bar'].set_fraction(cpu_usage / 100.0)
        self.progress_bars['cpu']['label'].set_text(f"{cpu_usage}%")
        
        self.progress_bars['gpu']['bar'].set_fraction(gpu_usage / 100.0)
        self.progress_bars['gpu']['label'].set_text(f"{gpu_usage}%")
        
        self.progress_bars['memory']['bar'].set_fraction(memory_usage / 100.0)
        self.progress_bars['memory']['label'].set_text(f"{memory_usage}%")
        
        # 更新坐标显示
        self.widgets['x_coord_value'].set_text(f"{self.system_state['x_coordinate']:.1f}mm")
        
        # 更新Modbus寄存器
        self.update_modbus_registers()
        
        return False
    
    def update_modbus_registers(self):
        """更新Modbus寄存器值"""
        if hasattr(self, 'register_values'):
            self.register_values['reg_40001'].set_text(str(1 if self.system_state['is_running'] else 0))
            self.register_values['reg_40002'].set_text(str(self.system_state['plc_command']))
            self.register_values['reg_40003'].set_text(str(self.system_state['coordinate_ready']))
            self.register_values['reg_40004'].set_text(str(int(self.system_state['x_coordinate'] * 10)))
            self.register_values['reg_40006'].set_text(str(self.system_state['cut_quality']))
            self.register_values['reg_40007'].set_text(str(self.system_state['heartbeat_counter']))
            self.register_values['reg_40009'].set_text(str(self.system_state['blade_selection']))
            self.register_values['reg_40010'].set_text("0")  # 健康状态正常
    
    def update_workflow_status(self, step):
        """更新工作流程状态"""
        for i, btn in enumerate(self.workflow_buttons):
            btn.get_style_context().remove_class("workflow-active")
            btn.get_style_context().remove_class("workflow-completed")
            btn.get_style_context().add_class("workflow-button")
            
            if i < step - 1:
                # 已完成
                btn.get_style_context().add_class("workflow-completed")
            elif i == step - 1:
                # 当前活动
                btn.get_style_context().add_class("workflow-active")
        
        self.system_state['current_step'] = step
        GLib.idle_add(self.widgets['process_label'].set_text, f"Current Process: {self.workflow_steps[step-1]['name']}")
    
    def simulate_workflow_step(self):
        """模拟工作流程步骤"""
        if not self.system_state['is_running']:
            return
        
        current_step = self.system_state['current_step']
        
        if current_step == 1:  # Feed Detection
            self.system_state['plc_command'] = 1
            time.sleep(3)
            if self.system_state['is_running']:
                self.update_workflow_status(2)
        elif current_step == 2:  # Vision Recognition
            self.system_state['plc_command'] = 0
            # 模拟竹子检测
            self.system_state['x_coordinate'] = 200 + random.random() * 600
            self.system_state['coordinate_ready'] = 1
            time.sleep(2)
            if self.system_state['is_running']:
                self.update_workflow_status(3)
        elif current_step == 3:  # Coordinate Transfer
            time.sleep(1)
            if self.system_state['is_running']:
                self.update_workflow_status(4)
        elif current_step == 4:  # Cut Prepare
            self.system_state['plc_command'] = 2
            time.sleep(4)
            if self.system_state['is_running']:
                self.update_workflow_status(5)
        elif current_step == 5:  # Execute Cut
            self.system_state['plc_command'] = 3
            time.sleep(3)
            if self.system_state['is_running']:
                # 完成循环，重新开始
                self.system_state['coordinate_ready'] = 0
                self.ai_data['today_detections'] += 1
                self.ai_data['total_detections'] += 1
                self.update_workflow_status(1)
    
    # 事件处理器
    def on_start_clicked(self, button):
        """启动按钮点击处理"""
        self.system_state['is_running'] = True
        self.update_workflow_status(1)
        
        # 更新按钮外观
        button.set_label("RUNNING")
        button.get_style_context().remove_class("success-button")
        button.get_style_context().add_class("warning-button")
    
    def on_pause_clicked(self, button):
        """暂停按钮点击处理"""
        if self.system_state['is_running']:
            self.system_state['is_running'] = False
            self.system_state['plc_command'] = 5  # 暂停命令
            
            # 更新启动按钮外观
            self.widgets['start_btn'].set_label("RESUME")
            self.widgets['start_btn'].get_style_context().remove_class("warning-button")
            self.widgets['start_btn'].get_style_context().add_class("success-button")
    
    def on_stop_clicked(self, button):
        """停止按钮点击处理"""
        self.system_state['is_running'] = False
        self.system_state['plc_command'] = 0
        self.system_state['coordinate_ready'] = 0
        self.update_workflow_status(1)
        
        # 更新启动按钮外观
        self.widgets['start_btn'].set_label("START")
        self.widgets['start_btn'].get_style_context().remove_class("warning-button")
        self.widgets['start_btn'].get_style_context().add_class("success-button")
    
    def on_emergency_clicked(self, button):
        """紧急停止按钮点击处理"""
        self.system_state['is_running'] = False
        self.system_state['plc_command'] = 6  # 紧急停止
        
        # 将所有工作流程步骤标记为紧急状态
        for btn in self.workflow_buttons:
            btn.get_style_context().remove_class("workflow-active")
            btn.get_style_context().remove_class("workflow-completed")
            btn.get_style_context().remove_class("workflow-button")
            btn.get_style_context().add_class("emergency-button")
        
        GLib.idle_add(self.widgets['process_label'].set_text, "Current Process: EMERGENCY STOP")
        
        # 更新启动按钮
        self.widgets['start_btn'].set_label("RECOVER")
        self.widgets['start_btn'].get_style_context().remove_class("success-button")
        self.widgets['start_btn'].get_style_context().add_class("warning-button")
    
    def on_power_clicked(self, button):
        """电源按钮点击处理"""
        # 在真实系统中，这里会启动关机流程
        pass
    
    def on_blade_select(self, button, blade_idx):
        """刀具选择处理"""
        # 更新按钮状态
        for i, btn in enumerate(self.blade_buttons):
            btn.get_style_context().remove_class("success-button")
            if i == blade_idx:
                btn.get_style_context().add_class("success-button")
        
        # 更新系统状态
        self.system_state['blade_selection'] = blade_idx + 1
        
        # 更新显示
        blade_names = ["Blade 1", "Blade 2", "Dual Blade"]
        self.widgets['blade_value'].set_text(blade_names[blade_idx])
    
    def on_window_destroy(self, widget):
        """窗口销毁处理"""
        self.running = False
        Gtk.main_quit()


def main():
    """主程序入口"""
    try:
        print("正在启动AI竹子识别系统 - GTK3界面...")
        
        # 创建并运行应用程序
        app = BambooSystemUI()
        
        print("GTK3界面初始化完成，正在启动主循环...")
        Gtk.main()
        
    except KeyboardInterrupt:
        print("\n用户中断，系统退出")
    except Exception as e:
        print(f"系统错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("AI竹子识别系统已停止")


if __name__ == "__main__":
    main()