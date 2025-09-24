#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Bamboo Recognition System - GTK4界面
基于GTK4的现代化工业级竹子识别切割系统界面
"""

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
from gi.repository import Gtk, Gdk, GLib, cairo, GdkPixbuf, Adw
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
        """初始化GTK4界面系统"""
        
        # GTK4主题色彩配置
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
        
        # 初始化Adwaita应用
        self.app = Adw.Application(application_id="com.bamboo.recognition")
        self.app.connect("activate", self.on_activate)
        
        # 启动后台更新线程
        self.start_update_threads()
    
    def on_activate(self, app):
        """GTK4应用激活回调"""
        # 创建主窗口
        self.window = Adw.ApplicationWindow(application=app)
        self.window.set_title("AI Bamboo Recognition System v2.1 - GTK4")
        self.window.set_default_size(1280, 800)
        self.window.connect("destroy", self.on_window_destroy)
        
        # 设置CSS样式
        self.setup_css_styling()
        
        # 创建主容器 - 使用Adwaita Clamp
        clamp = Adw.Clamp()
        clamp.set_maximum_size(1400)
        self.window.set_content(clamp)
        
        # 主要内容盒子
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        clamp.set_child(main_box)
        
        # 创建头部面板
        header_card = self.create_header_panel()
        main_box.append(header_card)
        
        # 创建中间内容区域
        content_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        content_box.set_margin_top(12)
        content_box.set_margin_bottom(12)
        content_box.set_margin_start(12)
        content_box.set_margin_end(12)
        content_box.set_hexpand(True)
        main_box.append(content_box)
        
        # 创建摄像头面板
        camera_card = self.create_camera_panel()
        content_box.append(camera_card)
        
        # 创建控制面板
        control_card = self.create_control_panel()
        content_box.append(control_card)
        
        # 创建底部控制栏
        footer_card = self.create_footer_panel()
        main_box.append(footer_card)
        
        # 显示窗口
        self.window.present()
    
    def setup_css_styling(self):
        """设置GTK4 CSS样式"""
        css_provider = Gtk.CssProvider()
        css_data = f"""
        window {{
            background-color: {self.colors['bg_main']};
            color: {self.colors['text_primary']};
        }}
        
        .header-card {{
            background-color: {self.colors['bg_panel']};
            border: 2px solid {self.colors['border']};
            border-radius: 12px;
            padding: 16px;
            margin: 12px;
        }}
        
        .control-card {{
            background-color: {self.colors['bg_panel']};
            border: 2px solid {self.colors['border']};
            border-radius: 12px;
            padding: 16px;
            min-width: 380px;
        }}
        
        .camera-card {{
            background-color: {self.colors['bg_panel']};
            border: 2px solid {self.colors['border']};
            border-radius: 12px;
            padding: 16px;
            min-height: 480px;
        }}
        
        .section-title {{
            color: {self.colors['accent']};
            font-weight: bold;
            font-size: 16px;
        }}
        
        .value-label {{
            color: {self.colors['text_primary']};
            font-weight: bold;
            font-size: 14px;
        }}
        
        .secondary-label {{
            color: {self.colors['text_secondary']};
            font-size: 12px;
        }}
        
        .success-button {{
            background-color: {self.colors['success']};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 20px;
            font-weight: bold;
        }}
        
        .warning-button {{
            background-color: {self.colors['warning']};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 20px;
            font-weight: bold;
        }}
        
        .error-button {{
            background-color: {self.colors['error']};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 20px;
            font-weight: bold;
        }}
        
        .emergency-button {{
            background-color: {self.colors['emergency']};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 20px;
            font-weight: bold;
        }}
        
        .workflow-button {{
            background-color: {self.colors['bg_main']};
            border: 2px solid {self.colors['border']};
            border-radius: 20px;
            padding: 8px 16px;
            margin: 4px;
            font-size: 11px;
        }}
        
        .workflow-active {{
            background-color: {self.colors['accent']};
            color: white;
            border-color: {self.colors['accent']};
        }}
        
        .workflow-completed {{
            background-color: {self.colors['success']};
            color: white;
            border-color: {self.colors['success']};
        }}
        
        .info-card {{
            background-color: rgba(45, 45, 45, 0.8);
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }}
        
        .progress-item {{
            margin: 8px 0;
        }}
        """
        
        css_provider.load_from_data(css_data.encode())
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
    
    def create_header_panel(self):
        """创建头部面板"""
        card = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        card.add_css_class("header-card")
        card.set_hexpand(True)
        
        # 系统标题
        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        title_label = Gtk.Label(label="AI Bamboo Recognition System v2.1")
        title_label.add_css_class("section-title")
        title_label.set_halign(Gtk.Align.START)
        title_box.append(title_label)
        
        subtitle_label = Gtk.Label(label="GTK4 Modern Industrial Interface")
        subtitle_label.add_css_class("secondary-label")
        subtitle_label.set_halign(Gtk.Align.START)
        title_box.append(subtitle_label)
        card.append(title_box)
        
        # 工作流程状态指示器
        workflow_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        workflow_box.set_halign(Gtk.Align.CENTER)
        workflow_box.set_hexpand(True)
        
        for step in self.workflow_steps:
            btn = Gtk.Button(label=step['name'][:8])
            btn.add_css_class("workflow-button")
            btn.set_size_request(80, 36)
            workflow_box.append(btn)
            self.workflow_buttons.append(btn)
        
        card.append(workflow_box)
        
        # 心跳监控
        heartbeat_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        heartbeat_box.set_halign(Gtk.Align.END)
        
        self.widgets['heartbeat_label'] = Gtk.Label(label=f"Heartbeat: {self.system_state['heartbeat_counter']}")
        self.widgets['heartbeat_label'].add_css_class("value-label")
        self.widgets['heartbeat_label'].set_halign(Gtk.Align.END)
        heartbeat_box.append(self.widgets['heartbeat_label'])
        
        self.widgets['response_label'] = Gtk.Label(label="Response: 12ms")
        self.widgets['response_label'].add_css_class("secondary-label")
        self.widgets['response_label'].set_halign(Gtk.Align.END)
        heartbeat_box.append(self.widgets['response_label'])
        
        card.append(heartbeat_box)
        
        return card
    
    def create_camera_panel(self):
        """创建摄像头面板"""
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        card.add_css_class("camera-card")
        card.set_hexpand(True)
        
        # 摄像头标题和信息
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        
        title_label = Gtk.Label(label="Real-time Detection View")
        title_label.add_css_class("section-title")
        title_label.set_halign(Gtk.Align.START)
        title_label.set_hexpand(True)
        header_box.append(title_label)
        
        info_label = Gtk.Label(label="Rail: 0-1000.0mm | Precision: 0.1mm | FPS: 28.5")
        info_label.add_css_class("secondary-label")
        info_label.set_halign(Gtk.Align.END)
        header_box.append(info_label)
        
        card.append(header_box)
        
        # 摄像头画布
        self.widgets['camera_canvas'] = Gtk.DrawingArea()
        self.widgets['camera_canvas'].set_size_request(750, 300)
        self.widgets['camera_canvas'].set_draw_func(self.on_camera_draw)
        self.widgets['camera_canvas'].set_hexpand(True)
        self.widgets['camera_canvas'].set_vexpand(True)
        card.append(self.widgets['camera_canvas'])
        
        # 坐标显示面板
        coord_card = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        coord_card.add_css_class("info-card")
        coord_card.set_homogeneous(True)
        
        # X坐标显示
        x_coord_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        x_coord_box.set_halign(Gtk.Align.CENTER)
        
        x_label = Gtk.Label(label="X Coordinate")
        x_label.add_css_class("secondary-label")
        x_coord_box.append(x_label)
        
        self.widgets['x_coord_value'] = Gtk.Label(label="245.8mm")
        self.widgets['x_coord_value'].add_css_class("value-label")
        x_coord_box.append(self.widgets['x_coord_value'])
        
        coord_card.append(x_coord_box)
        
        # 切割质量显示
        quality_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        quality_box.set_halign(Gtk.Align.CENTER)
        
        quality_label = Gtk.Label(label="Cut Quality")
        quality_label.add_css_class("secondary-label")
        quality_box.append(quality_label)
        
        self.widgets['quality_value'] = Gtk.Label(label="Normal")
        self.widgets['quality_value'].add_css_class("value-label")
        quality_box.append(self.widgets['quality_value'])
        
        coord_card.append(quality_box)
        
        # 刀具选择显示
        blade_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        blade_box.set_halign(Gtk.Align.CENTER)
        
        blade_label = Gtk.Label(label="Blade Selection")
        blade_label.add_css_class("secondary-label")
        blade_box.append(blade_label)
        
        self.widgets['blade_value'] = Gtk.Label(label="Dual Blade")
        self.widgets['blade_value'].add_css_class("value-label")
        blade_box.append(self.widgets['blade_value'])
        
        coord_card.append(blade_box)
        
        card.append(coord_card)
        
        return card
    
    def create_control_panel(self):
        """创建控制面板"""
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        card.add_css_class("control-card")
        
        # 创建滚动窗口
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
        card.append(scrolled)
        
        control_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        control_box.set_margin_top(8)
        control_box.set_margin_bottom(8)
        control_box.set_margin_start(8)
        control_box.set_margin_end(8)
        scrolled.set_child(control_box)
        
        # Modbus寄存器部分
        modbus_group = self.create_modbus_section()
        control_box.append(modbus_group)
        
        # PLC通信部分
        plc_group = self.create_plc_section()
        control_box.append(plc_group)
        
        # Jetson系统信息部分
        jetson_group = self.create_jetson_section()
        control_box.append(jetson_group)
        
        # AI模型状态部分
        ai_group = self.create_ai_section()
        control_box.append(ai_group)
        
        return card
    
    def create_modbus_section(self):
        """创建Modbus寄存器部分"""
        group = Adw.PreferencesGroup()
        group.set_title("Modbus Registers")
        group.set_description("PLC通信寄存器状态")
        
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
        
        for addr, desc, key in registers:
            row = Adw.ActionRow()
            row.set_title(f"{addr} - {desc}")
            
            value_label = Gtk.Label(label="0")
            value_label.add_css_class("value-label")
            value_label.set_valign(Gtk.Align.CENTER)
            row.add_suffix(value_label)
            
            group.add(row)
            self.register_values[key] = value_label
        
        return group
    
    def create_plc_section(self):
        """创建PLC通信部分"""
        group = Adw.PreferencesGroup()
        group.set_title("PLC Communication")
        group.set_description("西门子PLC通信状态")
        
        # PLC状态信息
        status_items = [
            ("Connection Status", "Connected"),
            ("PLC Address", "192.168.1.100"),
            ("Response Time", "12ms"),
            ("Total Cuts", "1,247")
        ]
        
        for title, value in status_items:
            row = Adw.ActionRow()
            row.set_title(title)
            
            value_label = Gtk.Label(label=value)
            value_label.add_css_class("value-label")
            value_label.set_valign(Gtk.Align.CENTER)
            row.add_suffix(value_label)
            
            group.add(row)
        
        # 刀具选择
        blade_row = Adw.ActionRow()
        blade_row.set_title("Blade Selection")
        blade_row.set_subtitle("选择切割刀具类型")
        
        blade_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        blade_names = ["Blade 1", "Blade 2", "Dual Blade"]
        
        for i, name in enumerate(blade_names):
            btn = Gtk.Button(label=name)
            btn.connect("clicked", self.on_blade_select, i)
            btn.set_size_request(70, 32)
            blade_box.append(btn)
            self.blade_buttons.append(btn)
            
            # 设置双刀为默认选择
            if i == 2:
                btn.add_css_class("success-button")
        
        blade_row.add_suffix(blade_box)
        group.add(blade_row)
        
        return group
    
    def create_jetson_section(self):
        """创建Jetson系统信息部分"""
        group = Adw.PreferencesGroup()
        group.set_title("Jetson Orin NX 16GB")
        group.set_description("15W模式 | JetPack 6.0 | CUDA 12.2")
        
        # 进度条项目
        progress_items = [
            ("CPU Usage", "cpu", "6-core ARM Cortex-A78AE"),
            ("GPU Usage", "gpu", "1024-core NVIDIA Ampere"),
            ("Memory Usage", "memory", "LPDDR5 8GB")
        ]
        
        for title, key, subtitle in progress_items:
            row = Adw.ActionRow()
            row.set_title(title)
            row.set_subtitle(subtitle)
            
            progress_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            
            progress = Gtk.ProgressBar()
            progress.set_fraction(0.45)
            progress.set_size_request(120, -1)
            progress.set_valign(Gtk.Align.CENTER)
            progress_box.append(progress)
            
            percent_label = Gtk.Label(label="45%")
            percent_label.add_css_class("value-label")
            percent_label.set_valign(Gtk.Align.CENTER)
            progress_box.append(percent_label)
            
            row.add_suffix(progress_box)
            group.add(row)
            
            self.progress_bars[key] = {'bar': progress, 'label': percent_label}
        
        # 系统信息
        info_items = [
            ("CPU Frequency", "1.5GHz"),
            ("GPU Frequency", "624MHz"),
            ("Memory Frequency", "2133MHz"),
            ("CPU Temperature", "52°C"),
            ("GPU Temperature", "49°C"),
            ("Fan Speed", "2150RPM"),
            ("Power Draw", "8.2W"),
            ("Voltage", "5.1V")
        ]
        
        for title, value in info_items:
            row = Adw.ActionRow()
            row.set_title(title)
            
            value_label = Gtk.Label(label=value)
            value_label.add_css_class("value-label")
            value_label.set_valign(Gtk.Align.CENTER)
            row.add_suffix(value_label)
            
            group.add(row)
        
        return group
    
    def create_ai_section(self):
        """创建AI模型状态部分"""
        group = Adw.PreferencesGroup()
        group.set_title("AI Model Status")
        group.set_description("YOLOv8推理引擎状态")
        
        ai_items = [
            ("Model Version", "YOLOv8n"),
            ("Inference Time", "15.3ms"),
            ("Confidence Threshold", "0.85"),
            ("Detection Accuracy", "94.2%"),
            ("Total Detections", "15,432"),
            ("Today Detections", "89")
        ]
        
        for title, value in ai_items:
            row = Adw.ActionRow()
            row.set_title(title)
            
            value_label = Gtk.Label(label=value)
            value_label.add_css_class("value-label")
            value_label.set_valign(Gtk.Align.CENTER)
            row.add_suffix(value_label)
            
            group.add(row)
        
        return group
    
    def create_footer_panel(self):
        """创建底部控制面板"""
        card = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        card.add_css_class("header-card")
        card.set_hexpand(True)
        
        # 控制按钮
        control_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        
        self.widgets['start_btn'] = Gtk.Button(label="START")
        self.widgets['start_btn'].add_css_class("success-button")
        self.widgets['start_btn'].connect("clicked", self.on_start_clicked)
        control_box.append(self.widgets['start_btn'])
        
        self.widgets['pause_btn'] = Gtk.Button(label="PAUSE")
        self.widgets['pause_btn'].add_css_class("warning-button")
        self.widgets['pause_btn'].connect("clicked", self.on_pause_clicked)
        control_box.append(self.widgets['pause_btn'])
        
        self.widgets['stop_btn'] = Gtk.Button(label="STOP")
        self.widgets['stop_btn'].add_css_class("error-button")
        self.widgets['stop_btn'].connect("clicked", self.on_stop_clicked)
        control_box.append(self.widgets['stop_btn'])
        
        card.append(control_box)
        
        # 状态信息
        status_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        status_box.set_halign(Gtk.Align.CENTER)
        status_box.set_hexpand(True)
        
        self.widgets['process_label'] = Gtk.Label(label="Current Process: Feed Detection")
        self.widgets['process_label'].add_css_class("value-label")
        status_box.append(self.widgets['process_label'])
        
        self.widgets['stats_label'] = Gtk.Label(label="Last Cut: 14:25:33 | Today: 89 cuts | Efficiency: 94.2%")
        self.widgets['stats_label'].add_css_class("secondary-label")
        status_box.append(self.widgets['stats_label'])
        
        card.append(status_box)
        
        # 紧急按钮
        emergency_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        
        self.widgets['emergency_btn'] = Gtk.Button(label="EMERGENCY")
        self.widgets['emergency_btn'].add_css_class("emergency-button")
        self.widgets['emergency_btn'].connect("clicked", self.on_emergency_clicked)
        emergency_box.append(self.widgets['emergency_btn'])
        
        self.widgets['power_btn'] = Gtk.Button(label="POWER")
        self.widgets['power_btn'].connect("clicked", self.on_power_clicked)
        emergency_box.append(self.widgets['power_btn'])
        
        card.append(emergency_box)
        
        return card
    
    def on_camera_draw(self, widget, cr, width, height, user_data):
        """摄像头画布绘制回调"""
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
        if 'heartbeat_label' in self.widgets:
            self.widgets['heartbeat_label'].set_label(f"Heartbeat: {self.system_state['heartbeat_counter']}")
        if 'camera_canvas' in self.widgets:
            self.widgets['camera_canvas'].queue_draw()  # 重绘摄像头画布
        return False
    
    def update_system_ui(self):
        """更新系统UI"""
        # 更新进度条
        cpu_usage = int(self.jetson_data['cpu']['usage'])
        gpu_usage = int(self.jetson_data['gpu']['usage'])
        memory_usage = int(self.jetson_data['memory']['used'] / self.jetson_data['memory']['total'] * 100)
        
        if 'cpu' in self.progress_bars:
            self.progress_bars['cpu']['bar'].set_fraction(cpu_usage / 100.0)
            self.progress_bars['cpu']['label'].set_label(f"{cpu_usage}%")
        
        if 'gpu' in self.progress_bars:
            self.progress_bars['gpu']['bar'].set_fraction(gpu_usage / 100.0)
            self.progress_bars['gpu']['label'].set_label(f"{gpu_usage}%")
        
        if 'memory' in self.progress_bars:
            self.progress_bars['memory']['bar'].set_fraction(memory_usage / 100.0)
            self.progress_bars['memory']['label'].set_label(f"{memory_usage}%")
        
        # 更新坐标显示
        if 'x_coord_value' in self.widgets:
            self.widgets['x_coord_value'].set_label(f"{self.system_state['x_coordinate']:.1f}mm")
        
        # 更新Modbus寄存器
        self.update_modbus_registers()
        
        return False
    
    def update_modbus_registers(self):
        """更新Modbus寄存器值"""
        register_updates = {
            'reg_40001': str(1 if self.system_state['is_running'] else 0),
            'reg_40002': str(self.system_state['plc_command']),
            'reg_40003': str(self.system_state['coordinate_ready']),
            'reg_40004': str(int(self.system_state['x_coordinate'] * 10)),
            'reg_40006': str(self.system_state['cut_quality']),
            'reg_40007': str(self.system_state['heartbeat_counter']),
            'reg_40009': str(self.system_state['blade_selection']),
            'reg_40010': "0"  # 健康状态正常
        }
        
        for key, value in register_updates.items():
            if key in self.register_values:
                self.register_values[key].set_label(value)
    
    def update_workflow_status(self, step):
        """更新工作流程状态"""
        for i, btn in enumerate(self.workflow_buttons):
            # 移除所有状态类
            btn.remove_css_class("workflow-active")
            btn.remove_css_class("workflow-completed")
            
            if i < step - 1:
                # 已完成
                btn.add_css_class("workflow-completed")
            elif i == step - 1:
                # 当前活动
                btn.add_css_class("workflow-active")
        
        self.system_state['current_step'] = step
        if 'process_label' in self.widgets:
            GLib.idle_add(self.widgets['process_label'].set_label, f"Current Process: {self.workflow_steps[step-1]['name']}")
    
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
        button.remove_css_class("success-button")
        button.add_css_class("warning-button")
    
    def on_pause_clicked(self, button):
        """暂停按钮点击处理"""
        if self.system_state['is_running']:
            self.system_state['is_running'] = False
            self.system_state['plc_command'] = 5  # 暂停命令
            
            # 更新启动按钮外观
            if 'start_btn' in self.widgets:
                self.widgets['start_btn'].set_label("RESUME")
                self.widgets['start_btn'].remove_css_class("warning-button")
                self.widgets['start_btn'].add_css_class("success-button")
    
    def on_stop_clicked(self, button):
        """停止按钮点击处理"""
        self.system_state['is_running'] = False
        self.system_state['plc_command'] = 0
        self.system_state['coordinate_ready'] = 0
        self.update_workflow_status(1)
        
        # 更新启动按钮外观
        if 'start_btn' in self.widgets:
            self.widgets['start_btn'].set_label("START")
            self.widgets['start_btn'].remove_css_class("warning-button")
            self.widgets['start_btn'].add_css_class("success-button")
    
    def on_emergency_clicked(self, button):
        """紧急停止按钮点击处理"""
        self.system_state['is_running'] = False
        self.system_state['plc_command'] = 6  # 紧急停止
        
        # 将所有工作流程步骤标记为紧急状态
        for btn in self.workflow_buttons:
            btn.remove_css_class("workflow-active")
            btn.remove_css_class("workflow-completed")
            btn.add_css_class("emergency-button")
        
        if 'process_label' in self.widgets:
            GLib.idle_add(self.widgets['process_label'].set_label, "Current Process: EMERGENCY STOP")
        
        # 更新启动按钮
        if 'start_btn' in self.widgets:
            self.widgets['start_btn'].set_label("RECOVER")
            self.widgets['start_btn'].remove_css_class("success-button")
            self.widgets['start_btn'].add_css_class("warning-button")
    
    def on_power_clicked(self, button):
        """电源按钮点击处理"""
        # 在真实系统中，这里会启动关机流程
        pass
    
    def on_blade_select(self, button, blade_idx):
        """刀具选择处理"""
        # 更新按钮状态
        for i, btn in enumerate(self.blade_buttons):
            btn.remove_css_class("success-button")
            if i == blade_idx:
                btn.add_css_class("success-button")
        
        # 更新系统状态
        self.system_state['blade_selection'] = blade_idx + 1
        
        # 更新显示
        blade_names = ["Blade 1", "Blade 2", "Dual Blade"]
        if 'blade_value' in self.widgets:
            self.widgets['blade_value'].set_label(blade_names[blade_idx])
    
    def on_window_destroy(self, widget):
        """窗口销毁处理"""
        self.running = False
        self.app.quit()
    
    def run(self):
        """运行GTK4应用程序"""
        return self.app.run(sys.argv)


def main():
    """主程序入口"""
    try:
        print("正在启动AI竹子识别系统 - GTK4界面...")
        
        # 检查是否有显示连接
        if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
            print("警告：没有检测到显示连接，启动无头模式...")
            from .headless_mode import run_headless_mode
            return run_headless_mode()
        
        # 检查GTK4是否可以初始化
        if not Gtk.init_check():
            print("警告：GTK4无法初始化，切换到无头模式...")
            from .headless_mode import run_headless_mode
            return run_headless_mode()
        
        # 创建并运行应用程序
        app = BambooSystemUI()
        
        print("GTK4界面初始化完成，正在启动主循环...")
        return app.run()
        
    except KeyboardInterrupt:
        print("\n用户中断，系统退出")
        return 1
    except Exception as e:
        print(f"系统错误: {e}")
        print("尝试切换到无头模式...")
        try:
            from .headless_mode import run_headless_mode
            return run_headless_mode()
        except:
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    exit(main())