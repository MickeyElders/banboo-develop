#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机 - 触摸界面主程序
基于GTK4的工业级触摸界面系统

主要功能：
1. 全屏触摸操作界面
2. 实时状态监控显示
3. 切割参数设置
4. 系统诊断和维护
5. 工业级UI设计
"""

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')

import sys
import os
import time
import threading
import logging
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from gi.repository import Gtk, Adw, GLib, Gdk, Gio
import cairo

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.communication.status_integration import PLCStatusIntegration, CoreStatusData, ExtendedStatusData
from src.vision.bamboo_detector import BambooDetector
from main import SmartBambooCuttingMachine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/bamboo_gui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BambooTouchInterface(Adw.Application):
    """智能切竹机触摸界面主应用"""
    
    def __init__(self):
        """初始化应用"""
        super().__init__(application_id='com.bamboocutter.touch')
        
        # 应用状态
        self.main_window: Optional[Gtk.ApplicationWindow] = None
        self.cutting_machine: Optional[SmartBambooCuttingMachine] = None
        self.status_integration: Optional[PLCStatusIntegration] = None
        
        # UI组件引用
        self.status_display = {}
        self.control_buttons = {}
        self.parameter_entries = {}
        
        # 当前状态
        self.current_status = {
            'device_state': 0,
            'position': 0.0,
            'cutting_force': 0.0,
            'temperature': 0.0,
            'emergency_stop': False
        }
        
        # UI更新定时器
        self.update_timer_id = None
        
        logger.info("触摸界面应用初始化完成")
    
    def do_activate(self):
        """应用激活时调用"""
        if not self.main_window:
            self.main_window = self.create_main_window()
            self.setup_style()
            self.init_cutting_machine()
            self.start_status_updates()
        
        self.main_window.present()
        self.main_window.fullscreen()  # 全屏显示
        
        logger.info("触摸界面应用已激活并全屏显示")
    
    def create_main_window(self) -> Gtk.ApplicationWindow:
        """创建主窗口"""
        window = Gtk.ApplicationWindow(application=self)
        window.set_title("智能切竹机控制系统")
        window.set_default_size(1024, 768)
        window.set_resizable(False)
        
        # 设置窗口属性以支持触摸
        window.set_decorated(False)  # 无边框
        
        # 创建主布局
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        window.set_child(main_box)
        
        # 创建头部状态栏
        header_bar = self.create_header_bar()
        main_box.append(header_bar)
        
        # 创建内容区域
        content_paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        content_paned.set_wide_handle(True)
        main_box.append(content_paned)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        content_paned.set_start_child(control_panel)
        
        # 右侧状态监控
        status_panel = self.create_status_panel()
        content_paned.set_end_child(status_panel)
        
        # 设置分割比例
        content_paned.set_position(400)
        
        return window
    
    def create_header_bar(self) -> Gtk.Box:
        """创建头部状态栏"""
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        header_box.set_margin_top(10)
        header_box.set_margin_bottom(10)
        header_box.set_margin_start(20)
        header_box.set_margin_end(20)
        header_box.add_css_class("header-bar")
        
        # 系统标题
        title_label = Gtk.Label(label="智能切竹机控制系统")
        title_label.add_css_class("title-label")
        header_box.append(title_label)
        
        # 弹性空间
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        header_box.append(spacer)
        
        # 系统状态指示器
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        
        # 设备状态
        self.status_display['device_status'] = Gtk.Label(label="设备状态: 未知")
        self.status_display['device_status'].add_css_class("status-label")
        status_box.append(self.status_display['device_status'])
        
        # 时间显示
        self.status_display['time'] = Gtk.Label()
        self.status_display['time'].add_css_class("time-label")
        status_box.append(self.status_display['time'])
        
        # 急停按钮
        emergency_button = Gtk.Button(label="急停")
        emergency_button.add_css_class("emergency-button")
        emergency_button.set_size_request(80, 50)
        emergency_button.connect("clicked", self.on_emergency_stop)
        status_box.append(emergency_button)
        
        header_box.append(status_box)
        
        return header_box
    
    def create_control_panel(self) -> Gtk.ScrolledWindow:
        """创建控制面板"""
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        
        control_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=20)
        control_box.set_margin_top(20)
        control_box.set_margin_bottom(20)
        control_box.set_margin_start(20)
        control_box.set_margin_end(20)
        
        # 操作控制区
        operation_group = self.create_operation_group()
        control_box.append(operation_group)
        
        # 参数设置区
        parameter_group = self.create_parameter_group()
        control_box.append(parameter_group)
        
        # 维护功能区
        maintenance_group = self.create_maintenance_group()
        control_box.append(maintenance_group)
        
        scrolled.set_child(control_box)
        return scrolled
    
    def create_operation_group(self) -> Gtk.Box:
        """创建操作控制组"""
        group_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        
        # 组标题
        title = Gtk.Label(label="操作控制")
        title.add_css_class("group-title")
        group_box.append(title)
        
        # 按钮网格
        button_grid = Gtk.Grid()
        button_grid.set_row_spacing(15)
        button_grid.set_column_spacing(15)
        button_grid.set_row_homogeneous(True)
        button_grid.set_column_homogeneous(True)
        
        # 启动/停止按钮
        self.control_buttons['start_stop'] = Gtk.Button(label="启动系统")
        self.control_buttons['start_stop'].add_css_class("control-button")
        self.control_buttons['start_stop'].add_css_class("start-button")
        self.control_buttons['start_stop'].set_size_request(150, 60)
        self.control_buttons['start_stop'].connect("clicked", self.on_start_stop_clicked)
        button_grid.attach(self.control_buttons['start_stop'], 0, 0, 2, 1)
        
        # 回零按钮
        home_button = Gtk.Button(label="回零")
        home_button.add_css_class("control-button")
        home_button.set_size_request(70, 50)
        home_button.connect("clicked", self.on_home_clicked)
        button_grid.attach(home_button, 0, 1, 1, 1)
        
        # 校准按钮
        calibrate_button = Gtk.Button(label="校准")
        calibrate_button.add_css_class("control-button")
        calibrate_button.set_size_request(70, 50)
        calibrate_button.connect("clicked", self.on_calibrate_clicked)
        button_grid.attach(calibrate_button, 1, 1, 1, 1)
        
        # 手动切割按钮
        manual_cut_button = Gtk.Button(label="手动切割")
        manual_cut_button.add_css_class("control-button")
        manual_cut_button.add_css_class("manual-button")
        manual_cut_button.set_size_request(150, 50)
        manual_cut_button.connect("clicked", self.on_manual_cut_clicked)
        button_grid.attach(manual_cut_button, 0, 2, 2, 1)
        
        group_box.append(button_grid)
        
        return group_box
    
    def create_parameter_group(self) -> Gtk.Box:
        """创建参数设置组"""
        group_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        
        # 组标题
        title = Gtk.Label(label="切割参数")
        title.add_css_class("group-title")
        group_box.append(title)
        
        # 参数输入网格
        param_grid = Gtk.Grid()
        param_grid.set_row_spacing(12)
        param_grid.set_column_spacing(10)
        
        # 目标长度
        length_label = Gtk.Label(label="目标长度 (mm):")
        length_label.set_halign(Gtk.Align.START)
        param_grid.attach(length_label, 0, 0, 1, 1)
        
        self.parameter_entries['target_length'] = Gtk.Entry()
        self.parameter_entries['target_length'].set_text("150")
        self.parameter_entries['target_length'].add_css_class("parameter-entry")
        param_grid.attach(self.parameter_entries['target_length'], 1, 0, 1, 1)
        
        # 切割速度
        speed_label = Gtk.Label(label="切割速度 (mm/s):")
        speed_label.set_halign(Gtk.Align.START)
        param_grid.attach(speed_label, 0, 1, 1, 1)
        
        self.parameter_entries['cutting_speed'] = Gtk.Entry()
        self.parameter_entries['cutting_speed'].set_text("50")
        self.parameter_entries['cutting_speed'].add_css_class("parameter-entry")
        param_grid.attach(self.parameter_entries['cutting_speed'], 1, 1, 1, 1)
        
        # 公差设置
        tolerance_label = Gtk.Label(label="长度公差 (mm):")
        tolerance_label.set_halign(Gtk.Align.START)
        param_grid.attach(tolerance_label, 0, 2, 1, 1)
        
        self.parameter_entries['tolerance'] = Gtk.Entry()
        self.parameter_entries['tolerance'].set_text("2.0")
        self.parameter_entries['tolerance'].add_css_class("parameter-entry")
        param_grid.attach(self.parameter_entries['tolerance'], 1, 2, 1, 1)
        
        group_box.append(param_grid)
        
        # 应用设置按钮
        apply_button = Gtk.Button(label="应用设置")
        apply_button.add_css_class("control-button")
        apply_button.add_css_class("apply-button")
        apply_button.connect("clicked", self.on_apply_parameters)
        group_box.append(apply_button)
        
        return group_box
    
    def create_maintenance_group(self) -> Gtk.Box:
        """创建维护功能组"""
        group_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        
        # 组标题
        title = Gtk.Label(label="系统维护")
        title.add_css_class("group-title")
        group_box.append(title)
        
        # 维护按钮
        maintenance_grid = Gtk.Grid()
        maintenance_grid.set_row_spacing(10)
        maintenance_grid.set_column_spacing(10)
        maintenance_grid.set_row_homogeneous(True)
        maintenance_grid.set_column_homogeneous(True)
        
        # 系统诊断
        diagnostic_button = Gtk.Button(label="系统诊断")
        diagnostic_button.add_css_class("maintenance-button")
        diagnostic_button.connect("clicked", self.on_diagnostic_clicked)
        maintenance_grid.attach(diagnostic_button, 0, 0, 1, 1)
        
        # 日志查看
        log_button = Gtk.Button(label="查看日志")
        log_button.add_css_class("maintenance-button")
        log_button.connect("clicked", self.on_log_clicked)
        maintenance_grid.attach(log_button, 1, 0, 1, 1)
        
        # 设置配置
        config_button = Gtk.Button(label="系统设置")
        config_button.add_css_class("maintenance-button")
        config_button.connect("clicked", self.on_config_clicked)
        maintenance_grid.attach(config_button, 0, 1, 1, 1)
        
        # 关机按钮
        shutdown_button = Gtk.Button(label="关机")
        shutdown_button.add_css_class("maintenance-button")
        shutdown_button.add_css_class("shutdown-button")
        shutdown_button.connect("clicked", self.on_shutdown_clicked)
        maintenance_grid.attach(shutdown_button, 1, 1, 1, 1)
        
        group_box.append(maintenance_grid)
        
        return group_box
    
    def create_status_panel(self) -> Gtk.Box:
        """创建状态监控面板"""
        status_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=20)
        status_box.set_margin_top(20)
        status_box.set_margin_bottom(20)
        status_box.set_margin_start(20)
        status_box.set_margin_end(20)
        
        # 实时状态区
        realtime_group = self.create_realtime_status_group()
        status_box.append(realtime_group)
        
        # 设备信息区
        device_group = self.create_device_info_group()
        status_box.append(device_group)
        
        # 统计信息区
        statistics_group = self.create_statistics_group()
        status_box.append(statistics_group)
        
        return status_box
    
    def create_realtime_status_group(self) -> Gtk.Box:
        """创建实时状态组"""
        group_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        
        # 组标题
        title = Gtk.Label(label="实时状态")
        title.add_css_class("group-title")
        group_box.append(title)
        
        # 状态网格
        status_grid = Gtk.Grid()
        status_grid.set_row_spacing(10)
        status_grid.set_column_spacing(15)
        
        # 当前位置
        pos_label = Gtk.Label(label="当前位置:")
        pos_label.set_halign(Gtk.Align.START)
        status_grid.attach(pos_label, 0, 0, 1, 1)
        
        self.status_display['position'] = Gtk.Label(label="0.00 mm")
        self.status_display['position'].add_css_class("status-value")
        self.status_display['position'].set_halign(Gtk.Align.END)
        status_grid.attach(self.status_display['position'], 1, 0, 1, 1)
        
        # 切割力
        force_label = Gtk.Label(label="切割力:")
        force_label.set_halign(Gtk.Align.START)
        status_grid.attach(force_label, 0, 1, 1, 1)
        
        self.status_display['cutting_force'] = Gtk.Label(label="0.0 N")
        self.status_display['cutting_force'].add_css_class("status-value")
        self.status_display['cutting_force'].set_halign(Gtk.Align.END)
        status_grid.attach(self.status_display['cutting_force'], 1, 1, 1, 1)
        
        # 电机温度
        temp_label = Gtk.Label(label="电机温度:")
        temp_label.set_halign(Gtk.Align.START)
        status_grid.attach(temp_label, 0, 2, 1, 1)
        
        self.status_display['temperature'] = Gtk.Label(label="0.0 °C")
        self.status_display['temperature'].add_css_class("status-value")
        self.status_display['temperature'].set_halign(Gtk.Align.END)
        status_grid.attach(self.status_display['temperature'], 1, 2, 1, 1)
        
        # 主轴转速
        rpm_label = Gtk.Label(label="主轴转速:")
        rpm_label.set_halign(Gtk.Align.START)
        status_grid.attach(rpm_label, 0, 3, 1, 1)
        
        self.status_display['spindle_rpm'] = Gtk.Label(label="0 RPM")
        self.status_display['spindle_rpm'].add_css_class("status-value")
        self.status_display['spindle_rpm'].set_halign(Gtk.Align.END)
        status_grid.attach(self.status_display['spindle_rpm'], 1, 3, 1, 1)
        
        group_box.append(status_grid)
        
        return group_box
    
    def create_device_info_group(self) -> Gtk.Box:
        """创建设备信息组"""
        group_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        
        # 组标题
        title = Gtk.Label(label="设备信息")
        title.add_css_class("group-title")
        group_box.append(title)
        
        # 信息网格
        info_grid = Gtk.Grid()
        info_grid.set_row_spacing(8)
        info_grid.set_column_spacing(10)
        
        # 运行时间
        runtime_label = Gtk.Label(label="运行时间:")
        runtime_label.set_halign(Gtk.Align.START)
        info_grid.attach(runtime_label, 0, 0, 1, 1)
        
        self.status_display['runtime'] = Gtk.Label(label="0 小时")
        self.status_display['runtime'].add_css_class("info-value")
        self.status_display['runtime'].set_halign(Gtk.Align.END)
        info_grid.attach(self.status_display['runtime'], 1, 0, 1, 1)
        
        # 切割次数
        cuts_label = Gtk.Label(label="切割次数:")
        cuts_label.set_halign(Gtk.Align.START)
        info_grid.attach(cuts_label, 0, 1, 1, 1)
        
        self.status_display['total_cuts'] = Gtk.Label(label="0 次")
        self.status_display['total_cuts'].add_css_class("info-value")
        self.status_display['total_cuts'].set_halign(Gtk.Align.END)
        info_grid.attach(self.status_display['total_cuts'], 1, 1, 1, 1)
        
        # 刀片磨损
        wear_label = Gtk.Label(label="刀片磨损:")
        wear_label.set_halign(Gtk.Align.START)
        info_grid.attach(wear_label, 0, 2, 1, 1)
        
        self.status_display['blade_wear'] = Gtk.Label(label="0.0 %")
        self.status_display['blade_wear'].add_css_class("info-value")
        self.status_display['blade_wear'].set_halign(Gtk.Align.END)
        info_grid.attach(self.status_display['blade_wear'], 1, 2, 1, 1)
        
        group_box.append(info_grid)
        
        return group_box
    
    def create_statistics_group(self) -> Gtk.Box:
        """创建统计信息组"""
        group_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        
        # 组标题
        title = Gtk.Label(label="生产统计")
        title.add_css_class("group-title")
        group_box.append(title)
        
        # 统计网格
        stats_grid = Gtk.Grid()
        stats_grid.set_row_spacing(8)
        stats_grid.set_column_spacing(10)
        
        # 成功率
        success_label = Gtk.Label(label="成功率:")
        success_label.set_halign(Gtk.Align.START)
        stats_grid.attach(success_label, 0, 0, 1, 1)
        
        self.status_display['success_rate'] = Gtk.Label(label="0.0 %")
        self.status_display['success_rate'].add_css_class("info-value")
        self.status_display['success_rate'].set_halign(Gtk.Align.END)
        stats_grid.attach(self.status_display['success_rate'], 1, 0, 1, 1)
        
        # 日产量
        daily_output_label = Gtk.Label(label="今日产量:")
        daily_output_label.set_halign(Gtk.Align.START)
        stats_grid.attach(daily_output_label, 0, 1, 1, 1)
        
        self.status_display['daily_output'] = Gtk.Label(label="0 根")
        self.status_display['daily_output'].add_css_class("info-value")
        self.status_display['daily_output'].set_halign(Gtk.Align.END)
        stats_grid.attach(self.status_display['daily_output'], 1, 1, 1, 1)
        
        # 平均效率
        efficiency_label = Gtk.Label(label="平均效率:")
        efficiency_label.set_halign(Gtk.Align.START)
        stats_grid.attach(efficiency_label, 0, 2, 1, 1)
        
        self.status_display['efficiency'] = Gtk.Label(label="0 根/小时")
        self.status_display['efficiency'].add_css_class("info-value")
        self.status_display['efficiency'].set_halign(Gtk.Align.END)
        stats_grid.attach(self.status_display['efficiency'], 1, 2, 1, 1)
        
        group_box.append(stats_grid)
        
        return group_box
    
    def setup_style(self):
        """设置界面样式"""
        css_provider = Gtk.CssProvider()
        css_data = """
        /* 工业风格界面样式 */
        .header-bar {
            background: linear-gradient(to bottom, #2c3e50, #34495e);
            color: white;
            border-bottom: 2px solid #3498db;
        }
        
        .title-label {
            font-size: 24px;
            font-weight: bold;
            color: #ecf0f1;
        }
        
        .status-label {
            font-size: 14px;
            color: #bdc3c7;
        }
        
        .time-label {
            font-size: 16px;
            font-weight: bold;
            color: #3498db;
        }
        
        .emergency-button {
            background: #e74c3c;
            color: white;
            font-weight: bold;
            font-size: 16px;
            border-radius: 8px;
        }
        
        .emergency-button:hover {
            background: #c0392b;
        }
        
        .group-title {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .control-button {
            font-size: 16px;
            font-weight: bold;
            min-height: 50px;
            border-radius: 8px;
            margin: 2px;
        }
        
        .start-button {
            background: #27ae60;
            color: white;
        }
        
        .start-button:hover {
            background: #2ecc71;
        }
        
        .stop-button {
            background: #e74c3c;
            color: white;
        }
        
        .stop-button:hover {
            background: #c0392b;
        }
        
        .manual-button {
            background: #f39c12;
            color: white;
        }
        
        .manual-button:hover {
            background: #e67e22;
        }
        
        .apply-button {
            background: #3498db;
            color: white;
        }
        
        .apply-button:hover {
            background: #2980b9;
        }
        
        .maintenance-button {
            background: #95a5a6;
            color: white;
            min-height: 40px;
            font-size: 14px;
        }
        
        .maintenance-button:hover {
            background: #7f8c8d;
        }
        
        .shutdown-button {
            background: #e74c3c;
            color: white;
        }
        
        .shutdown-button:hover {
            background: #c0392b;
        }
        
        .parameter-entry {
            font-size: 16px;
            min-height: 40px;
            padding: 8px;
            border-radius: 4px;
        }
        
        .status-value {
            font-size: 18px;
            font-weight: bold;
            color: #27ae60;
        }
        
        .info-value {
            font-size: 14px;
            color: #7f8c8d;
        }
        
        window {
            background: #ecf0f1;
        }
        """
        
        css_provider.load_from_data(css_data, -1)
        
        display = Gdk.Display.get_default()
        Gtk.StyleContext.add_provider_for_display(
            display, css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
    
    def init_cutting_machine(self):
        """初始化切割机后端"""
        # 测试阶段：注释掉后端初始化，使用模拟数据
        try:
            # TODO: 生产环境取消注释
            # config_path = "config/system_config.yaml"
            # self.cutting_machine = SmartBambooCuttingMachine(config_path)
            # 
            # if self.cutting_machine.load_config():
            #     self.cutting_machine.initialize_components()
            #     self.status_integration = self.cutting_machine.status_integration
            #     
            #     # 注册状态更新回调
            #     if self.status_integration:
            #         self.status_integration.add_status_callback(self.on_status_update)
            #         self.status_integration.add_alert_callback(self.on_alert_received)
            #     
            #     logger.info("切割机后端初始化成功")
            # else:
            #     logger.error("切割机配置加载失败")
            
                                 # 测试阶段：创建模拟数据
        self.init_test_data()
        logger.info("测试模式：使用模拟数据")
        
        # 也可以尝试连接真实PLC进行监控
        self.init_plc_monitor()
                 
         except Exception as e:
             logger.error(f"系统初始化失败: {e}")
             # 测试阶段：即使失败也继续运行
             self.init_test_data()
    
    def init_test_data(self):
        """初始化测试数据和模拟状态"""
        # 模拟状态数据
        self.test_data = {
            'device_state': 0,  # 0:空闲, 1:定位, 2:切割, 3:故障
            'actual_position': 0.0,
            'cutting_force': 0.0,
            'motor_temp': 25.0,
            'spindle_rpm': 0,
            'blade_wear_level': 5.2,
            'runtime_hours': 128,
            'total_cuts': 1567,
            'success_rate': 98.5,
            'daily_output': 45,
            'efficiency': 32
        }
        
        # 状态变化模拟
        self.test_running = False
        self.test_position_target = 0.0
        
        logger.info("测试数据初始化完成")
    
    def init_plc_monitor(self):
        """初始化PLC监控（纯监控模式）"""
        try:
            # 导入PLC监控模块
            from src.communication.plc_monitor import PLCMonitor
            
            # 创建PLC监控实例
            self.plc_monitor = PLCMonitor()
            
            # 尝试连接PLC
            if self.plc_monitor.connect():
                logger.info("PLC监控连接成功，将使用真实数据")
                # 启动监控线程
                self.plc_monitor.start_monitoring()
                self.use_real_plc_data = True
            else:
                logger.info("PLC连接失败，继续使用模拟数据")
                self.plc_monitor = None
                self.use_real_plc_data = False
                
        except Exception as e:
            logger.warning(f"PLC监控初始化失败: {e}，使用模拟数据")
            self.plc_monitor = None
            self.use_real_plc_data = False
    
    def start_status_updates(self):
        """启动状态更新定时器"""
        def update_ui():
            self.update_status_display()
            self.update_time_display()
            return True  # 继续定时器
        
        # 每500ms更新一次界面
        self.update_timer_id = GLib.timeout_add(500, update_ui)
        logger.info("状态更新定时器已启动")
    
    def update_status_display(self):
        """更新状态显示"""
        try:
            # 测试阶段：使用模拟数据
            if hasattr(self, 'test_data'):
                self.update_test_display()
            elif self.status_integration:
                current_status = self.status_integration.get_current_status()
                
                if current_status['core']:
                    core_data = current_status['core']
                    
                    # 更新设备状态
                    device_state_name = self.status_integration.get_device_state_name()
                    self.status_display['device_status'].set_text(f"设备状态: {device_state_name}")
                    
                    # 更新实时数据
                    self.status_display['position'].set_text(f"{core_data['actual_position']:.2f} mm")
                    self.status_display['cutting_force'].set_text(f"{core_data['cutting_force']:.1f} N")
                    self.status_display['temperature'].set_text(f"{core_data['motor_temp']:.1f} °C")
                    
                    # 更新按钮状态
                    self.update_control_buttons(core_data['device_state'])
                
                if current_status['extended']:
                    extended_data = current_status['extended']
                    self.status_display['spindle_rpm'].set_text(f"{extended_data['spindle_rpm']:.0f} RPM")
                    self.status_display['blade_wear'].set_text(f"{extended_data['blade_wear_level']:.1f} %")
                
                # 更新诊断数据
                diagnostic_data = self.status_integration.read_diagnostic_data()
                if diagnostic_data:
                    self.status_display['runtime'].set_text(f"{diagnostic_data.runtime_hours} 小时")
                    self.status_display['total_cuts'].set_text(f"{diagnostic_data.total_cuts} 次")
                    
                    # 计算成功率
                    if diagnostic_data.total_cuts > 0:
                        success_rate = (1 - diagnostic_data.fault_count / diagnostic_data.total_cuts) * 100
                        self.status_display['success_rate'].set_text(f"{success_rate:.1f} %")
                
                 except Exception as e:
             logger.error(f"状态显示更新失败: {e}")
    
    def update_test_display(self):
        """更新测试模式显示"""
        try:
            # 更新设备状态
            state_names = ["空闲", "定位中", "切割中", "故障"]
            device_state_name = state_names[self.test_data['device_state']]
            self.status_display['device_status'].set_text(f"设备状态: {device_state_name}")
            
            # 更新实时数据
            self.status_display['position'].set_text(f"{self.test_data['actual_position']:.2f} mm")
            self.status_display['cutting_force'].set_text(f"{self.test_data['cutting_force']:.1f} N")
            self.status_display['temperature'].set_text(f"{self.test_data['motor_temp']:.1f} °C")
            self.status_display['spindle_rpm'].set_text(f"{self.test_data['spindle_rpm']:.0f} RPM")
            self.status_display['blade_wear'].set_text(f"{self.test_data['blade_wear_level']:.1f} %")
            
            # 更新统计数据
            self.status_display['runtime'].set_text(f"{self.test_data['runtime_hours']} 小时")
            self.status_display['total_cuts'].set_text(f"{self.test_data['total_cuts']} 次")
            self.status_display['success_rate'].set_text(f"{self.test_data['success_rate']:.1f} %")
            self.status_display['daily_output'].set_text(f"{self.test_data['daily_output']} 根")
            self.status_display['efficiency'].set_text(f"{self.test_data['efficiency']} 根/小时")
            
            # 更新按钮状态
            self.update_control_buttons(self.test_data['device_state'])
            
            # 模拟数据变化
            if self.test_running:
                self.simulate_operation()
                
        except Exception as e:
            logger.error(f"测试显示更新失败: {e}")
    
    def simulate_operation(self):
        """模拟设备运行"""
        import random
        
        # 模拟位置变化
        if self.test_data['device_state'] == 1:  # 定位中
            if abs(self.test_data['actual_position'] - self.test_position_target) > 0.1:
                diff = self.test_position_target - self.test_data['actual_position']
                self.test_data['actual_position'] += diff * 0.1
            else:
                self.test_data['device_state'] = 2  # 切割
                
        elif self.test_data['device_state'] == 2:  # 切割中
            self.test_data['cutting_force'] = 15.0 + random.uniform(-2.0, 2.0)
            self.test_data['motor_temp'] = 35.0 + random.uniform(-1.0, 1.0)
            self.test_data['spindle_rpm'] = 1800 + random.uniform(-50, 50)
            
            # 模拟切割完成
            if random.random() < 0.05:  # 5%概率完成切割
                self.test_data['device_state'] = 0  # 空闲
                self.test_data['cutting_force'] = 0.0
                self.test_data['spindle_rpm'] = 0
                self.test_data['total_cuts'] += 1
                self.test_running = False
                
        # 模拟温度变化
        if self.test_data['device_state'] == 0:  # 空闲时降温
            self.test_data['motor_temp'] = max(25.0, self.test_data['motor_temp'] - 0.1)
    
    def update_time_display(self):
        """更新时间显示"""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.status_display['time'].set_text(current_time)
    
    def update_control_buttons(self, device_state: int):
        """更新控制按钮状态"""
        if device_state == 0:  # 空闲状态
            self.control_buttons['start_stop'].set_label("启动系统")
            self.control_buttons['start_stop'].remove_css_class("stop-button")
            self.control_buttons['start_stop'].add_css_class("start-button")
        else:  # 运行状态
            self.control_buttons['start_stop'].set_label("停止系统")
            self.control_buttons['start_stop'].remove_css_class("start-button")
            self.control_buttons['start_stop'].add_css_class("stop-button")
    
    def on_status_update(self, status_type: str, data: Any):
        """状态更新回调"""
        # 在主线程中更新UI
        GLib.idle_add(self.update_status_display)
    
    def on_alert_received(self, alert_message: str, data: Any):
        """告警接收回调"""
        def show_alert():
            dialog = Gtk.MessageDialog(
                transient_for=self.main_window,
                modal=True,
                message_type=Gtk.MessageType.WARNING,
                buttons=Gtk.ButtonsType.OK,
                text="系统告警"
            )
            dialog.format_secondary_text(alert_message)
            dialog.connect("response", lambda d, r: d.destroy())
            dialog.present()
        
        GLib.idle_add(show_alert)
    
    # 事件处理方法
    def on_emergency_stop(self, button):
        """急停按钮处理"""
        logger.warning("用户触发急停")
        if self.cutting_machine:
            self.cutting_machine.stop_event.set()
        
        # 显示确认对话框
        dialog = Gtk.MessageDialog(
            transient_for=self.main_window,
            modal=True,
            message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.OK,
            text="急停已触发"
        )
        dialog.format_secondary_text("系统已紧急停止，请检查设备状态")
        dialog.connect("response", lambda d, r: d.destroy())
        dialog.present()
    
    def on_start_stop_clicked(self, button):
        """启动/停止按钮处理"""
        # 测试阶段：使用模拟操作
        if hasattr(self, 'test_data'):
            if self.test_running:
                self.test_running = False
                self.test_data['device_state'] = 0  # 空闲
                logger.info("测试模式：用户停止系统")
            else:
                self.test_running = True
                self.test_data['device_state'] = 1  # 定位中
                # 模拟移动到随机位置
                import random
                self.test_position_target = random.uniform(50.0, 200.0)
                logger.info("测试模式：用户启动系统")
        # 生产模式：实际控制
        elif self.cutting_machine:
            if self.cutting_machine.running:
                self.cutting_machine.stop_event.set()
                logger.info("用户停止系统")
            else:
                # 启动系统
                threading.Thread(target=self.cutting_machine.run, daemon=True).start()
                logger.info("用户启动系统")
    
    def on_home_clicked(self, button):
        """回零按钮处理"""
        logger.info("执行回零操作")
        # TODO: 实现回零操作
    
    def on_calibrate_clicked(self, button):
        """校准按钮处理"""
        logger.info("执行校准操作")
        # TODO: 实现校准操作
    
    def on_manual_cut_clicked(self, button):
        """手动切割按钮处理"""
        logger.info("执行手动切割")
        # TODO: 实现手动切割
    
    def on_apply_parameters(self, button):
        """应用参数按钮处理"""
        try:
            target_length = float(self.parameter_entries['target_length'].get_text())
            cutting_speed = float(self.parameter_entries['cutting_speed'].get_text())
            tolerance = float(self.parameter_entries['tolerance'].get_text())
            
            logger.info(f"应用参数: 长度={target_length}mm, 速度={cutting_speed}mm/s, 公差={tolerance}mm")
            
            # TODO: 应用参数到系统
            
            # 显示确认
            dialog = Gtk.MessageDialog(
                transient_for=self.main_window,
                modal=True,
                message_type=Gtk.MessageType.INFO,
                buttons=Gtk.ButtonsType.OK,
                text="参数已应用"
            )
            dialog.connect("response", lambda d, r: d.destroy())
            dialog.present()
            
        except ValueError as e:
            logger.error(f"参数格式错误: {e}")
            # 显示错误对话框
            dialog = Gtk.MessageDialog(
                transient_for=self.main_window,
                modal=True,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text="参数格式错误"
            )
            dialog.format_secondary_text("请检查输入的数值格式")
            dialog.connect("response", lambda d, r: d.destroy())
            dialog.present()
    
    def on_diagnostic_clicked(self, button):
        """诊断按钮处理"""
        logger.info("显示系统诊断")
        # TODO: 显示诊断窗口
    
    def on_log_clicked(self, button):
        """日志按钮处理"""
        logger.info("显示系统日志")
        # TODO: 显示日志窗口
    
    def on_config_clicked(self, button):
        """配置按钮处理"""
        logger.info("显示系统配置")
        # TODO: 显示配置窗口
    
    def on_shutdown_clicked(self, button):
        """关机按钮处理"""
        dialog = Gtk.MessageDialog(
            transient_for=self.main_window,
            modal=True,
            message_type=Gtk.MessageType.QUESTION,
            buttons=Gtk.ButtonsType.YES_NO,
            text="确认关机"
        )
        dialog.format_secondary_text("确定要关闭系统吗？")
        
        def on_response(dialog, response):
            if response == Gtk.ResponseType.YES:
                logger.info("用户确认关机")
                if self.cutting_machine:
                    self.cutting_machine.shutdown()
                self.quit()
            dialog.destroy()
        
        dialog.connect("response", on_response)
        dialog.present()
    
    def do_shutdown(self):
        """应用关闭时调用"""
        if self.update_timer_id:
            GLib.source_remove(self.update_timer_id)
        
        if self.cutting_machine:
            self.cutting_machine.shutdown()
        
        logger.info("触摸界面应用已关闭")


def main():
    """主函数"""
    # 设置环境变量以支持触摸
    os.environ['GDK_BACKEND'] = 'wayland,x11'
    
    app = BambooTouchInterface()
    return app.run(sys.argv)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)