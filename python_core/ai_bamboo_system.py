# ai_bamboo_system.py
import time
import threading
import random
import math
import sys
import os

# Try to import graphics libraries with fallback
GRAPHICS_BACKEND = None
try:
    import lvgl as lv
    GRAPHICS_BACKEND = "LVGL"
    print("Using LVGL graphics backend")
except ImportError:
    try:
        import pygame
        GRAPHICS_BACKEND = "PYGAME"
        print("Using Pygame graphics backend")
    except ImportError:
        GRAPHICS_BACKEND = "CONSOLE"
        print("No graphics backend available, using console mode")

class BambooSystemUI:
    def __init__(self):
        # Initialize graphics backend
        self.backend = GRAPHICS_BACKEND
        
        if self.backend == "LVGL":
            lv.init()
            # LVGL Colors
            self.colors = {
                'bg_main': lv.color_hex(0x1E1E1E),
                'bg_panel': lv.color_hex(0x2D2D2D),
                'accent': lv.color_hex(0xFF6B35),
                'success': lv.color_hex(0x4CAF50),
                'warning': lv.color_hex(0xFFC107),
                'error': lv.color_hex(0xF44336),
                'text_primary': lv.color_hex(0xFFFFFF),
                'text_secondary': lv.color_hex(0xB0B0B0),
                'border': lv.color_hex(0x404040),
                'modbus_blue': lv.color_hex(0x2196F3),
                'emergency': lv.color_hex(0xFF1744),
                'power': lv.color_hex(0x9C27B0),
                'jetson_green': lv.color_hex(0x76B900)
            }
        elif self.backend == "PYGAME":
            pygame.init()
            # Pygame Colors (RGB tuples)
            self.colors = {
                'bg_main': (30, 30, 30),
                'bg_panel': (45, 45, 45),
                'accent': (255, 107, 53),
                'success': (76, 175, 80),
                'warning': (255, 193, 7),
                'error': (244, 67, 54),
                'text_primary': (255, 255, 255),
                'text_secondary': (176, 176, 176),
                'border': (64, 64, 64),
                'modbus_blue': (33, 150, 243),
                'emergency': (255, 23, 68),
                'power': (156, 39, 176),
                'jetson_green': (118, 185, 0)
            }
            self.screen = pygame.display.set_mode((1280, 720))
            pygame.display.set_caption("AI Bamboo Recognition System")
        else:
            # Console mode - no colors
            self.colors = {}
        
        # System state
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
        
        # Jetson data
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
        
        # AI model data
        self.ai_data = {
            'inference_time': 15.3,
            'detection_fps': 28.5,
            'total_detections': 15432,
            'today_detections': 89,
            'accuracy': 94.2
        }
        
        # Status mappings
        self.status_maps = {
            'system_status': ['Stopped', 'Running', 'Error', 'Paused', 'Emergency', 'Maintenance'],
            'plc_command': ['None', 'Feed Detection', 'Cut Prepare', 'Cut Complete', 'Start Feed', 'Pause', 'Emergency', 'Resume'],
            'cut_quality': ['Normal', 'Abnormal'],
            'blade_type': ['None', 'Blade 1', 'Blade 2', 'Dual Blade'],
            'health_status': ['Normal', 'Warning', 'Error', 'Critical']
        }
        
        # Workflow steps
        self.workflow_steps = [
            {'id': 1, 'name': 'Feed Detection', 'plc_cmd': 1},
            {'id': 2, 'name': 'Vision Recognition', 'plc_cmd': 0},
            {'id': 3, 'name': 'Coordinate Transfer', 'plc_cmd': 0},
            {'id': 4, 'name': 'Cut Prepare', 'plc_cmd': 2},
            {'id': 5, 'name': 'Execute Cut', 'plc_cmd': 3}
        ]
        
        # Create UI based on backend
        if self.backend in ["LVGL", "PYGAME"]:
            self.create_ui()
        else:
            self.create_console_ui()
        
        # Start update threads
        self.start_update_threads()
    
    def create_ui(self):
        """Create the main UI"""
        if self.backend == "LVGL":
            # Create main screen
            self.scr = lv.obj()
            lv.scr_load(self.scr)
            
            # Set background color
            self.scr.set_style_bg_color(self.colors['bg_main'], 0)
            
            # Create main layout
            self.create_header()
            self.create_camera_panel()
            self.create_control_panel()
            self.create_footer()
        elif self.backend == "PYGAME":
            # Pygame UI setup
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.title_font = pygame.font.Font(None, 32)
    
    def create_console_ui(self):
        """Create console-based UI"""
        print("=== AI Bamboo Recognition Cutting System ===")
        print("Running in console mode - no graphics backend available")
        print("System Status: Initializing...")
    
    def create_header(self):
        """Create header panel"""
        # Header container
        self.header = lv.obj(self.scr)
        self.header.set_size(lv.pct(100), 60)
        self.header.align(lv.ALIGN.TOP_MID, 0, 0)
        self.header.set_style_bg_color(self.colors['bg_panel'], 0)
        self.header.set_style_border_color(self.colors['border'], 0)
        self.header.set_style_border_width(2, 0)
        
        # System title
        self.title_label = lv.label(self.header)
        self.title_label.set_text("AI Bamboo Recognition Cutting System v2.1")
        self.title_label.align(lv.ALIGN.LEFT_MID, 20, 0)
        self.title_label.set_style_text_color(self.colors['text_primary'], 0)
        
        # Workflow status
        self.create_workflow_status()
        
        # Heartbeat monitor
        self.create_heartbeat_monitor()
    
    def create_workflow_status(self):
        """Create workflow status indicators"""
        self.workflow_container = lv.obj(self.header)
        self.workflow_container.set_size(400, 40)
        self.workflow_container.align(lv.ALIGN.CENTER, 0, 0)
        self.workflow_container.set_style_bg_opa(0, 0)
        self.workflow_container.set_style_border_opa(0, 0)
        self.workflow_container.set_flex_flow(lv.FLEX_FLOW.ROW)
        self.workflow_container.set_flex_align(lv.FLEX_ALIGN.SPACE_EVENLY, lv.FLEX_ALIGN.CENTER, lv.FLEX_ALIGN.CENTER)
        
        self.workflow_steps_ui = []
        for step in self.workflow_steps:
            step_btn = lv.btn(self.workflow_container)
            step_btn.set_size(70, 30)
            step_btn.set_style_bg_color(self.colors['bg_main'], 0)
            step_btn.set_style_border_color(self.colors['border'], 0)
            step_btn.set_style_radius(15, 0)
            
            step_label = lv.label(step_btn)
            step_label.set_text(step['name'][:8])  # Truncate for space
            step_label.center()
            step_label.set_style_text_font(lv.font_default(), 0)
            
            self.workflow_steps_ui.append({'btn': step_btn, 'label': step_label})
    
    def create_heartbeat_monitor(self):
        """Create heartbeat monitor"""
        self.heartbeat_container = lv.obj(self.header)
        self.heartbeat_container.set_size(200, 40)
        self.heartbeat_container.align(lv.ALIGN.RIGHT_MID, -20, 0)
        self.heartbeat_container.set_style_bg_opa(0, 0)
        self.heartbeat_container.set_style_border_opa(0, 0)
        
        # Heartbeat indicator
        self.heartbeat_led = lv.led(self.heartbeat_container)
        self.heartbeat_led.set_size(8, 8)
        self.heartbeat_led.align(lv.ALIGN.LEFT_MID, 0, -10)
        self.heartbeat_led.set_color(self.colors['success'])
        self.heartbeat_led.on()
        
        # Heartbeat counter
        self.heartbeat_label = lv.label(self.heartbeat_container)
        self.heartbeat_label.set_text(f"Heartbeat: {self.system_state['heartbeat_counter']}")
        self.heartbeat_label.align(lv.ALIGN.LEFT_MID, 15, -10)
        self.heartbeat_label.set_style_text_color(self.colors['success'], 0)
        
        # Response time
        self.response_label = lv.label(self.heartbeat_container)
        self.response_label.set_text("Response: 12ms")
        self.response_label.align(lv.ALIGN.LEFT_MID, 15, 10)
        self.response_label.set_style_text_color(self.colors['success'], 0)
    
    def create_camera_panel(self):
        """Create camera panel"""
        self.camera_panel = lv.obj(self.scr)
        self.camera_panel.set_size(lv.pct(70), 400)
        self.camera_panel.align(lv.ALIGN.TOP_LEFT, 10, 70)
        self.camera_panel.set_style_bg_color(self.colors['bg_panel'], 0)
        self.camera_panel.set_style_border_color(self.colors['border'], 0)
        self.camera_panel.set_style_border_width(2, 0)
        self.camera_panel.set_style_radius(8, 0)
        
        # Camera title
        self.camera_title = lv.label(self.camera_panel)
        self.camera_title.set_text("Real-time Detection View")
        self.camera_title.align(lv.ALIGN.TOP_LEFT, 15, 15)
        self.camera_title.set_style_text_color(self.colors['accent'], 0)
        
        # Detection info
        self.detection_info = lv.label(self.camera_panel)
        self.detection_info.set_text("Rail Range: 0-1000.0mm | Precision: 0.1mm | FPS: 28.5")
        self.detection_info.align(lv.ALIGN.TOP_RIGHT, -15, 15)
        self.detection_info.set_style_text_color(self.colors['text_secondary'], 0)
        
        # Camera canvas
        self.camera_canvas = lv.obj(self.camera_panel)
        self.camera_canvas.set_size(lv.pct(90), 250)
        self.camera_canvas.align(lv.ALIGN.TOP_MID, 0, 45)
        self.camera_canvas.set_style_bg_color(lv.color_hex(0x000000), 0)
        self.camera_canvas.set_style_border_color(self.colors['border'], 0)
        
        # Placeholder text
        self.camera_placeholder = lv.label(self.camera_canvas)
        self.camera_placeholder.set_text("Bamboo Detection View\n1280 x 720 | YOLOv8 Inference\nInference Time: 15.3ms")
        self.camera_placeholder.center()
        self.camera_placeholder.set_style_text_color(self.colors['text_secondary'], 0)
        self.camera_placeholder.set_style_text_align(lv.TEXT_ALIGN.CENTER, 0)
        
        # Rail indicator
        self.create_rail_indicator()
        
        # Coordinate display
        self.create_coordinate_display()
    
    def create_rail_indicator(self):
        """Create rail position indicator"""
        self.rail_indicator = lv.obj(self.camera_canvas)
        self.rail_indicator.set_size(lv.pct(80), 30)
        self.rail_indicator.align(lv.ALIGN.BOTTOM_MID, 0, -10)
        self.rail_indicator.set_style_bg_color(self.colors['modbus_blue'], lv.OPA._30)
        self.rail_indicator.set_style_border_color(self.colors['modbus_blue'], 0)
        
        self.rail_label = lv.label(self.rail_indicator)
        self.rail_label.set_text("X-axis Rail (0-1000.0mm)")
        self.rail_label.center()
        self.rail_label.set_style_text_color(self.colors['modbus_blue'], 0)
        
        # Cutting position indicator
        self.cutting_position = lv.obj(self.rail_indicator)
        self.cutting_position.set_size(2, 30)
        self.cutting_position.align(lv.ALIGN.LEFT_MID, 50, 0)  # Will be updated
        self.cutting_position.set_style_bg_color(self.colors['error'], 0)
    
    def create_coordinate_display(self):
        """Create coordinate display"""
        self.coord_panel = lv.obj(self.camera_panel)
        self.coord_panel.set_size(lv.pct(90), 60)
        self.coord_panel.align(lv.ALIGN.BOTTOM_MID, 0, -15)
        self.coord_panel.set_style_bg_color(lv.color_hex(0x1A1A1A), 0)
        self.coord_panel.set_style_border_color(self.colors['accent'], 0)
        
        # X coordinate
        self.x_coord_label = lv.label(self.coord_panel)
        self.x_coord_label.set_text("X Coordinate")
        self.x_coord_label.align(lv.ALIGN.LEFT_MID, 20, -15)
        self.x_coord_label.set_style_text_color(self.colors['text_secondary'], 0)
        
        self.x_coord_value = lv.label(self.coord_panel)
        self.x_coord_value.set_text("245.8mm")
        self.x_coord_value.align(lv.ALIGN.LEFT_MID, 20, 5)
        self.x_coord_value.set_style_text_color(self.colors['accent'], 0)
        
        # Cut quality
        self.quality_label = lv.label(self.coord_panel)
        self.quality_label.set_text("Cut Quality")
        self.quality_label.align(lv.ALIGN.CENTER, 0, -15)
        self.quality_label.set_style_text_color(self.colors['text_secondary'], 0)
        
        self.quality_value = lv.label(self.coord_panel)
        self.quality_value.set_text("Normal")
        self.quality_value.align(lv.ALIGN.CENTER, 0, 5)
        self.quality_value.set_style_text_color(self.colors['success'], 0)
        
        # Blade selection
        self.blade_label = lv.label(self.coord_panel)
        self.blade_label.set_text("Blade Selection")
        self.blade_label.align(lv.ALIGN.RIGHT_MID, -20, -15)
        self.blade_label.set_style_text_color(self.colors['text_secondary'], 0)
        
        self.blade_value = lv.label(self.coord_panel)
        self.blade_value.set_text("Dual Blade")
        self.blade_value.align(lv.ALIGN.RIGHT_MID, -20, 5)
        self.blade_value.set_style_text_color(self.colors['accent'], 0)
    
    def create_control_panel(self):
        """Create control panel"""
        self.control_panel = lv.obj(self.scr)
        self.control_panel.set_size(380, 400)
        self.control_panel.align(lv.ALIGN.TOP_RIGHT, -10, 70)
        self.control_panel.set_style_bg_color(self.colors['bg_panel'], 0)
        self.control_panel.set_style_border_color(self.colors['border'], 0)
        self.control_panel.set_style_border_width(2, 0)
        self.control_panel.set_style_radius(8, 0)
        
        # Create scrollable container
        self.scroll_container = lv.obj(self.control_panel)
        self.scroll_container.set_size(lv.pct(95), lv.pct(95))
        self.scroll_container.align(lv.ALIGN.TOP_MID, 0, 5)
        self.scroll_container.set_style_bg_opa(0, 0)
        self.scroll_container.set_style_border_opa(0, 0)
        self.scroll_container.set_scroll_dir(lv.DIR.VER)
        
        # Create sections
        self.create_modbus_section()
        self.create_plc_section()
        self.create_jetson_section()
        self.create_ai_section()
        self.create_communication_section()
    
    def create_modbus_section(self):
        """Create Modbus registers section"""
        section = self.create_section("Modbus Registers", self.colors['modbus_blue'])
        
        # Create table-like structure
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
        
        self.register_values = {}
        y_offset = 30
        
        for addr, desc, key in registers:
            # Address
            addr_label = lv.label(section)
            addr_label.set_text(addr)
            addr_label.align(lv.ALIGN.TOP_LEFT, 10, y_offset)
            addr_label.set_style_text_color(self.colors['modbus_blue'], 0)
            
            # Description
            desc_label = lv.label(section)
            desc_label.set_text(desc)
            desc_label.align(lv.ALIGN.TOP_LEFT, 80, y_offset)
            desc_label.set_style_text_color(self.colors['text_secondary'], 0)
            
            # Value
            value_label = lv.label(section)
            value_label.set_text("0")
            value_label.align(lv.ALIGN.TOP_RIGHT, -10, y_offset)
            value_label.set_style_text_color(self.colors['accent'], 0)
            
            self.register_values[key] = value_label
            y_offset += 20
    
    def create_plc_section(self):
        """Create PLC communication section"""
        section = self.create_section("PLC Communication", self.colors['success'])
        
        # Status items
        status_items = [
            ("Connection:", "Connected"),
            ("PLC Address:", "192.168.1.100"),
            ("Response Time:", "12ms"),
            ("Total Cuts:", "1,247")
        ]
        
        y_offset = 30
        for label, value in status_items:
            label_obj = lv.label(section)
            label_obj.set_text(label)
            label_obj.align(lv.ALIGN.TOP_LEFT, 10, y_offset)
            label_obj.set_style_text_color(self.colors['text_secondary'], 0)
            
            value_obj = lv.label(section)
            value_obj.set_text(value)
            value_obj.align(lv.ALIGN.TOP_RIGHT, -10, y_offset)
            value_obj.set_style_text_color(self.colors['text_primary'], 0)
            
            y_offset += 20
        
        # Blade selector
        self.create_blade_selector(section, y_offset + 10)
    
    def create_blade_selector(self, parent, y_offset):
        """Create blade selector buttons"""
        blade_container = lv.obj(parent)
        blade_container.set_size(lv.pct(90), 30)
        blade_container.align(lv.ALIGN.TOP_MID, 0, y_offset)
        blade_container.set_style_bg_opa(0, 0)
        blade_container.set_style_border_opa(0, 0)
        blade_container.set_flex_flow(lv.FLEX_FLOW.ROW)
        blade_container.set_flex_align(lv.FLEX_ALIGN.SPACE_EVENLY, lv.FLEX_ALIGN.CENTER, lv.FLEX_ALIGN.CENTER)
        
        self.blade_buttons = []
        blade_names = ["Blade 1", "Blade 2", "Dual Blade"]
        
        for i, name in enumerate(blade_names):
            btn = lv.btn(blade_container)
            btn.set_size(70, 25)
            btn.set_style_bg_color(self.colors['bg_main'], 0)
            btn.set_style_border_color(self.colors['border'], 0)
            
            label = lv.label(btn)
            label.set_text(name)
            label.center()
            
            # Set active state for blade 3 (dual blade)
            if i == 2:  # Dual blade
                btn.set_style_bg_color(self.colors['accent'], 0)
            
            btn.add_event_cb(lambda e, idx=i: self.on_blade_select(idx), lv.EVENT.CLICKED, None)
            self.blade_buttons.append(btn)
    
    def create_jetson_section(self):
        """Create Jetson system section"""
        section = self.create_section("Jetson Orin NX 16GB", self.colors['jetson_green'], height=200)
        
        # Performance mode indicator
        mode_label = lv.label(section)
        mode_label.set_text("15W")
        mode_label.align(lv.ALIGN.TOP_RIGHT, -10, 5)
        mode_label.set_style_text_color(self.colors['accent'], 0)
        mode_label.set_style_bg_color(self.colors['accent'], 0)
        mode_label.set_style_bg_opa(255, 0)
        mode_label.set_style_radius(3, 0)
        mode_label.set_style_pad_all(3, 0)
        
        # Progress bars
        self.create_progress_bar(section, "CPU (6-core ARM Cortex-A78AE)", 30, "cpu")
        self.create_progress_bar(section, "GPU (1024-core NVIDIA Ampere)", 60, "gpu")
        self.create_progress_bar(section, "Memory (LPDDR5)", 90, "memory")
        
        # System info grid
        self.create_jetson_info_grid(section, 120)
        
        # Version info
        version_label = lv.label(section)
        version_label.set_text("JetPack 6.0 | CUDA 12.2 | TensorRT 8.6.1\nPython 3.10.12 | PyTorch 2.1.0")
        version_label.align(lv.ALIGN.BOTTOM_LEFT, 10, -5)
        version_label.set_style_text_color(self.colors['text_secondary'], 0)
    
    def create_progress_bar(self, parent, title, y_offset, key):
        """Create progress bar with label"""
        # Title and percentage
        title_label = lv.label(parent)
        title_label.set_text(title)
        title_label.align(lv.ALIGN.TOP_LEFT, 10, y_offset)
        title_label.set_style_text_color(self.colors['text_secondary'], 0)
        
        percent_label = lv.label(parent)
        percent_label.set_text("45%")
        percent_label.align(lv.ALIGN.TOP_RIGHT, -10, y_offset)
        percent_label.set_style_text_color(self.colors['text_primary'], 0)
        
        # Progress bar
        progress = lv.bar(parent)
        progress.set_size(lv.pct(85), 6)
        progress.align(lv.ALIGN.TOP_LEFT, 10, y_offset + 15)
        progress.set_value(45, lv.ANIM.OFF)
        
        # Store references for updates
        if not hasattr(self, 'progress_bars'):
            self.progress_bars = {}
        self.progress_bars[key] = {'bar': progress, 'label': percent_label}
    
    def create_jetson_info_grid(self, parent, y_offset):
        """Create Jetson system info grid"""
        info_items = [
            ("CPU Freq:", "1.5GHz"),
            ("GPU Freq:", "624MHz"),
            ("EMC Freq:", "2133MHz"),
            ("CPU Temp:", "52°C"),
            ("GPU Temp:", "49°C"),
            ("Thermal:", "45°C"),
            ("Fan Speed:", "2150RPM"),
            ("Power:", "8.2W"),
            ("Voltage:", "5.1V")
        ]
        
        x_positions = [10, 130, 250]
        y_pos = y_offset
        
        for i, (label, value) in enumerate(info_items):
            x_pos = x_positions[i % 3]
            if i > 0 and i % 3 == 0:
                y_pos += 15
            
            label_obj = lv.label(parent)
            label_obj.set_text(label)
            label_obj.align(lv.ALIGN.TOP_LEFT, x_pos, y_pos)
            label_obj.set_style_text_color(self.colors['text_secondary'], 0)
            
            value_obj = lv.label(parent)
            value_obj.set_text(value)
            value_obj.align(lv.ALIGN.TOP_LEFT, x_pos + 50, y_pos)
            value_obj.set_style_text_color(self.colors['text_primary'], 0)
    
    def create_ai_section(self):
        """Create AI model section"""
        section = self.create_section("AI Model Status", self.colors['accent'])
        
        ai_items = [
            ("Model Version:", "YOLOv8n"),
            ("Inference Time:", "15.3ms"),
            ("Confidence:", "0.85"),
            ("Detection Accuracy:", "94.2%"),
            ("Total Detections:", "15,432"),
            ("Today Detections:", "89")
        ]
        
        y_offset = 30
        for label, value in ai_items:
            label_obj = lv.label(section)
            label_obj.set_text(label)
            label_obj.align(lv.ALIGN.TOP_LEFT, 10, y_offset)
            label_obj.set_style_text_color(self.colors['text_secondary'], 0)
            
            value_obj = lv.label(section)
            value_obj.set_text(value)
            value_obj.align(lv.ALIGN.TOP_RIGHT, -10, y_offset)
            value_obj.set_style_text_color(self.colors['text_primary'], 0)
            
            y_offset += 18
    
    def create_communication_section(self):
        """Create communication statistics section"""
        section = self.create_section("Communication Stats", self.colors['text_secondary'])
        
        comm_items = [
            ("Connection Time:", "2h 15m"),
            ("Data Packets:", "15,432"),
            ("Error Rate:", "0.02%"),
            ("Throughput:", "1.2KB/s")
        ]
        
        y_offset = 30
        for label, value in comm_items:
            label_obj = lv.label(section)
            label_obj.set_text(label)
            label_obj.align(lv.ALIGN.TOP_LEFT, 10, y_offset)
            label_obj.set_style_text_color(self.colors['text_secondary'], 0)
            
            value_obj = lv.label(section)
            value_obj.set_text(value)
            value_obj.align(lv.ALIGN.TOP_RIGHT, -10, y_offset)
            value_obj.set_style_text_color(self.colors['text_primary'], 0)
            
            y_offset += 18
    
    def create_section(self, title, color, height=120):
        """Create a section container"""
        section = lv.obj(self.scroll_container)
        section.set_size(lv.pct(95), height)
        section.set_style_bg_color(lv.color_hex(0x1A1A1A), 0)
        section.set_style_border_color(color, 0)
        section.set_style_border_width(1, 0)
        section.set_style_radius(5, 0)
        section.set_style_pad_all(8, 0)
        
        # Section title
        title_label = lv.label(section)
        title_label.set_text(title)
        title_label.align(lv.ALIGN.TOP_LEFT, 5, 5)
        title_label.set_style_text_color(color, 0)
        
        return section
    
    def create_footer(self):
        """Create footer with control buttons"""
        self.footer = lv.obj(self.scr)
        self.footer.set_size(lv.pct(100), 80)
        self.footer.align(lv.ALIGN.BOTTOM_MID, 0, 0)
        self.footer.set_style_bg_color(self.colors['bg_panel'], 0)
        self.footer.set_style_border_color(self.colors['border'], 0)
        self.footer.set_style_border_width(2, 0)
        
        # Control buttons
        self.create_control_buttons()
        
        # Status info
        self.create_status_info()
        
        # Emergency buttons
        self.create_emergency_buttons()
    
    def create_control_buttons(self):
        """Create main control buttons"""
        btn_container = lv.obj(self.footer)
        btn_container.set_size(400, 60)
        btn_container.align(lv.ALIGN.LEFT_MID, 20, 0)
        btn_container.set_style_bg_opa(0, 0)
        btn_container.set_style_border_opa(0, 0)
        btn_container.set_flex_flow(lv.FLEX_FLOW.ROW)
        btn_container.set_flex_align(lv.FLEX_ALIGN.SPACE_EVENLY, lv.FLEX_ALIGN.CENTER, lv.FLEX_ALIGN.CENTER)
        
        # Start button
        self.start_btn = lv.btn(btn_container)
        self.start_btn.set_size(100, 40)
        self.start_btn.set_style_bg_color(self.colors['success'], 0)
        self.start_btn.add_event_cb(self.on_start_clicked, lv.EVENT.CLICKED, None)
        
        start_label = lv.label(self.start_btn)
        start_label.set_text("START")
        start_label.center()
        
        # Pause button
        self.pause_btn = lv.btn(btn_container)
        self.pause_btn.set_size(100, 40)
        self.pause_btn.set_style_bg_color(self.colors['warning'], lv.STATE.DEFAULT)
        self.pause_btn.add_event_cb(self.on_pause_clicked, lv.EVENT.CLICKED, None)
        
        pause_label = lv.label(self.pause_btn)
        pause_label.set_text("PAUSE")
        pause_label.center()
        
        # Stop button
        self.stop_btn = lv.btn(btn_container)
        self.stop_btn.set_size(100, 40)
        self.stop_btn.set_style_bg_color(self.colors['error'], lv.STATE.DEFAULT)
        self.stop_btn.add_event_cb(self.on_stop_clicked, lv.EVENT.CLICKED, None)
        
        stop_label = lv.label(self.stop_btn)
        stop_label.set_text("STOP")
        stop_label.center()
    
    def create_status_info(self):
        """Create status information display"""
        info_container = lv.obj(self.footer)
        info_container.set_size(300, 60)
        info_container.align(lv.ALIGN.CENTER, 0, 0)
        info_container.set_style_bg_opa(0, 0)
        info_container.set_style_border_opa(0, 0)
        
        # Current process
        self.process_label = lv.label(info_container)
        self.process_label.set_text("Current Process: Feed Detection")
        self.process_label.align(lv.ALIGN.TOP_MID, 0, 10)
        self.process_label.set_style_text_color(self.colors['text_secondary'], 0)
        
        # Statistics
        self.stats_label = lv.label(info_container)
        self.stats_label.set_text("Last Cut: 14:25:33 | Today: 89 cuts | Efficiency: 94.2%")
        self.stats_label.align(lv.ALIGN.BOTTOM_MID, 0, -10)
        self.stats_label.set_style_text_color(self.colors['text_secondary'], 0)
    
    def create_emergency_buttons(self):
        """Create emergency and power buttons"""
        emergency_container = lv.obj(self.footer)
        emergency_container.set_size(250, 60)
        emergency_container.align(lv.ALIGN.RIGHT_MID, -20, 0)
        emergency_container.set_style_bg_opa(0, 0)
        emergency_container.set_style_border_opa(0, 0)
        emergency_container.set_flex_flow(lv.FLEX_FLOW.ROW)
        emergency_container.set_flex_align(lv.FLEX_ALIGN.SPACE_EVENLY, lv.FLEX_ALIGN.CENTER, lv.FLEX_ALIGN.CENTER)
        
        # Emergency stop
        self.emergency_btn = lv.btn(emergency_container)
        self.emergency_btn.set_size(110, 40)
        self.emergency_btn.set_style_bg_color(self.colors['emergency'], 0)
        self.emergency_btn.add_event_cb(self.on_emergency_clicked, lv.EVENT.CLICKED, None)
        
        emergency_label = lv.label(self.emergency_btn)
        emergency_label.set_text("EMERGENCY")
        emergency_label.center()
        
        # Power button
        self.power_btn = lv.btn(emergency_container)
        self.power_btn.set_size(100, 40)
        self.power_btn.set_style_bg_color(self.colors['power'], 0)
        self.power_btn.add_event_cb(self.on_power_clicked, lv.EVENT.CLICKED, None)
        
        power_label = lv.label(self.power_btn)
        power_label.set_text("POWER")
        power_label.center()
    
    def start_update_threads(self):
        """Start background update threads"""
        self.running = True
        
        # Heartbeat update thread (50Hz)
        heartbeat_thread = threading.Thread(target=self.heartbeat_update_loop)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        
        # System info update thread (2s interval)
        system_thread = threading.Thread(target=self.system_update_loop)
        system_thread.daemon = True
        system_thread.start()
        
        # Workflow simulation thread
        workflow_thread = threading.Thread(target=self.workflow_simulation_loop)
        workflow_thread.daemon = True
        workflow_thread.start()
    
    def heartbeat_update_loop(self):
        """Update heartbeat counter at 50Hz"""
        while self.running:
            self.system_state['heartbeat_counter'] += 1
            if self.system_state['heartbeat_counter'] > 4294967295:
                self.system_state['heartbeat_counter'] = 0
            
            # Update UI in main thread
            lv.timer_handler()
            time.sleep(0.02)  # 50Hz
    
    def system_update_loop(self):
        """Update system information every 2 seconds"""
        while self.running:
            self.update_system_data()
            time.sleep(2.0)
    
    def workflow_simulation_loop(self):
        """Simulate workflow when system is running"""
        while self.running:
            if self.system_state['is_running']:
                self.simulate_workflow_step()
            time.sleep(1.0)
    
    def update_system_data(self):
        """Update system data with random variations"""
        # Update Jetson data
        self.jetson_data['cpu']['usage'] = 40 + random.random() * 20
        self.jetson_data['gpu']['usage'] = 25 + random.random() * 25
        self.jetson_data['cpu']['temp'] = 45 + random.random() * 15
        self.jetson_data['gpu']['temp'] = 40 + random.random() * 20
        
        # Update AI data
        self.ai_data['inference_time'] = 12 + random.random() * 8
        self.ai_data['detection_fps'] = 25 + random.random() * 8
        
        # Update UI
        self.update_ui_elements()
    
    def update_ui_elements(self):
        """Update UI elements with current data"""
        # Update heartbeat
        self.heartbeat_label.set_text(f"Heartbeat: {self.system_state['heartbeat_counter']}")
        
        # Update progress bars
        if hasattr(self, 'progress_bars'):
            cpu_usage = int(self.jetson_data['cpu']['usage'])
            gpu_usage = int(self.jetson_data['gpu']['usage'])
            memory_usage = int(self.jetson_data['memory']['used'] / self.jetson_data['memory']['total'] * 100)
            
            self.progress_bars['cpu']['bar'].set_value(cpu_usage, lv.ANIM.OFF)
            self.progress_bars['cpu']['label'].set_text(f"{cpu_usage}%")
            
            self.progress_bars['gpu']['bar'].set_value(gpu_usage, lv.ANIM.OFF)
            self.progress_bars['gpu']['label'].set_text(f"{gpu_usage}%")
            
            self.progress_bars['memory']['bar'].set_value(memory_usage, lv.ANIM.OFF)
            self.progress_bars['memory']['label'].set_text(f"{memory_usage}%")
        
        # Update coordinate display
        self.x_coord_value.set_text(f"{self.system_state['x_coordinate']:.1f}mm")
        
        # Update cutting position indicator
        if hasattr(self, 'cutting_position'):
            position_percent = self.system_state['x_coordinate'] / 1000.0
            rail_width = self.rail_indicator.get_width()
            new_x = int(position_percent * rail_width * 0.8)  # 80% of rail width
            self.cutting_position.align(lv.ALIGN.LEFT_MID, new_x, 0)
        
        # Update Modbus registers
        self.update_modbus_registers()
    
    def update_modbus_registers(self):
        """Update Modbus register values"""
        if hasattr(self, 'register_values'):
            self.register_values['reg_40001'].set_text(str(1 if self.system_state['is_running'] else 0))
            self.register_values['reg_40002'].set_text(str(self.system_state['plc_command']))
            self.register_values['reg_40003'].set_text(str(self.system_state['coordinate_ready']))
            self.register_values['reg_40004'].set_text(str(int(self.system_state['x_coordinate'] * 10)))
            self.register_values['reg_40006'].set_text(str(self.system_state['cut_quality']))
            self.register_values['reg_40007'].set_text(str(self.system_state['heartbeat_counter']))
            self.register_values['reg_40009'].set_text(str(self.system_state['blade_selection']))
            self.register_values['reg_40010'].set_text("0")  # Health status normal
    
    def update_workflow_status(self, step):
        """Update workflow step indicators"""
        for i, step_ui in enumerate(self.workflow_steps_ui):
            if i < step - 1:
                # Completed
                step_ui['btn'].set_style_bg_color(self.colors['success'], 0)
            elif i == step - 1:
                # Active
                step_ui['btn'].set_style_bg_color(self.colors['accent'], 0)
            else:
                # Inactive
                step_ui['btn'].set_style_bg_color(self.colors['bg_main'], 0)
        
        self.system_state['current_step'] = step
        self.process_label.set_text(f"Current Process: {self.workflow_steps[step-1]['name']}")
    
    def simulate_workflow_step(self):
        """Simulate workflow progression"""
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
            # Simulate bamboo detection
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
                # Complete cycle, restart
                self.system_state['coordinate_ready'] = 0
                self.ai_data['today_detections'] += 1
                self.ai_data['total_detections'] += 1
                self.update_workflow_status(1)
    
    # Event handlers
    def on_start_clicked(self, e):
        """Handle start button click"""
        self.system_state['is_running'] = True
        self.update_workflow_status(1)
        
        # Update button appearance
        start_label = self.start_btn.get_child(0)
        start_label.set_text("RUNNING")
        self.start_btn.set_style_bg_color(self.colors['warning'], 0)
    
    def on_pause_clicked(self, e):
        """Handle pause button click"""
        if self.system_state['is_running']:
            self.system_state['is_running'] = False
            self.system_state['plc_command'] = 5  # Pause command
            
            # Update button appearance
            start_label = self.start_btn.get_child(0)
            start_label.set_text("RESUME")
            self.start_btn.set_style_bg_color(self.colors['success'], 0)
    
    def on_stop_clicked(self, e):
        """Handle stop button click"""
        self.system_state['is_running'] = False
        self.system_state['plc_command'] = 0
        self.system_state['coordinate_ready'] = 0
        self.update_workflow_status(1)
        
        # Update button appearance
        start_label = self.start_btn.get_child(0)
        start_label.set_text("START")
        self.start_btn.set_style_bg_color(self.colors['success'], 0)
        
        # Reset all workflow steps
        for step_ui in self.workflow_steps_ui:
            step_ui['btn'].set_style_bg_color(self.colors['bg_main'], 0)
    
    def on_emergency_clicked(self, e):
        """Handle emergency stop"""
        self.system_state['is_running'] = False
        self.system_state['plc_command'] = 6  # Emergency stop
        
        # Make all workflow steps red
        for step_ui in self.workflow_steps_ui:
            step_ui['btn'].set_style_bg_color(self.colors['emergency'], 0)
        
        self.process_label.set_text("Current Process: EMERGENCY STOP")
        
        # Update start button
        start_label = self.start_btn.get_child(0)
        start_label.set_text("RECOVER")
        self.start_btn.set_style_bg_color(self.colors['warning'], 0)
    
    def on_power_clicked(self, e):
        """Handle power button click"""
        # In a real system, this would initiate shutdown
        pass
    
    def on_blade_select(self, blade_idx):
        """Handle blade selection"""
        # Update button states
        for i, btn in enumerate(self.blade_buttons):
            if i == blade_idx:
                btn.set_style_bg_color(self.colors['accent'], 0)
            else:
                btn.set_style_bg_color(self.colors['bg_main'], 0)
        
        # Update system state
        self.system_state['blade_selection'] = blade_idx + 1
        
        # Update display
        blade_names = ["Blade 1", "Blade 2", "Dual Blade"]
        self.blade_value.set_text(blade_names[blade_idx])


# Main application entry point
def main():
    """Main application entry point"""
    try:
        # Create and run the application
        app = BambooSystemUI()
        
        print(f"Starting AI Bamboo System with {GRAPHICS_BACKEND} backend...")
        
        # Main loop based on backend
        if GRAPHICS_BACKEND == "LVGL":
            # LVGL main loop
            while True:
                lv.timer_handler()
                time.sleep(0.001)
        elif GRAPHICS_BACKEND == "PYGAME":
            # Pygame main loop
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                app.screen.fill(app.colors['bg_main'])
                
                # Simple text display for pygame mode
                title_text = app.title_font.render("AI Bamboo Recognition System", True, app.colors['text_primary'])
                app.screen.blit(title_text, (10, 10))
                
                status_text = app.font.render(f"System Running: {app.system_state['is_running']}", True, app.colors['text_secondary'])
                app.screen.blit(status_text, (10, 50))
                
                coord_text = app.font.render(f"X Coordinate: {app.system_state['x_coordinate']:.1f}mm", True, app.colors['accent'])
                app.screen.blit(coord_text, (10, 80))
                
                pygame.display.flip()
                app.clock.tick(60)
                
            pygame.quit()
        else:
            # Console mode main loop
            print("Running in console mode...")
            while True:
                # Print system status every 5 seconds
                print(f"System Status: {'Running' if app.system_state['is_running'] else 'Stopped'}")
                print(f"X Coordinate: {app.system_state['x_coordinate']:.1f}mm")
                print(f"Heartbeat: {app.system_state['heartbeat_counter']}")
                print("-" * 50)
                time.sleep(5.0)
                
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'app' in locals() and hasattr(app, 'running'):
            app.running = False
        print("Application cleanup completed")


if __name__ == "__main__":
    main()