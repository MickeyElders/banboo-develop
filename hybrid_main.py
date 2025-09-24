#!/usr/bin/env python3
"""
智能竹材切割系统 - Python+C++混合架构主程序
版本: v4.0-hybrid
"""

import sys
import time
import signal
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

# 导入混合架构模块
from python_core.ui_manager import UIManager
from python_core.config_manager import ConfigManager
from python_core.modbus_server import ModbusServerManager
from python_core.camera_controller import CameraController
from python_core.inference_engine import InferenceEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hybrid_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class SystemStats:
    """系统统计信息"""
    camera_fps: float = 0.0
    inference_fps: float = 0.0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    total_detections: int = 0
    plc_connected: bool = False
    uptime_seconds: int = 0


class HybridBambooSystem:
    """Python+C++混合架构竹子识别系统"""
    
    def __init__(self):
        self.shutdown_requested = threading.Event()
        self.system_started = False
        
        # 核心组件
        self.config_manager = None
        self.ui_manager = None
        self.camera_controller = None
        self.inference_engine = None
        self.modbus_server = None
        
        # 系统统计
        self.stats = SystemStats()
        self.start_time = time.time()
        
        # 线程管理
        self.inference_thread = None
        self.stats_thread = None
        self.ui_thread = None
        
    def setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"收到信号 {signum}，开始优雅关闭...")
            self.shutdown()
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def initialize(self) -> bool:
        """初始化系统"""
        logger.info("="*50)
        logger.info("智能竹材切割系统 v4.0-hybrid 启动")
        logger.info("Python+C++混合架构")
        logger.info("="*50)
        
        try:
            # 1. 初始化配置管理器
            logger.info("初始化配置管理器...")
            self.config_manager = ConfigManager()
            if not self.config_manager.load_config("config/system_config.yaml"):
                logger.error("配置文件加载失败")
                return False
                
            # 2. 初始化C++推理引擎
            logger.info("初始化C++推理引擎...")
            self.inference_engine = InferenceEngine(self.config_manager.get_ai_config())
            if not self.inference_engine.initialize():
                logger.error("推理引擎初始化失败")
                return False
                
            # 3. 初始化摄像头控制器
            logger.info("初始化摄像头控制器...")
            self.camera_controller = CameraController(self.config_manager.get_camera_config())
            if not self.camera_controller.initialize():
                logger.error("摄像头初始化失败")
                return False
                
            # 4. 初始化Modbus服务器
            logger.info("初始化Modbus服务器...")
            self.modbus_server = ModbusServerManager(self.config_manager.get_modbus_config())
            if not self.modbus_server.start():
                logger.error("Modbus服务器启动失败")
                return False
                
            # 5. 初始化UI管理器
            logger.info("初始化UI管理器...")
            self.ui_manager = UIManager(self.config_manager.get_ui_config())
            if not self.ui_manager.initialize():
                logger.error("UI管理器初始化失败") 
                return False
                
            logger.info("混合架构系统初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"系统初始化异常: {e}")
            return False
            
    def start_inference_thread(self):
        """启动推理线程"""
        def inference_loop():
            logger.info("推理线程启动")
            
            while not self.shutdown_requested.is_set():
                try:
                    # 从摄像头获取帧
                    frame_data = self.camera_controller.get_latest_frame()
                    if frame_data is None:
                        time.sleep(0.01)  # 10ms
                        continue
                        
                    # C++推理
                    detection_result = self.inference_engine.detect(frame_data)
                    
                    if detection_result and detection_result.success:
                        # 更新UI显示
                        self.ui_manager.update_detection_result(detection_result)
                        
                        # 发送到Modbus
                        self.modbus_server.update_detection_data(detection_result)
                        
                        # 更新统计
                        self.stats.total_detections += 1
                        
                    # 控制帧率 (30fps)
                    time.sleep(0.033)
                    
                except Exception as e:
                    logger.error(f"推理线程异常: {e}")
                    time.sleep(0.1)
                    
            logger.info("推理线程退出")
            
        self.inference_thread = threading.Thread(target=inference_loop, daemon=True)
        self.inference_thread.start()
        
    def start_stats_thread(self):
        """启动统计线程"""
        def stats_loop():
            logger.info("统计线程启动")
            
            last_detections = 0
            last_time = time.time()
            
            while not self.shutdown_requested.is_set():
                try:
                    current_time = time.time()
                    time_diff = current_time - last_time
                    
                    if time_diff >= 1.0:  # 每秒更新一次
                        # 计算FPS
                        detection_diff = self.stats.total_detections - last_detections
                        self.stats.inference_fps = detection_diff / time_diff
                        
                        # 获取其他统计信息
                        self.stats.camera_fps = self.camera_controller.get_fps()
                        self.stats.cpu_usage = self.get_cpu_usage()
                        self.stats.memory_usage_mb = self.get_memory_usage()
                        self.stats.plc_connected = self.modbus_server.is_connected()
                        self.stats.uptime_seconds = int(current_time - self.start_time)
                        
                        # 更新UI统计信息
                        self.ui_manager.update_system_stats(self.stats)
                        
                        last_detections = self.stats.total_detections
                        last_time = current_time
                        
                    time.sleep(0.5)  # 500ms检查间隔
                    
                except Exception as e:
                    logger.error(f"统计线程异常: {e}")
                    time.sleep(1.0)
                    
            logger.info("统计线程退出")
            
        self.stats_thread = threading.Thread(target=stats_loop, daemon=True)
        self.stats_thread.start()
        
    def run(self):
        """运行主循环"""
        if not self.initialize():
            logger.error("系统初始化失败")
            return False
            
        # 设置信号处理器
        self.setup_signal_handlers()
        
        try:
            # 启动后台线程
            self.start_inference_thread() 
            self.start_stats_thread()
            
            logger.info("系统启动完成")
            logger.info("按 Ctrl+C 退出系统")
            
            # 启动UI主循环 (阻塞)
            self.ui_manager.run_main_loop()
            
            return True
            
        except KeyboardInterrupt:
            logger.info("接收到键盘中断")
            self.shutdown()
            return True
        except Exception as e:
            logger.error(f"主循环异常: {e}")
            self.shutdown()
            return False
            
    def shutdown(self):
        """关闭系统"""
        if self.shutdown_requested.is_set():
            return
            
        logger.info("开始系统关闭...")
        self.shutdown_requested.set()
        
        # 停止UI
        if self.ui_manager:
            self.ui_manager.shutdown()
            
        # 停止Modbus服务器
        if self.modbus_server:
            self.modbus_server.stop()
            
        # 关闭摄像头
        if self.camera_controller:
            self.camera_controller.close()
            
        # 关闭推理引擎
        if self.inference_engine:
            self.inference_engine.cleanup()
            
        # 等待线程结束
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=5.0)
            
        if self.stats_thread and self.stats_thread.is_alive():
            self.stats_thread.join(timeout=2.0)
            
        logger.info("系统关闭完成")
        
    def get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            with open('/proc/loadavg', 'r') as f:
                load_avg = float(f.read().split()[0])
                return min(load_avg * 100 / 4, 100.0)  # 假设4核
        except:
            return 0.0
            
    def get_memory_usage(self) -> float:
        """获取内存使用量(MB)"""
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        # 提取内存大小(kB)并转换为MB
                        memory_kb = int(line.split()[1])
                        return memory_kb / 1024.0
        except:
            return 0.0
        

def main():
    """主函数"""
    # 创建日志目录
    Path("logs").mkdir(exist_ok=True)
    
    # 创建系统实例
    system = HybridBambooSystem()
    
    # 运行系统
    success = system.run()
    
    # 退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()