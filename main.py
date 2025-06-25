#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机主程序
整合AI视觉识别与PLC控制系统

主要功能：
1. 系统初始化与配置加载
2. 视觉识别与切割控制集成
3. 自动化工作流程管理
4. 异常处理与安全监控
"""

import os
import sys
import time
import threading
import signal
import yaml
import logging
from typing import List, Optional
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.vision.bamboo_detector import BambooDetector, BambooSegment, CuttingPoint
from src.communication.modbus_client import CuttingController, DeviceState

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bamboo_cutting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SmartBambooCuttingMachine:
    """智能切竹机主控制器"""
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        """
        初始化智能切竹机
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = {}
        self.running = False
        
        # 核心组件
        self.vision_detector: Optional[BambooDetector] = None
        self.cutting_controller: Optional[CuttingController] = None
        
        # 工作状态
        self.current_bamboo: Optional[List[BambooSegment]] = None
        self.cutting_queue: List[CuttingPoint] = []
        self.statistics = {
            'total_processed': 0,
            'successful_cuts': 0,
            'failed_cuts': 0,
            'total_runtime': 0,
            'start_time': None
        }
        
        # 线程控制
        self.main_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        logger.info("智能切竹机主控制器初始化")
    
    def load_config(self) -> bool:
        """
        加载配置文件
        Returns:
            是否成功加载
        """
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.error(f"配置文件不存在: {self.config_path}")
                return False
            
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            logger.info("配置文件加载成功")
            return True
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return False
    
    def initialize_components(self) -> bool:
        """
        初始化系统组件
        Returns:
            是否成功初始化
        """
        try:
            # 初始化AI视觉检测器
            vision_config = self.config.get('vision_ai', {})
            model_path = vision_config.get('model', {}).get('path')
            
            self.vision_detector = BambooDetector(
                model_path=model_path,
                device='cuda' if self.config.get('performance', {}).get('optimization', {}).get('enable_gpu') else 'cpu'
            )
            
            # 初始化切割控制器
            comm_config = self.config.get('communication', {}).get('modbus_tcp', {})
            plc_host = comm_config.get('plc_host', '192.168.1.20')
            
            self.cutting_controller = CuttingController(plc_host)
            
            # 启动状态监控
            self.cutting_controller.start_monitoring(
                interval=self.config.get('performance', {}).get('monitoring', {}).get('status_update_interval', 1.0)
            )
            
            logger.info("系统组件初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            return False
    
    def process_bamboo_stick(self, image_data) -> bool:
        """
        处理单根竹材
        Args:
            image_data: 图像数据
        Returns:
            是否处理成功
        """
        try:
            # 1. AI视觉识别
            logger.info("开始竹材识别...")
            segments = self.vision_detector.detect_bamboo_segments(image_data)
            
            if not segments:
                logger.warning("未检测到有效竹筒段")
                return False
            
            self.current_bamboo = segments
            
            # 2. 计算切割点
            target_length = self.config.get('cutting_parameters', {}).get('default', {}).get('target_length', 150.0)
            cutting_points = self.vision_detector.calculate_cutting_points(segments, target_length)
            
            if not cutting_points:
                logger.warning("未生成有效切割点")
                return False
            
            self.cutting_queue = cutting_points
            
            # 3. 执行切割序列
            return self._execute_cutting_sequence()
            
        except Exception as e:
            logger.error(f"竹材处理失败: {e}")
            return False
    
    def _execute_cutting_sequence(self) -> bool:
        """
        执行切割序列
        Returns:
            是否全部切割成功
        """
        success_count = 0
        total_cuts = len(self.cutting_queue)
        
        logger.info(f"开始执行切割序列，共{total_cuts}个切割点")
        
        # 获取切割参数
        cutting_config = self.config.get('cutting_parameters', {}).get('default', {})
        cutting_speed = cutting_config.get('cutting_speed', 50.0)
        
        for i, cutting_point in enumerate(self.cutting_queue):
            if self.stop_event.is_set():
                logger.info("收到停止信号，中断切割序列")
                break
            
            try:
                logger.info(f"执行第{i+1}/{total_cuts}次切割: 位置={cutting_point.position:.2f}mm, "
                          f"类型={cutting_point.cut_type}")
                
                # 检查设备状态
                if not self.cutting_controller.is_ready():
                    logger.error("设备未就绪，跳过此次切割")
                    continue
                
                # 执行切割
                success = self.cutting_controller.execute_cutting(
                    position=cutting_point.position,
                    speed=cutting_speed,
                    tool_id=0  # 使用主刀具
                )
                
                if success:
                    success_count += 1
                    logger.info(f"切割成功: {cutting_point.position:.2f}mm")
                else:
                    logger.error(f"切割失败: {cutting_point.position:.2f}mm")
                
                # 短暂延时，避免连续操作过快
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"切割点{i+1}执行异常: {e}")
        
        # 更新统计信息
        self.statistics['successful_cuts'] += success_count
        self.statistics['failed_cuts'] += (total_cuts - success_count)
        
        logger.info(f"切割序列完成: 成功{success_count}/{total_cuts}")
        return success_count == total_cuts
    
    def _work_cycle(self):
        """主工作循环"""
        logger.info("主工作循环启动")
        
        while not self.stop_event.is_set():
            try:
                # 检查设备状态
                if not self.cutting_controller.is_ready():
                    logger.debug("设备未就绪，等待...")
                    time.sleep(1.0)
                    continue
                
                # 模拟图像采集 (实际项目中从摄像头获取)
                # 这里使用模拟数据
                import numpy as np
                simulated_image = np.zeros((480, 640, 3), dtype=np.uint8)
                simulated_image[:] = (120, 160, 180)  # 模拟竹材图像
                
                # 处理竹材
                success = self.process_bamboo_stick(simulated_image)
                
                if success:
                    self.statistics['total_processed'] += 1
                    logger.info(f"竹材处理完成，累计处理: {self.statistics['total_processed']}")
                else:
                    logger.warning("竹材处理失败")
                
                # 等待下一根竹材 (实际项目中由传感器或手动触发)
                logger.info("等待下一根竹材...")
                time.sleep(5.0)  # 模拟等待时间
                
            except Exception as e:
                logger.error(f"工作循环异常: {e}")
                time.sleep(2.0)
        
        logger.info("主工作循环结束")
    
    def start(self) -> bool:
        """
        启动智能切竹机
        Returns:
            是否启动成功
        """
        try:
            # 加载配置
            if not self.load_config():
                return False
            
            # 初始化组件
            if not self.initialize_components():
                return False
            
            # 检查设备就绪状态
            if not self.cutting_controller.is_ready():
                logger.error("PLC设备未就绪，请检查连接")
                return False
            
            # 启动主工作线程
            self.running = True
            self.statistics['start_time'] = time.time()
            
            self.main_thread = threading.Thread(target=self._work_cycle, daemon=True)
            self.main_thread.start()
            
            logger.info("智能切竹机启动成功")
            return True
            
        except Exception as e:
            logger.error(f"启动失败: {e}")
            return False
    
    def stop(self):
        """停止智能切竹机"""
        logger.info("正在停止智能切竹机...")
        
        self.running = False
        self.stop_event.set()
        
        # 等待主线程结束
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5.0)
        
        # 停止组件
        if self.cutting_controller:
            self.cutting_controller.stop_monitoring()
            self.cutting_controller.client.disconnect()
        
        # 计算运行时间
        if self.statistics['start_time']:
            self.statistics['total_runtime'] = time.time() - self.statistics['start_time']
        
        logger.info("智能切竹机已停止")
    
    def emergency_stop(self):
        """紧急停止"""
        logger.warning("执行紧急停止！")
        
        if self.cutting_controller:
            self.cutting_controller.emergency_stop()
        
        self.stop()
    
    def get_status(self) -> dict:
        """
        获取系统状态
        Returns:
            系统状态信息
        """
        status = {
            'running': self.running,
            'device_ready': self.cutting_controller.is_ready() if self.cutting_controller else False,
            'statistics': self.statistics.copy(),
            'current_bamboo_segments': len(self.current_bamboo) if self.current_bamboo else 0,
            'cutting_queue_length': len(self.cutting_queue)
        }
        
        if self.cutting_controller:
            device_status = self.cutting_controller.get_current_status()
            if device_status:
                status['device_state'] = device_status.current_state.name
                status['device_position'] = device_status.actual_position
                status['device_temperature'] = device_status.motor_temp
        
        return status
    
    def print_status(self):
        """打印系统状态"""
        status = self.get_status()
        
        print("\n" + "="*50)
        print("智能切竹机系统状态")
        print("="*50)
        print(f"运行状态: {'运行中' if status['running'] else '已停止'}")
        print(f"设备就绪: {'是' if status['device_ready'] else '否'}")
        print(f"累计处理: {status['statistics']['total_processed']} 根")
        print(f"成功切割: {status['statistics']['successful_cuts']} 次")
        print(f"失败切割: {status['statistics']['failed_cuts']} 次")
        
        if 'device_state' in status:
            print(f"设备状态: {status['device_state']}")
            print(f"当前位置: {status['device_position']:.2f} mm")
            print(f"电机温度: {status['device_temperature']:.1f} ℃")
        
        if status['statistics']['total_runtime'] > 0:
            runtime_hours = status['statistics']['total_runtime'] / 3600
            print(f"运行时间: {runtime_hours:.2f} 小时")
        
        print("="*50)


def signal_handler(signum, frame):
    """信号处理器"""
    logger.info(f"收到信号 {signum}，准备退出...")
    if 'machine' in globals():
        machine.stop()
    sys.exit(0)


def main():
    """主函数"""
    print("智能切竹机控制系统")
    print("版本: v1.0")
    print("作者: 智能制造团队")
    print("-" * 50)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建智能切竹机实例
    global machine
    machine = SmartBambooCuttingMachine()
    
    try:
        # 启动系统
        if machine.start():
            print("系统启动成功！")
            
            # 主循环 - 监控和状态显示
            while machine.running:
                time.sleep(10)  # 每10秒显示一次状态
                machine.print_status()
        else:
            print("系统启动失败！")
            return 1
            
    except KeyboardInterrupt:
        print("\n收到中断信号...")
    except Exception as e:
        logger.error(f"运行异常: {e}")
        return 1
    finally:
        machine.stop()
        print("系统已安全关闭")
    
    return 0


if __name__ == "__main__":
    # 确保必要的目录存在
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    
    sys.exit(main()) 