#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机 - PLC状态监控模块
实现高效的状态数据收集和传输优化

主要功能：
1. 实时状态监控
2. 数据压缩传输
3. 状态变化检测
4. 异常告警处理
"""

import time
import threading
import struct
import zlib
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from queue import Queue, Empty
import json

from .modbus_client import ModbusTCPClient, DeviceState, FaultCode

logger = logging.getLogger(__name__)


class StatusPriority(Enum):
    """状态数据优先级"""
    CRITICAL = 0    # 关键状态 - 立即传输
    HIGH = 1        # 高优先级 - 快速传输
    NORMAL = 2      # 正常优先级 - 常规传输
    LOW = 3         # 低优先级 - 延迟传输


@dataclass
class PLCStatusData:
    """PLC状态数据结构 - 优化版本"""
    
    # 核心状态数据 (32字节) - 高频传输
    timestamp: int              # 时间戳 (8字节)
    device_state: int           # 设备状态 (4字节)
    actual_position: float      # 实际位置 (4字节)
    position_error: float       # 位置误差 (4字节)
    cutting_force: float        # 切割力 (4字节)
    motor_temp: float           # 电机温度 (4字节)
    emergency_stop: bool        # 急停状态 (1字节)
    fault_code: int             # 故障代码 (2字节)
    sequence_id: int            # 序列号 (1字节)
    
    # 扩展状态数据 (28字节) - 低频传输
    motor_current: float        # 电机电流 (4字节)
    spindle_rpm: float          # 主轴转速 (4字节)
    hydraulic_pressure: float   # 液压压力 (4字节)
    blade_wear_level: float     # 刀片磨损度 (4字节)
    vibration_level: float      # 振动等级 (4字节)
    coolant_temp: float         # 冷却液温度 (4字节)
    power_consumption: float    # 功耗 (4字节)
    
    # 状态标志位 (4字节)
    status_flags: int           # 组合状态标志
    
    def pack_core_data(self) -> bytes:
        """打包核心状态数据"""
        return struct.pack('>QIfffflHB',
                          self.timestamp,
                          self.device_state,
                          self.actual_position,
                          self.position_error,
                          self.cutting_force,
                          self.motor_temp,
                          self.emergency_stop,
                          self.fault_code,
                          self.sequence_id)
    
    def pack_extended_data(self) -> bytes:
        """打包扩展状态数据"""
        return struct.pack('>fffffffI',
                          self.motor_current,
                          self.spindle_rpm,
                          self.hydraulic_pressure,
                          self.blade_wear_level,
                          self.vibration_level,
                          self.coolant_temp,
                          self.power_consumption,
                          self.status_flags)
    
    @classmethod
    def unpack_core_data(cls, data: bytes) -> 'PLCStatusData':
        """解包核心状态数据"""
        values = struct.unpack('>QIfffflHB', data)
        status = cls(
            timestamp=values[0],
            device_state=values[1],
            actual_position=values[2],
            position_error=values[3],
            cutting_force=values[4],
            motor_temp=values[5],
            emergency_stop=values[6],
            fault_code=values[7],
            sequence_id=values[8],
            # 扩展数据设为默认值
            motor_current=0.0,
            spindle_rpm=0.0,
            hydraulic_pressure=0.0,
            blade_wear_level=0.0,
            vibration_level=0.0,
            coolant_temp=0.0,
            power_consumption=0.0,
            status_flags=0
        )
        return status


@dataclass
class StatusConfig:
    """状态监控配置"""
    core_update_interval: float = 0.1      # 核心数据更新间隔 (100ms)
    extended_update_interval: float = 1.0  # 扩展数据更新间隔 (1s)
    status_change_threshold: float = 0.01  # 状态变化阈值
    compression_enabled: bool = True       # 启用数据压缩
    max_queue_size: int = 1000            # 最大队列大小
    batch_size: int = 10                  # 批处理大小


class PLCStatusMonitor:
    """PLC状态监控器"""
    
    def __init__(self, 
                 modbus_client: ModbusTCPClient,
                 config: StatusConfig = None):
        """
        初始化状态监控器
        Args:
            modbus_client: Modbus客户端
            config: 监控配置
        """
        self.modbus_client = modbus_client
        self.config = config or StatusConfig()
        
        # 监控状态
        self.running = False
        self.last_core_status: Optional[PLCStatusData] = None
        self.last_extended_status: Optional[PLCStatusData] = None
        
        # 线程管理
        self.core_monitor_thread: Optional[threading.Thread] = None
        self.extended_monitor_thread: Optional[threading.Thread] = None
        self.data_processor_thread: Optional[threading.Thread] = None
        
        # 数据队列
        self.status_queue = Queue(maxsize=self.config.max_queue_size)
        self.alert_queue = Queue(maxsize=100)
        
        # 回调函数
        self.status_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # 统计信息
        self.stats = {
            'total_updates': 0,
            'compressed_updates': 0,
            'bytes_saved': 0,
            'alert_count': 0
        }
        
        logger.info("PLC状态监控器初始化完成")
    
    def add_status_callback(self, callback: Callable[[PLCStatusData], None]):
        """添加状态更新回调"""
        self.status_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[str, Any], None]):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """启动状态监控"""
        if self.running:
            logger.warning("状态监控已在运行")
            return
        
        self.running = True
        
        # 启动核心状态监控线程
        self.core_monitor_thread = threading.Thread(
            target=self._core_status_monitor,
            name="CoreStatusMonitor",
            daemon=True
        )
        self.core_monitor_thread.start()
        
        # 启动扩展状态监控线程
        self.extended_monitor_thread = threading.Thread(
            target=self._extended_status_monitor,
            name="ExtendedStatusMonitor",
            daemon=True
        )
        self.extended_monitor_thread.start()
        
        # 启动数据处理线程
        self.data_processor_thread = threading.Thread(
            target=self._data_processor,
            name="DataProcessor",
            daemon=True
        )
        self.data_processor_thread.start()
        
        logger.info("PLC状态监控已启动")
    
    def stop_monitoring(self):
        """停止状态监控"""
        self.running = False
        
        # 等待线程结束
        if self.core_monitor_thread and self.core_monitor_thread.is_alive():
            self.core_monitor_thread.join(timeout=2.0)
        
        if self.extended_monitor_thread and self.extended_monitor_thread.is_alive():
            self.extended_monitor_thread.join(timeout=2.0)
        
        if self.data_processor_thread and self.data_processor_thread.is_alive():
            self.data_processor_thread.join(timeout=2.0)
        
        logger.info("PLC状态监控已停止")
    
    def _core_status_monitor(self):
        """核心状态监控线程"""
        logger.info("核心状态监控线程启动")
        
        while self.running:
            try:
                # 读取核心状态数据
                status_data = self._read_core_status()
                
                if status_data:
                    # 检测状态变化
                    if self._is_status_changed(status_data, self.last_core_status):
                        # 添加到处理队列
                        self.status_queue.put({
                            'type': 'core',
                            'data': status_data,
                            'priority': self._get_status_priority(status_data),
                            'timestamp': time.time_ns()
                        })
                        
                        self.last_core_status = status_data
                        self.stats['total_updates'] += 1
                
                time.sleep(self.config.core_update_interval)
                
            except Exception as e:
                logger.error(f"核心状态监控错误: {e}")
                time.sleep(0.5)
    
    def _extended_status_monitor(self):
        """扩展状态监控线程"""
        logger.info("扩展状态监控线程启动")
        
        while self.running:
            try:
                # 读取扩展状态数据
                status_data = self._read_extended_status()
                
                if status_data:
                    # 检测状态变化
                    if self._is_extended_status_changed(status_data, self.last_extended_status):
                        # 添加到处理队列
                        self.status_queue.put({
                            'type': 'extended',
                            'data': status_data,
                            'priority': StatusPriority.LOW,
                            'timestamp': time.time_ns()
                        })
                        
                        self.last_extended_status = status_data
                
                time.sleep(self.config.extended_update_interval)
                
            except Exception as e:
                logger.error(f"扩展状态监控错误: {e}")
                time.sleep(1.0)
    
    def _data_processor(self):
        """数据处理线程"""
        logger.info("数据处理线程启动")
        batch_data = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                # 从队列获取数据
                try:
                    item = self.status_queue.get(timeout=0.1)
                    batch_data.append(item)
                except Empty:
                    pass
                
                # 批处理逻辑
                current_time = time.time()
                should_process = (
                    len(batch_data) >= self.config.batch_size or
                    (batch_data and current_time - last_batch_time > 0.5) or
                    any(item['priority'] == StatusPriority.CRITICAL for item in batch_data)
                )
                
                if should_process and batch_data:
                    self._process_batch_data(batch_data)
                    batch_data.clear()
                    last_batch_time = current_time
                
            except Exception as e:
                logger.error(f"数据处理错误: {e}")
    
    def _read_core_status(self) -> Optional[PLCStatusData]:
        """读取核心状态数据"""
        try:
            # 读取核心状态寄存器 (0x2000-0x2008)
            response = self.modbus_client.read_holding_registers(0x2000, 16)
            
            if response and len(response) >= 32:
                # 解析数据
                timestamp = time.time_ns()
                device_state = struct.unpack('>I', response[0:4])[0]
                actual_position = struct.unpack('>f', response[4:8])[0]
                position_error = struct.unpack('>f', response[8:12])[0]
                cutting_force = struct.unpack('>f', response[12:16])[0]
                motor_temp = struct.unpack('>f', response[16:20])[0]
                emergency_stop = bool(struct.unpack('>H', response[20:22])[0])
                fault_code = struct.unpack('>H', response[22:24])[0]
                
                status_data = PLCStatusData(
                    timestamp=timestamp,
                    device_state=device_state,
                    actual_position=actual_position,
                    position_error=position_error,
                    cutting_force=cutting_force,
                    motor_temp=motor_temp,
                    emergency_stop=emergency_stop,
                    fault_code=fault_code,
                    sequence_id=(self.stats['total_updates'] % 256),
                    # 扩展数据设为默认值
                    motor_current=0.0,
                    spindle_rpm=0.0,
                    hydraulic_pressure=0.0,
                    blade_wear_level=0.0,
                    vibration_level=0.0,
                    coolant_temp=0.0,
                    power_consumption=0.0,
                    status_flags=0
                )
                
                return status_data
                
        except Exception as e:
            logger.error(f"读取核心状态失败: {e}")
        
        return None
    
    def _read_extended_status(self) -> Optional[PLCStatusData]:
        """读取扩展状态数据"""
        try:
            # 读取扩展状态寄存器 (0x2010-0x201F)
            response = self.modbus_client.read_holding_registers(0x2010, 16)
            
            if response and len(response) >= 32:
                # 解析扩展数据
                motor_current = struct.unpack('>f', response[0:4])[0]
                spindle_rpm = struct.unpack('>f', response[4:8])[0]
                hydraulic_pressure = struct.unpack('>f', response[8:12])[0]
                blade_wear_level = struct.unpack('>f', response[12:16])[0]
                vibration_level = struct.unpack('>f', response[16:20])[0]
                coolant_temp = struct.unpack('>f', response[20:24])[0]
                power_consumption = struct.unpack('>f', response[24:28])[0]
                status_flags = struct.unpack('>I', response[28:32])[0]
                
                # 创建扩展状态数据
                if self.last_core_status:
                    status_data = PLCStatusData(
                        # 复制核心数据
                        timestamp=self.last_core_status.timestamp,
                        device_state=self.last_core_status.device_state,
                        actual_position=self.last_core_status.actual_position,
                        position_error=self.last_core_status.position_error,
                        cutting_force=self.last_core_status.cutting_force,
                        motor_temp=self.last_core_status.motor_temp,
                        emergency_stop=self.last_core_status.emergency_stop,
                        fault_code=self.last_core_status.fault_code,
                        sequence_id=self.last_core_status.sequence_id,
                        # 扩展数据
                        motor_current=motor_current,
                        spindle_rpm=spindle_rpm,
                        hydraulic_pressure=hydraulic_pressure,
                        blade_wear_level=blade_wear_level,
                        vibration_level=vibration_level,
                        coolant_temp=coolant_temp,
                        power_consumption=power_consumption,
                        status_flags=status_flags
                    )
                    
                    return status_data
                
        except Exception as e:
            logger.error(f"读取扩展状态失败: {e}")
        
        return None
    
    def _is_status_changed(self, 
                          current: PLCStatusData, 
                          last: Optional[PLCStatusData]) -> bool:
        """检测核心状态是否发生变化"""
        if not last:
            return True
        
        # 检查关键状态变化
        if (current.device_state != last.device_state or
            current.emergency_stop != last.emergency_stop or
            current.fault_code != last.fault_code):
            return True
        
        # 检查数值变化（使用阈值）
        threshold = self.config.status_change_threshold
        return (
            abs(current.actual_position - last.actual_position) > threshold or
            abs(current.position_error - last.position_error) > threshold or
            abs(current.cutting_force - last.cutting_force) > threshold * 10 or
            abs(current.motor_temp - last.motor_temp) > threshold * 5
        )
    
    def _is_extended_status_changed(self, 
                                   current: PLCStatusData, 
                                   last: Optional[PLCStatusData]) -> bool:
        """检测扩展状态是否发生变化"""
        if not last:
            return True
        
        threshold = self.config.status_change_threshold
        return (
            abs(current.motor_current - last.motor_current) > threshold or
            abs(current.spindle_rpm - last.spindle_rpm) > threshold * 50 or
            abs(current.hydraulic_pressure - last.hydraulic_pressure) > threshold * 5 or
            abs(current.vibration_level - last.vibration_level) > threshold
        )
    
    def _get_status_priority(self, status: PLCStatusData) -> StatusPriority:
        """获取状态数据优先级"""
        # 关键状态判断
        if (status.emergency_stop or 
            status.fault_code != 0 or
            status.device_state == DeviceState.FAULT.value):
            return StatusPriority.CRITICAL
        
        # 高优先级状态
        if (status.device_state in [DeviceState.POSITIONING.value, DeviceState.CUTTING.value] or
            abs(status.position_error) > 1.0 or
            status.cutting_force > 400.0 or
            status.motor_temp > 70.0):
            return StatusPriority.HIGH
        
        return StatusPriority.NORMAL
    
    def _process_batch_data(self, batch_data: List[Dict]):
        """处理批量数据"""
        try:
            # 按优先级排序
            batch_data.sort(key=lambda x: x['priority'].value)
            
            for item in batch_data:
                status_data = item['data']
                
                # 触发状态更新回调
                for callback in self.status_callbacks:
                    try:
                        callback(status_data)
                    except Exception as e:
                        logger.error(f"状态回调错误: {e}")
                
                # 检查告警条件
                self._check_alerts(status_data)
            
            # 数据压缩统计
            if self.config.compression_enabled:
                self._update_compression_stats(batch_data)
            
        except Exception as e:
            logger.error(f"批量数据处理错误: {e}")
    
    def _check_alerts(self, status: PLCStatusData):
        """检查告警条件"""
        alerts = []
        
        # 故障告警
        if status.fault_code != 0:
            alerts.append(f"设备故障: {FaultCode(status.fault_code).name}")
        
        # 温度告警
        if status.motor_temp > 80.0:
            alerts.append(f"电机温度过高: {status.motor_temp:.1f}°C")
        
        # 位置误差告警
        if abs(status.position_error) > 2.0:
            alerts.append(f"位置误差过大: {status.position_error:.2f}mm")
        
        # 切割力告警
        if status.cutting_force > 500.0:
            alerts.append(f"切割力过载: {status.cutting_force:.1f}N")
        
        # 发送告警
        for alert_msg in alerts:
            self.alert_queue.put({
                'message': alert_msg,
                'status': status,
                'timestamp': time.time()
            })
            
            # 触发告警回调
            for callback in self.alert_callbacks:
                try:
                    callback(alert_msg, status)
                except Exception as e:
                    logger.error(f"告警回调错误: {e}")
            
            self.stats['alert_count'] += 1
    
    def _update_compression_stats(self, batch_data: List[Dict]):
        """更新压缩统计信息"""
        original_size = 0
        compressed_size = 0
        
        for item in batch_data:
            status_data = item['data']
            
            # 计算原始大小
            original_data = json.dumps(asdict(status_data)).encode('utf-8')
            original_size += len(original_data)
            
            # 计算压缩大小
            compressed_data = zlib.compress(original_data)
            compressed_size += len(compressed_data)
        
        # 更新统计
        self.stats['compressed_updates'] += len(batch_data)
        self.stats['bytes_saved'] += (original_size - compressed_size)
    
    def get_current_status(self) -> Optional[PLCStatusData]:
        """获取当前状态"""
        return self.last_core_status
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取监控统计信息"""
        return {
            **self.stats,
            'queue_size': self.status_queue.qsize(),
            'alert_queue_size': self.alert_queue.qsize(),
            'compression_ratio': (
                (self.stats['bytes_saved'] / max(self.stats['compressed_updates'], 1))
                if self.stats['compressed_updates'] > 0 else 0
            )
        }


# 工具函数
def create_optimized_status_monitor(plc_host: str = "192.168.1.20") -> PLCStatusMonitor:
    """创建优化的状态监控器"""
    from .modbus_client import ModbusTCPClient
    
    # 创建Modbus客户端
    modbus_client = ModbusTCPClient(plc_host)
    
    # 优化配置
    config = StatusConfig(
        core_update_interval=0.05,      # 50ms更新核心状态
        extended_update_interval=2.0,   # 2s更新扩展状态
        status_change_threshold=0.005,  # 更敏感的变化检测
        compression_enabled=True,       # 启用压缩
        max_queue_size=500,            # 减小队列大小
        batch_size=5                   # 减小批处理大小
    )
    
    monitor = PLCStatusMonitor(modbus_client, config)
    
    logger.info("已创建优化的PLC状态监控器")
    return monitor 