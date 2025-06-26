#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机 - 状态集成模块
实现PLC状态监控与视觉系统的高效集成

主要功能：
1. 分层状态数据管理
2. 优化通信策略
3. 实时状态监控
4. 异常处理和告警
"""

import time
import threading
import struct
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from queue import Queue, Empty
import json

from .modbus_client import ModbusTCPClient, DeviceState, FaultCode

logger = logging.getLogger(__name__)


@dataclass
class CoreStatusData:
    """核心状态数据 - 32字节高频传输"""
    timestamp: int              # 时间戳 (8字节)
    device_state: int           # 设备状态 (4字节)
    actual_position: float      # 实际位置 (4字节)
    position_error: float       # 位置误差 (4字节)
    cutting_force: float        # 切割力 (4字节)
    motor_temp: float           # 电机温度 (4字节)
    emergency_stop: bool        # 急停状态 (1字节)
    fault_code: int             # 故障代码 (2字节)
    sequence_id: int            # 序列号 (1字节)
    
    def pack(self) -> bytes:
        """打包数据"""
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
    
    @classmethod
    def unpack(cls, data: bytes) -> 'CoreStatusData':
        """解包数据"""
        if len(data) < 31:
            raise ValueError("数据长度不足")
        
        values = struct.unpack('>QIfffflHB', data[:31])
        return cls(*values)


@dataclass
class ExtendedStatusData:
    """扩展状态数据 - 28字节低频传输"""
    motor_current: float        # 电机电流 (4字节)
    spindle_rpm: float          # 主轴转速 (4字节)
    hydraulic_pressure: float   # 液压压力 (4字节)
    blade_wear_level: float     # 刀片磨损度 (4字节)
    vibration_level: float      # 振动等级 (4字节)
    coolant_temp: float         # 冷却液温度 (4字节)
    power_consumption: float    # 功耗 (4字节)
    
    def pack(self) -> bytes:
        """打包数据"""
        return struct.pack('>fffffff',
                          self.motor_current,
                          self.spindle_rpm,
                          self.hydraulic_pressure,
                          self.blade_wear_level,
                          self.vibration_level,
                          self.coolant_temp,
                          self.power_consumption)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'ExtendedStatusData':
        """解包数据"""
        if len(data) < 28:
            raise ValueError("数据长度不足")
        
        values = struct.unpack('>fffffff', data[:28])
        return cls(*values)


@dataclass
class DiagnosticData:
    """诊断数据 - 20字节按需传输"""
    runtime_hours: int          # 运行时间 (4字节)
    total_cuts: int             # 切割次数 (4字节)
    fault_count: int            # 故障次数 (4字节)
    maintenance_countdown: int  # 维护倒计时 (4字节)
    calibration_error: float    # 校准偏差 (4字节)
    
    def pack(self) -> bytes:
        """打包数据"""
        return struct.pack('>IIIIf',
                          self.runtime_hours,
                          self.total_cuts,
                          self.fault_count,
                          self.maintenance_countdown,
                          self.calibration_error)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'DiagnosticData':
        """解包数据"""
        if len(data) < 20:
            raise ValueError("数据长度不足")
        
        values = struct.unpack('>IIIIf', data[:20])
        return cls(*values)


class StatusPriority(Enum):
    """状态优先级"""
    CRITICAL = 0    # 关键 - 立即传输
    HIGH = 1        # 高优先级 - 快速传输
    NORMAL = 2      # 正常 - 常规传输
    LOW = 3         # 低优先级 - 延迟传输


class PLCStatusIntegration:
    """PLC状态集成管理器"""
    
    def __init__(self, modbus_client: ModbusTCPClient):
        """
        初始化状态集成器
        Args:
            modbus_client: Modbus TCP客户端
        """
        self.modbus_client = modbus_client
        
        # 当前状态
        self.current_core_status: Optional[CoreStatusData] = None
        self.current_extended_status: Optional[ExtendedStatusData] = None
        self.current_diagnostic_data: Optional[DiagnosticData] = None
        
        # 上次状态（用于变化检测）
        self.last_core_status: Optional[CoreStatusData] = None
        self.last_extended_status: Optional[ExtendedStatusData] = None
        
        # 监控线程
        self.running = False
        self.core_monitor_thread: Optional[threading.Thread] = None
        self.extended_monitor_thread: Optional[threading.Thread] = None
        
        # 状态变化回调
        self.status_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # 变化检测阈值
        self.core_thresholds = {
            'position': 0.01,      # 位置变化阈值 (mm)
            'error': 0.01,         # 误差变化阈值 (mm)
            'force': 1.0,          # 切割力变化阈值 (N)
            'temp': 0.5            # 温度变化阈值 (°C)
        }
        
        self.extended_thresholds = {
            'current': 0.1,        # 电流变化阈值 (A)
            'rpm': 50.0,           # 转速变化阈值 (RPM)
            'pressure': 0.05,      # 压力变化阈值 (MPa)
            'vibration': 0.1       # 振动变化阈值 (m/s²)
        }
        
        # 统计信息
        self.stats = {
            'core_updates': 0,
            'extended_updates': 0,
            'diagnostic_updates': 0,
            'alerts_sent': 0,
            'data_bytes_saved': 0
        }
        
        logger.info("PLC状态集成器初始化完成")
    
    def add_status_callback(self, callback: Callable[[str, Any], None]):
        """添加状态变化回调"""
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
        
        # 启动核心状态监控线程 (50ms更新)
        self.core_monitor_thread = threading.Thread(
            target=self._core_status_monitor,
            name="CoreStatusMonitor",
            daemon=True
        )
        self.core_monitor_thread.start()
        
        # 启动扩展状态监控线程 (2s更新)
        self.extended_monitor_thread = threading.Thread(
            target=self._extended_status_monitor,
            name="ExtendedStatusMonitor",
            daemon=True
        )
        self.extended_monitor_thread.start()
        
        logger.info("PLC状态监控已启动")
    
    def stop_monitoring(self):
        """停止状态监控"""
        self.running = False
        
        if self.core_monitor_thread and self.core_monitor_thread.is_alive():
            self.core_monitor_thread.join(timeout=2.0)
        
        if self.extended_monitor_thread and self.extended_monitor_thread.is_alive():
            self.extended_monitor_thread.join(timeout=2.0)
        
        logger.info("PLC状态监控已停止")
    
    def _core_status_monitor(self):
        """核心状态监控线程"""
        logger.info("核心状态监控线程启动")
        
        while self.running:
            try:
                # 读取核心状态数据
                core_data = self._read_core_status()
                
                if core_data:
                    # 检测状态变化
                    if self._is_core_status_changed(core_data):
                        self.current_core_status = core_data
                        self.stats['core_updates'] += 1
                        
                        # 触发状态更新回调
                        self._notify_status_change('core', core_data)
                        
                        # 检查告警条件
                        self._check_core_alerts(core_data)
                        
                        self.last_core_status = core_data
                
                time.sleep(0.05)  # 50ms更新间隔
                
            except Exception as e:
                logger.error(f"核心状态监控错误: {e}")
                time.sleep(0.5)
    
    def _extended_status_monitor(self):
        """扩展状态监控线程"""
        logger.info("扩展状态监控线程启动")
        
        while self.running:
            try:
                # 读取扩展状态数据
                extended_data = self._read_extended_status()
                
                if extended_data:
                    # 检测状态变化
                    if self._is_extended_status_changed(extended_data):
                        self.current_extended_status = extended_data
                        self.stats['extended_updates'] += 1
                        
                        # 触发状态更新回调
                        self._notify_status_change('extended', extended_data)
                        
                        # 检查告警条件
                        self._check_extended_alerts(extended_data)
                        
                        self.last_extended_status = extended_data
                
                time.sleep(2.0)  # 2s更新间隔
                
            except Exception as e:
                logger.error(f"扩展状态监控错误: {e}")
                time.sleep(5.0)
    
    def _read_core_status(self) -> Optional[CoreStatusData]:
        """读取核心状态数据"""
        try:
            # 读取核心状态寄存器 (0x2000-0x200B)
            response = self.modbus_client.read_holding_registers(0x2000, 12)
            
            if response and len(response) >= 24:
                # 解析数据
                timestamp = time.time_ns()
                
                # 解析时间戳 (8字节)
                timestamp_high = struct.unpack('>I', response[0:4])[0]
                timestamp_low = struct.unpack('>I', response[4:8])[0]
                plc_timestamp = (timestamp_high << 32) | timestamp_low
                
                # 解析其他数据
                device_state = struct.unpack('>I', response[8:12])[0]
                actual_position = struct.unpack('>f', response[12:16])[0]
                position_error = struct.unpack('>f', response[16:20])[0]
                cutting_force = struct.unpack('>f', response[20:24])[0]
                
                # 如果有更多数据
                if len(response) >= 28:
                    motor_temp = struct.unpack('>f', response[24:28])[0]
                else:
                    motor_temp = 0.0
                
                # 解析状态标志
                if len(response) >= 30:
                    status_flags = struct.unpack('>H', response[28:30])[0]
                    emergency_stop = bool(status_flags & 0x01)
                    fault_code = (status_flags >> 1) & 0x7FFF
                else:
                    emergency_stop = False
                    fault_code = 0
                
                sequence_id = self.stats['core_updates'] % 256
                
                core_data = CoreStatusData(
                    timestamp=plc_timestamp if plc_timestamp > 0 else timestamp,
                    device_state=device_state,
                    actual_position=actual_position,
                    position_error=position_error,
                    cutting_force=cutting_force,
                    motor_temp=motor_temp,
                    emergency_stop=emergency_stop,
                    fault_code=fault_code,
                    sequence_id=sequence_id
                )
                
                return core_data
                
        except Exception as e:
            logger.error(f"读取核心状态失败: {e}")
        
        return None
    
    def _read_extended_status(self) -> Optional[ExtendedStatusData]:
        """读取扩展状态数据"""
        try:
            # 读取扩展状态寄存器 (0x2010-0x201C)
            response = self.modbus_client.read_holding_registers(0x2010, 14)
            
            if response and len(response) >= 28:
                # 解析扩展数据
                motor_current = struct.unpack('>f', response[0:4])[0]
                spindle_rpm = struct.unpack('>f', response[4:8])[0]
                hydraulic_pressure = struct.unpack('>f', response[8:12])[0]
                blade_wear_level = struct.unpack('>f', response[12:16])[0]
                vibration_level = struct.unpack('>f', response[16:20])[0]
                coolant_temp = struct.unpack('>f', response[20:24])[0]
                power_consumption = struct.unpack('>f', response[24:28])[0]
                
                extended_data = ExtendedStatusData(
                    motor_current=motor_current,
                    spindle_rpm=spindle_rpm,
                    hydraulic_pressure=hydraulic_pressure,
                    blade_wear_level=blade_wear_level,
                    vibration_level=vibration_level,
                    coolant_temp=coolant_temp,
                    power_consumption=power_consumption
                )
                
                return extended_data
                
        except Exception as e:
            logger.error(f"读取扩展状态失败: {e}")
        
        return None
    
    def read_diagnostic_data(self) -> Optional[DiagnosticData]:
        """读取诊断数据 - 按需调用"""
        try:
            # 读取诊断数据寄存器 (0x2020-0x2029)
            response = self.modbus_client.read_holding_registers(0x2020, 10)
            
            if response and len(response) >= 20:
                runtime_hours = struct.unpack('>I', response[0:4])[0]
                total_cuts = struct.unpack('>I', response[4:8])[0]
                fault_count = struct.unpack('>I', response[8:12])[0]
                maintenance_countdown = struct.unpack('>I', response[12:16])[0]
                calibration_error = struct.unpack('>f', response[16:20])[0]
                
                diagnostic_data = DiagnosticData(
                    runtime_hours=runtime_hours,
                    total_cuts=total_cuts,
                    fault_count=fault_count,
                    maintenance_countdown=maintenance_countdown,
                    calibration_error=calibration_error
                )
                
                self.current_diagnostic_data = diagnostic_data
                self.stats['diagnostic_updates'] += 1
                
                return diagnostic_data
                
        except Exception as e:
            logger.error(f"读取诊断数据失败: {e}")
        
        return None
    
    def _is_core_status_changed(self, current: CoreStatusData) -> bool:
        """检测核心状态是否变化"""
        if not self.last_core_status:
            return True
        
        last = self.last_core_status
        
        # 检查离散状态变化
        if (current.device_state != last.device_state or
            current.emergency_stop != last.emergency_stop or
            current.fault_code != last.fault_code):
            return True
        
        # 检查连续值变化
        return (
            abs(current.actual_position - last.actual_position) > self.core_thresholds['position'] or
            abs(current.position_error - last.position_error) > self.core_thresholds['error'] or
            abs(current.cutting_force - last.cutting_force) > self.core_thresholds['force'] or
            abs(current.motor_temp - last.motor_temp) > self.core_thresholds['temp']
        )
    
    def _is_extended_status_changed(self, current: ExtendedStatusData) -> bool:
        """检测扩展状态是否变化"""
        if not self.last_extended_status:
            return True
        
        last = self.last_extended_status
        
        return (
            abs(current.motor_current - last.motor_current) > self.extended_thresholds['current'] or
            abs(current.spindle_rpm - last.spindle_rpm) > self.extended_thresholds['rpm'] or
            abs(current.hydraulic_pressure - last.hydraulic_pressure) > self.extended_thresholds['pressure'] or
            abs(current.vibration_level - last.vibration_level) > self.extended_thresholds['vibration']
        )
    
    def _notify_status_change(self, status_type: str, data: Any):
        """通知状态变化"""
        for callback in self.status_callbacks:
            try:
                callback(status_type, data)
            except Exception as e:
                logger.error(f"状态回调错误: {e}")
    
    def _check_core_alerts(self, core_data: CoreStatusData):
        """检查核心状态告警"""
        alerts = []
        
        # 故障检查
        if core_data.fault_code != 0:
            alerts.append(f"设备故障: 代码{core_data.fault_code:04X}")
        
        # 急停检查
        if core_data.emergency_stop:
            alerts.append("急停按钮被触发")
        
        # 温度告警
        if core_data.motor_temp > 80.0:
            alerts.append(f"电机温度过高: {core_data.motor_temp:.1f}°C")
        
        # 位置误差告警
        if abs(core_data.position_error) > 2.0:
            alerts.append(f"位置误差过大: {core_data.position_error:.2f}mm")
        
        # 切割力告警
        if core_data.cutting_force > 500.0:
            alerts.append(f"切割力过载: {core_data.cutting_force:.1f}N")
        
        self._send_alerts(alerts, core_data)
    
    def _check_extended_alerts(self, extended_data: ExtendedStatusData):
        """检查扩展状态告警"""
        alerts = []
        
        # 电流告警
        if extended_data.motor_current > 15.0:
            alerts.append(f"电机电流过高: {extended_data.motor_current:.1f}A")
        
        # 振动告警
        if extended_data.vibration_level > 2.0:
            alerts.append(f"设备振动异常: {extended_data.vibration_level:.2f}m/s²")
        
        # 刀片磨损告警
        if extended_data.blade_wear_level > 80.0:
            alerts.append(f"刀片磨损严重: {extended_data.blade_wear_level:.1f}%")
        
        # 液压压力告警
        if extended_data.hydraulic_pressure < 2.0:
            alerts.append(f"液压压力不足: {extended_data.hydraulic_pressure:.2f}MPa")
        
        self._send_alerts(alerts, extended_data)
    
    def _send_alerts(self, alerts: List[str], data: Any):
        """发送告警信息"""
        for alert_msg in alerts:
            self.stats['alerts_sent'] += 1
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert_msg, data)
                except Exception as e:
                    logger.error(f"告警回调错误: {e}")
            
            logger.warning(f"PLC告警: {alert_msg}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前完整状态"""
        return {
            'core': asdict(self.current_core_status) if self.current_core_status else None,
            'extended': asdict(self.current_extended_status) if self.current_extended_status else None,
            'diagnostic': asdict(self.current_diagnostic_data) if self.current_diagnostic_data else None,
            'timestamp': time.time_ns()
        }
    
    def get_device_state_name(self) -> str:
        """获取设备状态名称"""
        if not self.current_core_status:
            return "未知"
        
        state_map = {
            0: "空闲",
            1: "定位中",
            2: "切割中",
            3: "故障",
            4: "回零中",
            5: "校准中",
            6: "维护模式"
        }
        
        return state_map.get(self.current_core_status.device_state, "未知状态")
    
    def is_ready_for_cutting(self) -> bool:
        """检查设备是否准备好进行切割"""
        if not self.current_core_status:
            return False
        
        return (
            self.current_core_status.device_state == 0 and  # 空闲状态
            not self.current_core_status.emergency_stop and  # 急停未触发
            self.current_core_status.fault_code == 0 and    # 无故障
            abs(self.current_core_status.position_error) < 0.1  # 位置误差小
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'core_update_rate': self.stats['core_updates'] / max(time.time() - getattr(self, '_start_time', time.time()), 1),
            'extended_update_rate': self.stats['extended_updates'] / max(time.time() - getattr(self, '_start_time', time.time()), 1),
            'estimated_bandwidth_saved': self.stats['data_bytes_saved']
        }


# 工具函数
def create_status_integration(plc_host: str = "192.168.1.20") -> PLCStatusIntegration:
    """创建PLC状态集成器"""
    from .modbus_client import ModbusTCPClient
    
    modbus_client = ModbusTCPClient(plc_host)
    integration = PLCStatusIntegration(modbus_client)
    integration._start_time = time.time()
    
    logger.info("已创建PLC状态集成器")
    return integration 