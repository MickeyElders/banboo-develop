#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机 - PLC状态监控模块
专门用于监控PLC状态数据，不发送控制指令

根据机械设计提供的监控功能：
1. 滑台位置和速度监控
2. 主轴转速和切割力监控
3. 电机温度和电流监控
4. 限位开关和安全状态监控
5. 夹爪和工件检测监控
"""

import time
import struct
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from pymodbus.client.sync import ModbusTcpClient
    from pymodbus.exceptions import ModbusException
    PYMODBUS_AVAILABLE = True
except ImportError:
    PYMODBUS_AVAILABLE = False
    print("pymodbus not available, using mock client")

# 配置日志
logger = logging.getLogger(__name__)


class DeviceState(Enum):
    """设备运行状态"""
    STOPPED = 0      # 停止状态
    HOMING = 1       # 回零中
    IDLE = 2         # 空闲待命
    RUNNING = 3      # 运行中
    FAULT = 4        # 故障状态


class GripperState(Enum):
    """夹爪状态"""
    UNKNOWN = 0      # 未知状态
    OPEN = 1         # 夹爪张开
    CLOSED = 2       # 夹爪闭合
    GRIPPING = 3     # 夹持工件


class CoolingState(Enum):
    """冷却系统状态"""
    OFF = 0          # 关闭
    STANDBY = 1      # 待机
    RUNNING = 2      # 运行中
    FAULT = 3        # 故障


@dataclass
class CoreMonitoringData:
    """核心监控数据 (50ms更新)"""
    timestamp: int               # PLC时间戳 (ms)
    device_state: DeviceState    # 设备运行状态
    slide_position: float        # 滑台位置 (mm)
    slide_velocity: float        # 滑台速度 (mm/s)
    spindle_rpm: int            # 主轴转速 (RPM)
    cutting_force: float         # 切割力 (N)
    limit_switches: int          # 限位开关状态 (位字段)
    emergency_stop: bool         # 急停状态
    fault_code: int             # 故障代码


@dataclass
class ExtendedMonitoringData:
    """扩展监控数据 (2s更新)"""
    servo_motor_current: float   # 伺服电机电流 (A)
    servo_motor_temp: float      # 伺服电机温度 (°C)
    spindle_motor_current: float # 主轴电机电流 (A)
    spindle_motor_temp: float    # 主轴电机温度 (°C)
    gripper_state: GripperState  # 夹爪状态
    air_pressure: float          # 气压 (kPa)
    workpiece_detected: bool     # 工件检测
    power_consumption: float     # 总功耗 (kW)
    cooling_system: CoolingState # 冷却系统状态


@dataclass
class DiagnosticData:
    """诊断数据 (按需读取)"""
    runtime_hours: int           # 运行时间 (小时)
    total_cuts: int             # 总切割次数
    fault_count: int            # 故障次数
    maintenance_countdown: int   # 维护倒计时 (小时)
    calibration_offset_x: float  # X轴校准偏差 (mm)
    calibration_offset_y: float  # Y轴校准偏差 (mm)


class PLCMonitor:
    """PLC监控客户端 - 纯监控模式"""
    
    # 寄存器地址映射
    CORE_DATA_START = 0x2000      # 核心数据起始地址
    CORE_DATA_COUNT = 13          # 核心数据寄存器数量
    
    EXTENDED_DATA_START = 0x2010  # 扩展数据起始地址
    EXTENDED_DATA_COUNT = 15      # 扩展数据寄存器数量
    
    DIAGNOSTIC_DATA_START = 0x2020  # 诊断数据起始地址
    DIAGNOSTIC_DATA_COUNT = 10      # 诊断数据寄存器数量
    
    def __init__(self, host: str = "192.168.1.10", port: int = 502, 
                 slave_id: int = 1, timeout: float = 3.0):
        """
        初始化PLC监控客户端
        
        Args:
            host: PLC IP地址
            port: Modbus TCP端口
            slave_id: 从站ID
            timeout: 超时时间
        """
        self.host = host
        self.port = port
        self.slave_id = slave_id
        self.timeout = timeout
        
        # 初始化Modbus客户端
        if PYMODBUS_AVAILABLE:
            self.client = ModbusTcpClient(host=host, port=port, timeout=timeout)
        else:
            self.client = None
            logger.warning("pymodbus不可用，将使用模拟数据")
        
        # 连接状态
        self.connected = False
        
        # 数据缓存
        self.core_data: Optional[CoreMonitoringData] = None
        self.extended_data: Optional[ExtendedMonitoringData] = None
        self.diagnostic_data: Optional[DiagnosticData] = None
        
        # 监控线程控制
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'core_reads': 0,
            'extended_reads': 0,
            'diagnostic_reads': 0,
            'read_errors': 0,
            'last_error': None
        }
        
        logger.info(f"PLC监控客户端初始化: {host}:{port}")
    
    def connect(self) -> bool:
        """
        连接到PLC
        
        Returns:
            bool: 连接是否成功
        """
        try:
            if not PYMODBUS_AVAILABLE:
                self.connected = True
                logger.info("模拟模式：PLC连接成功")
                return True
                
            if self.client.connect():
                self.connected = True
                logger.info(f"成功连接到PLC: {self.host}:{self.port}")
                return True
            else:
                logger.error(f"连接PLC失败: {self.host}:{self.port}")
                return False
                
        except Exception as e:
            logger.error(f"PLC连接异常: {e}")
            self.stats['last_error'] = str(e)
            return False
    
    def disconnect(self):
        """断开PLC连接"""
        try:
            if PYMODBUS_AVAILABLE and self.client:
                self.client.close()
            self.connected = False
            logger.info("PLC连接已断开")
        except Exception as e:
            logger.error(f"断开连接异常: {e}")
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connected
    
    def read_core_monitoring_data(self) -> Optional[CoreMonitoringData]:
        """
        读取核心监控数据
        
        Returns:
            CoreMonitoringData: 核心监控数据或None
        """
        try:
            if not self.is_connected():
                return None
            
            if not PYMODBUS_AVAILABLE:
                return self._get_mock_core_data()
            
            # 读取寄存器
            result = self.client.read_holding_registers(
                self.CORE_DATA_START, 
                self.CORE_DATA_COUNT,
                unit=self.slave_id
            )
            
            if result.isError():
                logger.error(f"读取核心数据失败: {result}")
                self.stats['read_errors'] += 1
                return None
            
            # 解析数据
            registers = result.registers
            core_data = self._parse_core_data(registers)
            
            with self.lock:
                self.core_data = core_data
                self.stats['core_reads'] += 1
            
            return core_data
            
        except Exception as e:
            logger.error(f"读取核心数据异常: {e}")
            self.stats['read_errors'] += 1
            self.stats['last_error'] = str(e)
            return None
    
    def read_extended_monitoring_data(self) -> Optional[ExtendedMonitoringData]:
        """
        读取扩展监控数据
        
        Returns:
            ExtendedMonitoringData: 扩展监控数据或None
        """
        try:
            if not self.is_connected():
                return None
            
            if not PYMODBUS_AVAILABLE:
                return self._get_mock_extended_data()
            
            # 读取寄存器
            result = self.client.read_holding_registers(
                self.EXTENDED_DATA_START,
                self.EXTENDED_DATA_COUNT,
                unit=self.slave_id
            )
            
            if result.isError():
                logger.error(f"读取扩展数据失败: {result}")
                self.stats['read_errors'] += 1
                return None
            
            # 解析数据
            registers = result.registers
            extended_data = self._parse_extended_data(registers)
            
            with self.lock:
                self.extended_data = extended_data
                self.stats['extended_reads'] += 1
            
            return extended_data
            
        except Exception as e:
            logger.error(f"读取扩展数据异常: {e}")
            self.stats['read_errors'] += 1
            self.stats['last_error'] = str(e)
            return None
    
    def read_diagnostic_data(self) -> Optional[DiagnosticData]:
        """
        读取诊断数据
        
        Returns:
            DiagnosticData: 诊断数据或None
        """
        try:
            if not self.is_connected():
                return None
            
            if not PYMODBUS_AVAILABLE:
                return self._get_mock_diagnostic_data()
            
            # 读取寄存器
            result = self.client.read_holding_registers(
                self.DIAGNOSTIC_DATA_START,
                self.DIAGNOSTIC_DATA_COUNT,
                unit=self.slave_id
            )
            
            if result.isError():
                logger.error(f"读取诊断数据失败: {result}")
                self.stats['read_errors'] += 1
                return None
            
            # 解析数据
            registers = result.registers
            diagnostic_data = self._parse_diagnostic_data(registers)
            
            with self.lock:
                self.diagnostic_data = diagnostic_data
                self.stats['diagnostic_reads'] += 1
            
            return diagnostic_data
            
        except Exception as e:
            logger.error(f"读取诊断数据异常: {e}")
            self.stats['read_errors'] += 1
            self.stats['last_error'] = str(e)
            return None
    
    def _parse_core_data(self, registers: List[int]) -> CoreMonitoringData:
        """解析核心数据寄存器"""
        # 解析32位时间戳
        timestamp = (registers[0] << 16) | registers[1]
        
        # 解析设备状态
        device_state = DeviceState(registers[2])
        
        # 解析32位浮点数 - 滑台位置
        pos_bytes = struct.pack('>HH', registers[3], registers[4])
        slide_position = struct.unpack('>f', pos_bytes)[0]
        
        # 解析32位浮点数 - 滑台速度
        vel_bytes = struct.pack('>HH', registers[5], registers[6])
        slide_velocity = struct.unpack('>f', vel_bytes)[0]
        
        # 主轴转速
        spindle_rpm = registers[7]
        
        # 解析32位浮点数 - 切割力
        force_bytes = struct.pack('>HH', registers[8], registers[9])
        cutting_force = struct.unpack('>f', force_bytes)[0]
        
        # 限位开关状态
        limit_switches = registers[10]
        
        # 急停状态
        emergency_stop = bool(registers[11])
        
        # 故障代码
        fault_code = registers[12]
        
        return CoreMonitoringData(
            timestamp=timestamp,
            device_state=device_state,
            slide_position=slide_position,
            slide_velocity=slide_velocity,
            spindle_rpm=spindle_rpm,
            cutting_force=cutting_force,
            limit_switches=limit_switches,
            emergency_stop=emergency_stop,
            fault_code=fault_code
        )
    
    def _parse_extended_data(self, registers: List[int]) -> ExtendedMonitoringData:
        """解析扩展数据寄存器"""
        # 伺服电机电流
        servo_current_bytes = struct.pack('>HH', registers[0], registers[1])
        servo_motor_current = struct.unpack('>f', servo_current_bytes)[0]
        
        # 伺服电机温度
        servo_temp_bytes = struct.pack('>HH', registers[2], registers[3])
        servo_motor_temp = struct.unpack('>f', servo_temp_bytes)[0]
        
        # 主轴电机电流
        spindle_current_bytes = struct.pack('>HH', registers[4], registers[5])
        spindle_motor_current = struct.unpack('>f', spindle_current_bytes)[0]
        
        # 主轴电机温度
        spindle_temp_bytes = struct.pack('>HH', registers[6], registers[7])
        spindle_motor_temp = struct.unpack('>f', spindle_temp_bytes)[0]
        
        # 夹爪状态
        gripper_state = GripperState(registers[8])
        
        # 气压
        air_pressure_bytes = struct.pack('>HH', registers[9], registers[10])
        air_pressure = struct.unpack('>f', air_pressure_bytes)[0]
        
        # 工件检测
        workpiece_detected = bool(registers[11])
        
        # 功耗
        power_bytes = struct.pack('>HH', registers[12], registers[13])
        power_consumption = struct.unpack('>f', power_bytes)[0]
        
        # 冷却系统状态
        cooling_system = CoolingState(registers[14])
        
        return ExtendedMonitoringData(
            servo_motor_current=servo_motor_current,
            servo_motor_temp=servo_motor_temp,
            spindle_motor_current=spindle_motor_current,
            spindle_motor_temp=spindle_motor_temp,
            gripper_state=gripper_state,
            air_pressure=air_pressure,
            workpiece_detected=workpiece_detected,
            power_consumption=power_consumption,
            cooling_system=cooling_system
        )
    
    def _parse_diagnostic_data(self, registers: List[int]) -> DiagnosticData:
        """解析诊断数据寄存器"""
        # 运行时间 (32位)
        runtime_hours = (registers[0] << 16) | registers[1]
        
        # 总切割次数 (32位)
        total_cuts = (registers[2] << 16) | registers[3]
        
        # 故障次数
        fault_count = registers[4]
        
        # 维护倒计时
        maintenance_countdown = registers[5]
        
        # X轴校准偏差
        cal_x_bytes = struct.pack('>HH', registers[6], registers[7])
        calibration_offset_x = struct.unpack('>f', cal_x_bytes)[0]
        
        # Y轴校准偏差
        cal_y_bytes = struct.pack('>HH', registers[8], registers[9])
        calibration_offset_y = struct.unpack('>f', cal_y_bytes)[0]
        
        return DiagnosticData(
            runtime_hours=runtime_hours,
            total_cuts=total_cuts,
            fault_count=fault_count,
            maintenance_countdown=maintenance_countdown,
            calibration_offset_x=calibration_offset_x,
            calibration_offset_y=calibration_offset_y
        )
    
    def _get_mock_core_data(self) -> CoreMonitoringData:
        """生成模拟核心数据"""
        import random
        return CoreMonitoringData(
            timestamp=int(time.time() * 1000) % (2**32),
            device_state=DeviceState.IDLE,
            slide_position=random.uniform(0, 1000),
            slide_velocity=random.uniform(-50, 50),
            spindle_rpm=random.randint(0, 3000),
            cutting_force=random.uniform(0, 50),
            limit_switches=0,  # 无限位触发
            emergency_stop=False,
            fault_code=0
        )
    
    def _get_mock_extended_data(self) -> ExtendedMonitoringData:
        """生成模拟扩展数据"""
        import random
        return ExtendedMonitoringData(
            servo_motor_current=random.uniform(2, 8),
            servo_motor_temp=random.uniform(25, 45),
            spindle_motor_current=random.uniform(1, 6),
            spindle_motor_temp=random.uniform(30, 50),
            gripper_state=GripperState.OPEN,
            air_pressure=random.uniform(600, 800),
            workpiece_detected=False,
            power_consumption=random.uniform(1, 5),
            cooling_system=CoolingState.STANDBY
        )
    
    def _get_mock_diagnostic_data(self) -> DiagnosticData:
        """生成模拟诊断数据"""
        return DiagnosticData(
            runtime_hours=128,
            total_cuts=1567,
            fault_count=3,
            maintenance_countdown=872,
            calibration_offset_x=0.02,
            calibration_offset_y=-0.01
        )
    
    def start_monitoring(self, core_interval: float = 0.05, 
                        extended_interval: float = 2.0):
        """
        启动监控线程
        
        Args:
            core_interval: 核心数据读取间隔 (秒)
            extended_interval: 扩展数据读取间隔 (秒)
        """
        if self.monitoring_active:
            logger.warning("监控已经在运行")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(core_interval, extended_interval),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("PLC监控线程已启动")
    
    def stop_monitoring(self):
        """停止监控线程"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            self.monitor_thread = None
        logger.info("PLC监控线程已停止")
    
    def _monitor_loop(self, core_interval: float, extended_interval: float):
        """监控主循环"""
        last_extended_read = 0
        
        while self.monitoring_active:
            try:
                # 读取核心数据 (高频)
                self.read_core_monitoring_data()
                
                # 读取扩展数据 (低频)
                current_time = time.time()
                if current_time - last_extended_read >= extended_interval:
                    self.read_extended_monitoring_data()
                    last_extended_read = current_time
                
                time.sleep(core_interval)
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(1.0)  # 异常时延长等待时间
    
    def get_latest_data(self) -> Dict[str, Any]:
        """
        获取最新的所有监控数据
        
        Returns:
            Dict: 包含所有最新数据的字典
        """
        with self.lock:
            return {
                'core': self.core_data,
                'extended': self.extended_data,
                'diagnostic': self.diagnostic_data,
                'stats': self.stats.copy(),
                'connected': self.connected
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def __enter__(self):
        """上下文管理器进入"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop_monitoring()
        self.disconnect()


def main():
    """测试PLC监控功能"""
    logging.basicConfig(level=logging.INFO)
    
    # 测试PLC监控
    monitor = PLCMonitor()
    
    try:
        # 连接PLC
        if monitor.connect():
            print("PLC连接成功")
            
            # 启动监控
            monitor.start_monitoring()
            
            # 测试读取数据
            for i in range(10):
                time.sleep(1)
                
                # 获取最新数据
                data = monitor.get_latest_data()
                
                if data['core']:
                    core = data['core']
                    print(f"核心数据 #{i+1}:")
                    print(f"  设备状态: {core.device_state}")
                    print(f"  滑台位置: {core.slide_position:.2f} mm")
                    print(f"  主轴转速: {core.spindle_rpm} RPM")
                    print(f"  切割力: {core.cutting_force:.1f} N")
                    print(f"  急停状态: {core.emergency_stop}")
                
                if data['extended'] and i % 3 == 0:  # 每3秒显示一次扩展数据
                    ext = data['extended']
                    print(f"扩展数据:")
                    print(f"  伺服温度: {ext.servo_motor_temp:.1f} °C")
                    print(f"  夹爪状态: {ext.gripper_state}")
                    print(f"  功耗: {ext.power_consumption:.2f} kW")
                
                print(f"统计: {data['stats']}")
                print("-" * 40)
        
        else:
            print("PLC连接失败")
    
    except KeyboardInterrupt:
        print("用户中断")
    
    finally:
        monitor.stop_monitoring()
        monitor.disconnect()
        print("测试结束")


if __name__ == "__main__":
    main() 