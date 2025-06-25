#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机 - Modbus TCP通信客户端
实现与PLC控制器的实时通信

主要功能：
1. Modbus TCP连接管理
2. 切割指令发送
3. 设备状态查询
4. 安全校验与错误处理
"""

import socket
import struct
import time
import hashlib
import threading
from typing import Optional, Dict, Any, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import crc32c

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceState(Enum):
    """设备状态枚举"""
    IDLE = 0         # 空闲
    POSITIONING = 1  # 定位中
    CUTTING = 2      # 切割中
    FAULT = 3        # 故障


class FaultCode(Enum):
    """故障代码枚举"""
    POSITION_LIMIT = 0x0301    # 位置超限
    CUTTING_OVERLOAD = 0x0302  # 切割力过载
    MOTOR_OVERHEAT = 0x0303    # 电机过热
    COMM_TIMEOUT = 0x0304      # 通信超时
    EMERGENCY_STOP = 0x0305    # 急停触发
    TOOL_FAULT = 0x0306        # 刀具故障


@dataclass
class CuttingCommand:
    """切割指令数据类"""
    sequence_id: int           # 序列号
    timestamp: int             # 时间戳 (纳秒)
    target_position: float     # 目标位置 (mm)
    cutting_speed: float       # 切割速度 (mm/s)
    tool_id: int              # 刀具选择 (0:主刀 1:备用刀)


@dataclass  
class DeviceStatus:
    """设备状态数据类"""
    current_state: DeviceState # 当前状态
    actual_position: float     # 实际位置 (mm)
    position_error: float      # 位置误差 (mm)
    cutting_force: float       # 切割力 (N)
    motor_temp: float          # 电机温度 (℃)
    emergency_stop: bool       # 急停状态


class ModbusFrame:
    """Modbus TCP帧结构"""
    
    def __init__(self):
        self.transaction_id = 0    # 事务标识符
        self.protocol_id = 0       # 协议标识符 (固定0x0000)
        self.length = 0            # 长度字段
        self.unit_id = 1           # 单元标识符
        
        # 扩展安全头部
        self.sequence_id = 0       # 序列号
        self.timestamp = 0         # 时间戳
        self.crc32 = 0             # CRC32校验
        self.signature = b''       # 数字签名
    
    def pack_header(self) -> bytes:
        """打包Modbus TCP头部"""
        header = struct.pack('>HHHB', 
                           self.transaction_id,
                           self.protocol_id,
                           self.length,
                           self.unit_id)
        return header
    
    def pack_security_header(self) -> bytes:
        """打包扩展安全头部"""
        security_header = struct.pack('>IQ', 
                                    self.sequence_id,
                                    self.timestamp)
        security_header += struct.pack('>I', self.crc32)
        security_header += self.signature  # 32字节签名
        return security_header


class ModbusTCPClient:
    """Modbus TCP客户端"""
    
    def __init__(self, host: str = '192.168.1.20', port: int = 502):
        """
        初始化Modbus TCP客户端
        Args:
            host: PLC IP地址
            port: Modbus TCP端口
        """
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.connected = False
        
        # 通信参数
        self.timeout = 5.0  # 超时时间 (秒)
        self.retry_count = 3  # 重试次数
        self.sequence_counter = 0  # 序列号计数器
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'last_error': None
        }
        
        logger.info(f"Modbus TCP客户端初始化，目标: {host}:{port}")
    
    def connect(self) -> bool:
        """
        建立TCP连接
        Returns:
            连接是否成功
        """
        try:
            if self.socket:
                self.disconnect()
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            
            self.connected = True
            logger.info(f"成功连接到PLC: {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"连接PLC失败: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """断开TCP连接"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            finally:
                self.socket = None
                self.connected = False
        logger.info("已断开PLC连接")
    
    def _get_next_sequence_id(self) -> int:
        """获取下一个序列号"""
        with self.lock:
            self.sequence_counter += 1
            return self.sequence_counter
    
    def _calculate_crc32(self, data: bytes) -> int:
        """计算CRC32校验值"""
        return crc32c.crc32c(data)
    
    def _generate_signature(self, data: bytes) -> bytes:
        """生成数字签名 (简化实现)"""
        # 实际项目中应使用ECDSA P-256
        hash_obj = hashlib.sha256(data)
        signature = hash_obj.digest()
        # 填充到32字节
        return signature + b'\x00' * (32 - len(signature))
    
    def _send_request(self, function_code: int, data: bytes) -> Optional[bytes]:
        """
        发送Modbus请求
        Args:
            function_code: 功能码
            data: 数据部分
        Returns:
            响应数据或None
        """
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            # 构建Modbus帧
            frame = ModbusFrame()
            frame.transaction_id = self._get_next_sequence_id()
            frame.sequence_id = frame.transaction_id
            frame.timestamp = int(time.time_ns())
            
            # 构建请求数据
            request_data = struct.pack('B', function_code) + data
            
            # 计算CRC校验
            frame.crc32 = self._calculate_crc32(request_data)
            
            # 生成数字签名
            frame.signature = self._generate_signature(request_data)
            
            # 计算总长度
            security_header_len = 48  # 4+8+4+32
            frame.length = 1 + len(request_data) + security_header_len
            
            # 打包完整请求
            request = frame.pack_header()
            request += frame.pack_security_header()
            request += request_data
            
            # 发送请求
            self.socket.sendall(request)
            
            # 接收响应
            response_header = self.socket.recv(7)  # Modbus TCP头部
            if len(response_header) < 7:
                raise Exception("响应头部不完整")
            
            # 解析响应长度
            _, _, length, _ = struct.unpack('>HHHB', response_header)
            
            # 接收响应数据
            response_data = self.socket.recv(length - 1)
            
            self.stats['successful_requests'] += 1
            return response_data
            
        except Exception as e:
            logger.error(f"Modbus请求失败: {e}")
            self.stats['failed_requests'] += 1
            self.stats['last_error'] = str(e)
            self.connected = False
            return None
        
        finally:
            self.stats['total_requests'] += 1
    
    def write_cutting_command(self, command: CuttingCommand) -> bool:
        """
        写入切割指令
        Args:
            command: 切割指令
        Returns:
            是否成功
        """
        try:
            # 构建写入数据 (功能码0x10 - 写多个保持寄存器)
            start_address = 0x1000  # 起始地址
            register_count = 8      # 寄存器数量
            byte_count = 16         # 字节数
            
            # 打包参数数据
            param_data = struct.pack('>HHB', start_address, register_count, byte_count)
            param_data += struct.pack('>f', command.target_position)  # 目标位置
            param_data += struct.pack('>f', command.cutting_speed)    # 切割速度
            param_data += struct.pack('>I', command.tool_id)          # 刀具选择
            param_data += b'\x00\x00\x00\x00'                        # 保留字段
            
            # 发送请求
            response = self._send_request(0x10, param_data)
            
            if response:
                logger.info(f"切割指令发送成功: 位置={command.target_position}mm, "
                          f"速度={command.cutting_speed}mm/s")
                return True
            else:
                logger.error("切割指令发送失败")
                return False
                
        except Exception as e:
            logger.error(f"写入切割指令异常: {e}")
            return False
    
    def read_device_status(self) -> Optional[DeviceStatus]:
        """
        读取设备状态
        Returns:
            设备状态或None
        """
        try:
            # 构建读取数据 (功能码0x03 - 读保持寄存器)
            start_address = 0x2000  # 起始地址
            register_count = 16     # 寄存器数量
            
            read_data = struct.pack('>HH', start_address, register_count)
            
            # 发送请求
            response = self._send_request(0x03, read_data)
            
            if response and len(response) >= 33:  # 功能码(1) + 字节数(1) + 数据(32)
                # 解析响应数据
                byte_count = response[0]
                data = response[1:byte_count+1]
                
                if len(data) >= 32:
                    # 解析状态数据
                    state_raw = struct.unpack('>I', data[0:4])[0]
                    current_state = DeviceState(state_raw & 0xFF)
                    
                    actual_position = struct.unpack('>f', data[4:8])[0]
                    position_error = struct.unpack('>f', data[8:12])[0]
                    cutting_force = struct.unpack('>f', data[12:16])[0]
                    motor_temp = struct.unpack('>f', data[16:20])[0]
                    emergency_stop = struct.unpack('>I', data[20:24])[0] != 0
                    
                    status = DeviceStatus(
                        current_state=current_state,
                        actual_position=actual_position,
                        position_error=position_error,
                        cutting_force=cutting_force,
                        motor_temp=motor_temp,
                        emergency_stop=emergency_stop
                    )
                    
                    logger.debug(f"设备状态: {current_state.name}, "
                               f"位置: {actual_position:.2f}mm")
                    return status
            
            logger.error("设备状态响应数据不完整")
            return None
            
        except Exception as e:
            logger.error(f"读取设备状态异常: {e}")
            return None
    
    def send_reset_command(self) -> bool:
        """
        发送复位指令
        Returns:
            是否成功
        """
        try:
            # 构建复位指令
            reset_data = struct.pack('>HHI', 0x1005, 1, 0xFFFF0000)  # 复位命令
            
            response = self._send_request(0x06, reset_data)  # 功能码0x06 - 写单个寄存器
            
            if response:
                logger.info("设备复位指令发送成功")
                return True
            else:
                logger.error("设备复位指令发送失败")
                return False
                
        except Exception as e:
            logger.error(f"发送复位指令异常: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取通信统计信息"""
        success_rate = 0.0
        if self.stats['total_requests'] > 0:
            success_rate = self.stats['successful_requests'] / self.stats['total_requests'] * 100
        
        return {
            'connected': self.connected,
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'success_rate': f"{success_rate:.1f}%",
            'last_error': self.stats['last_error']
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()


class CuttingController:
    """切割控制器 - 高级封装"""
    
    def __init__(self, plc_host: str = '192.168.1.20'):
        """
        初始化切割控制器
        Args:
            plc_host: PLC主机地址
        """
        self.client = ModbusTCPClient(plc_host)
        self.current_status: Optional[DeviceStatus] = None
        self.last_update_time = 0
        
        # 状态监控线程
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval: float = 1.0):
        """
        开始状态监控
        Args:
            interval: 监控间隔 (秒)
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("开始设备状态监控")
    
    def stop_monitoring(self):
        """停止状态监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("停止设备状态监控")
    
    def _monitor_loop(self, interval: float):
        """状态监控循环"""
        while self.monitoring:
            try:
                status = self.client.read_device_status()
                if status:
                    self.current_status = status
                    self.last_update_time = time.time()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"状态监控异常: {e}")
                time.sleep(interval)
    
    def execute_cutting(self, position: float, speed: float = 50.0, 
                       tool_id: int = 0) -> bool:
        """
        执行切割操作
        Args:
            position: 切割位置 (mm)
            speed: 切割速度 (mm/s)
            tool_id: 刀具选择
        Returns:
            是否成功
        """
        # 检查设备状态
        if not self.is_ready():
            logger.error("设备未就绪，无法执行切割")
            return False
        
        # 构建切割指令
        command = CuttingCommand(
            sequence_id=int(time.time()),
            timestamp=int(time.time_ns()),
            target_position=position,
            cutting_speed=speed,
            tool_id=tool_id
        )
        
        # 发送指令
        if not self.client.write_cutting_command(command):
            return False
        
        # 等待执行完成
        return self._wait_for_completion(timeout=30.0)
    
    def _wait_for_completion(self, timeout: float = 30.0) -> bool:
        """
        等待操作完成
        Args:
            timeout: 超时时间 (秒)
        Returns:
            是否成功完成
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.client.read_device_status()
            if not status:
                time.sleep(0.1)
                continue
            
            if status.current_state == DeviceState.IDLE:
                logger.info("切割操作完成")
                return True
            elif status.current_state == DeviceState.FAULT:
                logger.error("切割操作失败，设备故障")
                return False
            
            time.sleep(0.5)
        
        logger.error("切割操作超时")
        return False
    
    def is_ready(self) -> bool:
        """检查设备是否就绪"""
        if not self.client.connected:
            self.client.connect()
        
        status = self.client.read_device_status()
        if not status:
            return False
        
        return (status.current_state == DeviceState.IDLE and 
                not status.emergency_stop)
    
    def emergency_stop(self) -> bool:
        """紧急停止"""
        logger.warning("执行紧急停止")
        return self.client.send_reset_command()
    
    def get_current_status(self) -> Optional[DeviceStatus]:
        """获取当前设备状态"""
        return self.current_status


def main():
    """主函数 - 测试代码"""
    # 创建切割控制器
    controller = CuttingController('192.168.1.20')
    
    try:
        # 开始监控
        controller.start_monitoring()
        
        # 检查设备状态
        if controller.is_ready():
            print("设备就绪，开始切割测试")
            
            # 执行切割
            success = controller.execute_cutting(position=150.0, speed=30.0)
            if success:
                print("切割测试完成")
            else:
                print("切割测试失败")
        else:
            print("设备未就绪")
        
        # 显示统计信息
        stats = controller.client.get_statistics()
        print(f"通信统计: {stats}")
        
    finally:
        # 停止监控
        controller.stop_monitoring()
        controller.client.disconnect()


if __name__ == "__main__":
    main() 