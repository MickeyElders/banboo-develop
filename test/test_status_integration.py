#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机 - 状态集成测试程序
测试PLC状态监控和数据传输优化功能

测试内容：
1. 状态数据读取和解析
2. 分层传输机制验证
3. 变化检测算法测试
4. 告警机制验证
5. 性能和带宽优化测试
"""

import time
import threading
import struct
import random
import logging
from typing import Dict, List, Any
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.communication.status_integration import (
    PLCStatusIntegration, CoreStatusData, ExtendedStatusData, DiagnosticData,
    create_status_integration
)
from src.communication.modbus_client import ModbusTCPClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockModbusTCPClient(ModbusTCPClient):
    """模拟Modbus TCP客户端，用于测试"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 502):
        super().__init__(host, port)
        
        # 模拟寄存器数据
        self.mock_registers = {}
        self._init_mock_data()
        
        # 模拟变化
        self.simulation_thread = None
        self.simulation_running = False
    
    def _init_mock_data(self):
        """初始化模拟数据"""
        # 核心状态寄存器 (0x2000-0x200B)
        timestamp = time.time_ns()
        self.mock_registers.update({
            0x2000: (timestamp >> 32) & 0xFFFFFFFF,    # 时间戳高位
            0x2001: timestamp & 0xFFFFFFFF,            # 时间戳低位
            0x2002: 0,                                 # 设备状态 (空闲)
            0x2003: 0x42C80000,                        # 实际位置 (100.0mm)
            0x2004: 0x3DCCCCCD,                        # 位置误差 (0.1mm)
            0x2005: 0x43480000,                        # 切割力 (200.0N)
            0x2006: 0x42480000,                        # 电机温度 (50.0°C)
            0x2007: 0x0000,                            # 急停+故障状态
        })
        
        # 扩展状态寄存器 (0x2010-0x201C)
        self.mock_registers.update({
            0x2010: 0x41200000,    # 电机电流 (10.0A)
            0x2011: 0x459C4000,    # 主轴转速 (5000.0RPM)
            0x2012: 0x40400000,    # 液压压力 (3.0MPa)
            0x2013: 0x42200000,    # 刀片磨损度 (40.0%)
            0x2014: 0x3F000000,    # 振动等级 (0.5m/s²)
            0x2015: 0x42280000,    # 冷却液温度 (42.0°C)
            0x2016: 0x40A00000,    # 功耗 (5.0kW)
        })
        
        # 诊断数据寄存器 (0x2020-0x2029)
        self.mock_registers.update({
            0x2020: 1000,         # 运行时间 (1000小时)
            0x2021: 5000,         # 切割次数 (5000次)
            0x2022: 10,           # 故障次数 (10次)
            0x2023: 100,          # 维护倒计时 (100小时)
            0x2024: 0x3DCCCCCD,   # 校准偏差 (0.1mm)
        })
    
    def connect(self) -> bool:
        """模拟连接"""
        self.connected = True
        logger.info("模拟Modbus连接成功")
        return True
    
    def disconnect(self):
        """模拟断开连接"""
        self.connected = False
        self.stop_simulation()
        logger.info("模拟Modbus连接断开")
    
    def read_holding_registers(self, start_address: int, count: int) -> bytes:
        """模拟读取保持寄存器"""
        if not self.connected:
            return None
        
        try:
            data = b''
            for i in range(count):
                addr = start_address + i
                if addr in self.mock_registers:
                    value = self.mock_registers[addr]
                    if isinstance(value, int):
                        if value <= 0xFFFF:
                            data += struct.pack('>H', value)
                        else:
                            data += struct.pack('>I', value)
                    else:
                        data += struct.pack('>f', value)
                else:
                    data += b'\x00\x00'
            
            return data
            
        except Exception as e:
            logger.error(f"模拟读取寄存器错误: {e}")
            return None
    
    def start_simulation(self):
        """启动数据变化模拟"""
        if self.simulation_running:
            return
        
        self.simulation_running = True
        self.simulation_thread = threading.Thread(
            target=self._simulate_data_changes,
            daemon=True
        )
        self.simulation_thread.start()
        logger.info("启动数据变化模拟")
    
    def stop_simulation(self):
        """停止数据变化模拟"""
        self.simulation_running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=1.0)
        logger.info("停止数据变化模拟")
    
    def _simulate_data_changes(self):
        """模拟数据变化"""
        state_sequence = [0, 1, 2, 0]  # 空闲->定位->切割->空闲
        state_index = 0
        cycle_count = 0
        
        while self.simulation_running:
            try:
                # 更新时间戳
                timestamp = time.time_ns()
                self.mock_registers[0x2000] = (timestamp >> 32) & 0xFFFFFFFF
                self.mock_registers[0x2001] = timestamp & 0xFFFFFFFF
                
                # 状态机模拟
                current_state = state_sequence[state_index]
                self.mock_registers[0x2002] = current_state
                
                if current_state == 0:  # 空闲状态
                    # 位置回到原点
                    self.mock_registers[0x2003] = struct.unpack('>I', struct.pack('>f', 0.0))[0]
                    self.mock_registers[0x2004] = struct.unpack('>I', struct.pack('>f', 0.0))[0]
                    self.mock_registers[0x2005] = struct.unpack('>I', struct.pack('>f', 0.0))[0]
                    
                elif current_state == 1:  # 定位状态
                    # 模拟移动到目标位置
                    target_pos = 100.0 + random.uniform(-5.0, 5.0)
                    position_error = random.uniform(-0.2, 0.2)
                    self.mock_registers[0x2003] = struct.unpack('>I', struct.pack('>f', target_pos))[0]
                    self.mock_registers[0x2004] = struct.unpack('>I', struct.pack('>f', position_error))[0]
                    
                elif current_state == 2:  # 切割状态
                    # 模拟切割过程
                    cutting_force = 200.0 + random.uniform(-50.0, 100.0)
                    motor_temp = 50.0 + random.uniform(-5.0, 15.0)
                    self.mock_registers[0x2005] = struct.unpack('>I', struct.pack('>f', cutting_force))[0]
                    self.mock_registers[0x2006] = struct.unpack('>I', struct.pack('>f', motor_temp))[0]
                
                # 模拟扩展数据变化
                self._simulate_extended_data()
                
                # 状态切换
                time.sleep(2.0)  # 每个状态持续2秒
                state_index = (state_index + 1) % len(state_sequence)
                
                if state_index == 0:
                    cycle_count += 1
                    logger.info(f"完成第{cycle_count}个工作循环")
                
                # 偶尔模拟告警情况
                if random.random() < 0.1:  # 10%概率
                    self._simulate_alert_condition()
                
            except Exception as e:
                logger.error(f"数据模拟错误: {e}")
                time.sleep(1.0)
    
    def _simulate_extended_data(self):
        """模拟扩展数据变化"""
        # 电机电流变化
        current = 10.0 + random.uniform(-2.0, 3.0)
        self.mock_registers[0x2010] = struct.unpack('>I', struct.pack('>f', current))[0]
        
        # 主轴转速变化
        rpm = 5000.0 + random.uniform(-200.0, 200.0)
        self.mock_registers[0x2011] = struct.unpack('>I', struct.pack('>f', rpm))[0]
        
        # 刀片磨损累积
        current_wear = struct.unpack('>f', struct.pack('>I', self.mock_registers[0x2013]))[0]
        new_wear = min(100.0, current_wear + random.uniform(0.0, 0.1))
        self.mock_registers[0x2013] = struct.unpack('>I', struct.pack('>f', new_wear))[0]
        
        # 振动等级变化
        vibration = 0.5 + random.uniform(-0.2, 0.3)
        self.mock_registers[0x2014] = struct.unpack('>I', struct.pack('>f', vibration))[0]
    
    def _simulate_alert_condition(self):
        """模拟告警条件"""
        alert_type = random.choice(['temperature', 'force', 'vibration', 'fault'])
        
        if alert_type == 'temperature':
            # 模拟温度过高
            high_temp = 85.0 + random.uniform(0.0, 10.0)
            self.mock_registers[0x2006] = struct.unpack('>I', struct.pack('>f', high_temp))[0]
            logger.warning(f"模拟温度告警: {high_temp:.1f}°C")
            
        elif alert_type == 'force':
            # 模拟切割力过载
            high_force = 520.0 + random.uniform(0.0, 100.0)
            self.mock_registers[0x2005] = struct.unpack('>I', struct.pack('>f', high_force))[0]
            logger.warning(f"模拟切割力告警: {high_force:.1f}N")
            
        elif alert_type == 'vibration':
            # 模拟振动异常
            high_vibration = 2.5 + random.uniform(0.0, 1.0)
            self.mock_registers[0x2014] = struct.unpack('>I', struct.pack('>f', high_vibration))[0]
            logger.warning(f"模拟振动告警: {high_vibration:.2f}m/s²")
            
        elif alert_type == 'fault':
            # 模拟设备故障
            self.mock_registers[0x2002] = 3  # 故障状态
            self.mock_registers[0x2007] = 0x0302  # 切割力过载故障
            logger.warning("模拟设备故障")


class StatusIntegrationTester:
    """状态集成测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.mock_client = MockModbusTCPClient()
        self.status_integration = PLCStatusIntegration(self.mock_client)
        
        # 测试统计
        self.test_stats = {
            'core_updates': 0,
            'extended_updates': 0,
            'alerts_received': 0,
            'test_start_time': time.time()
        }
        
        # 注册回调
        self.status_integration.add_status_callback(self._on_status_update)
        self.status_integration.add_alert_callback(self._on_alert)
        
        logger.info("状态集成测试器初始化完成")
    
    def _on_status_update(self, status_type: str, data: Any):
        """状态更新回调"""
        if status_type == 'core':
            self.test_stats['core_updates'] += 1
            core_data: CoreStatusData = data
            logger.info(f"核心状态更新: 设备状态={core_data.device_state}, "
                       f"位置={core_data.actual_position:.2f}mm, "
                       f"切割力={core_data.cutting_force:.1f}N")
                       
        elif status_type == 'extended':
            self.test_stats['extended_updates'] += 1
            extended_data: ExtendedStatusData = data
            logger.info(f"扩展状态更新: 电流={extended_data.motor_current:.1f}A, "
                       f"转速={extended_data.spindle_rpm:.0f}RPM")
    
    def _on_alert(self, alert_message: str, data: Any):
        """告警回调"""
        self.test_stats['alerts_received'] += 1
        logger.warning(f"收到告警: {alert_message}")
    
    def run_basic_test(self, duration: int = 30):
        """运行基础功能测试"""
        logger.info(f"开始基础功能测试，持续{duration}秒...")
        
        # 连接模拟客户端
        self.mock_client.connect()
        
        # 启动状态监控
        self.status_integration.start_monitoring()
        
        # 启动数据模拟
        self.mock_client.start_simulation()
        
        try:
            # 运行测试
            time.sleep(duration)
            
            # 测试诊断数据读取
            logger.info("测试诊断数据读取...")
            diagnostic_data = self.status_integration.read_diagnostic_data()
            if diagnostic_data:
                logger.info(f"诊断数据: 运行时间={diagnostic_data.runtime_hours}h, "
                           f"切割次数={diagnostic_data.total_cuts}")
            
            # 显示测试结果
            self._show_test_results()
            
        finally:
            # 清理
            self.status_integration.stop_monitoring()
            self.mock_client.stop_simulation()
            self.mock_client.disconnect()
    
    def run_performance_test(self, duration: int = 60):
        """运行性能测试"""
        logger.info(f"开始性能测试，持续{duration}秒...")
        
        self.mock_client.connect()
        self.status_integration.start_monitoring()
        self.mock_client.start_simulation()
        
        start_time = time.time()
        last_stats_time = start_time
        
        try:
            while time.time() - start_time < duration:
                current_time = time.time()
                
                # 每10秒显示一次统计信息
                if current_time - last_stats_time >= 10.0:
                    self._show_performance_stats()
                    last_stats_time = current_time
                
                time.sleep(1.0)
            
            # 最终性能报告
            self._show_final_performance_report()
            
        finally:
            self.status_integration.stop_monitoring()
            self.mock_client.stop_simulation()
            self.mock_client.disconnect()
    
    def test_alert_conditions(self):
        """测试告警条件"""
        logger.info("开始告警条件测试...")
        
        self.mock_client.connect()
        self.status_integration.start_monitoring()
        
        try:
            # 模拟各种告警条件
            alert_scenarios = [
                ('温度过高', lambda: self._set_register_float(0x2006, 90.0)),
                ('切割力过载', lambda: self._set_register_float(0x2005, 550.0)),
                ('位置误差过大', lambda: self._set_register_float(0x2004, 3.0)),
                ('振动异常', lambda: self._set_register_float(0x2014, 3.0)),
                ('设备故障', lambda: self._set_register_int(0x2002, 3)),
            ]
            
            for scenario_name, set_condition in alert_scenarios:
                logger.info(f"测试{scenario_name}告警...")
                set_condition()
                time.sleep(3.0)  # 等待告警触发
                
                # 恢复正常状态
                self._reset_to_normal()
                time.sleep(2.0)
            
            logger.info("告警条件测试完成")
            
        finally:
            self.status_integration.stop_monitoring()
            self.mock_client.disconnect()
    
    def _set_register_float(self, address: int, value: float):
        """设置浮点寄存器值"""
        self.mock_client.mock_registers[address] = struct.unpack('>I', struct.pack('>f', value))[0]
    
    def _set_register_int(self, address: int, value: int):
        """设置整数寄存器值"""
        self.mock_client.mock_registers[address] = value
    
    def _reset_to_normal(self):
        """重置到正常状态"""
        self._set_register_int(0x2002, 0)      # 空闲状态
        self._set_register_float(0x2004, 0.1)  # 正常位置误差
        self._set_register_float(0x2005, 200.0)  # 正常切割力
        self._set_register_float(0x2006, 50.0)   # 正常温度
        self._set_register_float(0x2014, 0.5)    # 正常振动
    
    def _show_test_results(self):
        """显示测试结果"""
        elapsed_time = time.time() - self.test_stats['test_start_time']
        
        print("\n" + "="*60)
        print("状态集成测试结果")
        print("="*60)
        print(f"测试时长: {elapsed_time:.1f}秒")
        print(f"核心状态更新次数: {self.test_stats['core_updates']}")
        print(f"扩展状态更新次数: {self.test_stats['extended_updates']}")
        print(f"告警接收次数: {self.test_stats['alerts_received']}")
        
        # 获取系统统计信息
        sys_stats = self.status_integration.get_statistics()
        print(f"系统核心更新: {sys_stats['core_updates']}")
        print(f"系统扩展更新: {sys_stats['extended_updates']}")
        print(f"系统告警数量: {sys_stats['alerts_sent']}")
        
        print("="*60)
    
    def _show_performance_stats(self):
        """显示性能统计"""
        stats = self.status_integration.get_statistics()
        current_status = self.status_integration.get_current_status()
        
        print(f"\n性能统计 - 核心更新: {stats['core_updates']}, "
              f"扩展更新: {stats['extended_updates']}, "
              f"告警: {stats['alerts_sent']}")
        
        if current_status['core']:
            core = current_status['core']
            print(f"当前状态 - 设备: {core['device_state']}, "
                  f"位置: {core['actual_position']:.2f}mm, "
                  f"温度: {core['motor_temp']:.1f}°C")
    
    def _show_final_performance_report(self):
        """显示最终性能报告"""
        stats = self.status_integration.get_statistics()
        elapsed_time = time.time() - self.test_stats['test_start_time']
        
        print("\n" + "="*60)
        print("性能测试最终报告")
        print("="*60)
        print(f"测试时长: {elapsed_time:.1f}秒")
        print(f"核心状态更新频率: {stats['core_updates']/elapsed_time:.2f} Hz")
        print(f"扩展状态更新频率: {stats['extended_updates']/elapsed_time:.2f} Hz")
        print(f"总数据更新: {stats['core_updates'] + stats['extended_updates']}")
        print(f"告警触发率: {stats['alerts_sent']/elapsed_time:.4f} alerts/s")
        
        # 估算带宽消耗
        core_bytes = stats['core_updates'] * 32  # 核心数据32字节
        extended_bytes = stats['extended_updates'] * 28  # 扩展数据28字节
        total_bytes = core_bytes + extended_bytes
        bandwidth_bps = (total_bytes * 8) / elapsed_time
        
        print(f"数据传输量: {total_bytes} 字节")
        print(f"平均带宽: {bandwidth_bps:.1f} bps ({bandwidth_bps/1000:.2f} Kbps)")
        print("="*60)


def main():
    """主测试函数"""
    print("智能切竹机 - 状态集成测试程序")
    print("="*50)
    
    tester = StatusIntegrationTester()
    
    try:
        # 运行基础功能测试
        print("\n1. 基础功能测试")
        tester.run_basic_test(duration=20)
        
        print("\n" + "-"*50)
        
        # 运行告警测试
        print("\n2. 告警机制测试")
        tester.test_alert_conditions()
        
        print("\n" + "-"*50)
        
        # 运行性能测试
        print("\n3. 性能优化测试")
        tester.run_performance_test(duration=30)
        
        print("\n状态集成测试全部完成！")
        
    except KeyboardInterrupt:
        logger.info("用户中断测试")
    except Exception as e:
        logger.error(f"测试程序错误: {e}")
        raise


if __name__ == "__main__":
    main() 