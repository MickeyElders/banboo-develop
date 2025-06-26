# PLC监控数据规范

## 概述

本文档详细说明智能切竹机系统中PLC需要提供的监控数据接口、寄存器映射和通信协议要求。

**注意**：本系统为纯监控模式，Jetson Nano只读取PLC状态信息，不向PLC发送控制指令。

## 1. 通信接口要求

### 1.1 通信协议
- **协议**: Modbus TCP
- **端口**: 502
- **从站ID**: 1
- **字节序**: 大端字节序 (Big Endian)
- **超时时间**: 3秒

### 1.2 网络配置
- **PLC IP地址**: 192.168.1.10 (默认，可配置)
- **子网掩码**: 255.255.255.0
- **网关**: 192.168.1.1
- **Jetson Nano IP**: 192.168.1.100

### 1.3 通信模式
- **模式**: 只读模式（Read-Only）
- **功能**: Jetson Nano仅读取PLC状态数据
- **限制**: 不发送任何控制指令到PLC

## 2. 数据接口规范

### 2.1 核心监控数据 (高频传输 - 50ms)

#### 寄存器映射表 (起始地址: 0x2000)

| 寄存器地址 | 数据类型 | 变量名 | 单位 | 取值范围 | 描述 |
|-----------|----------|---------|------|----------|------|
| 0x2000 | UINT32 | timestamp | ms | 0-4294967295 | PLC时间戳 |
| 0x2002 | UINT16 | device_state | - | 0-4 | 设备运行状态 |
| 0x2003 | FLOAT32 | slide_position | mm | 0-1000 | 滑台在导轨位置 |
| 0x2005 | FLOAT32 | slide_velocity | mm/s | -200 to +200 | 滑台运动速度 |
| 0x2007 | UINT16 | spindle_rpm | RPM | 0-3000 | 主轴实际转速 |
| 0x2008 | FLOAT32 | cutting_force | N | 0-100 | 切割力反馈 |
| 0x200A | UINT16 | limit_switches | - | Bit位 | 限位开关状态 |
| 0x200B | UINT16 | emergency_stop | - | 0-1 | 急停按钮状态 |
| 0x200C | UINT16 | fault_code | - | 0-65535 | 故障代码 |

#### 设备状态定义 (device_state)
```
0 = STOPPED     // 停止状态
1 = HOMING      // 回零中
2 = IDLE        // 空闲待命
3 = RUNNING     // 运行中
4 = FAULT       // 故障状态
```

#### 限位开关状态定义 (limit_switches)
```
Bit 0: 滑台负限位 (0=未触发, 1=触发)
Bit 1: 滑台正限位 (0=未触发, 1=触发)  
Bit 2: 滑台原点 (0=未触发, 1=触发)
Bit 3: 切割深度限位 (0=未触发, 1=触发)
Bit 4: 安全门开关 (0=关闭, 1=打开)
Bit 5: 防护罩状态 (0=关闭, 1=打开)
Bit 6-15: 保留
```

#### 故障代码定义 (fault_code)
```
0x0000 = 无故障
0x0001 = 伺服电机超温
0x0002 = 位置跟随误差过大
0x0003 = 切割力超限
0x0004 = 通信超时
0x0005 = 急停触发
0x0006 = 限位开关触发
0x0007 = 电源电压异常
0x0008 = 编码器故障
0x0009 = 刀具磨损严重
0x000A = 液压系统故障
```

### 2.2 扩展监控数据 (低频传输 - 2s)

#### 寄存器映射表 (起始地址: 0x2010)

| 寄存器地址 | 数据类型 | 变量名 | 单位 | 取值范围 | 描述 |
|-----------|----------|---------|------|----------|------|
| 0x2010 | FLOAT32 | servo_motor_current | A | 0-20 | 伺服电机电流 |
| 0x2012 | FLOAT32 | servo_motor_temp | °C | -40 to +120 | 伺服电机温度 |
| 0x2014 | FLOAT32 | spindle_motor_current | A | 0-15 | 主轴电机电流 |
| 0x2016 | FLOAT32 | spindle_motor_temp | °C | -40 to +120 | 主轴电机温度 |
| 0x2018 | UINT16 | gripper_state | - | 0-3 | 夹爪状态 |
| 0x2019 | FLOAT32 | air_pressure | kPa | 0-1000 | 夹爪气压 |
| 0x201B | UINT16 | workpiece_detected | - | 0-1 | 工件检测 |
| 0x201C | FLOAT32 | power_consumption | kW | 0-10 | 总功耗 |
| 0x201E | UINT16 | cooling_system | - | 0-3 | 冷却系统状态 |

#### 夹爪状态定义 (gripper_state)
```
0 = UNKNOWN     // 未知状态
1 = OPEN        // 夹爪张开
2 = CLOSED      // 夹爪闭合
3 = GRIPPING    // 夹持工件
```

#### 冷却系统状态定义 (cooling_system)
```
0 = OFF         // 关闭
1 = STANDBY     // 待机
2 = RUNNING     // 运行中
3 = FAULT       // 故障
```

### 2.3 诊断数据 (按需传输)

#### 寄存器映射表 (起始地址: 0x2020)

| 寄存器地址 | 数据类型 | 变量名 | 单位 | 取值范围 | 描述 |
|-----------|----------|---------|------|----------|------|
| 0x2020 | UINT32 | runtime_hours | 小时 | 0-4294967295 | 运行时间 |
| 0x2022 | UINT32 | total_cuts | 次 | 0-4294967295 | 总切割次数 |
| 0x2024 | UINT16 | fault_count | 次 | 0-65535 | 故障次数 |
| 0x2025 | UINT16 | maintenance_countdown | 小时 | 0-1000 | 维护倒计时 |
| 0x2026 | FLOAT32 | calibration_offset_x | mm | -5 to +5 | X轴校准偏差 |
| 0x2028 | FLOAT32 | calibration_offset_y | mm | -5 to +5 | Y轴校准偏差 |

## 3. 数据格式要求

### 3.1 浮点数格式
- **标准**: IEEE 754单精度浮点 (32位)
- **字节序**: 大端字节序
- **精度要求**:
  - 位置数据: 0.01mm
  - 力值数据: 0.1N
  - 温度数据: 0.1°C
  - 压力数据: 0.01MPa

### 3.2 时间戳格式
- **类型**: 32位无符号整数
- **单位**: 毫秒
- **基准**: PLC系统启动时间
- **更新频率**: 每个采样周期

### 3.3 状态位定义
- **布尔值**: 0=False, 1=True
- **多状态值**: 使用枚举定义
- **保留位**: 设置为0

## 4. 通信时序要求

### 4.1 数据读取周期
```
核心状态数据: 50ms (20Hz)
扩展状态数据: 2000ms (0.5Hz)
诊断数据: 按需读取
```

### 4.2 响应时间要求
```
数据更新响应: < 50ms
急停状态反馈: < 10ms (硬件级)
故障状态反馈: < 100ms
```

### 4.3 超时处理
```
读取超时: 3秒
连接超时: 5秒
重连间隔: 1秒
```

## 5. 安全要求

### 5.1 状态监控安全
- **急停状态**: 实时反馈急停按钮状态
- **限位状态**: 及时报告限位开关触发
- **故障状态**: 快速反馈设备故障信息

### 5.2 通信安全
- **连接监控**: 监控Modbus连接状态
- **数据校验**: 使用CRC校验确保数据完整性
- **超时处理**: 通信超时时的安全处理

### 5.3 数据完整性
- **时间戳校验**: 检测数据是否及时更新
- **范围检查**: 验证传感器数据合理性
- **状态一致性**: 确保相关状态逻辑一致

## 6. 测试要求

### 6.1 功能测试
```python
# 测试脚本示例 - 纯监控模式
def test_plc_monitoring():
    # 1. 连接测试
    assert connect_to_plc() == True
    
    # 2. 读取核心监控数据
    core_data = read_core_monitoring_data()
    assert core_data is not None
    assert 0 <= core_data.device_state <= 4
    assert 0 <= core_data.slide_position <= 1000
    
    # 3. 读取扩展监控数据
    extended_data = read_extended_monitoring_data()
    assert extended_data is not None
    assert extended_data.servo_motor_temp < 80  # 温度检查
    
    # 4. 急停状态检测
    assert core_data.emergency_stop in [0, 1]
    
    # 5. 限位开关状态检测
    limit_status = core_data.limit_switches
    assert isinstance(limit_status, int)
```

### 6.2 性能测试
- **通信延迟**: < 10ms
- **数据读取速率**: > 20Hz (核心数据)
- **连续运行**: 24小时无通信中断

### 6.3 稳定性测试
- **高频读取**: 连续1小时满负荷读取
- **异常恢复**: 模拟网络中断自动恢复
- **边界条件**: 极限数值处理

## 7. 配置示例

### 7.1 PLC网络配置
```yaml
plc_monitoring_config:
  ip_address: "192.168.1.10"
  port: 502
  slave_id: 1
  timeout: 3.0
  retry_count: 3
  byte_order: "big"
  read_only: true  # 纯监控模式
```

### 7.2 寄存器映射配置
```yaml
register_map:
  core_monitoring:
    start_address: 0x2000
    register_count: 13
    read_frequency: 50  # ms
    data_type: "mixed"
  
  extended_monitoring:
    start_address: 0x2010
    register_count: 15
    read_frequency: 2000  # ms
    data_type: "mixed"
  
  diagnostic_data:
    start_address: 0x2020
    register_count: 10
    read_frequency: 0  # on-demand
    data_type: "mixed"
```

## 8. 故障排除

### 8.1 常见问题
| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 连接超时 | 网络配置错误 | 检查IP地址和端口 |
| 数据异常 | 字节序不匹配 | 确认大端字节序 |
| 读取失败 | 寄存器地址错误 | 验证寄存器映射 |
| 时间戳不更新 | PLC程序问题 | 检查PLC时间戳更新逻辑 |

### 8.2 调试工具
- **Modbus调试软件**: ModbusPoll, QModBus
- **网络抓包**: Wireshark
- **PLC监控**: 各厂商专用软件

## 9. 系统架构

### 9.1 数据流向
```
PLC传感器系统 → PLC处理器 → Modbus TCP → Jetson Nano → AI分析
```

### 9.2 职责分工
- **PLC系统**: 设备控制、数据采集、状态管理
- **Jetson Nano**: 数据监控、AI视觉分析、界面显示
- **通信系统**: 仅用于状态数据传输

## 10. 版本历史

| 版本 | 日期 | 修改内容 | 作者 |
|------|------|----------|------|
| 1.0 | 2024-01-01 | 初始版本 | 开发团队 |
| 1.1 | 2024-01-15 | 增加诊断数据 | 开发团队 |
| 2.0 | 2024-02-01 | 调整为纯监控模式 | 开发团队 |

---

**重要说明**：
1. 本系统为纯监控模式，Jetson Nano不控制PLC
2. 所有数据类型必须严格按照规范实现
3. 通信频率不可超过规定值，避免网络拥塞
4. PLC程序更新前必须进行兼容性验证
5. 急停和安全功能完全依赖PLC硬件实现 