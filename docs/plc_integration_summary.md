# PLC集成实施总结

## 系统架构调整

根据您的要求"不需要控制PLC，只需要PLC系统信息"，已将系统调整为**纯监控模式**。

### 架构变更

**之前：双向控制架构**
```
Jetson Nano ←→ Modbus TCP ←→ PLC
     (控制 + 监控)
```

**现在：单向监控架构**  
```
Jetson Nano ← Modbus TCP ← PLC
     (仅监控)
```

---

## 数据接口规范

### 核心监控数据 (50ms高频)
| 数据项 | 寄存器 | 类型 | 范围 | 说明 |
|--------|--------|------|------|------|
| PLC时间戳 | 0x2000-0x2001 | UINT32 | 0-4294967295 | 毫秒时间戳 |
| 设备状态 | 0x2002 | UINT16 | 0-4 | 停止/回零/空闲/运行/故障 |
| 滑台位置 | 0x2003-0x2004 | FLOAT32 | 0-1000mm | 在1米导轨位置 |
| 滑台速度 | 0x2005-0x2006 | FLOAT32 | ±200mm/s | 运动速度 |
| 主轴转速 | 0x2007 | UINT16 | 0-3000 RPM | 实际转速 |
| 切割力 | 0x2008-0x2009 | FLOAT32 | 0-100N | 切割力反馈 |
| 限位开关 | 0x200A | UINT16 | 位字段 | 6个限位状态 |
| 急停状态 | 0x200B | UINT16 | 0-1 | 急停按钮 |
| 故障代码 | 0x200C | UINT16 | 0-65535 | 故障信息 |

### 扩展监控数据 (2s低频)
| 数据项 | 寄存器 | 类型 | 范围 | 说明 |
|--------|--------|------|------|------|
| 伺服电机电流 | 0x2010-0x2011 | FLOAT32 | 0-20A | 负载电流 |
| 伺服电机温度 | 0x2012-0x2013 | FLOAT32 | -40~120°C | 温度监控 |
| 主轴电机电流 | 0x2014-0x2015 | FLOAT32 | 0-15A | 负载电流 |
| 主轴电机温度 | 0x2016-0x2017 | FLOAT32 | -40~120°C | 温度监控 |
| 夹爪状态 | 0x2018 | UINT16 | 0-3 | 未知/张开/闭合/夹持 |
| 气压 | 0x2019-0x201A | FLOAT32 | 0-1000kPa | 夹爪气压 |
| 工件检测 | 0x201B | UINT16 | 0-1 | 是否有工件 |
| 总功耗 | 0x201C-0x201D | FLOAT32 | 0-10kW | 系统功耗 |
| 冷却系统 | 0x201E | UINT16 | 0-3 | 关闭/待机/运行/故障 |

### 诊断数据 (按需读取)
| 数据项 | 寄存器 | 类型 | 范围 | 说明 |
|--------|--------|------|------|------|
| 运行时间 | 0x2020-0x2021 | UINT32 | 0-4294967295 | 累计运行小时 |
| 切割次数 | 0x2022-0x2023 | UINT32 | 0-4294967295 | 总切割次数 |
| 故障次数 | 0x2024 | UINT16 | 0-65535 | 故障统计 |
| 维护倒计时 | 0x2025 | UINT16 | 0-1000 | 维护提醒 |
| X轴校准偏差 | 0x2026-0x2027 | FLOAT32 | ±5mm | 校准偏差 |
| Y轴校准偏差 | 0x2028-0x2029 | FLOAT32 | ±5mm | 校准偏差 |

---

## 状态定义

### 设备状态 (device_state)
```
0 = STOPPED     // 停止状态
1 = HOMING      // 回零中  
2 = IDLE        // 空闲待命
3 = RUNNING     // 运行中
4 = FAULT       // 故障状态
```

### 限位开关位字段 (limit_switches)
```
Bit 0: 滑台负限位 (0=正常, 1=触发)
Bit 1: 滑台正限位 (0=正常, 1=触发)
Bit 2: 滑台原点   (0=正常, 1=触发)
Bit 3: 切割深度限位 (0=正常, 1=触发)
Bit 4: 安全门开关 (0=关闭, 1=打开)
Bit 5: 防护罩状态 (0=关闭, 1=打开)
Bit 6-15: 保留
```

### 故障代码 (fault_code)
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

---

## 技术实现

### 1. 创建的新模块

**`src/communication/plc_monitor.py`** - 专用PLC监控客户端
- 纯读取模式，无写入功能
- 支持分层数据读取（核心50ms，扩展2s，诊断按需）
- 自动重连和错误恢复
- 模拟数据支持（无PLC硬件时）

### 2. 数据结构

```python
@dataclass
class CoreMonitoringData:
    """核心监控数据"""
    timestamp: int
    device_state: DeviceState
    slide_position: float
    slide_velocity: float
    spindle_rpm: int
    cutting_force: float
    limit_switches: int
    emergency_stop: bool
    fault_code: int

@dataclass
class ExtendedMonitoringData:
    """扩展监控数据"""
    servo_motor_current: float
    servo_motor_temp: float
    spindle_motor_current: float
    spindle_motor_temp: float
    gripper_state: GripperState
    air_pressure: float
    workpiece_detected: bool
    power_consumption: float
    cooling_system: CoolingState
```

### 3. 使用示例

```python
from src.communication.plc_monitor import PLCMonitor

# 创建监控实例
monitor = PLCMonitor(host="192.168.1.10")

# 连接PLC
if monitor.connect():
    # 启动监控线程
    monitor.start_monitoring()
    
    # 读取数据
    core_data = monitor.read_core_monitoring_data()
    extended_data = monitor.read_extended_monitoring_data()
    
    # 获取最新数据
    latest = monitor.get_latest_data()
    print(f"设备状态: {latest['core'].device_state}")
    print(f"滑台位置: {latest['core'].slide_position:.2f} mm")
```

---

## 测试方法

### 1. 无硬件测试
```bash
# 启动测试模式（使用模拟数据）
python3 test_touch_interface.py
```

### 2. 有PLC硬件测试
```bash
# 测试PLC监控功能
python3 src/communication/plc_monitor.py
```

### 3. 集成测试
系统会自动尝试连接PLC：
- 连接成功 → 使用真实PLC数据
- 连接失败 → 降级到模拟数据

---

## 配置文件

### 网络配置
```yaml
plc_monitoring:
  host: "192.168.1.10"
  port: 502
  slave_id: 1
  timeout: 3.0
  read_only: true  # 纯监控模式
  
  # 读取频率配置
  core_interval: 0.05      # 50ms
  extended_interval: 2.0   # 2s
```

---

## 系统优势

### 1. 职责分离
- **PLC**: 专注设备控制和数据采集
- **Jetson Nano**: 专注AI视觉分析和界面显示
- **通信**: 仅用于状态数据传输

### 2. 安全性
- 无控制指令，避免误操作
- 急停等安全功能完全依赖PLC硬件
- 通信中断不影响设备安全运行

### 3. 可靠性
- 单向数据流，简化通信逻辑
- 自动降级到模拟数据
- 多层错误恢复机制

### 4. 可维护性
- 清晰的数据接口定义
- 模块化设计，易于测试
- 完整的调试和监控工具

---

## 下一步工作

1. **PLC端实现**：根据本规范在PLC中实现相应的寄存器映射
2. **网络配置**：确保PLC和Jetson Nano在同一网段
3. **数据验证**：验证PLC提供的数据格式和精度
4. **集成测试**：在实际硬件环境中测试监控功能
5. **性能优化**：根据实际需求调整读取频率

---

**重要提醒**：本系统为纯监控模式，Jetson Nano不会向PLC发送任何控制指令。所有设备控制逻辑需要在PLC端独立实现。 