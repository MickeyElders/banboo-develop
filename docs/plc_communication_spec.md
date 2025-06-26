# 智能切竹机PLC通信协议规范 v2.0
**面向PLC开发人员**

## 1. 项目概述

### 1.1 系统架构
```
┌─────────────────┐    Modbus TCP    ┌──────────────────┐
│ Jetson Nano     │ ←───────────────  │ PLC控制器        │
│ AI视觉处理中心   │    只读模式       │ 设备控制中心      │
└─────────────────┘                  └──────────────────┘
```

### 1.2 通信特点
- **单向通信**: Jetson Nano仅读取PLC数据，不发送控制指令
- **分层传输**: 根据数据重要性分为核心、扩展、诊断三层
- **高可靠性**: 工业级通信要求，支持7×24小时连续运行

---

## 2. 网络配置要求

### 2.1 基础网络参数
| 参数 | 数值 | 说明 |
|------|------|------|
| **PLC IP地址** | 192.168.1.10 | 固定IP，可配置 |
| **子网掩码** | 255.255.255.0 | 标准C类网络 |
| **网关** | 192.168.1.1 | 局域网网关 |
| **Modbus端口** | 502 | 标准Modbus TCP端口 |
| **从站ID** | 1 | PLC站号 |

### 2.2 客户端信息
| 参数 | 数值 | 说明 |
|------|------|------|
| **客户端IP** | 192.168.1.100 | Jetson Nano IP |
| **连接模式** | 主动连接 | Jetson Nano主动连接PLC |
| **访问权限** | 只读 | 仅读取，不写入 |

---

## 3. Modbus协议规范

### 3.1 基础协议参数
```
协议类型: Modbus TCP/IP
字节序: 大端字节序 (Big Endian)
功能码: 0x03 (读保持寄存器)
数据编码: 
  - 16位整数: UINT16, INT16
  - 32位整数: UINT32 (高位在前)
  - 32位浮点: IEEE 754单精度 (高位在前)
```

### 3.2 数据类型说明
| 类型 | 字节数 | 寄存器数 | 说明 | 示例 |
|------|--------|----------|------|------|
| UINT16 | 2 | 1 | 无符号16位整数 | 0-65535 |
| INT16 | 2 | 1 | 有符号16位整数 | -32768~32767 |
| UINT32 | 4 | 2 | 无符号32位整数 | 0-4294967295 |
| FLOAT32 | 4 | 2 | IEEE 754单精度浮点 | ±3.4E±38 |

### 3.3 浮点数编码示例
```c
// PLC端浮点数编码示例 (C语言)
float value = 123.45f;
uint32_t encoded = *(uint32_t*)&value;
uint16_t high_word = (encoded >> 16) & 0xFFFF;  // 寄存器N
uint16_t low_word = encoded & 0xFFFF;           // 寄存器N+1
```

---

## 4. 寄存器映射规范

### 4.1 核心监控数据区 (高优先级 - 50ms更新)
**起始地址**: 0x2000  
**寄存器数量**: 13  
**更新频率**: 50ms (20Hz)

| 寄存器地址 | 数据类型 | 字段名称 | 单位 | 数值范围 | 功能描述 |
|-----------|----------|----------|------|----------|----------|
| 0x2000-0x2001 | UINT32 | plc_timestamp | ms | 0-4294967295 | PLC系统时间戳 |
| 0x2002 | UINT16 | device_state | - | 0-4 | 设备运行状态 |
| 0x2003-0x2004 | FLOAT32 | slide_position | mm | 0-1000.0 | 滑台在导轨的绝对位置 |
| 0x2005-0x2006 | FLOAT32 | slide_velocity | mm/s | -200.0~+200.0 | 滑台运动速度(正向/负向) |
| 0x2007 | UINT16 | spindle_rpm | RPM | 0-3000 | 主轴实际转速 |
| 0x2008-0x2009 | FLOAT32 | cutting_force | N | 0-100.0 | 切割力传感器反馈 |
| 0x200A | UINT16 | limit_switches | 位字段 | 0x0000-0x003F | 限位开关状态集合 |
| 0x200B | UINT16 | emergency_stop | - | 0-1 | 急停按钮状态 |
| 0x200C | UINT16 | fault_code | - | 0-65535 | 当前故障代码 |

### 4.2 扩展监控数据区 (中优先级 - 2s更新)
**起始地址**: 0x2010  
**寄存器数量**: 15  
**更新频率**: 2000ms (0.5Hz)

| 寄存器地址 | 数据类型 | 字段名称 | 单位 | 数值范围 | 功能描述 |
|-----------|----------|----------|------|----------|----------|
| 0x2010-0x2011 | FLOAT32 | servo_motor_current | A | 0-20.0 | 伺服电机实际电流 |
| 0x2012-0x2013 | FLOAT32 | servo_motor_temp | °C | -40.0~+120.0 | 伺服电机温度 |
| 0x2014-0x2015 | FLOAT32 | spindle_motor_current | A | 0-15.0 | 主轴电机实际电流 |
| 0x2016-0x2017 | FLOAT32 | spindle_motor_temp | °C | -40.0~+120.0 | 主轴电机温度 |
| 0x2018 | UINT16 | gripper_state | - | 0-3 | 夹爪状态 |
| 0x2019-0x201A | FLOAT32 | air_pressure | kPa | 0-1000.0 | 气动系统压力 |
| 0x201B | UINT16 | workpiece_detected | - | 0-1 | 工件检测传感器 |
| 0x201C-0x201D | FLOAT32 | power_consumption | kW | 0-10.0 | 系统总功耗 |
| 0x201E | UINT16 | cooling_system | - | 0-3 | 冷却系统状态 |

### 4.3 诊断数据区 (低优先级 - 按需读取)
**起始地址**: 0x2020  
**寄存器数量**: 10  
**更新频率**: 按需 (通常每分钟一次)

| 寄存器地址 | 数据类型 | 字段名称 | 单位 | 数值范围 | 功能描述 |
|-----------|----------|----------|------|----------|----------|
| 0x2020-0x2021 | UINT32 | runtime_hours | 小时 | 0-4294967295 | 设备累计运行时间 |
| 0x2022-0x2023 | UINT32 | total_cuts | 次 | 0-4294967295 | 累计切割次数 |
| 0x2024 | UINT16 | fault_count | 次 | 0-65535 | 历史故障总数 |
| 0x2025 | UINT16 | maintenance_countdown | 小时 | 0-1000 | 维护提醒倒计时 |
| 0x2026-0x2027 | FLOAT32 | calibration_offset_x | mm | -5.0~+5.0 | X轴校准偏差值 |
| 0x2028-0x2029 | FLOAT32 | calibration_offset_y | mm | -5.0~+5.0 | Y轴校准偏差值 |

---

## 5. 枚举值定义

### 5.1 设备状态枚举 (device_state)
```c
typedef enum {
    DEVICE_STOPPED = 0,     // 设备停止状态
    DEVICE_HOMING = 1,      // 回零操作中
    DEVICE_IDLE = 2,        // 空闲待命状态
    DEVICE_RUNNING = 3,     // 正常运行中
    DEVICE_FAULT = 4        // 故障状态
} device_state_t;
```

### 5.2 限位开关位字段 (limit_switches)
```c
#define LIMIT_SLIDE_NEG     (1 << 0)    // Bit 0: 滑台负限位
#define LIMIT_SLIDE_POS     (1 << 1)    // Bit 1: 滑台正限位  
#define LIMIT_SLIDE_HOME    (1 << 2)    // Bit 2: 滑台原点
#define LIMIT_CUTTING_DEPTH (1 << 3)    // Bit 3: 切割深度限位
#define SAFETY_DOOR         (1 << 4)    // Bit 4: 安全门开关
#define PROTECTION_COVER    (1 << 5)    // Bit 5: 防护罩状态
// Bit 6-15: 保留
```

### 5.3 夹爪状态枚举 (gripper_state)
```c
typedef enum {
    GRIPPER_UNKNOWN = 0,    // 未知状态
    GRIPPER_OPEN = 1,       // 夹爪张开
    GRIPPER_CLOSED = 2,     // 夹爪闭合
    GRIPPER_GRIPPING = 3    // 夹持工件中
} gripper_state_t;
```

### 5.4 冷却系统状态枚举 (cooling_system)
```c
typedef enum {
    COOLING_OFF = 0,        // 冷却关闭
    COOLING_STANDBY = 1,    // 冷却待机
    COOLING_RUNNING = 2,    // 冷却运行
    COOLING_FAULT = 3       // 冷却故障
} cooling_state_t;
```

### 5.5 标准故障代码定义 (fault_code)
```c
#define FAULT_NONE                  0x0000  // 无故障
#define FAULT_SERVO_OVERHEAT        0x0001  // 伺服电机超温
#define FAULT_POSITION_ERROR        0x0002  // 位置跟随误差过大
#define FAULT_CUTTING_OVERLOAD      0x0003  // 切割力超限
#define FAULT_COMM_TIMEOUT          0x0004  // 通信超时
#define FAULT_EMERGENCY_STOP        0x0005  // 急停触发
#define FAULT_LIMIT_TRIGGERED       0x0006  // 限位开关触发
#define FAULT_POWER_VOLTAGE         0x0007  // 电源电压异常
#define FAULT_ENCODER_ERROR         0x0008  // 编码器故障
#define FAULT_TOOL_WEAR             0x0009  // 刀具磨损严重
#define FAULT_HYDRAULIC_SYSTEM      0x000A  // 液压系统故障
```

---

## 6. 数据质量要求

### 6.1 精度要求
| 数据类型 | 精度要求 | 说明 |
|----------|----------|------|
| 位置数据 | ±0.01mm | 滑台位置精度 |
| 速度数据 | ±0.1mm/s | 速度测量精度 |
| 力值数据 | ±0.1N | 切割力传感器精度 |
| 温度数据 | ±0.1°C | 温度传感器精度 |
| 压力数据 | ±0.01MPa | 压力传感器精度 |
| 电流数据 | ±0.01A | 电流传感器精度 |

### 6.2 时间戳要求
```c
// 时间戳更新示例 (PLC代码参考)
static uint32_t plc_timestamp = 0;

void update_timestamp() {
    // 每个扫描周期自增，单位：毫秒
    plc_timestamp += SCAN_CYCLE_MS;
    
    // 防止溢出回绕
    if (plc_timestamp == 0xFFFFFFFF) {
        plc_timestamp = 0;
    }
}
```

### 6.3 数据有效性检查
```c
// 数据范围检查示例
bool validate_slide_position(float position) {
    return (position >= 0.0f && position <= 1000.0f);
}

bool validate_motor_temp(float temp) {
    return (temp >= -40.0f && temp <= 120.0f);
}
```

---

## 7. 通信时序要求

### 7.1 读取时序
```
核心数据区 (0x2000):  每50ms读取一次
扩展数据区 (0x2010):  每2000ms读取一次  
诊断数据区 (0x2020):  按需读取 (通常1分钟)
```

### 7.2 响应时间要求
| 操作类型 | 最大响应时间 | 说明 |
|----------|--------------|------|
| 数据读取响应 | < 10ms | 正常读取响应 |
| 急停状态更新 | < 5ms | 安全关键信号 |
| 故障状态反馈 | < 50ms | 故障检测响应 |

### 7.3 超时处理
```c
// PLC端超时监控示例
#define COMM_TIMEOUT_MS     5000    // 5秒通信超时

static uint32_t last_comm_time = 0;

void check_communication_timeout() {
    if ((plc_timestamp - last_comm_time) > COMM_TIMEOUT_MS) {
        // 通信超时处理
        fault_code = FAULT_COMM_TIMEOUT;
        // 进入安全状态
    }
}
```

---

## 8. PLC端实现指南

### 8.1 寄存器数据结构示例
```c
// PLC内部数据结构参考
typedef struct {
    // 核心监控数据 (0x2000开始)
    uint32_t timestamp;
    uint16_t device_state;
    float slide_position;
    float slide_velocity;
    uint16_t spindle_rpm;
    float cutting_force;
    uint16_t limit_switches;
    uint16_t emergency_stop;
    uint16_t fault_code;
    
    // 扩展监控数据 (0x2010开始)
    float servo_motor_current;
    float servo_motor_temp;
    float spindle_motor_current;
    float spindle_motor_temp;
    uint16_t gripper_state;
    float air_pressure;
    uint16_t workpiece_detected;
    float power_consumption;
    uint16_t cooling_system;
    
    // 诊断数据 (0x2020开始)
    uint32_t runtime_hours;
    uint32_t total_cuts;
    uint16_t fault_count;
    uint16_t maintenance_countdown;
    float calibration_offset_x;
    float calibration_offset_y;
} plc_data_t;
```

### 8.2 数据更新建议
```c
void update_core_data() {
    // 高频数据更新 (每个扫描周期)
    core_data.timestamp = get_system_time_ms();
    core_data.device_state = get_device_state();
    core_data.slide_position = read_encoder_position();
    core_data.slide_velocity = calculate_velocity();
    core_data.spindle_rpm = read_spindle_encoder();
    core_data.cutting_force = read_force_sensor();
    core_data.limit_switches = read_limit_switches();
    core_data.emergency_stop = read_emergency_button();
    core_data.fault_code = get_current_fault();
}

void update_extended_data() {
    // 中频数据更新 (每2秒)
    extended_data.servo_motor_current = read_servo_current();
    extended_data.servo_motor_temp = read_servo_temperature();
    // ... 其他扩展数据
}
```

### 8.3 Modbus服务器配置
```c
// Modbus TCP服务器配置示例
void init_modbus_server() {
    modbus_server_config_t config = {
        .ip_address = "192.168.1.10",
        .port = 502,
        .slave_id = 1,
        .max_connections = 1,      // 只允许一个客户端连接
        .read_only = true,         // 只允许读操作
        .holding_register_count = 100
    };
    
    // 映射内存到Modbus寄存器
    map_data_to_registers(&plc_data, 0x2000);
}
```

---

## 9. 测试与验证

### 9.1 通信测试工具
| 工具名称 | 用途 | 获取方式 |
|----------|------|----------|
| **ModbusPoll** | Modbus客户端测试 | 商业软件 |
| **QModBus** | 开源Modbus工具 | GitHub开源 |
| **Wireshark** | 网络数据包分析 | 官方免费 |
| **Modbus Doctor** | 协议调试工具 | 免费版本可用 |

### 9.2 测试步骤
```bash
# 1. 网络连通性测试
ping 192.168.1.10

# 2. 端口连通性测试  
telnet 192.168.1.10 502

# 3. Modbus功能测试
# 使用ModbusPoll读取寄存器0x2000-0x200C
```

### 9.3 数据验证清单
- [ ] 时间戳正常递增
- [ ] 设备状态枚举值正确
- [ ] 浮点数据格式符合IEEE 754
- [ ] 限位开关位字段正确
- [ ] 故障代码定义一致
- [ ] 数据更新频率符合要求
- [ ] 通信超时处理正常

---

## 10. 常见问题解答

### 10.1 字节序问题
**Q**: 浮点数在寄存器中如何存储？  
**A**: 使用大端字节序，高16位存储在较小的寄存器地址中。

```c
// 正确的浮点数存储方式
float value = 123.45f;
uint32_t encoded = *(uint32_t*)&value;
registers[addr] = (encoded >> 16) & 0xFFFF;      // 高16位
registers[addr+1] = encoded & 0xFFFF;            // 低16位
```

### 10.2 时间戳同步
**Q**: 时间戳是否需要与外部时钟同步？  
**A**: 不需要。时间戳主要用于检测数据更新，使用PLC内部相对时间即可。

### 10.3 故障处理
**Q**: 通信中断时PLC应如何处理？  
**A**: PLC应继续正常工作，不依赖外部通信。通信中断不应影响设备安全运行。

### 10.4 性能优化
**Q**: 如何优化Modbus通信性能？  
**A**: 
- 核心数据使用连续寄存器地址
- 避免频繁的小数据包传输
- 合理设置超时时间
- 使用批量读取减少协议开销

---

## 11. 技术支持联系方式

### 11.1 开发团队联系方式
- **项目负责人**: [联系信息]
- **通信协议支持**: [联系信息]  
- **系统集成支持**: [联系信息]

### 11.2 文档版本信息
- **文档版本**: v2.0
- **最后更新**: 2024年2月
- **适用PLC**: 支持Modbus TCP的通用PLC
- **兼容性**: 向后兼容v1.x版本

---

**重要提醒**: 本文档为技术规范，请严格按照寄存器映射和数据格式要求实现。如有疑问，请及时联系开发团队确认。 