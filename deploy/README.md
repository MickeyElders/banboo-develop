# 智能切竹机控制系统部署指南

## 系统架构

- **后端**: C++后端使用libmodbus与PLC通信，通过UNIX Domain Socket提供数据
- **前端**: LVGL图形界面，连接后端获取实时数据
- **通信**: UNIX Domain Socket (`/tmp/bamboo_socket`)
- **服务管理**: systemd服务自动启动和监控

## 快速部署

### 1. 一键部署
```bash
sudo ./deploy.sh
```

### 2. 手动部署
```bash
# 构建项目
make all

# 安装到系统
make install

# 安装systemd服务
make install-service

# 启动服务
make start
```

## 服务管理

### 使用Makefile命令
```bash
make start      # 启动服务
make stop       # 停止服务  
make restart    # 重启服务
make status     # 查看状态
make logs       # 查看日志
```

### 使用systemctl命令
```bash
# 后端服务
systemctl start bamboo-backend
systemctl stop bamboo-backend
systemctl status bamboo-backend
systemctl enable bamboo-backend

# 前端服务
systemctl start bamboo-frontend
systemctl stop bamboo-frontend
systemctl status bamboo-frontend
systemctl enable bamboo-frontend
```

## 监控和诊断

### 系统监控
```bash
# 一次性检查
./deploy/monitor.sh

# 实时监控
./deploy/monitor.sh realtime

# 查看日志
./deploy/monitor.sh logs
```

### 日志查看
```bash
# 查看服务日志
journalctl -u bamboo-backend -f
journalctl -u bamboo-frontend -f

# 查看所有相关日志
journalctl -u bamboo-backend -u bamboo-frontend -f
```

## 配置文件

- **服务配置**: `/etc/bamboo/`
- **二进制文件**: `/opt/bamboo/bin/`
- **日志文件**: `/var/log/bamboo/`
- **Socket文件**: `/tmp/bamboo_socket`

## 环境变量

### 后端服务环境变量
- `BAMBOO_CONFIG_DIR`: 配置目录 (默认: `/etc/bamboo`)
- `BAMBOO_LOG_DIR`: 日志目录 (默认: `/var/log/bamboo`)
- `BAMBOO_SOCKET_PATH`: Socket路径 (默认: `/tmp/bamboo_socket`)
- `MODBUS_PLC_IP`: PLC IP地址 (默认: `192.168.1.100`)
- `MODBUS_PLC_PORT`: PLC端口 (默认: `502`)

### 前端服务环境变量
- `DISPLAY`: X11显示器 (默认: `:0`)
- `LVGL_DISPLAY_DRV`: 显示驱动 (默认: `fbdev`)
- `LVGL_INPUT_DEV`: 输入设备 (默认: `/dev/input/event0`)

## 故障排除

### 常见问题

1. **前端无法启动**
   ```bash
   # 检查图形环境
   echo $DISPLAY
   xauth list
   
   # 检查设备权限
   ls -l /dev/fb0 /dev/input/event*
   ```

2. **后端无法连接PLC**
   ```bash
   # 测试网络连接
   ping 192.168.1.100
   telnet 192.168.1.100 502
   ```

3. **Socket通信失败**
   ```bash
   # 检查Socket文件
   ls -l /tmp/bamboo_socket
   
   # 测试Socket连接
   nc -U /tmp/bamboo_socket
   ```

### 日志分析

```bash
# 查看启动错误
journalctl -u bamboo-backend --since "5 minutes ago"
journalctl -u bamboo-frontend --since "5 minutes ago"

# 查看详细错误信息
journalctl -u bamboo-backend -p err
journalctl -u bamboo-frontend -p err
```

## 开发和调试

### 调试模式编译
```bash
make clean
make BUILD_TYPE=Debug
```

### 手动运行（用于调试）
```bash
# 停止系统服务
make stop

# 手动运行后端
/opt/bamboo/bin/bamboo_cut_backend

# 手动运行前端
DISPLAY=:0 /opt/bamboo/bin/bamboo_controller_lvgl
```

## 卸载

```bash
# 停止服务并完全卸载
make uninstall
```

这将移除所有安装的文件、服务和配置。