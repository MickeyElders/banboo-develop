# 智能切竹机部署验证指南

## 1. 系统部署完成检查

### 验证服务状态
```bash
sudo systemctl status bamboo-cut
```
✅ 期望状态：`active (running)`

### 验证日志输出
```bash
sudo journalctl -u bamboo-cut -f
```
✅ 期望看到：
- "🚀 智能切竹机系统启动中..."
- "✅ C++后端启动成功"
- "✅ Qt前端启动成功"

## 2. 核心功能验证

### 检查模型文件
```bash
ls -la /opt/bamboo-cut/models/
```
✅ 期望文件：
- `best.onnx` - OpenCV兼容的ONNX模型
- `best.engine` - TensorRT优化模型（可选）

### 检查可执行文件
```bash
ls -la /opt/bamboo-cut/bin/
```
✅ 期望文件：
- `bamboo_cut_backend` - C++后端
- `bamboo_cut_frontend` - Qt前端
- `optimize_models.sh` - 模型优化脚本
- `optimize_performance.sh` - 性能优化脚本

## 3. 性能验证

### GPU性能检查
```bash
sudo tegrastats
```
✅ 期望看到：GPU使用率和频率信息

### 内存使用检查
```bash
free -h
```
✅ 期望：系统内存合理使用，有足够可用内存

## 4. 网络连接验证

### 检查Qt前端界面
- 访问显示器或VNC连接
- 确认Qt界面正常显示
- 测试触摸响应（如适用）

### 检查Modbus通信
```bash
# 检查Modbus端口是否监听
sudo netstat -tlnp | grep 502
```
✅ 期望看到：bamboo_cut_backend进程监听502端口

## 5. 常见问题排查

### 问题1：TensorRT优化失败
**现象**：日志显示"trtexec 未找到"
**解决**：
```bash
sudo apt install tensorrt-dev
sudo /opt/bamboo-cut/bin/optimize_models.sh
```

### 问题2：Qt前端无法显示
**现象**：黑屏或Qt错误
**解决**：
```bash
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
sudo systemctl restart bamboo-cut
```

### 问题3：权限问题
**现象**：访问被拒绝
**解决**：
```bash
sudo chown -R bamboo-cut:bamboo-cut /opt/bamboo-cut/
sudo chmod +x /opt/bamboo-cut/bin/*
```

### 问题4：模型加载失败
**现象**：OpenCV DNN错误
**解决**：
```bash
# 重新转换模型
cd /opt/bamboo-cut/models/
python3 /opt/bamboo-cut/scripts/convert_pytorch_to_onnx.py
```

## 6. 系统重启验证

重启系统后，验证自动启动：
```bash
sudo reboot
# 重启后检查
sudo systemctl status bamboo-cut
```

## 7. 性能基准测试

### CPU性能
```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```
✅ 期望输出：所有CPU显示`performance`

### GPU性能
```bash
cat /sys/devices/gpu.0/railgate_enable
```
✅ 期望输出：`0`（禁用GPU节能）

## 8. 完整系统测试

1. **摄像头测试**：确认图像采集正常
2. **检测测试**：验证竹节检测功能
3. **PLC通信测试**：测试Modbus指令
4. **切割测试**：验证完整工作流程

## 联系支持

如遇到问题，请提供以下信息：
- 系统版本：`cat /etc/os-release`
- JetPack版本：`sudo apt show nvidia-jetpack`
- 错误日志：`sudo journalctl -u bamboo-cut --since "1 hour ago"`
- 硬件信息：`sudo tegrastats --interval 1000 --logfile /tmp/tegrastats.log`

---

**部署成功标志**：
✅ 系统服务正常运行
✅ Qt界面正常显示  
✅ 模型加载成功
✅ 网络通信正常
✅ 性能优化生效