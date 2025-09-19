# 智能切竹机一键重新部署指南

## 快速使用

### 最简单的方式 - 一键重新部署

```bash
sudo ./jetpack_deploy.sh
```

**这个命令会自动完成：**
- 🛑 停止所有运行中的服务和进程
- 🧹 清理历史版本和配置
- 🔄 强制重新编译所有组件
- 📦 重新部署Qt前端和AI模型
- 🚀 自动启动智能切竹机服务

## 部署过程说明

### 1. 进程清理阶段
脚本会自动停止并清理：
- `bamboo-cut-jetpack` 服务
- 所有相关的后端和前端进程
- 历史版本的配置文件和日志

### 2. 编译阶段
- 清理旧的编译缓存
- 重新编译C++后端
- 重新编译Qt前端

### 3. 部署阶段
- 创建systemd服务
- 配置Qt环境
- 部署AI模型
- 设置权限和配置

### 4. 启动阶段
- 启动 `bamboo-cut-jetpack` 服务
- 验证服务状态
- 显示操作摘要

## 常用检查命令

### 检查服务状态
```bash
sudo systemctl status bamboo-cut-jetpack
```

### 查看实时日志
```bash
sudo journalctl -u bamboo-cut-jetpack -f
```

### 重启服务
```bash
sudo systemctl restart bamboo-cut-jetpack
```

### 停止服务
```bash
sudo systemctl stop bamboo-cut-jetpack
```

## 可选参数

如果需要自定义部署行为：

```bash
# 安装依赖包
sudo ./jetpack_deploy.sh --install-deps

# 编译调试版本
sudo ./jetpack_deploy.sh --type Debug

# 不备份当前版本
sudo ./jetpack_deploy.sh --no-backup

# 查看帮助
./jetpack_deploy.sh --help
```

## 故障排除

### 如果部署失败

1. **检查编译错误**：
   ```bash
   # 查看详细错误
   sudo ./jetpack_deploy.sh --type Debug
   ```

2. **检查依赖包**：
   ```bash
   # 重新安装依赖
   sudo ./jetpack_deploy.sh --install-deps
   ```

3. **手动清理**：
   ```bash
   # 如果进程清理不彻底
   sudo pkill -f bamboo
   sudo systemctl stop bamboo-cut-jetpack
   ```

### 如果服务启动失败

1. **查看详细日志**：
   ```bash
   sudo journalctl -u bamboo-cut-jetpack -n 50
   ```

2. **检查权限**：
   ```bash
   sudo chown -R root:root /opt/bamboo-cut
   sudo chmod +x /opt/bamboo-cut/*.sh
   ```

3. **手动启动测试**：
   ```bash
   cd /opt/bamboo-cut
   sudo ./start_bamboo_cut_jetpack.sh --debug
   ```

## 重要提醒

- ⚠️ 运行脚本需要 sudo 权限
- ⚠️ 脚本会停止所有运行中的智能切竹机进程
- ⚠️ 编译过程可能需要几分钟时间
- ✅ 脚本会自动备份当前版本（除非使用 --no-backup）
- ✅ 支持在无摄像头环境下运行（模拟模式）

## 验证部署成功

部署完成后，你应该看到：

```
🎉 JetPack SDK 重新部署完成!
✅ 已停止所有运行中的进程
✅ 已清理历史版本
✅ 已强制重新编译
✅ 已重新部署服务
✅ 已启动智能切竹机服务
```

如果看到以上信息，说明部署成功。可以通过以下命令确认：

```bash
sudo systemctl status bamboo-cut-jetpack
```

应该显示 `Active: active (running)`。