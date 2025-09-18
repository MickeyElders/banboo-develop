# JetPack SDK 部署服务启动问题修复

## 问题分析

从详细日志可以看出，问题的根本原因是：
1. **`bamboo-cut` 用户没有sudo权限**
2. **性能配置脚本 `power_config.sh` 包含需要sudo权限的命令**
3. **systemd服务以 `bamboo-cut` 用户身份运行，无法执行系统调优命令**

## 立即修复方案

### 方案1：修改systemd服务为root用户运行（推荐）

```bash
# 编辑systemd服务配置
sudo systemctl edit --full bamboo-cut-jetpack

# 将User行从 bamboo-cut 改为 root
# 修改前：
# User=bamboo-cut
# 修改后：
# User=root

# 或者直接替换服务文件
sudo tee /etc/systemd/system/bamboo-cut-jetpack.service > /dev/null << 'EOF'
[Unit]
Description=智能切竹机系统 (JetPack SDK)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bamboo-cut
ExecStart=/opt/bamboo-cut/start_bamboo_cut_jetpack.sh
Restart=always
RestartSec=10
Environment=DISPLAY=:0
Environment=QT_QPA_PLATFORM=eglfs
Environment=QT_QPA_EGLFS_KMS_CONFIG=/opt/bamboo-cut/config/kms.conf

[Install]
WantedBy=multi-user.target
EOF

# 重新加载并启动服务
sudo systemctl daemon-reload
sudo systemctl restart bamboo-cut-jetpack
sudo systemctl status bamboo-cut-jetpack
```

### 方案2：给bamboo-cut用户添加sudo权限

```bash
# 创建sudoers规则文件
sudo tee /etc/sudoers.d/bamboo-cut > /dev/null << 'EOF'
# 允许bamboo-cut用户执行系统调优命令，无需密码
bamboo-cut ALL=(ALL) NOPASSWD: /usr/bin/tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
bamboo-cut ALL=(ALL) NOPASSWD: /usr/bin/tee /proc/sys/vm/overcommit_memory
bamboo-cut ALL=(ALL) NOPASSWD: /usr/bin/tee /proc/sys/vm/swappiness
bamboo-cut ALL=(ALL) NOPASSWD: /usr/bin/tee /sys/devices/platform/host1x/*/gpu/power/autosuspend_delay_ms
bamboo-cut ALL=(ALL) NOPASSWD: /usr/bin/tee /proc/sys/net/core/netdev_max_backlog
bamboo-cut ALL=(ALL) NOPASSWD: /usr/sbin/nvpmodel
bamboo-cut ALL=(ALL) NOPASSWD: /usr/bin/jetson_clocks
EOF

# 检查语法
sudo visudo -c -f /etc/sudoers.d/bamboo-cut

# 重启服务
sudo systemctl restart bamboo-cut-jetpack
```

### 方案3：修改启动脚本，跳过需要sudo的命令

```bash
# 备份原脚本
sudo cp /opt/bamboo-cut/start_bamboo_cut_jetpack.sh /opt/bamboo-cut/start_bamboo_cut_jetpack.sh.backup

# 创建修改版本的启动脚本
sudo tee /opt/bamboo-cut/start_bamboo_cut_jetpack.sh > /dev/null << 'EOF'
#!/bin/bash
# 智能切竹机 JetPack SDK 启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 加载 Qt 环境 (如果存在)
if [ -f "./qt_libs/setup_qt_env.sh" ]; then
    source "./qt_libs/setup_qt_env.sh"
fi

# 跳过性能优化脚本（需要sudo权限）
# if [ -f "./power_config.sh" ]; then
#     ./power_config.sh
# fi

# 输出信息提示
echo "注意: 性能优化脚本已跳过，请手动以root权限执行 /opt/bamboo-cut/power_config.sh"

# 优化模型 (如果存在且需要)
if [ -f "./models/optimize_models.sh" ] && [ ! -f "./models/tensorrt/optimized.flag" ]; then
    echo "首次运行，正在优化 AI 模型..."
    ./models/optimize_models.sh
    touch "./models/tensorrt/optimized.flag"
fi

# 设置环境变量
export LD_LIBRARY_PATH="./qt_libs:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0

# 启动后端
echo "启动 C++ 后端..."
./bamboo_cut_backend &
BACKEND_PID=$!

# 等待后端启动
sleep 3

# 启动前端
echo "启动 Qt 前端..."
./bamboo_cut_frontend &
FRONTEND_PID=$!

# 等待进程
wait $FRONTEND_PID
kill $BACKEND_PID 2>/dev/null || true

echo "智能切竹机已停止"
EOF

# 设置执行权限
sudo chmod +x /opt/bamboo-cut/start_bamboo_cut_jetpack.sh

# 重启服务
sudo systemctl restart bamboo-cut-jetpack
```

## 测试和验证

### 1. 验证服务状态
```bash
sudo systemctl status bamboo-cut-jetpack
sudo journalctl -u bamboo-cut-jetpack -f
```

### 2. 手动应用性能优化（如果使用方案3）
```bash
sudo /opt/bamboo-cut/power_config.sh
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 3. 检查应用程序是否正常运行
```bash
# 检查后端进程
ps aux | grep bamboo_cut_backend

# 检查前端进程
ps aux | grep bamboo_cut_frontend

# 检查网络端口
netstat -tlnp | grep :502
```

## 推荐的最终解决方案

**建议使用方案1（修改为root用户运行）**，因为：

1. **最简单直接**：不需要复杂的权限配置
2. **功能完整**：所有性能优化功能都能正常工作
3. **符合系统服务惯例**：许多系统级服务都以root身份运行
4. **安全性可控**：应用程序在受控环境中运行

执行以下命令应用修复：

```bash
# 1. 更新systemd服务为root用户
sudo tee /etc/systemd/system/bamboo-cut-jetpack.service > /dev/null << 'EOF'
[Unit]
Description=智能切竹机系统 (JetPack SDK)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bamboo-cut
ExecStart=/opt/bamboo-cut/start_bamboo_cut_jetpack.sh
Restart=always
RestartSec=10
Environment=DISPLAY=:0
Environment=QT_QPA_PLATFORM=eglfs
Environment=QT_QPA_EGLFS_KMS_CONFIG=/opt/bamboo-cut/config/kms.conf

[Install]
WantedBy=multi-user.target
EOF

# 2. 重新加载配置
sudo systemctl daemon-reload

# 3. 重启服务
sudo systemctl restart bamboo-cut-jetpack

# 4. 检查状态
sudo systemctl status bamboo-cut-jetpack

# 5. 查看日志
sudo journalctl -u bamboo-cut-jetpack -f
```

执行以上命令后，服务应该能够正常启动，并且所有性能优化功能都会生效。