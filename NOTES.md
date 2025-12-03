# 项目现状与待办清单

## 已完成的关键改造
- 本地一体化编译：Makefile 编译/安装仓内 `jetson-inference`（ENABLE_GSTREAMER=ON），运行/服务优先使用本地 build 的 Python 绑定和库。
- 后端接口：
  - `/api/status` 提供检测/PLC 状态（实时展示到前端）。
  - `/api/jetson` 提供 CPU/内存/温度/uptime 基础监控。
  - `/api/control` 接收 start/pause/stop/emergency，写入 shared_state。
  - `/api/power` 支持 reboot/shutdown/restart_service（systemd）。
  - `/api/wifi` / `/api/wifi/restart` 通过 nmcli 真实应用 Wi-Fi 配置、查询状态、重启网络。
  - Modbus 自动发现：host 为空且 auto_discover=true 时扫描 scan_subnet 寻找 Modbus 设备。
- 前端 UI：
  - 去掉模拟数据，检测/PLC/Jetson 状态全部来自后端接口。
  - 按钮（运行/暂停/停止/急停、Wi-Fi 应用/重启、电源操作）调用后端真实 API。
  - 去掉切割质量显示与 HLS 播放相关 UI/逻辑，避免摆设。
- 运行与部署：`make run` 后台启动写日志；`make redeploy` 重新编译 jetson-inference + 项目并重启服务。TensorRT 引擎仅在缺失时构建，避免重复耗时。

## 待实现（优先级从高到低）
1. **控制指令落地到推理/PLC**
   - 在 app/pipeline 主循环中响应 shared_state.control，真正暂停/恢复推理与输出。
   - 将 start/stop/急停动作映射到 PLC 寄存器写入（参考 PLC.md），而不仅是状态存储。

2. **视频预览（无 NVENC）**
   - 选用 jetson-inference 支持的软编输出（rtp:// 或 webrtc://，或软编 x264 → mediamtx/HLS）提供浏览器可看的实时画面。

3. **更完整的 Jetson 监控**
   - `/api/jetson` 增加 GPU 利用率/功耗/风扇/NVPModel（可解析 tegrastats/NVML），前端显示真实值。

4. **Wi‑Fi 兼容与提示**
   - 若无 nmcli，提供 connmanctl/networkctl 备用实现；前端使用统一提示条代替 alert，显示执行成功/失败。

5. **校准持久化与反馈**
   - 确认 calibration.persist 可写，前端显示保存结果状态（成功/失败提示条）。

6. **依赖/源码检查**
   - 在 `make deps` 前检测 `jetson-inference` 源码完整性，缺失时输出明确提示而非卡构建。

7. **前端提示体验**
   - 用统一的消息条/状态区域替代 alert，集中显示执行结果并避免重复弹窗。

## 运行提示
- 配置：`config/runtime.yaml` 可开启 Modbus 自动发现（host 为空 + auto_discover=true + scan_subnet），或直接填 PLC IP。
- 服务：`make redeploy` 编译安装并重启；`make logs` 查看运行日志。
- 前端：访问 `http://<jetson_ip>:8080`，数据来自后端实时接口。按钮会调用真实控制/电源/Wi‑Fi API。
