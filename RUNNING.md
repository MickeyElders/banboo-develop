# Bamboo Vision (jetson-inference) 快速运行指南

## 1. 依赖
- JetPack 自带的 `jetson-inference` Python 轮子（`python3 -c "import jetson.inference, jetson.utils"` 应返回成功）。
- 其他依赖：`pymodbus`、`pyyaml`、`flask`  
  ```bash
  sudo apt-get install -y python3-pip
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements.txt
  ```

## 2. 配置
- 编辑 `config/runtime.yaml`：至少填写 Modbus `host/port/slave_id`，以及校准后的 `pixel_to_mm/offset_mm`。  
- CSI 相机默认 `csi://0`，如需自定义 GStreamer 管线，替换 `camera.pipeline`。
- RTSP 默认输出到 `rtsp://@:8554/live`，本地 HDMI 通过 `display://0`。
- HTTP 服务：默认 `http://0.0.0.0:8080/` 提供 `bamboo.html` 及 `/api/status`。

## 3. 运行
### 本地直接运行
```bash
python3 -m bamboo_vision.app --config config/runtime.yaml      # HDMI+RTSP
python3 -m bamboo_vision.app --config config/runtime.yaml --headless   # 仅 RTSP
```
或使用 Makefile：
```bash
make deps
make run
```

### 安装为 systemd 服务（Jetson 上）
```bash
make service
```
服务名：`bamboo-vision.service`，默认工作目录 `/opt/bamboo-vision`，配置 `/opt/bamboo-vision/config/runtime.yaml`。

## 4. 功能说明
- 捕获 CSI/GStreamer 视频，使用 `detectNet`（ONNX/engine）检测竹节。
- 同时输出到 HDMI 和 RTSP（可被 MediaMTX/ffplay/VLC 拉流）。
- HTTP 动态服务：`/`/`/bamboo.html` 提供前端页面；`/api/status` 返回最近检测、PLC 状态、FPS；`/api/control` 为预留控制入口（按键可调用）。
- 在线标定：`/api/calibration` GET/POST，前端悬浮面板可实时调整 `pixel_to_mm/offset_mm/latency_ms/belt_speed_mm_s`，可选 `persist=true` 写回 `config/runtime.yaml`。
- 编码器：默认硬件 `nvv4l2h264enc`；若缺失则自动降级为软件 `x264enc`（需 `gstreamer1.0-plugins-good/bad`、`libx264-dev`）。`GST_PLUGIN_PATH` 在程序内默认包含 `/usr/lib/aarch64-linux-gnu/gstreamer-1.0:/usr/lib/aarch64-linux-gnu/tegra`。
- 备用：`output.raw_udp` 可启用原始帧 UDP 输出（RGBA->I420，无编码）到 `raw_udp_host:raw_udp_port`，便于外部轻量 x264/RTSP 组合（例如：`udpsrc ! videoconvert ! x264enc ! rtph264pay ! rtspclientsink`）。

## 外部软编推流示例（raw_udp 默认开启）
```bash
UDP_PORT=5600 WIDTH=1280 HEIGHT=720 FPS=30 RTSP_URL=rtsp://127.0.0.1:8554/live \
  bash deploy/scripts/udp_x264_rtsp.sh
```
- 简单像素→毫米换算（按 `runtime.yaml`），写入 Modbus：  
  - 相机→PLC：0x07D0 通信请求保持 1，0x07D1 状态=1，0x07D2 坐标(float，高低字)，0x07D4 结果码(1 成功/2 失败)。  
  - PLC→相机：0x0834 心跳，0x0835 状态(1=可接收坐标，2=送料中)，0x0836 当前位置(float，高低字)。  
- 真实标定与节拍补偿请在 `calibration` 段落填入实测值。

## 5. 调试建议
- 无相机可先用文件/测试源：`camera.pipeline: "videotestsrc is-live=true ! video/x-raw,width=1280,height=720,framerate=30/1 ! videoconvert ! video/x-raw,format=BGRx ! appsink"`。
- 如 RTSP 拉不到，确认 8554 端口未被占用；可改 `output.rtsp_uri` 为其他端口。
- Modbus 侧可用 `modpoll`/`pymodbus.console` 验证寄存器读写。
