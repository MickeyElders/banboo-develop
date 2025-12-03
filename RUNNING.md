# Bamboo Vision (jetson-inference) 快速运行指南

## 1. 依赖
- JetPack 自带的 `jetson-inference` Python 轮子（`python3 -c "import jetson.inference, jetson.utils"` 应返回成功）。
- `pymodbus`、`pyyaml`：  
  ```bash
  sudo apt-get install -y python3-pip
  python3 -m pip install --upgrade pip
  python3 -m pip install pymodbus pyyaml
  ```

## 2. 配置
- 编辑 `config/runtime.yaml`：至少填写 Modbus `host/port/slave_id`，以及校准后的 `pixel_to_mm/offset_mm`。  
- CSI 相机默认 `csi://0`，如需自定义 GStreamer 管线，替换 `camera.pipeline`。
- RTSP 默认输出到 `rtsp://@:8554/live`，本地 HDMI 通过 `display://0`。

## 3. 运行
```bash
python3 bamboo_vision.py --config config/runtime.yaml
```
常用参数：
- `--headless`：仅 RTSP，不占用 HDMI。

## 4. 功能说明
- 捕获 CSI/GStreamer 视频，使用 `detectNet`（ONNX/engine）检测竹节。
- 同时输出到 HDMI 和 RTSP（可被 MediaMTX/ffplay/VLC 拉流）。
- 简单像素→毫米换算（按 `runtime.yaml`），写入 Modbus：  
  - 相机→PLC：0x07D0 通信请求保持 1，0x07D1 状态=1，0x07D2 坐标(float，高低字)，0x07D4 结果码(1 成功/2 失败)。  
  - PLC→相机：0x0834 心跳，0x0835 状态(1=可接收坐标，2=送料中)，0x0836 当前位置(float，高低字)。  
- 真实标定与节拍补偿请在 `calibration` 段落填入实测值。

## 5. 调试建议
- 无相机可先用文件/测试源：`camera.pipeline: "videotestsrc is-live=true ! video/x-raw,width=1280,height=720,framerate=30/1 ! videoconvert ! video/x-raw,format=BGRx ! appsink"`。
- 如 RTSP 拉不到，确认 8554 端口未被占用；可改 `output.rtsp_uri` 为其他端口。
- Modbus 侧可用 `modpoll`/`pymodbus.console` 验证寄存器读写。
