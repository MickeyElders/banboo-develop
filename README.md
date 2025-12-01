# Jetson Orin NX 竹节识别与 PLC 切割系统

基于 C++、LVGL 和 DeepStream 的一体化视觉系统，在 Jetson Orin NX 上运行 YOLO 识别竹节位置，并通过 Modbus TCP 将坐标推送给下游 PLC 完成切割。主入口为 `bamboo_integrated`（Wayland/LVGL 单进程架构）。

## 关键特性
- C++/LVGL 一体化：`integrated_main.cpp` 将推理、界面和通信整合为单进程，支持 Wayland (nvweston)。
- 推理管线：YOLOv8 ->（计划 TensorRT/DeepStream）-> 数据桥接，当前默认 OpenCV DNN，DeepStream 支持 Wayland subsurface 叠加。
- PLC 通信：Modbus TCP 客户端在视觉端发起连接，周期推送寄存器并接收指令。
- 辅助模块：双目立体深度、Jetson 监控、摄像头诊断、模型转换 (PyTorch->ONNX/TensorRT)。

## 目录结构
- `integrated_main.cpp`：Wayland/LVGL 集成入口，封装 DeepStream 渲染与 UI 线程。
- `cpp_backend/`：核心组件
  - `core/`：`bamboo_system.cpp` 主控制器、`data_bridge.cpp` 线程安全数据桥。
  - `inference/`：`bamboo_detector.cpp` YOLO 推理线程（OpenCV DNN，TensorRT 路径待补全）。
  - `deepstream/`：`deepstream_manager.cpp` GStreamer/DeepStream 管线与 Wayland subsurface 输出。
  - `communication/`：`modbus_interface.cpp` Modbus TCP 客户端。
  - `ui/`：`lvgl_wayland_interface.cpp` LVGL Wayland 界面与画布更新。
  - `vision/`：`stereo_vision.cpp` 等立体视觉与轻量化算子。
- `cpp_inference/`：`inference_core.cpp` TensorRT 引擎构建与推理（未接入主流程）。
- `config/`：`system_config.yaml` 主配置，`ai_optimization.yaml`、`performance_config.yaml` 等优化与显示/Wayland 配置。
- `deploy/`：`systemd/bamboo-cpp-lvgl.service.in` 服务模板，`scripts/` 针对 Jetson 的驱动/显示修复，`monitor.sh` 监控脚本。
- `models/`：`best.pt` 训练模型，`convert_model.py` 转换脚本。
- `docs/`：`cpp_plc_communication_protocol.md`、`PLC.md` 详细寄存器与时序说明。

## 运行架构
1) 摄像头 -> GStreamer/DeepStream（`deepstream_manager.cpp`）采集与渲染；  
2) 推理线程（`inference::InferenceThread` 或 `InferenceWorkerThread`）执行 YOLO，生成 `DetectionResult`；  
3) `core::DataBridge` 在线程间传递帧、检测和系统统计；  
4) LVGL Wayland 界面（`ui::LVGLWaylandInterface`）展示视频/状态；  
5) Modbus 接口（`communication::ModbusInterface`）按协议写 40001+ 寄存器并读取 PLC 指令。

## 构建与运行
Jetson Orin NX（JetPack 6，Wayland/nvweston）已默认支持 CUDA/TensorRT、GStreamer。

```bash
# 构建
cmake -B build -S . ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DENABLE_LVGL=ON -DENABLE_TENSORRT=ON -DENABLE_CUDA=ON -DENABLE_MODBUS=ON -DENABLE_GSTREAMER=ON
cmake --build build --config Release -- -j

# 安装到 /opt/bamboo-cut（与 systemd 服务路径一致）
sudo cmake --install build

# 直接运行（本地构建）
./build/bamboo_integrated --verbose --config config/system_config.yaml --test

# 以安装版本运行（确保 nvweston 已启动）
sudo XDG_RUNTIME_DIR=/run/nvidia-wayland WAYLAND_DISPLAY=wayland-0 \
  /opt/bamboo-cut/bin/bamboo_integrated --verbose \
  --config /opt/bamboo-cut/share/bamboo/config/system_config.yaml
```

> Makefile 还提供了自动化命令：`make deploy`（构建+安装+启服务）、`make start|status|logs`、`make camera-diag` 等。

## 配置
- `config/system_config.yaml`：摄像头/AI/Modbus/UI/安全参数（当前 `SystemConfig::loadFromFile` 未解析此文件，运行时仍使用默认值，需要补全解析逻辑后配置才生效）。
- `config/ai_optimization.yaml`：TensorRT、NAM/GhostConv/VoV-GSCSP、SAHI 切片等模型优化参数。
- `config/performance_config.yaml`：推理批次、线程、零拷贝等性能开关。
- `config/display_config.yaml`、`config/wayland_config.yaml`：显示/Wayland 环境预设。

## PLC 协议
- 详见 `docs/cpp_plc_communication_protocol.md` 与 `PLC.md`，覆盖 40001-40019 及尾料、心跳、复检等时序。
- 代码侧 Modbus 寄存器映射定义：`cpp_backend/include/bamboo_cut/core/data_bridge.h`。

## 模型与转换
- 默认模型：`models/best.pt`。
- 转换：`python models/convert_model.py --config <cfg>` 支持 PyTorch->ONNX->TensorRT，包含 GhostConv/VoV-GSCSP/SAHI 等选项。

## 部署与运维
- systemd 模板：`deploy/systemd/bamboo-cpp-lvgl.service.in`（Wayland 环境变量、nvargus 重启、输入组权限）。
- Jetson 辅助脚本：`deploy/scripts/camera_fix.sh`、`enable_nvidia_drm.sh`、`force_nvidia_drm.sh`。
- 监控：`deploy/monitor.sh`（CPU/GPU/温度/FPS）。

## 待补充/已知缺口
- `cpp_backend/src/core/bamboo_system.cpp`：`SystemConfig::loadFromFile/saveToFile` 仍是 TODO，当前运行不会应用 `config/system_config.yaml`；需按 YAML 映射填充 `detector_config/ui_config/modbus_config` 等字段。
- `cpp_backend/src/core/bamboo_system.cpp`：`WorkflowManager::executeWorkflowStep/isStepCompleted` 留空，未实现送料->识别->坐标下发->切割的状态机与超时处理。
- `cpp_backend/src/communication/modbus_interface.cpp` 与 `core/data_bridge.cpp`：推理结果未写入 `ModbusRegisters`（坐标、就绪标志、尾料等），PLC 侧只收到默认值；需在推理线程映射检测坐标并维护寄存器状态机。
- `cpp_backend/src/inference/bamboo_detector.cpp`：TensorRT 路径未完成（`initializeTensorRT`/`tensorRTInference` 返回 false），仅 OpenCV DNN 可用；需接入 TensorRT/engine 缓存并与 DeepStream 管线对齐。
- `integrated_main.cpp`：`InferenceWorkerThread::processFrame` 仅生成模拟帧，未执行真实检测或 Modbus 交互；需改为调用 DeepStream/YOLO 推理、通过 `IntegratedDataBridge` 更新 UI 与 PLC。
- `cpp_inference/inference_core.cpp`：独立 TensorRT 推理实现未接入主二进制或导出接口，需决定集成方式或移除。

变更留存（截至 2025-12-02）

qt_ui/main.cpp：fixArgusDeadlockInHeadless() 现在创建并常驻 surfaceless EGLDisplay/EGLContext/PBuffer，不再销毁；确保 headless 下 Argus 能拿到有效 EGLDisplay。继续强制 GST_GL_PLATFORM=egl / GST_GL_API=gles2 / EGL_PLATFORM=surfaceless。
qt_ui/DeepStreamRunner.cpp：
DeepStream 放入子线程，主线程仅跑 Qt 事件循环与 Web 服务。
默认强制软件编码 x264（DS_FORCE_SW_ENC=1），若手动置 0 且有 nvv4l2h264enc 才尝试 NVENC。
自动管线（无 DS_PIPELINE 时）：nvarguscamerasrc → nvvidconv →（可选 nvstreammux+nvinfer+nvdsosd）→ nvvideoconvert → x264enc → rtph264pay，RTSP 服务 8554。WebRTC 拉 rtsp://127.0.0.1:8554，libnice 依赖 gstreamer1.0-nice。
DS_PIPELINE 若存在则原样使用。
deploy/systemd/bamboo-qt-ui.service：
offscreen 环境、GStreamer 插件路径、gst-plugin-scanner 软链到 /usr/libexec/gstreamer-1.0/gst-plugin-scanner。
默认 DS_AUTOSTART=1，DS_SINK=rtsp，DS_FORCE_SW_ENC=1。
为保证通路，暂时将 DS_PIPELINE 设为最小相机→nvvidconv→I420→x264enc→rtph264pay。
Makefile：必装依赖包含 gstreamer1.0-nice/libnice10、gstreamer1.0-x、libx264-dev、nvidia-l4t-gstreamer/nvidia-l4t-multimedia 等；Qt6/QML/Multimedia/Quick 控件依赖齐全。
现象与原因

8080/9000 监听正常，但无视频；RTSP 503/超时。nvarguscamerasrc 报 “No cameras available / Unable to open BW Ioctl FD”。
根因：Argus 相机栈未成功启动；在纯 headless 下需要有效 EGLDisplay，且 Argus 服务/DTB/overlay/电源接口需正常。
已采取的解决路径

持久化 EGL 上下文，避免 headless EGL_NO_DISPLAY 让 Argus 静默失败。
将 DeepStream 从主线程移到子线程，防止 GStreamer 内部阻塞 Qt 事件循环。
固定 x264 软件编码，绕过 NVENC 设备缺失。
硬设 GStreamer 环境变量与 plugin-scanner 路径，避免扫描失败。
下一步排查与操作指引

先让 Argus 出帧（关键）：
export GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0:/usr/lib/aarch64-linux-gnu/tegra
export GST_PLUGIN_SCANNER=/usr/lib/aarch64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-plugin-scanner
export GST_GL_PLATFORM=egl GST_GL_API=gles2 EGL_PLATFORM=surfaceless XDG_RUNTIME_DIR=/run/bamboo-qt
gst-launch-1.0 -v nvarguscamerasrc sensor-id=0 num-buffers=30 ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12' ! fakesink
若失败，检查 Jetson-IO/DTB 选择正确 CSI overlay，重启 nvargus-daemon，必要时重新刷机/同步 BSP。
验证链路时可临时设置测试源，不依赖相机：
sudo systemctl set-environment DS_PIPELINE="videotestsrc is-live=true ! video/x-raw,width=1280,height=720,framerate=30/1 ! x264enc tune=zerolatency bitrate=4000 speed-preset=superfast ! h264parse config-interval=1 ! rtph264pay name=pay0 pt=96"
sudo systemctl restart bamboo-qt-ui
确认 RTSP/Web/WebRTC 全链路可达后，再去掉 DS_PIPELINE 用相机。
WebRTC 需安装 libnice/gstreamer1.0-nice，确保 webrtcbin 可用。
相机恢复后，可移除服务里强制的最小 DS_PIPELINE，恢复自动管线（含 nvinfer/nvdsosd），模型路径已指向 /opt/bamboo-qt/models/best.onnx。