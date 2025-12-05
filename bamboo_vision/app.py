import argparse
import logging
import signal
import sys
import time
from pathlib import Path
import os
import glob

# Prefer local jetson-inference source/build tree
_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_JI_PY_SRC = _ROOT / "jetson-inference" / "python"
_LOCAL_JI_UTILS_PY = _ROOT / "jetson-inference" / "utils" / "python" / "python"
_LOCAL_JI_PY_BUILD = _ROOT / "jetson-inference" / "build"
_LOCAL_JI_LIB = _LOCAL_JI_PY_BUILD / "aarch64" / "lib"

# Collect possible python package roots
_extra_py_paths = [_LOCAL_JI_PY_SRC, _LOCAL_JI_UTILS_PY]
# collect any python* dirs under build/**
_extra_py_paths += [p for p in (Path(_ROOT) / "jetson-inference" / "build").rglob("python*") if p.is_dir()]

for _p in _extra_py_paths:
    if _p.is_dir():
        sys.path.insert(0, str(_p))

# Ensure local libs are visible (for freshly built jetson-inference in-tree)
if _LOCAL_JI_LIB.is_dir():
    os.environ["LD_LIBRARY_PATH"] = str(_LOCAL_JI_LIB) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# Require local jetson_utils (no system fallback)
try:
    import jetson_utils as ju
except ImportError:
    print("jetson_utils not found in local jetson-inference build. Please run `make install-jetson` to build/install bindings.", file=sys.stderr)
    sys.exit(1)

from .config_loader import load_config
from .pipeline import build_net, build_outputs
from .shared_state import SharedState
from .modbus import ModbusBridge
from .http_server import start_http_server
from .ws_stream import FrameBroadcaster
from .calibration import CalibrationManager
from .preflight import preflight_checks


def main():
    # Ensure GStreamer plugin path includes Tegra encoders
    os.environ.setdefault("GST_PLUGIN_PATH", "/usr/lib/aarch64-linux-gnu/gstreamer-1.0:/usr/lib/aarch64-linux-gnu/tegra")
    os.environ.setdefault("GST_PLUGIN_SCANNER", "/usr/lib/aarch64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-plugin-scanner")
    # Prepend tegra libs for NV encoders
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    tegra_path = "/usr/lib/aarch64-linux-gnu/tegra"
    if tegra_path not in ld:
        os.environ["LD_LIBRARY_PATH"] = tegra_path + (":" + ld if ld else "")
    parser = argparse.ArgumentParser(description="Bamboo vision with jetson-inference + Modbus + HTTP")
    parser.add_argument("--config", default="config/runtime.yaml", help="Path to runtime config YAML")
    parser.add_argument("--headless", action="store_true", help="Disable HDMI output")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = load_config(cfg_path)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Starting bamboo vision service with %s", cfg_path)

    cam_cfg = cfg.get("camera", {})
    # 固定双摄 csi://0 与 csi://1，分辨率 1280x960（若配置里有则优先配置）
    width = cam_cfg.get("width", 1280)
    height = cam_cfg.get("height", 960)
    fps = cam_cfg.get("fps", 30)
    left_uri = "csi://0"
    right_uri = "csi://1"
    out_cfg = cfg.get("output", {})
    calib_cfg = cfg.get("calibration", {})
    calib_file = calib_cfg.get("file")
    if calib_file:
        calib_file = Path(calib_file)
        if not calib_file.is_absolute():
            calib_file = Path("/opt/bamboo-vision") / calib_file
    calib_manager = CalibrationManager(calib_cfg, calib_file)
    if args.headless or os.environ.get("HEADLESS") == "1" or not os.environ.get("DISPLAY"):
        out_cfg = dict(out_cfg)
        out_cfg["hdmi"] = False
        logging.info("Headless mode detected; disabling HDMI output")

    preflight_checks(cfg, _LOCAL_JI_PY_BUILD, _LOCAL_JI_LIB)

    # 初始化双摄
    base_ws = int(cfg.get("http", {}).get("ws_port", 8765))
    try:
        left = ju.videoSource(left_uri, argv=[f"--input-width={width}", f"--input-height={height}", f"--framerate={fps}"])
        right = ju.videoSource(right_uri, argv=[f"--input-width={width}", f"--input-height={height}", f"--framerate={fps}"])
        logging.info("Cameras initialized: %s & %s", left_uri, right_uri)
    except Exception as e:
        logging.error("Failed to init dual cameras: %s", e)
        sys.exit(1)

    broadcaster = FrameBroadcaster(host="0.0.0.0", port=base_ws)
    broadcaster.start()
    logging.info("WebSocket output on ws://0.0.0.0:%d", base_ws)

    outputs = []  # videoOutput 全部禁用，只保留推理与 WebSocket JPEG 推送
    logging.info("Outputs disabled: inference + websocket JPEG only.")

    net = build_net(cfg)
    font = ju.cudaFont()
    state = SharedState()
    mb = ModbusBridge(cfg, state)
    base_dir = Path(__file__).resolve().parent.parent
    start_http_server(cfg, state, base_dir, calib_manager)
    state.update_calibration(calib_manager.get())
    if calib_file and not calib_manager.status()["loaded"]:
        logging.warning("Calibration file missing: %s, please run calibration tool via UI to generate it.", calib_file)

    running = True
    control = state.get_control()
    last_control_mode = control.get("mode")

    def handle_sig(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    last_publish = 0.0
    publish_interval = float(cfg.get("app", {}).get("publish_interval_ms", 100)) / 1000.0

    combo = None  # 合成的左右并排图

    while running:
        now = time.time()
        control = state.get_control()
        if control.get("mode") != last_control_mode:
            logging.info("Control change: %s", control)
            last_control_mode = control.get("mode")
        mb.sync_control(control)
        mb.step(now, control)

        if not left.IsStreaming() or not right.IsStreaming():
            logging.warning("Input stream not streaming, skipping frame")
            time.sleep(0.01)
            continue

        l_img = left.Capture()
        r_img = right.Capture()
        if l_img is None or r_img is None:
            continue

        calib_missing = not calib_manager.status().get("loaded", False)

        if not control.get("running", True):
            font.OverlayText(
                l_img,
                text=f"paused ({control.get('mode')})",
                x=5,
                y=5,
                color=ju.makeColor(255, 200, 64, 255),
                bg_color=ju.makeColor(0, 0, 0, 180),
            )
        else:
            detections = net.Detect(l_img, overlay="box,labels,conf")
            best_det = None
            for det in detections:
                if best_det is None or det.Confidence > best_det.Confidence:
                    best_det = det

            x_mm = None
            result_code = 2  # default: no target
            conf = 0.0
            if best_det:
                cx = 0.5 * (best_det.Left + best_det.Right)
                x_mm = calib_manager.to_mm(cx)
                result_code = 1
                conf = best_det.Confidence
                font.OverlayText(
                    l_img,
                    text=f"x={x_mm:.1f}mm conf={best_det.Confidence:.2f}",
                    x=5,
                    y=5,
                    color=ju.makeColor(0, 255, 0, 255),
                    bg_color=ju.makeColor(0, 0, 0, 160),
                )
            else:
                font.OverlayText(
                    l_img,
                    text="no target",
                    x=5,
                    y=5,
                    color=ju.makeColor(255, 64, 64, 255),
                    bg_color=ju.makeColor(0, 0, 0, 160),
                )

            state.update_detection(x_mm, conf, result_code)
            state.update_fps(net.GetNetworkFPS())
            if now - last_publish >= publish_interval:
                mb.publish_detection(x_mm, result_code)
                last_publish = now

        if calib_missing:
            font.OverlayText(
                l_img,
                text="未标定：请在前端点击标定生成文件",
                x=5,
                y=30,
                color=ju.makeColor(255, 200, 64, 255),
                bg_color=ju.makeColor(0, 0, 0, 180),
            )

        # 合成到并排画面
        if combo is None or combo.width != (l_img.width + r_img.width) or combo.height != l_img.height:
            combo = ju.cudaAllocMapped(l_img.width + r_img.width, l_img.height, l_img.format)
        ju.cudaMemcpy2D(dest=combo, destX=0, src=l_img)
        ju.cudaMemcpy2D(dest=combo, destX=l_img.width, src=r_img)

        try:
            jpeg_bytes = ju.cudaEncodeImage(combo, "jpg")
            broadcaster.push(bytes(jpeg_bytes))
        except Exception as e:
            logging.debug("WebSocket frame encode/broadcast failed: %s", e)

        # 无本地渲染，纯推理推流

    mb.close()
    logging.info("Shutting down")


if __name__ == "__main__":
    main()
