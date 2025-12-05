import argparse
import logging
import signal
import sys
import time
from pathlib import Path
import os

# Prefer local jetson-inference source/build tree
_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_JI_PY_SRC = _ROOT / "jetson-inference" / "python"
_LOCAL_JI_PY_BUILD = _ROOT / "jetson-inference" / "build" / "aarch64" / "python"
_LOCAL_JI_PY_BUILD_LIB = _ROOT / "jetson-inference" / "build" / "aarch64" / "lib" / "python"
_LOCAL_JI_LIB = _ROOT / "jetson-inference" / "build" / "aarch64" / "lib"

for _p in (_LOCAL_JI_PY_BUILD, _LOCAL_JI_PY_BUILD_LIB, _LOCAL_JI_PY_SRC):
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
    cam_uri = cam_cfg.get("pipeline") or "csi://0"
    out_cfg = cfg.get("output", {})
    calib_cfg = cfg.get("calibration", {})
    calib_manager = CalibrationManager(calib_cfg, cfg_path if cfg_path.exists() else None)
    if args.headless or os.environ.get("HEADLESS") == "1" or not os.environ.get("DISPLAY"):
        out_cfg = dict(out_cfg)
        out_cfg["hdmi"] = False
        logging.info("Headless mode detected; disabling HDMI output")

    preflight_checks(cfg, _LOCAL_JI_PY_BUILD, _LOCAL_JI_LIB)

    input_stream = ju.videoSource(cam_uri)
    outputs = build_outputs(out_cfg, cam_cfg)
    if not outputs:
        logging.warning("No outputs available (RTSP/HDMI). Continuing without rendering.")

    net = build_net(cfg)
    font = ju.cudaFont()
    state = SharedState()
    mb = ModbusBridge(cfg, state)
    base_dir = Path(__file__).resolve().parent.parent
    start_http_server(cfg, state, base_dir, calib_manager)
    state.update_calibration(calib_manager.get())

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

    while running:
        now = time.time()
        control = state.get_control()
        if control.get("mode") != last_control_mode:
            logging.info("Control change: %s", control)
            last_control_mode = control.get("mode")
        mb.sync_control(control)
        mb.step(now, control)

        if not input_stream.IsStreaming():
            logging.warning("Input stream not streaming, breaking loop")
            break
        img = input_stream.Capture()
        if img is None:
            continue

        if not control.get("running", True):
            # Skip inference/output but keep heartbeats and status updates alive
            font.OverlayText(
                img,
                text=f"paused ({control.get('mode')})",
                x=5,
                y=5,
                color=ju.makeColor(255, 200, 64, 255),
                bg_color=ju.makeColor(0, 0, 0, 180),
            )
            for out in outputs:
                out.Render(img)
                out.SetStatus(f"Paused ({control.get('mode')})")
            state.update_detection(None, 0.0, 0)
            state.update_fps(0.0)
            continue

        detections = net.Detect(img, overlay="box,labels,conf")
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
                img,
                text=f"x={x_mm:.1f}mm conf={best_det.Confidence:.2f}",
                x=5,
                y=5,
                color=ju.makeColor(0, 255, 0, 255),
                bg_color=ju.makeColor(0, 0, 0, 160),
            )
        else:
            font.OverlayText(
                img,
                text="no target",
                x=5,
                y=5,
                color=ju.makeColor(255, 64, 64, 255),
                bg_color=ju.makeColor(0, 0, 0, 160),
            )

        for out in outputs:
            out.Render(img)
            out.SetStatus(f"FPS {net.GetNetworkFPS():.1f}")

        state.update_detection(x_mm, conf, result_code)
        state.update_fps(net.GetNetworkFPS())
        if now - last_publish >= publish_interval:
            mb.publish_detection(x_mm, result_code)
            last_publish = now

        if outputs and not outputs[0].IsStreaming():
            running = False

    mb.close()
    logging.info("Shutting down")


if __name__ == "__main__":
    main()
