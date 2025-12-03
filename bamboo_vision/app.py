import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import jetson.utils as ju

from .config_loader import load_config
from .pipeline import build_net, build_outputs, pixel_to_mm
from .shared_state import SharedState
from .modbus import ModbusBridge
from .http_server import start_http_server


def main():
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
    if args.headless:
        out_cfg = dict(out_cfg)
        out_cfg["hdmi"] = False

    input_stream = ju.videoSource(cam_uri)
    outputs = build_outputs(out_cfg)
    if not outputs:
        outputs.append(ju.videoOutput("rtsp://@:8554/live"))

    net = build_net(cfg)
    font = ju.cudaFont()
    state = SharedState()
    mb = ModbusBridge(cfg, state)
    base_dir = Path(__file__).resolve().parent.parent
    start_http_server(cfg, state, base_dir)

    running = True

    def handle_sig(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    last_publish = 0.0
    publish_interval = float(cfg.get("app", {}).get("publish_interval_ms", 100)) / 1000.0

    while running:
        if not input_stream.IsStreaming():
            logging.warning("Input stream not streaming, breaking loop")
            break
        img = input_stream.Capture()
        if img is None:
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
            x_mm = pixel_to_mm(cx, calib_cfg)
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

        now = time.time()
        state.update_detection(x_mm, conf, result_code)
        state.update_fps(net.GetNetworkFPS())
        mb.step(now)
        if now - last_publish >= publish_interval:
            mb.publish_detection(x_mm, result_code)
            last_publish = now

        if not outputs[0].IsStreaming():
            running = False

    mb.close()
    logging.info("Shutting down")


if __name__ == "__main__":
    main()
