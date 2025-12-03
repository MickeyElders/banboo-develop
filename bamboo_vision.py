#!/usr/bin/env python3
"""
Minimal bamboo vision service using jetson-inference (Python wheel).
Features:
- Capture from CSI or custom GStreamer pipeline
- YOLO detection via detectNet (ONNX/engine)
- Dual output: HDMI display://0 and RTSP rtsp://@:8554/live
- Modbus TCP bridge to PLC (register map in PLC.md)
Calibration is intentionally simple; replace with real values when available.
"""
import argparse
import logging
import signal
import sys
import time
from pathlib import Path
import struct
import yaml

try:
    from pymodbus.client import ModbusTcpClient
except ImportError as exc:  # pragma: no cover
    print("Missing dependency: pymodbus (pip install pymodbus)", file=sys.stderr)
    raise

import jetson.inference as ji
import jetson.utils as ju


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def float_to_regs_be(value: float) -> tuple[int, int]:
    """Pack float into two 16-bit registers (big endian high/low)."""
    packed = struct.pack(">f", float(value))
    high = int.from_bytes(packed[0:2], byteorder="big", signed=False)
    low = int.from_bytes(packed[2:4], byteorder="big", signed=False)
    return high, low


def regs_to_float_be(high: int, low: int) -> float:
    packed = high.to_bytes(2, "big") + low.to_bytes(2, "big")
    return struct.unpack(">f", packed)[0]


class ModbusBridge:
    """Lightweight Modbus TCP client matching PLC.md map."""

    def __init__(self, cfg: dict):
        mcfg = cfg.get("modbus", {})
        self.host = mcfg.get("host", "127.0.0.1")
        self.port = int(mcfg.get("port", 502))
        self.slave_id = int(mcfg.get("slave_id", 1))
        self.poll_ms = int(mcfg.get("poll_ms", 50))
        self.hb_ms = int(mcfg.get("write_heartbeat_ms", 20))
        self.addr_cam = mcfg.get("addr_cam_to_plc", {})
        self.addr_plc = mcfg.get("addr_plc_to_cam", {})
        self.client = ModbusTcpClient(host=self.host, port=self.port, unit_id=self.slave_id, timeout=1)
        self.connected = False
        self.last_poll = 0.0
        self.last_hb = 0.0
        self.plc_ready = False
        self.plc_state = 0
        self.plc_pos = 0.0

    def ensure_connected(self) -> bool:
        if self.connected and self.client.connected:
            return True
        self.connected = self.client.connect()
        if not self.connected:
            logging.warning("Modbus connect failed %s:%s", self.host, self.port)
        return self.connected

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
        self.connected = False

    def step(self, now: float):
        if not self.ensure_connected():
            return

        # Heartbeat/communication ack
        if now - self.last_hb >= self.hb_ms / 1000.0:
            ack_addr = self.addr_cam.get("comm", 0x07D0)
            status_addr = self.addr_cam.get("status", 0x07D1)
            self.client.write_register(address=ack_addr, value=1, slave=self.slave_id)
            self.client.write_register(address=status_addr, value=1, slave=self.slave_id)  # 1=normal
            self.last_hb = now

        # Poll PLC state/position
        if now - self.last_poll >= self.poll_ms / 1000.0:
            start = self.addr_plc.get("heartbeat", 0x0834)
            count = 4  # 0834..0837 covers heartbeat/state/pos(float)
            resp = self.client.read_holding_registers(address=start, count=count, slave=self.slave_id)
            if not resp.isError():
                hb_plc, state, pos_hi, pos_lo = resp.registers
                self.plc_state = state
                self.plc_pos = regs_to_float_be(pos_hi, pos_lo)
                self.plc_ready = state == 1  # 1=ready to receive coordinate
            else:
                logging.warning("Modbus read error: %s", resp)
            self.last_poll = now

    def publish_detection(self, x_mm: float | None, result_code: int):
        """
        Write detection to PLC if connected.
        result_code: 1=success, 2=fail/no target (per PLC.md D2004)
        """
        if not self.ensure_connected():
            return
        # Respect PLC gating if provided
        if not self.plc_ready:
            logging.debug("PLC not ready (state=%s), skip publish", self.plc_state)
            return
        coord_addr = self.addr_cam.get("coord", 0x07D2)
        result_addr = self.addr_cam.get("result", 0x07D4)
        coord_val = x_mm if x_mm is not None else 0.0
        hi, lo = float_to_regs_be(coord_val)
        self.client.write_registers(address=coord_addr, values=[hi, lo], slave=self.slave_id)
        self.client.write_register(address=result_addr, value=result_code, slave=self.slave_id)
        logging.info("Published to PLC: x=%.2f mm result=%d", coord_val, result_code)


def build_net(cfg: dict):
    mcfg = cfg.get("model", {})
    model_path = mcfg.get("onnx", "models/best.onnx")
    engine_path = mcfg.get("engine", "")
    threshold = float(mcfg.get("threshold", 0.5))
    nms = float(mcfg.get("nms", 0.45))
    extra_args = []
    if engine_path:
        extra_args += ["--engine", engine_path]
    # Allow overriding labels/output names if needed
    labels = mcfg.get("labels")
    if labels:
        extra_args += ["--labels", labels]
    input_blob = mcfg.get("input_blob")
    if input_blob:
        extra_args += ["--input-blob", input_blob]
    output_cvg = mcfg.get("output_cvg")
    if output_cvg:
        extra_args += ["--output-cvg", output_cvg]
    output_bbox = mcfg.get("output_bbox")
    if output_bbox:
        extra_args += ["--output-bbox", output_bbox]
    net = ji.detectNet(model=model_path, threshold=threshold, argv=extra_args)
    net.SetNMS(nms)
    return net


def pixel_to_mm(x_px: float, img_w: int, calib: dict) -> float:
    """
    Simple pixel->mm conversion.
    Replace with real calibration later.
    """
    px_to_mm = float(calib.get("pixel_to_mm", 1.0))
    offset_mm = float(calib.get("offset_mm", 0.0))
    return x_px * px_to_mm + offset_mm


def main():
    parser = argparse.ArgumentParser(description="Bamboo vision with jetson-inference + Modbus")
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

    # Video IO
    input_stream = ju.videoSource(cam_uri)
    outputs = []
    if out_cfg.get("hdmi", True) and not args.headless:
        outputs.append(ju.videoOutput("display://0"))
    if out_cfg.get("rtsp", True):
        rtsp_uri = out_cfg.get("rtsp_uri", "rtsp://@:8554/live")
        outputs.append(ju.videoOutput(rtsp_uri))
    if not outputs:
        outputs.append(ju.videoOutput("rtsp://@:8554/live"))

    net = build_net(cfg)
    font = ju.cudaFont()
    mb = ModbusBridge(cfg)

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
        if best_det:
            cx = 0.5 * (best_det.Left + best_det.Right)
            x_mm = pixel_to_mm(cx, img.width, calib_cfg)
            result_code = 1
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

        # Render to all outputs
        for out in outputs:
            out.Render(img)
            out.SetStatus(f"FPS {net.GetNetworkFPS():.1f}")

        now = time.time()
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
