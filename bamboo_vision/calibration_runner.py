import subprocess
import threading
import os
from pathlib import Path


class CalibrationRunner:
    """Manage jetson-utils calibrate-camera subprocess."""

    def __init__(self, cfg: dict):
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        tool_cfg = cfg.get("calibration_tool", {})
        self.binary = tool_cfg.get("binary", "calibrate-camera")
        self.board_rows = int(tool_cfg.get("board_rows", 7))
        self.board_cols = int(tool_cfg.get("board_cols", 10))
        self.square_size = float(tool_cfg.get("square_size", 0.024))  # meters
        self.output_dir = Path(tool_cfg.get("output_dir", "/opt/bamboo-vision/calibration"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self, camera: str) -> dict:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return {"ok": False, "error": "calibration already running"}
            output_file = self.output_dir / f"calib_{self._sanitize(camera)}.json"
            args = [
                self.binary,
                f"--input={camera}",
                f"--rows={self.board_rows}",
                f"--cols={self.board_cols}",
                f"--size={self.square_size}",
                f"--output={output_file}",
            ]
            env = os.environ.copy()
            env.setdefault("DISPLAY", "")
            env.setdefault("GST_PLUGIN_PATH", "/usr/lib/aarch64-linux-gnu/gstreamer-1.0:/usr/lib/aarch64-linux-gnu/tegra")
            env.setdefault("GST_PLUGIN_SCANNER", "/usr/lib/aarch64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-plugin-scanner")
            self._proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
            return {"ok": True, "pid": self._proc.pid, "output": str(output_file)}

    def stop(self) -> dict:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                return {"ok": True, "stopped": True}
            return {"ok": False, "error": "not running"}

    def status(self) -> dict:
        with self._lock:
            if self._proc is None:
                return {"ok": True, "running": False}
            code = self._proc.poll()
            return {"ok": True, "running": code is None, "returncode": code, "pid": self._proc.pid}

    def _sanitize(self, uri: str) -> str:
        return uri.replace("://", "_").replace("/", "_").replace(":", "_")
