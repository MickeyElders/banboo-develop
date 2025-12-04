import threading
from pathlib import Path
import yaml


class CalibrationManager:
    """Thread-safe calibration storage and pixel->mm conversion."""

    def __init__(self, calib_cfg: dict, persist_path: Path | None = None):
        self._lock = threading.Lock()
        self._calib = {
            "pixel_to_mm": float(calib_cfg.get("pixel_to_mm", 1.0)),
            "offset_mm": float(calib_cfg.get("offset_mm", 0.0)),
            "latency_ms": float(calib_cfg.get("latency_ms", 0.0)),
            "belt_speed_mm_s": float(calib_cfg.get("belt_speed_mm_s", 0.0)),
        }
        self._persist_path = persist_path

    def to_mm(self, x_px: float) -> float:
        with self._lock:
            return x_px * self._calib["pixel_to_mm"] + self._calib["offset_mm"]

    def get(self) -> dict:
        with self._lock:
            return dict(self._calib)

    def update(self, data: dict):
        with self._lock:
            for key in ("pixel_to_mm", "offset_mm", "latency_ms", "belt_speed_mm_s"):
                if key in data:
                    self._calib[key] = float(data[key])

    def persist(self):
        if not self._persist_path:
            return False
        try:
            with self._persist_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            cfg = {}
        if "calibration" not in cfg:
            cfg["calibration"] = {}
        cfg["calibration"].update(self.get())
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self._persist_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True)
            return True
        except Exception:
            return False
