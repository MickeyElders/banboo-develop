import threading
from pathlib import Path
import yaml


class CalibrationManager:
    """Thread-safe calibration storage and pixel->mm conversion."""

    def __init__(self, calib_cfg: dict, persist_file: Path | None = None):
        self._lock = threading.Lock()
        self._persist_file = persist_file
        self._calib = {
            "pixel_to_mm": float(calib_cfg.get("pixel_to_mm", 1.0)),
            "offset_mm": float(calib_cfg.get("offset_mm", 0.0)),
            "latency_ms": float(calib_cfg.get("latency_ms", 0.0)),
            "belt_speed_mm_s": float(calib_cfg.get("belt_speed_mm_s", 0.0)),
        }
        self._loaded_from_file = False
        # Attempt to load from persisted file (json/yaml). If missing, keep defaults.
        if self._persist_file:
            self._loaded_from_file = self._load_from_file()

    def to_mm(self, x_px: float) -> float:
        with self._lock:
            return x_px * self._calib["pixel_to_mm"] + self._calib["offset_mm"]

    def get(self) -> dict:
        with self._lock:
            data = dict(self._calib)
            data["loaded"] = self._loaded_from_file
            data["file"] = str(self._persist_file) if self._persist_file else ""
            return data

    def status(self) -> dict:
        with self._lock:
            return {
                "file": str(self._persist_file) if self._persist_file else "",
                "loaded": self._loaded_from_file,
                "missing": not self._loaded_from_file,
            }

    def update(self, data: dict):
        with self._lock:
            for key in ("pixel_to_mm", "offset_mm", "latency_ms", "belt_speed_mm_s"):
                if key in data:
                    self._calib[key] = float(data[key])
            # manual update counts as loaded
            self._loaded_from_file = True

    def persist(self):
        if not self._persist_file:
            return False
        try:
            self._persist_file.parent.mkdir(parents=True, exist_ok=True)
            with self._persist_file.open("w", encoding="utf-8") as f:
                yaml.safe_dump(self._calib, f, allow_unicode=True)
            self._loaded_from_file = True
            return True
        except Exception:
            return False

    def _load_from_file(self) -> bool:
        try:
            if not self._persist_file.exists():
                return False
            text = self._persist_file.read_text(encoding="utf-8")
            data = None
            # Try JSON then YAML
            try:
                import json
                data = json.loads(text)
            except Exception:
                try:
                    data = yaml.safe_load(text)
                except Exception:
                    data = None
            if not isinstance(data, dict):
                return False
            with self._lock:
                for key in ("pixel_to_mm", "offset_mm", "latency_ms", "belt_speed_mm_s"):
                    if key in data:
                        self._calib[key] = float(data[key])
            return True
        except Exception:
            return False
