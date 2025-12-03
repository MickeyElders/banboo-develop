import json
import threading
import time


class SharedState:
    """Thread-safe runtime state for HTTP API and status reporting."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = {
            "last_detection": None,
            "plc": {"state": 0, "ready": False, "pos": 0.0},
            "fps": 0.0,
            "heartbeat": 0,
            "ts": time.time(),
        }

    def update_detection(self, x_mm, conf, result_code):
        with self.lock:
            self.data["last_detection"] = {
                "x_mm": x_mm,
                "conf": conf,
                "result": result_code,
                "ts": time.time(),
            }

    def update_plc(self, state, ready, pos, heartbeat):
        with self.lock:
            self.data["plc"] = {
                "state": state,
                "ready": ready,
                "pos": pos,
            }
            self.data["heartbeat"] = heartbeat
            self.data["ts"] = time.time()

    def update_fps(self, fps):
        with self.lock:
            self.data["fps"] = fps
            self.data["ts"] = time.time()

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.data))
