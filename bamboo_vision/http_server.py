import logging
from functools import partial
import threading
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request
import os
import multiprocessing

from .shared_state import SharedState
from .calibration import CalibrationManager
from . import system_control


def start_http_server(cfg: dict, state: SharedState, base_dir: Path, calib: CalibrationManager):
    http_cfg = cfg.get("http", {})
    if not http_cfg.get("enable", True):
        logging.info("HTTP server disabled via config.http.enable")
        return None
    host = http_cfg.get("host", "0.0.0.0")
    port = int(http_cfg.get("port", 8080))

    app = Flask(__name__, static_folder=None)

    @app.route("/")
    def index():
        return send_from_directory(base_dir, "bamboo.html")

    @app.route("/bamboo.html")
    def html():
        return send_from_directory(base_dir, "bamboo.html")

    @app.route("/api/status")
    def api_status():
        return jsonify(state.snapshot())

    @app.route("/api/control", methods=["POST"])
    def api_control():
        body = request.get_json(force=True, silent=True) or {}
        action = body.get("action", "noop")
        # Map actions to shared state flags or future hooks
        if action == "start":
            state.update_control({"running": True, "mode": "start"})
        elif action == "pause":
            state.update_control({"running": False, "mode": "pause"})
        elif action == "stop":
            state.update_control({"running": False, "mode": "stop"})
        elif action == "emergency_stop":
            state.update_control({"running": False, "mode": "emergency"})
        return jsonify({"ok": True, "action": action})

    @app.route("/api/calibration", methods=["GET", "POST"])
    def api_calibration():
        if request.method == "GET":
            return jsonify({"ok": True, "calibration": calib.get()})
        body = request.get_json(force=True, silent=True) or {}
        calib.update(body)
        if body.get("persist"):
            calib.persist()
            state.update_calibration(calib.get())
        return jsonify({"ok": True, "calibration": calib.get()})

    @app.route("/api/jetson")
    def api_jetson():
        # Lightweight system stats without extra deps
        try:
            load1, _, _ = os.getloadavg()
            cpus = max(1, multiprocessing.cpu_count())
            cpu_pct = min(100.0, (load1 / cpus) * 100.0)
        except Exception:
            cpu_pct = 0.0

        mem_total_mb = mem_free_mb = mem_avail_mb = 0.0
        try:
            with open("/proc/meminfo", "r") as f:
                info = f.read().splitlines()
            kv = {}
            for line in info:
                parts = line.replace(":", "").split()
                if len(parts) >= 2:
                    kv[parts[0]] = float(parts[1])  # kB
            mem_total_mb = kv.get("MemTotal", 0.0) / 1024.0
            mem_avail_mb = kv.get("MemAvailable", 0.0) / 1024.0
            mem_free_mb = kv.get("MemFree", 0.0) / 1024.0
        except Exception:
            pass
        mem_used_mb = max(0.0, mem_total_mb - mem_avail_mb)

        temp_c = 0.0
        try:
            import glob
            temps = []
            for path in glob.glob("/sys/devices/virtual/thermal/thermal_zone*/temp"):
                try:
                    with open(path, "r") as f:
                        val = float(f.read().strip()) / 1000.0
                        temps.append(val)
                except Exception:
                    continue
            if temps:
                temp_c = max(temps)
        except Exception:
            pass

        uptime_sec = 0.0
        try:
            with open("/proc/uptime", "r") as f:
                uptime_sec = float(f.read().split()[0])
        except Exception:
            pass

        return jsonify(
            {
                "cpu": cpu_pct,
                "gpu": 0.0,  # GPU util not parsed here
                "mem_total_mb": mem_total_mb,
                "mem_used_mb": mem_used_mb,
                "mem_free_mb": mem_free_mb,
                "temp_c": temp_c,
                "uptime_sec": uptime_sec,
                "power_w": 0.0,
                "emc_mhz": 0,
                "fan_rpm": 0,
            }
        )

    @app.route("/api/power", methods=["POST"])
    def api_power():
        body = request.get_json(force=True, silent=True) or {}
        action = body.get("action")
        if action == "reboot":
            rc, out, err = system_control.reboot()
        elif action == "shutdown":
            rc, out, err = system_control.shutdown()
        elif action == "restart_service":
            rc, out, err = system_control.restart_service()
        else:
            return jsonify({"ok": False, "error": "unknown action"})
        return jsonify({"ok": rc == 0, "stdout": out, "stderr": err})

    @app.route("/api/wifi", methods=["GET", "POST"])
    def api_wifi():
        if request.method == "GET":
            return jsonify(system_control.wifi_status())
        body = request.get_json(force=True, silent=True) or {}
        return jsonify(system_control.wifi_apply(body))

    @app.route("/api/wifi/restart", methods=["POST"])
    def api_wifi_restart():
        return jsonify(system_control.wifi_restart())

    thread = threading.Thread(
        target=partial(app.run, host=host, port=port, threaded=True, use_reloader=False),
        daemon=True,
        name="http-server",
    )
    thread.start()
    logging.info("HTTP server started on http://%s:%d", host, port)
    return thread
