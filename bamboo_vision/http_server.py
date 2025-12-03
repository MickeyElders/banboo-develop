import logging
from functools import partial
import threading
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request

from .shared_state import SharedState
from .calibration import CalibrationManager


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
        # Placeholder for button actions; extend with actual commands.
        body = request.get_json(force=True, silent=True) or {}
        action = body.get("action", "noop")
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

    thread = threading.Thread(
        target=partial(app.run, host=host, port=port, threaded=True, use_reloader=False),
        daemon=True,
        name="http-server",
    )
    thread.start()
    logging.info("HTTP server started on http://%s:%d", host, port)
    return thread
