import logging
import subprocess
import os
import sys
from pathlib import Path
import glob

# Require local jetson-inference build (no system fallback)
_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_JI_PY_SRC = _ROOT / "jetson-inference" / "python"
_LOCAL_JI_UTILS_PY = _ROOT / "jetson-inference" / "utils" / "python" / "python"
_LOCAL_JI_PY_BUILD_ROOT = _ROOT / "jetson-inference" / "build"
_LOCAL_JI_LIB = _LOCAL_JI_PY_BUILD_ROOT / "aarch64" / "lib"

_extra_py_paths = [_LOCAL_JI_PY_SRC, _LOCAL_JI_UTILS_PY]
_extra_py_paths += [p for p in _LOCAL_JI_PY_BUILD_ROOT.rglob("python*") if p.is_dir()]

for _p in _extra_py_paths:
    if _p.is_dir():
        sys.path.insert(0, str(_p))
if _LOCAL_JI_LIB.is_dir():
    os.environ["LD_LIBRARY_PATH"] = str(_LOCAL_JI_LIB) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

try:
    import jetson_inference as ji  # prefer local bindings
    import jetson_utils as ju
except ImportError:
    print("jetson_inference/jetson_utils not found in local build. Please run `make install-jetson`.", file=sys.stderr)
    sys.exit(1)


def build_net(cfg: dict):
    mcfg = cfg.get("model", {})
    base_dir = Path(__file__).resolve().parent.parent
    model_path = Path(mcfg.get("onnx", "models/best.onnx"))
    if not model_path.is_absolute():
        model_path = base_dir / model_path
    engine_path = mcfg.get("engine", "")
    if engine_path:
        engine_path = Path(engine_path)
        if not engine_path.is_absolute():
            engine_path = base_dir / engine_path
    threshold = float(mcfg.get("threshold", 0.5))
    nms = float(mcfg.get("nms", 0.45))
    input_shape = mcfg.get("input_shape")  # e.g. "1x3x960x960" to avoid dynamic profiles
    workspace = mcfg.get("workspace_mb")
    extra_args = [f"--model={model_path}"]
    if engine_path:
        extra_args += [f"--engine={engine_path}"]
    labels = mcfg.get("labels")
    if labels:
        labels_path = Path(labels)
        if not labels_path.is_absolute():
            labels_path = base_dir / labels_path
        extra_args += [f"--labels={labels_path}"]
    input_blob = mcfg.get("input_blob")
    if input_blob:
        extra_args += [f"--input-blob={input_blob}"]
    output_cvg = mcfg.get("output_cvg")
    if output_cvg:
        extra_args += [f"--output-cvg={output_cvg}"]
    output_bbox = mcfg.get("output_bbox")
    if output_bbox:
        extra_args += [f"--output-bbox={output_bbox}"]
    # If no coverage layer provided, reuse bbox to satisfy detectNet expectations for single-output models.
    if not output_cvg and output_bbox:
        extra_args += [f"--output-cvg={output_bbox}"]
    if input_shape:
        extra_args += [f"--input-shape={input_shape}"]
    if workspace:
        extra_args += [f"--workspace={workspace}"]
    logging.info("Loading model: %s", model_path)
    net = ji.detectNet(argv=extra_args, threshold=threshold)
    net.SetNMS(nms)
    return net


def build_outputs(out_cfg: dict, cam_cfg: dict):
    outputs = []
    # Detect encoder availability (for RTSP/HLS decisions)
    try:
        env = os.environ.copy()
        env.setdefault("GST_PLUGIN_PATH", "/usr/lib/aarch64-linux-gnu/gstreamer-1.0:/usr/lib/aarch64-linux-gnu/tegra")
        env.setdefault("GST_PLUGIN_SCANNER", "/usr/lib/aarch64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-plugin-scanner")
        env.setdefault("LD_LIBRARY_PATH", "/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:" + env.get("LD_LIBRARY_PATH", ""))
        have_nvenc = subprocess.run(["gst-inspect-1.0", "nvv4l2h264enc"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env).returncode == 0
        have_x264 = subprocess.run(["gst-inspect-1.0", "x264enc"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env).returncode == 0
    except FileNotFoundError:
        have_nvenc = False
        have_x264 = True  # assume software stack present

    # Prefer WebRTC (built-in jetson.utils server) if enabled, then RTSP/HDMI; keep a single output to avoid multiple pipelines on CSI cameras.
    selected = None
    webrtc_enabled = out_cfg.get("webrtc", False)
    if webrtc_enabled:
        uri = out_cfg.get("webrtc_uri", "webrtc://@:9000/live")
        try:
            selected = ju.videoOutput(uri)
            logging.info("WebRTC enabled via jetson.utils videoOutput: %s", uri)
        except Exception as e:
            logging.error("Failed to create WebRTC output (%s): %s", uri, e)
            selected = None

    rtsp_enabled = out_cfg.get("rtsp", False)
    if selected is None and rtsp_enabled:
        host = out_cfg.get("rtsp_host", "127.0.0.1")
        port = out_cfg.get("rtsp_port", 8554)
        path = out_cfg.get("rtsp_path", "live")
        rtsp_uri = out_cfg.get("rtsp_uri", f"rtsp://{host}:{port}/{path}")
        if rtsp_uri.startswith("rtsp://@:"):
            rtsp_uri = "rtsp://127.0.0.1:" + rtsp_uri.split("@:", 1)[-1]
        if not have_nvenc and not out_cfg.get("software_rtsp", False):
            logging.error("RTSP disabled: nvv4l2h264enc not present; enable software_rtsp or install NVENC packages.")
        elif have_nvenc and not out_cfg.get("software_rtsp", False):
            try:
                selected = ju.videoOutput(rtsp_uri)
                logging.info("RTSP enabled via jetson.utils videoOutput: %s", rtsp_uri)
            except Exception as e:
                logging.error("Failed to create RTSP output (%s): %s", rtsp_uri, e)
                selected = None
        elif out_cfg.get("software_rtsp", False):
            if not have_x264:
                logging.error("Software RTSP requested but x264enc missing; install gstreamer1.0-plugins-*/libx264.")
            else:
                # Use built-in MJPEG output; browser <img> 可直接拉取 http://host:port
                mjpeg_uri = f"mjpeg://{host}:{port}"
                try:
                    selected = ju.videoOutput(mjpeg_uri)
                    logging.info("Software MJPEG output via jetson.utils: %s", mjpeg_uri)
                except Exception as e:
                    logging.error("Failed to create software MJPEG output: %s", e)
                    selected = None

    if selected is None and out_cfg.get("hdmi", True):
        try:
            selected = ju.videoOutput("display://0")
            logging.info("HDMI output enabled (display://0)")
        except Exception as e:
            logging.warning("Failed to create HDMI output (display://0): %s; running headless", e)

    if selected:
        outputs.append(selected)
    else:
        logging.warning("No outputs available (RTSP/HDMI).")
    return outputs
