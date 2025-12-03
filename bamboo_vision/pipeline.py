import logging
import subprocess
import os
import jetson.inference as ji
import jetson.utils as ju
from pathlib import Path


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
    extra_args = []
    # model path is passed via detectNet(model=...)
    if engine_path:
        extra_args += ["--engine", str(engine_path)]
    labels = mcfg.get("labels")
    if labels:
        labels_path = Path(labels)
        if not labels_path.is_absolute():
            labels_path = base_dir / labels_path
        extra_args += ["--labels", str(labels_path)]
    input_blob = mcfg.get("input_blob")
    if input_blob:
        extra_args += ["--input-blob", input_blob]
    output_cvg = mcfg.get("output_cvg")
    if output_cvg:
        extra_args += ["--output-cvg", output_cvg]
    output_bbox = mcfg.get("output_bbox")
    if output_bbox:
        extra_args += ["--output-bbox", output_bbox]
    logging.info("Loading model: %s", model_path)
    # Pass model via dedicated parameter, argv carries auxiliary options only
    net = ji.detectNet(model=str(model_path), threshold=threshold, argv=extra_args)
    net.SetNMS(nms)
    return net


def build_outputs(out_cfg: dict, cam_cfg: dict):
    outputs = []
    # Detect encoder availability
    use_x264 = False
    have_x264 = False
    try:
        env = os.environ.copy()
        env.setdefault("GST_PLUGIN_PATH", "/usr/lib/aarch64-linux-gnu/gstreamer-1.0:/usr/lib/aarch64-linux-gnu/tegra")
        env.setdefault("GST_PLUGIN_SCANNER", "/usr/lib/aarch64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-plugin-scanner")
        env.setdefault("LD_LIBRARY_PATH", "/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:" + env.get("LD_LIBRARY_PATH", ""))
        if subprocess.run(["gst-inspect-1.0", "nvv4l2h264enc"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env).returncode != 0:
            use_x264 = True
        have_x264 = subprocess.run(["gst-inspect-1.0", "x264enc"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env).returncode == 0
    except FileNotFoundError:
        # gst-inspect missing; be conservative and use x264
        use_x264 = True
        have_x264 = True  # assume available in good plugins

    if use_x264:
        logging.warning("nvv4l2h264enc not available; will try software x264enc for RTSP")
    if out_cfg.get("hdmi", True):
        try:
            outputs.append(ju.videoOutput("display://0"))
        except Exception as e:
            logging.warning("Failed to create HDMI output (display://0): %s; continuing without HDMI", e)
    # RTSP output (prefer x264 pipeline when HW encoder missing)
    rtsp_enabled = out_cfg.get("rtsp", False) or out_cfg.get("software_rtsp", False)
    if rtsp_enabled:
        host = out_cfg.get("rtsp_host", "127.0.0.1")
        port = out_cfg.get("rtsp_port", 8554)
        path = out_cfg.get("rtsp_path", "live")
        rtsp_uri = out_cfg.get("rtsp_uri", f"rtsp://{host}:{port}/{path}")
        # normalise rtsp_uri host placeholder like rtsp://@:8554/live
        if rtsp_uri.startswith("rtsp://@:"):
            rtsp_uri = "rtsp://127.0.0.1:" + rtsp_uri.split("@:", 1)[-1]
        width = cam_cfg.get("width", 1280)
        height = cam_cfg.get("height", 720)
        fr = cam_cfg.get("fps", 30)
        # force software pipeline when HW encoder missing or software_rtsp set
        if use_x264 or out_cfg.get("software_rtsp", False):
            if not have_x264:
                logging.error("x264enc not available; RTSP disabled")
            else:
                pipeline = (
                    f"appsrc name=mysource is-live=true do-timestamp=true format=3 ! "
                    f"video/x-raw,format=RGBA,width={width},height={height},framerate={fr}/1 ! "
                    "videoconvert ! video/x-raw,format=I420 ! "
                    "x264enc tune=zerolatency bitrate=4000 speed-preset=superfast key-int-max=30 ! "
                    "rtph264pay config-interval=1 pt=96 ! "
                    f"rtspclientsink location={rtsp_uri}"
                )
                try:
                    outputs.append(ju.videoOutput("gstreamer://" + pipeline))
                    logging.info("Software RTSP (x264) enabled: %s", rtsp_uri)
                except Exception as e:
                    logging.error("Failed to create software RTSP output: %s", e)
        else:
            try:
                outputs.append(ju.videoOutput(rtsp_uri))
                logging.info("RTSP enabled via NVENC: %s", rtsp_uri)
            except Exception as e:
                logging.error("Failed to create RTSP output (%s): %s", rtsp_uri, e)

    # Raw UDP (optional)
    if out_cfg.get("raw_udp", False):
        width = cam_cfg.get("width", 1280)
        height = cam_cfg.get("height", 720)
        fr = cam_cfg.get("fps", 30)
        host = out_cfg.get("raw_udp_host", "127.0.0.1")
        port = out_cfg.get("raw_udp_port", 5600)
        pipeline = (
            f"appsrc name=mysource is-live=true do-timestamp=true format=3 ! "
            f"video/x-raw,format=RGBA,width={width},height={height},framerate={fr}/1 ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            f"udpsink host={host} port={port} sync=false"
        )
        try:
            outputs.append(ju.videoOutput("gstreamer://" + pipeline))
            logging.info("Raw UDP output enabled to udp://%s:%d (RGBA->I420, no encoder)", host, port)
        except Exception as e:
            logging.error("Failed to create raw UDP output: %s", e)
    return outputs
