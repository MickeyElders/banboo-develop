import logging
import subprocess
import os
import jetson.inference as ji
import jetson.utils as ju


def build_net(cfg: dict):
    mcfg = cfg.get("model", {})
    model_path = mcfg.get("onnx", "models/best.onnx")
    engine_path = mcfg.get("engine", "")
    threshold = float(mcfg.get("threshold", 0.5))
    nms = float(mcfg.get("nms", 0.45))
    extra_args = []
    if engine_path:
        extra_args += ["--engine", engine_path]
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


def build_outputs(out_cfg: dict):
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

    if out_cfg.get("hdmi", True):
        try:
            outputs.append(ju.videoOutput("display://0"))
        except Exception as e:
            logging.warning("Failed to create HDMI output (display://0): %s; continuing without HDMI", e)
    if out_cfg.get("rtsp", True):
        rtsp_uri = out_cfg.get("rtsp_uri", "rtsp://@:8554/live")
        argv = []
        if use_x264:
            if not have_x264:
                logging.error("Neither nvv4l2h264enc nor x264enc is available; RTSP output disabled")
            else:
                logging.warning("nvv4l2h264enc not available; falling back to software x264enc for RTSP")
                os.environ["GST_ENCODER"] = "x264enc"
                if "?" in rtsp_uri:
                    rtsp_uri = rtsp_uri + "&encoder=x264enc"
                else:
                    rtsp_uri = rtsp_uri + "?encoder=x264enc"
                argv = ["--encoder=x264enc"]
                try:
                    outputs.append(ju.videoOutput(rtsp_uri, argv=argv))
                except Exception as e:
                    logging.error("Failed to create RTSP output with x264 (%s): %s", rtsp_uri, e)
        else:
            try:
                outputs.append(ju.videoOutput(rtsp_uri, argv=argv))
            except Exception as e:
                logging.error("Failed to create RTSP output (%s): %s", rtsp_uri, e)
    return outputs
