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
    if out_cfg.get("hdmi", True):
        outputs.append(ju.videoOutput("display://0"))
    if out_cfg.get("rtsp", True):
        rtsp_uri = out_cfg.get("rtsp_uri", "rtsp://@:8554/live")
        outputs.append(ju.videoOutput(rtsp_uri))
    return outputs
