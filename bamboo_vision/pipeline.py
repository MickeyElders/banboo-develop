import logging
import os
import subprocess
import sys
from pathlib import Path

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
    if not output_cvg and output_bbox:
        extra_args += [f"--output-cvg={output_bbox}"]
    if input_shape:
        extra_args += [f"--input-shape={input_shape}"]
    if workspace:
        extra_args += [f"--workspace={workspace}"]
    # Minimal safety check: ensure output shape is [1, 18900, 6]; if it's [1, 6, 18900] warn and exit to avoid SEGV.
    try:
        import onnx  # type: ignore

        model_onnx = onnx.load(str(model_path))
        if model_onnx.graph.output:
            dims = model_onnx.graph.output[0].type.tensor_type.shape.dim
            if len(dims) == 3:
                d0 = dims[0].dim_value
                d1 = dims[1].dim_value
                d2 = dims[2].dim_value
                if d0 == 1 and d1 == 6 and d2 == 18900:
                    msg = (
                        "Model output shape is [1,6,18900]; expected [1,18900,6]. "
                        "Please transpose the output (perm=[0,2,1]) and rebuild engine to avoid TensorRT SEGV."
                    )
                    logging.error(msg)
                    raise SystemExit(msg)
    except ImportError:
        logging.warning("onnx package not available; skipping output shape sanity check.")
    except Exception as e:
        logging.warning("Output shape sanity check failed/skipped: %s", e)
    logging.info("Loading model: %s", model_path)
    net = ji.detectNet(argv=extra_args, threshold=threshold)
    net.SetNMS(nms)
    return net


def build_outputs(out_cfg: dict, cam_cfg: dict):
    """Outputs禁用：只做推理+合成，不创建任何 jetson.utils videoOutput 管线。"""
    logging.info("Output disabled: inference-only (no RTSP/HDMI/WebRTC).")
    return []
