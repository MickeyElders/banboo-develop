import logging
import os
import subprocess
from pathlib import Path
from typing import Optional


def _check_gst():
    env = os.environ.copy()
    env.setdefault("GST_PLUGIN_PATH", "/usr/lib/aarch64-linux-gnu/gstreamer-1.0:/usr/lib/aarch64-linux-gnu/tegra")
    env.setdefault("GST_PLUGIN_SCANNER", "/usr/lib/aarch64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-plugin-scanner")
    env.setdefault("LD_LIBRARY_PATH", "/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:" + env.get("LD_LIBRARY_PATH", ""))
    def has(elem: str) -> bool:
        return subprocess.run(["gst-inspect-1.0", elem], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

    have_nvenc = has("nvv4l2h264enc")
    have_x264 = has("x264enc")
    if have_nvenc:
        logging.info("Preflight: NVENC encoder available (nvv4l2h264enc)")
    else:
        logging.warning("Preflight: NVENC encoder missing (nvv4l2h264enc)")
    if have_x264:
        logging.info("Preflight: software encoder available (x264enc)")
    else:
        logging.warning("Preflight: software encoder missing (x264enc)")
    return have_nvenc, have_x264


def _check_model(cfg: dict, base_dir: Path):
    mcfg = cfg.get("model", {})
    model_path = Path(mcfg.get("onnx", "models/best.onnx"))
    if not model_path.is_absolute():
        model_path = base_dir / model_path
    if model_path.exists():
        logging.info("Preflight: model ONNX found at %s", model_path)
    else:
        logging.error("Preflight: model ONNX missing at %s", model_path)
    engine_path = mcfg.get("engine")
    if engine_path:
        ep = Path(engine_path)
        if not ep.is_absolute():
            ep = base_dir / ep
        if ep.exists():
            logging.info("Preflight: engine found at %s", ep)
        else:
            logging.warning("Preflight: engine not found at %s (will rebuild from ONNX)", ep)


def preflight_checks(cfg: dict, ji_py_build: Optional[Path], ji_lib: Optional[Path]):
    base_dir = Path(__file__).resolve().parent.parent
    logging.info("Preflight: python path uses jetson-inference build at %s", ji_py_build)
    if ji_lib and ji_lib.exists():
        logging.info("Preflight: LD_LIBRARY_PATH includes %s", ji_lib)
    _check_model(cfg, base_dir)
    _check_gst()
