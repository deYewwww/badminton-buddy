"""CoreML copies of the two networks, so they run on the Mac's Neural Engine / GPU through
CoreML instead of PyTorch-on-MPS. TrackNet is ~85 % of the analysis compute.

    uv run --with numpy==2.3.5 hud export-coreml      # once, on the Mac

writes models/tracknetv3/TrackNet.mlpackage and models/yolo11n-pose.mlpackage. They are
used when present. The TrackNet copy is checked against PyTorch on the first window of every
video (shuttle.run_tracknet) and dropped if the heatmaps disagree, so a bad export can never
silently change results. Converting works anywhere; running needs macOS.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRACKNET_ML = ROOT / "models" / "tracknetv3" / "TrackNet.mlpackage"
YOLO_ML = ROOT / "models" / "yolo11n-pose.mlpackage"
H, W = 288, 512


def export(imgsz=960):
    import coremltools as ct
    import torch

    from hud.detection import shuttle
    (net, L), _ = shuttle._load_models("cpu")
    x = torch.rand(1, (L + 1) * 3, H, W)
    traced = torch.jit.trace(net.eval(), x)
    m = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape)], outputs=[ct.TensorType(name="heat")],
                   compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS13,
                   convert_to="mlprogram")
    m.save(str(TRACKNET_ML))
    print(f"Wrote {TRACKNET_ML}")
    import numpy
    if tuple(int(x) for x in numpy.__version__.split(".")[:2]) > (2, 3):
        # Ultralytics' CoreML export needs numpy <= 2.3.5: with 2.4 its attention block fails to
        # convert ("only 0-dimensional arrays can be converted to Python scalars")
        print(f"YOLO 的 CoreML 匯出需要 numpy <= 2.3.5（目前 {numpy.__version__}），這次先跳過，"
              f"球員偵測繼續用 PyTorch。要匯出 YOLO 請改用：\n  uv run --with numpy==2.3.5 hud export-coreml")
        return
    from ultralytics import YOLO

    from hud.detection.detect import MODEL
    try:  # YOLO is ~15 % of the compute; a failed export just leaves it on PyTorch
        out = Path(YOLO(str(MODEL), task="pose").export(format="coreml", imgsz=imgsz, half=True))
        if out != YOLO_ML:
            out.replace(YOLO_ML)
        print(f"Wrote {YOLO_ML}")
    except Exception as e:  # anything else: PyTorch it is
        print(f"YOLO CoreML export failed ({e}); the pose model stays on PyTorch")


def tracknet():
    """-> callable (batch, C, 288, 512) float32 in [0, 1] -> heat (batch, L, 288, 512), or None
    when not on macOS, not exported, or coremltools is missing."""
    if sys.platform != "darwin" or not TRACKNET_ML.exists():
        return None
    try:
        import coremltools as ct
        m = ct.models.MLModel(str(TRACKNET_ML), compute_units=ct.ComputeUnit.ALL)
    except Exception as e:  # missing package, unsupported macOS: PyTorch it is
        print(f"CoreML TrackNet unavailable ({e}); using PyTorch")
        return None
    return lambda x: np.concatenate([m.predict({"x": xi[None].astype(np.float32)})["heat"] for xi in x])
