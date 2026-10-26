"""Person detection + ByteTrack IDs + ankle keypoints (Ultralytics pose model).

Why pose and not a plain detector: from behind the baseline, near players often hide the
far players' legs. The box bottom then sits on the near player's shoulder and the
"foot" projects metres off. Ankle confidence tells us whether the feet are actually visible.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

MODEL = Path(__file__).resolve().parents[1] / "models" / "yolo11n-pose.pt"
L_WRIST, R_WRIST, L_ANKLE, R_ANKLE = 9, 10, 15, 16  # COCO keypoints
ANKLE_CONF, WRIST_CONF = 0.5, 0.3


class PlayerDetector:
    def __init__(self, model=None, imgsz=960, conf=0.25, device=None):
        from ultralytics import YOLO
        # a CoreML export (Neural Engine) runs ~20% faster than PyTorch-on-MPS with the same
        # detections; used when present (`hud export-coreml`, see hud/coreml.py).
        # nano: ~10x lighter than the medium model the first version used; the far-half crop
        # pass below wins back most of the small far players it would otherwise miss
        import sys

        from hud.coreml import YOLO_ML
        use_ml = model is None and sys.platform == "darwin" and YOLO_ML.exists()
        self.model = YOLO(str(YOLO_ML) if use_ml else (model or str(MODEL)), task="pose")
        self.imgsz, self.conf = imgsz, conf
        self.device = None if use_ml else (device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.far_box = None  # set to far_region(...) to enable the far-half crop pass
        self.main_box = None  # set to court_region(...): the main pass sees only the court and its players

    def __call__(self, frame):
        """-> boxes (N,4) xyxy px, track_ids (N,) (-1 if none yet), feet (N,2) px, feet_ok (N,) bool,
        wrists (N,2,2) px [left, right], NaN where not visible (swing detection uses them)."""
        mx0, my0, mx1, my1 = self.main_box or (0, 0, frame.shape[1], frame.shape[0])
        r = self.model.track(frame[my0:my1, mx0:mx1], persist=True, tracker="bytetrack.yaml", classes=[0],
                             imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        out = _parse(r, (mx0, my0))
        if self.far_box is None:
            return out
        # second pass on the far half, cropped: far players are small, and on 480p / blurry video the
        # full-frame pass misses ~45% of them (35% -> 11% missed with the crop, README "低畫質").
        x0, y0, x1, y1 = self.far_box
        r = self.model.predict(frame[y0:y1, x0:x1], classes=[0], imgsz=self.imgsz, conf=self.conf,
                               device=self.device, verbose=False)[0]
        extra = _parse(r, (x0, y0))
        new = [i for i, b in enumerate(extra[0]) if not len(out[0]) or _iou(b, out[0]).max() < 0.3]
        if not new:
            return out
        extra = (extra[0][new], np.full(len(new), -1), *(e[new] for e in extra[2:]))
        return tuple(np.concatenate([a, b]) for a, b in zip(out, extra))


def player_region(det, frames, H, size, inside, pad=0.15, court_px=None):
    """Pixel box (x0, y0, x1, y1) where the players are, from a few sampled frames: the union of
    the person boxes whose feet are on court (inside(court xy) -> bool), padded for lunges and
    jumps. On a portrait or stabilised frame the court is a fraction of the picture (ACE canvas:
    the players live in ~30 % of it), and YOLO shrinks the whole frame to imgsz; cropped, the
    players are ~1.5-2x bigger and the nano model stops missing clearly visible ones (it missed
    a sharp, unoccluded near player in 4 of 20 sampled frames). None when the crop wouldn't
    save at least 15 % of the frame or nobody was found."""
    import cv2
    w, h = size
    got = []
    for f in frames:
        r = det.model.predict(f, classes=[0], imgsz=det.imgsz, conf=det.conf, device=det.device, verbose=False)[0]
        boxes, _, feet, ok, _ = _parse(r)
        sx, sy = w / f.shape[1], h / f.shape[0]
        if not len(boxes):
            continue
        xy = cv2.perspectiveTransform((feet * [sx, sy]).reshape(-1, 1, 2), np.asarray(H)).reshape(-1, 2)
        got += [b * [sx, sy, sx, sy] for b, q, o in zip(boxes, xy, ok) if o and inside(q)]
    if len(got) < 3:
        return None
    b = np.array(got)
    tall = (b[:, 3] - b[:, 1]).max()
    if court_px is not None:  # the painted court too: a player who never went near a line in the samples still will
        c = np.clip(np.asarray(court_px, float), 0, [w, h])
        b = np.r_[b, np.c_[c, c]]
    x0, y0 = b[:, :2].min(0)
    x1, y1 = b[:, 2:].max(0)
    x0, x1 = x0 - pad * (x1 - x0) - 0.3 * tall, x1 + pad * (x1 - x0) + 0.3 * tall
    y0, y1 = y0 - 0.5 * tall, y1 + 0.25 * tall          # jumps, raised rackets / lunges forward
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if (x1 - x0) * (y1 - y0) > 0.85 * w * h:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def far_region(H, size, margin_m=1.5):
    """Pixel box (x0, y0, x1, y1) around the far half of the court plus head room, from the
    pixel->court homography H."""
    from hud.court import COURT_L, COURT_W, NET_Y
    far = np.array([[-margin_m, NET_Y], [COURT_W + margin_m, NET_Y], [COURT_W + margin_m, COURT_L + margin_m],
                    [-margin_m, COURT_L + margin_m]], np.float64)
    px = cv2.perspectiveTransform(far.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    w, h = size
    x0, y0 = np.maximum(px.min(0) - [0, h * 0.25], 0).astype(int)
    x1, y1 = np.minimum(px.max(0), [w, h]).astype(int)
    return (x0, y0, x1, y1) if x1 - x0 > 32 and y1 - y0 > 32 else None


def _iou(b, boxes):
    ix = np.clip(np.minimum(b[2], boxes[:, 2]) - np.maximum(b[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(b[3], boxes[:, 3]) - np.maximum(b[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(b) + area(boxes) - inter)


def _parse(r, off=(0, 0)):
    n = len(r.boxes)
    if n == 0:
        return np.zeros((0, 4)), np.zeros(0, int), np.zeros((0, 2)), np.zeros(0, bool), np.zeros((0, 2, 2))
    shift = np.array(off, np.float64)
    boxes = r.boxes.xyxy.cpu().numpy() + np.tile(shift, 2)
    ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else np.full(n, -1)
    kpts = r.keypoints.data.cpu().numpy()
    kpts[..., :2] += shift
    ank = kpts[:, [L_ANKLE, R_ANKLE]]  # (N,2,[x,y,conf])
    vis = ank[:, :, 2] >= ANKLE_CONF
    feet_ok = vis.any(axis=1)
    # x: visible-ankle mean (box centre is dragged sideways by the racket arm in lunges)
    # y: box bottom = shoe soles, unless an ankle sits lower
    ax = np.where(vis, ank[:, :, 0], np.nan)
    ay = np.where(vis, ank[:, :, 1], -np.inf)
    with np.errstate(invalid="ignore"):
        fx = np.where(feet_ok, np.nanmean(ax, axis=1), (boxes[:, 0] + boxes[:, 2]) / 2)
    fy = np.maximum(boxes[:, 3], ay.max(axis=1))
    kp = kpts[:, [L_WRIST, R_WRIST]]
    wrists = np.where(kp[:, :, 2:] >= WRIST_CONF, kp[:, :, :2], np.nan)
    return boxes, ids, np.c_[fx, fy], feet_ok, wrists
