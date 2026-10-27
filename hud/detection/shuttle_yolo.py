"""Fill TrackNet's gaps with a frame-level shuttle detector (models/shuttle_yolo.pt, trained by
scripts/train_shuttle.py).

TrackNet finds the shuttle from its motion over 3 frames plus a background, which handles a
tiny blurred shuttle but loses some that are plainly visible (against a banner, by the net
tape: on the user's ACE clip a hosted detector trained on still images found one TrackNet
missed). So only TrackNet's short gaps are searched, and only near the shuttle's path:

  * a gap = frames between two TrackNet sightings at most MAX_GAP_S apart
  * per gap frame, one TILE x TILE crop at full resolution around where the shuttle's path puts
    it (a quadratic through the sightings either side; the detector sees native pixels)
  * a detection counts if it is confident (CONF_HI) and within PATH_PX of that path, or less
    confident (CONF_LO) and within half of it; the path tolerance grows with the distance to
    the nearest sighting. Outside the air above the court: ignored. A tile with more than
    MAX_CANDS candidates is ignored (a 1-epoch model fired ~100 times per tile and still
    "filled" 9 frames of a 4 s clip under the loose first tolerances).
Everything it adds is marked as a sighting; shuttle.clean then drops whatever still jumps off
the track. Off without the weights file, or with [shuttle] yolo_fill = false.
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "shuttle_yolo.pt"
TILE = 640
MAX_GAP_S = 1.0
CONF_HI, CONF_LO = 0.35, 0.15  # hosted model on the ACE frames: the missed shuttle 0.45, false ones <= 0.25
PATH_PX = 25          # at 1080p: tolerance at the gap ends ...
PATH_GROW_PX = 8      # ... plus this per frame towards the middle of the gap
PATH_PTS = 4          # sightings each side of a gap the path is fitted to (quadratic in time)
MAX_CANDS = 4         # a tile with more candidates than this says nothing (the detector fires everywhere)
BATCH = 16
LAND_SEARCH_S = 1.5   # look for the shuttle lying on the floor this long after a track ends ...
LAND_EVERY = 3        # ... in every 3rd frame
STILL_PX = 12         # at 1080p: two detections this close are the same lying shuttle (hits.STILL_PX)
PRE_S = (2.0, 1.5)    # ... and none of them may already lie there this long before the track ended


class Detector:
    """tiles (list of BGR arrays) -> list of (N, 3) [x, y, conf] arrays, tile pixels."""

    def __init__(self, weights=MODEL, device=None, conf=CONF_LO):
        import torch
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.conf = conf
        names = getattr(self.model, "names", {}) or {}
        # our weights: one class. A COCO model (used to test the plumbing): "sports ball"
        self.classes = None if len(names) <= 1 else [k for k, v in names.items() if v == "sports ball" or "shuttle" in str(v)]

    def __call__(self, tiles):
        out = []
        for k in range(0, len(tiles), BATCH):
            for r in self.model.predict(tiles[k:k + BATCH], imgsz=TILE, conf=self.conf, classes=self.classes,
                                        device=self.device, verbose=False):
                b = r.boxes
                out.append(np.c_[b.xywh.cpu().numpy()[:, :2], b.conf.cpu().numpy()] if len(b) else np.zeros((0, 3)))
        return out


def available():
    return MODEL.exists()


def gaps(seen, max_len):
    """(a, b) pairs: seen[a] and seen[b], nothing seen strictly between, 1 <= b - a - 1 <= max_len."""
    idx = np.flatnonzero(seen)
    return [(int(a), int(b)) for a, b in zip(idx, idx[1:]) if 1 < b - a <= max_len + 1]


def gap_path(idx, P, a, b, k=PATH_PTS, max_step=3):
    """Where the shuttle is expected inside the gap (a, b): a quadratic in time through up to k
    sightings on each side (a flight curves), from the same flight only (no jump of more than
    max_step frames between them); a straight line when there are too few."""
    i = np.searchsorted(idx, a)
    left, right = [a], [b]
    for j in range(i - 1, max(-1, i - k), -1):
        if left[-1] - idx[j] > max_step:
            break
        left.append(int(idx[j]))
    j0 = np.searchsorted(idx, b)
    for j in range(j0 + 1, min(len(idx), j0 + k)):
        if idx[j] - right[-1] > max_step:
            break
        right.append(int(idx[j]))
    fs = np.array(sorted(left + right))
    if len(fs) < 5:
        return lambda f: P[a] + (P[b] - P[a]) * (f - a) / (b - a)
    cx, cy = np.polyfit(fs, P[fs, 0], 2), np.polyfit(fs, P[fs, 1], 2)
    return lambda f: np.array([np.polyval(cx, f), np.polyval(cy, f)])


def tolerance(f, a, b, scale):
    return scale * (PATH_PX + PATH_GROW_PX * min(f - a, b - f))


def choose(cands, expect, tol):
    """Best [x, y, conf] among cands for a frame whose path point is expect, or None."""
    if not len(cands):
        return None
    d = np.hypot(*(cands[:, :2] - expect).T)
    ok = ((cands[:, 2] >= CONF_HI) & (d <= tol)) | ((cands[:, 2] >= CONF_LO) & (d <= tol / 2))
    if not ok.any():
        return None
    c = cands[ok]
    return c[np.argmax(c[:, 2] - d[ok] / (4 * tol))]  # confidence first, closeness breaks ties


def track_ends(seen, gap):
    """Frames where a track ends: seen, then nothing for more than gap frames."""
    idx = np.flatnonzero(seen)
    nxt = np.r_[idx[1:], len(seen) + gap + 1]
    return [int(e) for e, q in zip(idx, nxt) if q - e > gap + 1]


def landing_spot(dets, scale):
    """dets: [(frame, x, y, conf)] confident detections after a track ended. -> (first frame,
    last frame, (x, y)) of a spot seen in at least 2 sampled frames (a shuttle lying still),
    or None."""
    best = None
    for i, (f, x, y, c) in enumerate(dets):
        near = [d for d in dets if np.hypot(d[1] - x, d[2] - y) <= STILL_PX * scale]
        if len({d[0] for d in near}) >= 2 and (best is None or len(near) > len(best)):
            best = near
    if best is None:
        return None
    xy = np.median([[d[1], d[2]] for d in best], 0)
    return min(d[0] for d in best), max(d[0] for d in best), xy


def fill(video, raw, fps, img_h, detector=None, inside=None, read=None):
    """raw: run_tracknet output (Frame, Visibility, X, Y, inpainted), frame pixels.
    -> (raw with frames filled, number filled). inside(px) -> bool: the court's air box.
    read(frames) -> iterable of (frame, BGR image) in order (tests); default decodes the video.
    Two searches, one decoding pass:
      gaps     between two TrackNet sightings, along the path (module docstring)
      landings after a track ends, every LAND_EVERY-th frame for LAND_SEARCH_S, around and below
               the last sighting: TrackNet finds the shuttle by its motion, so it loses it once it
               lies on the floor, where the point is decided. A spot found in >= 2 samples is
               written into every frame between them, standing still, which is what hits.landed_at
               looks for. Unless a shuttle already lay on that spot before the shot (spares by
               the net post), checked in the same tile PRE_S before the track ended.
    Streams: one batch of tiles is held at a time (a whole gap list of stabilised 2K frames
    would not fit an 8 GB Mac)."""
    real = ((raw.Visibility == 1) & ~raw.inpainted).to_numpy()
    P = raw[["X", "Y"]].to_numpy(float)
    n_frames = len(raw)
    gap = round(MAX_GAP_S * fps)
    scale = img_h / 1080
    todo = {}
    idx = np.flatnonzero(real)
    for a, b in gaps(real, gap):
        path = gap_path(idx, P, a, b)
        for f in range(a + 1, b):
            todo[f] = ("gap", path(f), a, b)
    ends = track_ends(real, gap)
    for e in ends:
        for f in range(e + LAND_EVERY, min(n_frames, e + 1 + round(LAND_SEARCH_S * fps)), LAND_EVERY):
            if f not in todo:
                todo[f] = ("land", P[e] + [0, TILE / 4], e, None)
        # the same tile before the shot: a spare shuttle already lying there is not this landing
        for s_ in PRE_S:
            f = e - round(s_ * fps)
            if f >= 0 and f not in todo:
                todo[f] = ("pre", P[e] + [0, TILE / 4], e, None)
    if not todo:
        return raw, 0
    detector = detector or Detector()
    out = raw.astype({"X": float, "Y": float})  # TrackNet's are float32: pandas refuses float64 writes
    n = 0
    batch, after_end, before_end = [], {}, {}

    def flush():
        nonlocal n
        if not batch:
            return
        for (f, x0, y0, _), cands in zip(batch, detector([t[3] for t in batch])):
            kind, expect, a, b = todo[f]
            cands = cands + [x0, y0, 0] if len(cands) else cands
            if inside is not None and len(cands):
                cands = cands[[inside(p[:2]) for p in cands]]
            if (cands[:, 2] >= CONF_LO).sum() > MAX_CANDS if len(cands) else False:
                continue
            if kind == "gap":
                best = choose(cands, expect, tolerance(f, a, b, scale))
                if best is not None:
                    out.loc[f, ["X", "Y"]] = best[:2]
                    out.loc[f, "Visibility"], out.loc[f, "inpainted"] = 1, False
                    n += 1
            elif kind == "land":  # landing search: confident, and not above where the track ended
                for x, y, c in cands:
                    if c >= CONF_HI and y >= P[a][1] - 20 * scale:
                        after_end.setdefault(a, []).append((f, x, y, c))
            else:
                before_end.setdefault(a, []).extend((x, y) for x, y, c in cands if c >= CONF_HI)
        batch.clear()
    for f, img in (read or (lambda fs: _read(video, fs)))(sorted(todo)):
        h, w = img.shape[:2]
        c = todo[f][1]
        x0 = int(np.clip(c[0] - TILE / 2, 0, max(w - TILE, 0)))
        y0 = int(np.clip(c[1] - TILE / 2, 0, max(h - TILE, 0)))
        batch.append((f, x0, y0, np.ascontiguousarray(img[y0:y0 + TILE, x0:x0 + TILE])))
        if len(batch) == BATCH:
            flush()
    flush()
    for e, dets in after_end.items():
        spot = landing_spot(dets, scale)
        if spot is None:
            continue
        f0, f1, xy = spot
        if any(np.hypot(x - xy[0], y - xy[1]) <= 2 * STILL_PX * scale for x, y in before_end.get(e, ())):
            continue  # lying there already before the shot
        frames = [f for f in range(f0, f1 + 1) if not real[f]]
        out.loc[frames, "X"], out.loc[frames, "Y"] = xy[0], xy[1]
        out.loc[frames, "Visibility"], out.loc[frames, "inpainted"] = 1, False
        n += len(frames)
        # the fall in between, straight from the last sighting: a guess (inpainted), but it joins
        # the track to the spot, which is how hits tells a landing from a lock onto a shoe
        fall = list(range(e + 1, f0))
        if fall:
            w = (np.array(fall) - e) / (f0 - e)
            out.loc[fall, "X"] = P[e][0] + (xy[0] - P[e][0]) * w
            out.loc[fall, "Y"] = P[e][1] + (xy[1] - P[e][1]) * w
            out.loc[fall, "Visibility"], out.loc[fall, "inpainted"] = 1, True
    return out, n


def _read(video, wanted):
    """(frame, BGR) for the wanted frame indices, decoding the video once, in order."""
    want = set(wanted)
    cap = cv2.VideoCapture(str(video))
    i, last = 0, max(want)
    try:
        while i <= last and cap.grab():
            if i in want:
                ok, img = cap.retrieve()
                if ok:
                    yield i, img
            i += 1
    finally:
        cap.release()
