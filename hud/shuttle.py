"""Shuttle pixel track with TrackNetV3 (github.com/qaz812345/TrackNetV3, MIT; pretrained on
rear-view broadcast badminton), then cleaned: single-frame spikes removed, short gaps
linearly interpolated.

We import its model definitions and checkpoints but not its predict.py: that script resizes
every full-res frame 8 times and buffers heatmaps, which grew to 8 GB and swapped an 8 GB Mac
on a 16 s clip. Here each frame is resized once and streamed in non-overlapping 8-frame
windows, so memory stays flat for any video length.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from hud.video import frames, open_video

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "models" / "tracknetv3"
W, H = 512, 288           # TrackNet input size
COOR_TH = 50 / np.hypot(W, H)  # InpaintNet output below this (normalised) = no shuttle, as upstream
MEDIAN_SAMPLES = 60
MAX_GAP_S = 0.1   # interpolate invisible runs up to this long, if detected on both sides
MAX_INPAINT_S = 0.4  # InpaintNet only bridges gaps this short (shuttle behind a body). Unbounded,
                     # it drew whole fake rallies through 5 s idle gaps on the oblique test clip
SPIKE_PX = 50     # at 1080p: min jump (px/frame) that can start a glitch run
MAX_GLITCH_S = 0.15  # glitch runs up to this long are removed


def _joined(path):
    """GitHub caps files at 100 MB, so the 136 MB TrackNet checkpoint is stored as
    <name>.part0, .part1...; stitch them back together on first use."""
    parts = sorted(path.parent.glob(path.name + ".part*"))
    if not path.exists() and parts:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as out:
            for p in parts:
                out.write(p.read_bytes())
        tmp.replace(path)
    return path


def _load_models(device):
    from hud.tracknet.model import InpaintNet, TrackNet
    in_dim = {"subtract": 1, "subtract_concat": 4}
    nets = []
    for name in ("TrackNet", "InpaintNet"):
        path = _joined(CKPT_DIR / f"{name}_best.pt")
        if not path.exists():
            raise SystemExit(f"Missing {path}. Download the TrackNetV3 checkpoints, see README.")
        ck = torch.load(path, map_location="cpu", weights_only=False)  # upstream pickles a param dict
        p = ck["param_dict"]
        L = p["seq_len"]
        net = (TrackNet(in_dim=(L + 1) * 3 if p["bg_mode"] == "concat" else L * in_dim.get(p["bg_mode"], 3), out_dim=L)
               if name == "TrackNet" else InpaintNet())
        net.load_state_dict(ck["model"])
        nets.append((net.to(device).eval(), p["seq_len"]))
    return nets


def _frames(video, every=None):
    return frames(video, (W, H), every or 1, rgb=True)


def _median(video, n):
    frames = list(_frames(video, every=max(1, n // MEDIAN_SAMPLES)))
    if not frames:
        raise SystemExit(f"No frames could be read from {video}")
    return np.median(frames, axis=0).astype(np.uint8)


def _locate(heat):
    """Centre of the largest blob in a binary heatmap (upstream's rule), or None."""
    cnts, _ = cv2.findContours(heat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = max((cv2.boundingRect(c) for c in cnts), key=lambda r: r[2] * r[3])
    return x + w / 2, y + h / 2


def _inpaint_mask(y, vis, th, max_len=None):
    """Port of upstream generate_inpaint_mask: fill gaps that start and end inside the frame
    (y > th); gaps touching the top edge mean the shuttle flew out of view, leave those."""
    mask, i, n = np.zeros(len(y)), 0, len(y)
    while i < n:
        if vis[i]:
            i += 1
            continue
        j = i
        while j < n and not vis[j]:
            j += 1
        before_ok = i == 0 or y[i - 1] > th
        after_ok = j < n and y[j] > th
        if before_ok and after_ok and (max_len is None or j - i <= max_len):
            mask[i:j] = 1
        i = j
    return mask


@torch.no_grad()
def run_tracknet(video, batch=1):
    """Every frame (feeding 60 fps footage at 30 fps lost 2 of 3 rallies on the oblique clip)."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    (tracknet, L), (inpaintnet, L2) = _load_models(device)
    dtype = torch.float16 if device == "mps" else torch.float32  # half precision: ~6% faster, same detections
    tracknet = tracknet.to(dtype)
    cap = open_video(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw, fh = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    stride = 1
    median = np.moveaxis(_median(video, n), -1, 0)
    xy = []  # normalised (x, y) per frame, (0, 0) = not seen
    windows, done = [], False
    src = _frames(video)
    while not done:
        frames = []
        while len(frames) < L:
            f = next(src, None)
            if f is None:
                done = True
                break
            frames.append(np.moveaxis(f, -1, 0))
        if frames:
            real = len(frames)
            frames += [frames[-1]] * (L - real)  # pad the last window
            windows.append((np.concatenate([median] + frames), real))
        if windows and (len(windows) == batch or done):
            x = torch.from_numpy(np.stack([w for w, _ in windows])).to(device).to(dtype).div(255)
            heat = (tracknet(x) > 0.5).cpu().numpy().astype(np.uint8)
            for k, (_, real) in enumerate(windows):
                for t in range(real):
                    c = _locate(heat[k, t])
                    xy.append((c[0] / W, c[1] / H) if c else (0.0, 0.0))
            windows = []
    xy = np.array(xy, np.float32).reshape(-1, 2)
    vis = (xy != 0).any(axis=1)
    # InpaintNet: rectify occluded stretches in non-overlapping windows of L2 frames
    mask = _inpaint_mask(xy[:, 1], vis, 0.05, max(1, round(MAX_INPAINT_S * fps / stride)))
    m = len(xy)
    pad = (-m) % L2
    xp = np.concatenate([xy, np.repeat(xy[-1:], pad, 0)]).reshape(-1, L2, 2)
    mp = np.concatenate([mask, np.repeat(mask[-1:], pad)]).reshape(-1, L2, 1).astype(np.float32)
    out = inpaintnet(torch.from_numpy(xp).to(device), torch.from_numpy(mp).to(device)).cpu().numpy()
    fixed = (out * mp + xp * (1 - mp)).reshape(-1, 2)[:m]
    fixed[(fixed < COOR_TH).all(axis=1)] = 0
    seen = (fixed != 0).any(axis=1)
    return pd.DataFrame({"Frame": np.arange(m) * stride, "Visibility": seen.astype(int),
                         "X": fixed[:, 0] * fw, "Y": fixed[:, 1] * fh,
                         "inpainted": seen & ~vis})


def drop_glitches(p, spike, max_len):
    """Remove short runs where TrackNet locked onto something else: a run of <= max_len
    frames that jumps away from the track and back, while the points on either side join
    up without it (closer to each other than to the run)."""
    p = p.copy()
    idx = np.flatnonzero(np.isfinite(p[:, 0]))
    if len(idx) < 3:
        return p
    step = np.hypot(*np.diff(p[idx], axis=0).T) / np.diff(idx)  # px per frame between sightings
    local = pd.Series(step).rolling(15, center=True, min_periods=1).median().to_numpy()
    brk = np.flatnonzero(step > np.maximum(spike, 4 * local))  # break between idx[k] and idx[k+1]
    starts, ends = np.r_[0, brk + 1], np.r_[brk, len(idx) - 1]
    k = 1  # piece 0 has nothing before it
    while k < len(starts) - 1:
        a = starts[k]
        # a glitch can itself be split into several short pieces: try spans of 1..4 pieces
        for m in range(k, min(k + 4, len(starts) - 1)):
            b = ends[m]
            if idx[b] - idx[a] + 1 > max_len:
                break
            before, after = p[idx[a - 1]], p[idx[b + 1]]
            jump_in, jump_out = np.hypot(*(p[idx[a]] - before)), np.hypot(*(after - p[idx[b]]))
            if np.hypot(*(after - before)) < min(jump_in, jump_out):
                p[idx[a:b + 1]] = np.nan
                k = m
                break
        k += 1
    return p


def clean(raw, times, height=1080, fps=50.0):
    """raw: (Frame, Visibility, X, Y[, inpainted]) -> frame, timestamp, x_px, y_px, visible, interpolated, inpainted."""
    n = len(times)
    p = np.full((n, 2), np.nan)
    r = raw[(raw.Visibility == 1) & (raw.Frame < n)]
    p[r.Frame.values] = r[["X", "Y"]].values
    p = drop_glitches(p, SPIKE_PX * height / 1080, max(1, round(MAX_GLITCH_S * fps)))
    detected = ~np.isnan(p[:, 0])
    s = pd.DataFrame(p, columns=["x_px", "y_px"])
    # limit_area="inside" fills only gaps bounded on both sides; then undo the over-long ones
    filled = s.interpolate(limit_area="inside")
    gap_len = pd.Series(~detected).groupby(detected.cumsum()).transform("sum").values
    filled[~detected & (gap_len > round(MAX_GAP_S * fps))] = np.nan
    out = pd.DataFrame({"frame": np.arange(n), "timestamp": np.round(times, 4),
                        "x_px": filled.x_px.round(1), "y_px": filled.y_px.round(1)})
    out["visible"] = out.x_px.notna()
    out["interpolated"] = out.visible & ~detected
    # InpaintNet-rectified points (shuttle hidden behind a player): model guesses, not sightings
    inp = np.zeros(n, bool)
    if "inpainted" in raw:
        ri = raw[raw.inpainted & (raw.Frame < n)].Frame.values
        inp[ri] = True
    out["inpainted"] = out.visible & ~out.interpolated & inp
    return out


def run(video, out):
    from hud.video import frame_times
    times = frame_times(video)
    if not times:
        raise SystemExit(f"No frames could be read from {video}")
    cap = open_video(video)
    height, fps = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    df = clean(run_tracknet(video), times, height, fps)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)  # overwrite: idempotent
    det = int((df.visible & ~df.interpolated & ~df.inpainted).sum())
    print(f"Wrote {out}: {len(df)} frames, shuttle in {df.visible.mean():.1%} "
          f"({det} detected, {int(df.inpainted.sum())} inpainted, {int(df.interpolated.sum())} interpolated)")
    return df
