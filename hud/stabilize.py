"""Make a panning (gimbal-tracked) or cut-together clip look like one fixed camera.

The whole analysis assumes one fixed camera. A phone on a tracking gimbal only turns about
its own centre, and for a pure rotation every scene point moves by the same homography, so
each frame can be warped exactly into the view of one reference frame: the court stops
moving and the existing pipeline runs unchanged on the result.

Per frame: ORB features matched to the reference frame, RANSAC homography. Frames that don't
match directly (fast pan, motion blur) chain through the previous frame. All frames are
warped onto one canvas that holds the reference view plus everything the pan reached; the
calibration of the reference frame carries over with a shift.

The background (walls, ceiling: most of the features) fixes the rotation. Where the phone was
also moved between takes (edited clips), the near floor has parallax against the walls and
lands a few px off; so per segment between cuts the court model is snapped onto that
segment's own painted lines and the floor-plane correction folded into the warp (players
stand on the floor, which is what has to be right).

Limits: overlays burnt into the video (titles, watermarks) don't move with the scene, and a
camera walked to the other side of the court won't match the reference at all.
"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from hud.video import probe

SCALE = 0.5           # features are found at half size
MIN_INLIERS = 40
MAX_CANVAS = 1.6      # canvas at most this many times the frame, each way


class Matcher:
    def __init__(self, ref):
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref = self._feat(ref)

    def _feat(self, img):
        g = cv2.cvtColor(cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        return self.orb.detectAndCompute(g, None)

    def homography(self, a, b):
        """features a -> features b: (3x3 at full size, inliers) or (None, n)."""
        (ka, da), (kb, db) = a, b
        if da is None or db is None or len(ka) < MIN_INLIERS or len(kb) < MIN_INLIERS:
            return None, 0
        good = [m for m, *r in self.bf.knnMatch(da, db, k=2) if r and m.distance < 0.75 * r[0].distance]
        if len(good) < MIN_INLIERS:
            return None, len(good)
        pa = np.float32([ka[m.queryIdx].pt for m in good]) / SCALE
        pb = np.float32([kb[m.trainIdx].pt for m in good]) / SCALE
        A, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
        n = int(inl.sum()) if inl is not None else 0
        return (A, n) if A is not None and n >= MIN_INLIERS else (None, n)


def _read(video):
    cap = cv2.VideoCapture(str(video))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


CUT_DIFF, CUT_CORR = 30.0, 0.3  # thumbnail mean |diff| above / phase correlation below: a cut


def _thumb(f):
    return cv2.cvtColor(cv2.resize(f, (180, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)[:200].astype(np.float32)


def homographies(video, ref_index):
    """-> (list of (3x3 frame px -> reference px, inliers, how) per frame, cut frame indices)."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_index)
    ok, ref = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read reference frame {ref_index} of {video}")
    m = Matcher(ref)
    out, prev, prev_H, cuts, prev_t = [], None, None, [], None
    for i, f in enumerate(_read(video)):
        th = _thumb(f)
        if prev_t is not None:
            (_, _), corr = cv2.phaseCorrelate(prev_t, th)
            if np.abs(th - prev_t).mean() > CUT_DIFF and corr < CUT_CORR:
                cuts.append(i)
        prev_t = th
        feat = m._feat(f)
        H, n = m.homography(feat, m.ref)
        how = "direct"
        if H is None and prev is not None and prev_H is not None:  # chain through the last frame
            F, n = m.homography(feat, prev)
            H, how = (prev_H @ F, "chained") if F is not None else (None, "lost")
        elif H is None:
            how = "lost"
        out.append((H, n, how))
        prev, prev_H = feat, H if H is not None else prev_H
    return out, cuts


def _frame_at(cap, i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = cap.read()
    return f if ok else None


def floor_fixes(video, Hs, cuts, T, size, corners, min_len=30, samples=7, chunk=None):
    """Per segment between cuts: 3x3 canvas->canvas correction putting that segment's painted
    lines on the reference court (identity when snapping doesn't improve the fit)."""
    from hud import court
    n = len(Hs)
    bounds = [0, *cuts, n]
    if chunk:  # optional finer pieces; measured on the user's clip they gained no line fit (0.625 -> 0.639) and add jitter
        bounds = sorted({int(x) for a, b in zip(bounds, bounds[1:]) for x in np.linspace(a, b, max(2, int(np.ceil((b - a) / chunk)) + 1))})
    cap = cv2.VideoCapture(str(video))
    fixes = []
    ref = np.float32(corners)
    for a, b in zip(bounds, bounds[1:]):
        K, shift = np.eye(3), 0.0
        if b - a >= min_len:
            imgs = []
            for i in np.linspace(a, b - 1, samples).astype(int):
                f, H = _frame_at(cap, i), Hs[i][0]
                if f is not None and H is not None:
                    imgs.append(cv2.warpPerspective(f, T @ H, size))
            if imgs:
                bg = np.median(imgs, axis=0).astype(np.uint8)
                snapped, _ = court.snap(bg, corners, max_shift=0.08)
                shift = float(np.abs(snapped - corners).max())
                if shift > 0.5:
                    K = cv2.getPerspectiveTransform(np.float32(snapped), ref).astype(np.float64)
        fixes.append((a, b, K, shift))
    cap.release()
    return fixes


def canvas(Hs, size):
    """Offset (ox, oy) and canvas size holding every warped frame, capped at MAX_CANVAS."""
    w, h = size
    box = np.float64([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    pts = np.concatenate([cv2.perspectiveTransform(box, H).reshape(-1, 2) for H, *_ in Hs if H is not None] + [box.reshape(-1, 2)])
    lo = np.maximum(pts.min(0), [-(MAX_CANVAS - 1) / 2 * w, -(MAX_CANVAS - 1) / 2 * h])
    hi = np.minimum(pts.max(0), [w * (1 + (MAX_CANVAS - 1) / 2), h * (1 + (MAX_CANVAS - 1) / 2)])
    ox, oy = -np.floor(lo).astype(int)
    cw, ch = (np.ceil(hi).astype(int) + [ox, oy] + 1) // 2 * 2
    return (int(ox), int(oy)), (int(cw), int(ch))


def run(video, out, ref_time, corners=None):
    """Write the stabilised video to out (.mp4) and out.json (per-frame homographies, offset).
    corners: the reference frame's court corners -> returned shifted onto the canvas."""
    w, h, fps, _ = probe(video)
    ref_index = int(round(ref_time * fps))
    Hs, cuts = homographies(video, ref_index)
    (ox, oy), (cw, ch) = canvas(Hs, (w, h))
    T = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1.0]])
    hows = [x[2] for x in Hs]
    print(f"Stabilised {len(Hs)} frames: {hows.count('direct')} direct, {hows.count('chained')} chained, "
          f"{hows.count('lost')} lost; {len(cuts)} cuts; canvas {cw}x{ch}, reference at +({ox},{oy})")
    K_of = np.repeat(np.eye(3)[None], len(Hs), 0)
    if corners is not None:  # floor-plane correction per segment (camera moved between takes)
        fixes = floor_fixes(video, Hs, cuts, T, (cw, ch), np.float64(corners) + [ox, oy])
        for a, b, K, _ in fixes:
            K_of[a:b] = K
        big = [f"{a / fps:.1f}s {sh:.0f}px" for a, b, K, sh in fixes if sh > 5]
        print(f"Floor corrected in {sum(sh > 0.5 for *_, sh in fixes)}/{len(fixes)} pieces; > 5 px at: {', '.join(big) or 'none'}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cw}x{ch}",
                           "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-crf", "17", "-preset", "veryfast",
                           "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    last = None
    for i, (f, (H, _, _)) in enumerate(zip(_read(video), Hs)):
        H = H if H is not None else last  # lost: hold the last good warp
        last = H if H is not None else last
        img = cv2.warpPerspective(f, K_of[i] @ T @ H, (cw, ch)) if H is not None else np.zeros((ch, cw, 3), np.uint8)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    meta = dict(video=str(video), ref_time=ref_time, offset=[ox, oy], canvas=[cw, ch], cuts=cuts,
                H=[None if H is None else np.round(K_of[i] @ T @ H, 6).tolist() for i, (H, *_) in enumerate(Hs)],
                how=hows)
    out.with_suffix(".json").write_text(json.dumps(meta))
    return None if corners is None else np.float64(corners) + [ox, oy]


def residual(video_stab, n=40, region=None):
    """Median frame-to-frame background shift (px) left in a stabilised video: phase correlation
    of the top part (walls, ceiling), sampled over the clip. Should be ~0-1 px."""
    cap = cv2.VideoCapture(str(video_stab))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    shifts = []
    for i in np.linspace(1, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i) - 1)
        ok1, a = cap.read()
        ok2, b = cap.read()
        if not (ok1 and ok2):
            continue
        y0, y1 = region or (0, a.shape[0] // 2)
        ga, gb = (cv2.cvtColor(x[y0:y1], cv2.COLOR_BGR2GRAY).astype(np.float32) for x in (a, b))
        (dx, dy), resp = cv2.phaseCorrelate(ga, gb)
        if resp > 0.3:
            shifts.append(np.hypot(dx, dy))
    cap.release()
    return float(np.median(shifts)) if shifts else float("nan")
