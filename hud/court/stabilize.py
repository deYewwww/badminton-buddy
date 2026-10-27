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

from hud.media.video import probe

SCALE = 0.5           # features are found at half size (of the working size below) ...
FEAT_SIDE = 640       # ... and on an image no bigger than this (long side)
MAX_SIDE = 1920       # frames are stabilised at most this big (long side): a 4K clip warped and re-encoded
                      # at full size took hours; nothing downstream needs more (YOLO 960, TrackNet 512)
MIN_INLIERS = 40
N_FEAT = 3000         # ORB features per frame (1500 moved warps by ~12 px median on the DT clip)
KEY_EVERY = 4         # match every 4th frame, interpolate between (a phone pans smoothly); cuts,
                      # lost frames and jerks are matched frame by frame. DT clip: frame-by-frame warps
                      # jittered ~16 px (median 2nd difference), this ~0.1 px, and it is ~3x faster
MAX_CANVAS = 1.6      # canvas at most this many times the frame, each way


class Matcher:
    def __init__(self, ref):
        self.orb = cv2.ORB_create(N_FEAT)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # features on a FEAT_SIDE image whatever the clip's size: found on half of a 1920 frame,
        # RANSAC accepted wrong homographies before a cut (jumps every few frames); at the size
        # tuned on the user's 720x1280 clip it doesn't
        self.scale = min(SCALE, FEAT_SIDE / max(ref.shape[:2]))
        self.ref = self._feat(ref)

    def _feat(self, img, small=False):
        """small: img is already at feature size (decoded that small by ffmpeg)."""
        g = img if small else cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return self.orb.detectAndCompute(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY), None)

    def homography(self, a, b):
        """features a -> features b: (3x3 at full size, inliers) or (None, n)."""
        (ka, da), (kb, db) = a, b
        if da is None or db is None or len(ka) < MIN_INLIERS or len(kb) < MIN_INLIERS:
            return None, 0
        good = [m for m, *r in self.bf.knnMatch(da, db, k=2) if r and m.distance < 0.75 * r[0].distance]
        if len(good) < MIN_INLIERS:
            return None, len(good)
        pa = np.float32([ka[m.queryIdx].pt for m in good]) / self.scale
        pb = np.float32([kb[m.trainIdx].pt for m in good]) / self.scale
        A, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0 * SCALE / self.scale)
        n = int(inl.sum()) if inl is not None else 0
        return (A, n) if A is not None and n >= MIN_INLIERS else (None, n)


CUT_DIFF, CUT_CORR = 30.0, 0.3  # thumbnail mean |diff| above / phase correlation below: a cut
JUMP_CUT = 0.05       # ... or the view direction jumps this much of the frame width in one frame.
                      # User's clip: real cuts 41-228 px (720 wide), the fastest gimbal pans <= 31 px


def _thumb(f):
    return cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (180, 320), interpolation=cv2.INTER_AREA)[:200].astype(np.float32)


def sane(H, size, reach=0.6, area=(0.4, 2.5)):
    """A frame -> reference warp that can be real for a phone turning about its centre: the frame
    lands as a convex quad of about its own area, its centre within reach x the frame of the
    reference. The user's clip turns away from the court (people on the bench): chaining frame
    to frame there drifted to warps sending the frame off the canvas or folding it."""
    if H is None or not np.isfinite(H).all():
        return False
    w, h = size
    q = cv2.perspectiveTransform(np.float64([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H).reshape(-1, 2)
    if not cv2.isContourConvex(np.float32(q)):
        return False
    a = cv2.contourArea(np.float32(q)) / (w * h)
    cx, cy = q.mean(0)
    return area[0] < a < area[1] and abs(cx / w - 0.5) < reach and abs(cy / h - 0.5) < reach


def work_size(w, h, max_side=MAX_SIDE):
    """(scale, (w, h)) the clip is stabilised at."""
    k = min(1.0, max_side / max(w, h))
    return k, (int(round(w * k / 2) * 2), int(round(h * k / 2) * 2))


def _scaled(f, size):
    return f if f is None or f.shape[1::-1] == tuple(size) else cv2.resize(f, size, interpolation=cv2.INTER_AREA)


def homographies(video, ref_index, size=None, progress=None):
    """-> (list of (3x3 frame px -> reference px, inliers, how) per frame, cut frame indices),
    on frames of the given size (decoded and shrunk by the hardware where it can).
    progress(fraction) every 100 frames."""
    from hud.media.video import frames
    w, h, _, n = probe(video)
    size = size or (w, h)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_index)
    ok, ref = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read reference frame {ref_index} of {video}")
    m = Matcher(_scaled(ref, size))
    # decoded straight at feature size (ffmpeg/VideoToolbox): resizing 1920 px frames in Python was
    # 40 % of this pass; homographies found there are scaled back to `size` px
    fs = (int(round(size[0] * m.scale)), int(round(size[1] * m.scale)))
    U = np.diag([fs[0] / size[0], fs[1] / size[1], 1.0])
    m.scale = 1.0
    box = np.float32([[0, 0], [fs[0], 0], [fs[0], fs[1]], [0, fs[1]]])

    def place(f, prev, prev_H):
        """-> (features, H in feature px or None, inliers, how); chained through prev when the
        reference doesn't match."""
        feat = m._feat(f, small=True)
        H, k = m.homography(feat, m.ref)
        H = H if sane(H, fs) else None
        how = "direct"
        if H is None and prev is not None and prev_H is not None:
            F, k = m.homography(feat, prev)
            H = prev_H @ F if F is not None else None
            H, how = (H, "chained") if sane(H, fs) else (None, "lost")
        elif H is None:
            how = "lost"
        return feat, H, k, how

    def corners(H):
        return cv2.perspectiveTransform(box.reshape(-1, 1, 2), H).reshape(-1, 2)

    res = []                 # per frame (H in feature px, inliers, how)
    cuts, prev_t = [], None
    key = (None, None)       # last placed frame's (features, H): what the next one chains through
    pending = []             # (index, frame) since the last key frame
    end = object()

    def flush_one_by_one():
        nonlocal key
        for j, g in pending:
            feat, H, k, how = place(g, *key)
            res[j] = (H, k, how)
            key = (feat, H)
        pending.clear()

    src = frames(video, fs)
    i = -1
    while True:
        f = next(src, end)
        if f is end:
            flush_one_by_one()
            break
        i += 1
        res.append(None)
        if progress and i % 100 == 0:
            progress(i / max(n, 1))
        th = _thumb(f)
        cut = False
        if prev_t is not None:
            (_, _), corr = cv2.phaseCorrelate(prev_t, th)
            cut = np.abs(th - prev_t).mean() > CUT_DIFF and corr < CUT_CORR
        prev_t = th
        if cut:
            cuts.append(i)
            flush_one_by_one()
        if i % KEY_EVERY and not cut:
            pending.append((i, f))
            continue
        # key frame: matched; the frames since the last key are interpolated between the two when
        # both were placed and the view moved smoothly, else matched one by one
        kH = key[1]
        feat, H, k, how = place(f, *key)
        if pending and kH is not None and H is not None and \
                np.abs(corners(H) - corners(kH)).max() < JUMP_CUT * fs[0] * (len(pending) + 1):
            a, b = corners(kH), corners(H)
            j0 = pending[0][0] - 1
            for j, _ in pending:
                u = (j - j0) / (i - j0)
                res[j] = (cv2.getPerspectiveTransform(box, np.float32((1 - u) * a + u * b)).astype(np.float64), 0, "interpolated")
            pending.clear()
        else:
            flush_one_by_one()
            feat, H, k, how = place(f, *key)  # chained through the frame just before
        res[i] = (H, k, how)
        key = (feat, H)
    Ui = np.linalg.inv(U)
    return [(None if H is None else Ui @ H @ U, k, how) for H, k, how in res], cuts


def jump_cuts(Hs, size, cuts):
    """Add frames where the frame centre jumps on the reference view (a cut keeping the scene, so
    the image-difference test misses it) to cuts; duplicates within 3 frames merged."""
    w, h = size
    c = np.float64([[w / 2, h / 2]]).reshape(-1, 1, 2)
    pos = np.array([cv2.perspectiveTransform(c, H).ravel() if H is not None else [np.nan, np.nan] for H, *_ in Hs])
    jump = np.r_[0, np.hypot(*np.diff(pos, axis=0).T)]
    back = np.flatnonzero(np.isnan(pos[:-1, 0]) & np.isfinite(pos[1:, 0])) + 1  # view found again after being lost
    found = sorted(set(cuts) | set(np.flatnonzero(np.nan_to_num(jump) > JUMP_CUT * w).tolist()) | set(back.tolist()))
    out = []
    for i in found:
        if not out or i - out[-1] > 3:
            out.append(int(i))
    return out


MOVE_FRAC = 0.02      # the view drifting more than this much of the frame width: stabilise first


def moving(video, ref_index, n=24):
    """Does the camera pan or cut? n frames spread over the clip matched to the reference frame.
    -> (moving, largest frame-centre shift as a fraction of the width, samples not matched).
    A tripod clip stays well under MOVE_FRAC (players and the shuttle are a small part of the
    features); a sample that doesn't match at all is a cut to another view or a fast pan."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref = _frame_at(cap, ref_index)
    if ref is None:
        cap.release()
        return False, 0.0, 0
    h, w = ref.shape[:2]
    m = Matcher(ref)
    c = np.float64([[w / 2, h / 2]]).reshape(-1, 1, 2)
    shift, lost = 0.0, 0
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        f = _frame_at(cap, i)
        if f is None:
            continue
        H, _ = m.homography(m._feat(f), m.ref)
        if H is None:
            lost += 1
            continue
        shift = max(shift, float(np.hypot(*(cv2.perspectiveTransform(c, H).ravel() - c.ravel()))) / w)
    cap.release()
    return bool(shift > MOVE_FRAC or lost >= 2), round(shift, 3), lost


def frame_maps(video):
    """Per frame 3x3 original px -> stabilised px of a stabilised video (lost frames hold the last
    good one), or None for a normal video."""
    meta = Path(video).with_suffix(".json")
    try:
        Hs = json.loads(meta.read_text())["H"] if meta.exists() else None
    except (OSError, ValueError, KeyError):
        return None
    if Hs is None:
        return None
    out, last = [], np.eye(3)
    for H in Hs:
        last = np.array(H) if H is not None else last
        out.append(last)
    return np.array(out)


def cuts_of(video):
    """Cut frame indices of a stabilised video (from its .json), else []."""
    meta = Path(video).with_suffix(".json")
    try:
        return list(json.loads(meta.read_text()).get("cuts", [])) if meta.exists() else []
    except (OSError, ValueError):
        return []


def _frame_at(cap, i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = cap.read()
    return f if ok else None


def floor_fixes(video, Hs, cuts, T, size, corners, min_len=30, samples=7, chunk=None, work=None):
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
                f, H = _scaled(_frame_at(cap, i), work) if work else _frame_at(cap, i), Hs[i][0]
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


def run(video, out, ref_time, corners=None, progress=None, max_side=MAX_SIDE):
    """Write the stabilised video to out (.mp4) and out.json (per-frame homographies from the
    ORIGINAL frame's pixels to the stabilised ones, offset, the working scale).
    corners: the reference frame's court corners (original px) -> returned on the canvas.
    progress(fraction, what) while it works: two passes over the clip (measure, then write)."""
    W0, H0, fps, n = probe(video)
    k, (w, h) = work_size(W0, H0, max_side)
    S = np.diag([k, k, 1.0])  # original px -> working px
    corners = None if corners is None else np.float64(corners) * k
    ref_index = int(round(ref_time * fps))
    say = progress or (lambda x, what: None)
    Hs, cuts = homographies(video, ref_index, (w, h), lambda x: say(0.5 * x, "measure"))
    cuts = jump_cuts(Hs, (w, h), cuts)
    (ox, oy), (cw, ch) = canvas(Hs, (w, h))
    r = max(cw, ch) / max_side
    if r > 1:  # the canvas (frame + all the pan reached) at most max_side too: warping and encoding a
        # 3074 px canvas was a third of this step, and YOLO / TrackNet shrink it to 960 / 512 anyway
        k2, (w2, h2) = work_size(W0, H0, max(w, h) / r)
        R = np.diag([w2 / w, h2 / h, 1.0])
        Hs = [(None if H is None else R @ H @ np.linalg.inv(R), *x) for H, *x in Hs]
        corners = None if corners is None else corners * [w2 / w, h2 / h]
        k, w, h = k * w2 / w, w2, h2
        S = np.diag([k, k, 1.0])
        (ox, oy), (cw, ch) = canvas(Hs, (w, h))
    T = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1.0]])
    hows = [x[2] for x in Hs]
    print(f"Stabilised {len(Hs)} frames: {hows.count('direct')} direct, {hows.count('chained')} chained, {hows.count('interpolated')} interpolated, "
          f"{hows.count('lost')} lost; {len(cuts)} cuts; canvas {cw}x{ch}, reference at +({ox},{oy})")
    K_of = np.repeat(np.eye(3)[None], len(Hs), 0)
    if corners is not None:  # floor-plane correction per segment (camera moved between takes)
        fixes = floor_fixes(video, Hs, cuts, T, (cw, ch), np.float64(corners) + [ox, oy], work=(w, h))
        for a, b, K, _ in fixes:
            K_of[a:b] = K
        big = [f"{a / fps:.1f}s {sh:.0f}px" for a, b, K, sh in fixes if sh > 5]
        print(f"Floor corrected in {sum(sh > 0.5 for *_, sh in fixes)}/{len(fixes)} pieces; > 5 px at: {', '.join(big) or 'none'}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from hud.visualization.render import _has_videotoolbox
    from hud.media.video import frames
    # Mac: the hardware encoder (the CPU one was most of the time on long clips); high bitrate, the
    # shuttle is a few pixels
    enc = (["-c:v", "h264_videotoolbox", "-b:v", "24M"] if _has_videotoolbox()
           else ["-c:v", "libx264", "-crf", "17", "-preset", "veryfast"])
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cw}x{ch}",
                           "-r", f"{fps:.4f}", "-i", "-", *enc, "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    last, since = None, 0
    hold = int(fps / 2)
    for i, (f, (H, _, _)) in enumerate(zip(frames(video, (w, h)), Hs)):
        since = 0 if H is not None else since + 1
        last = H if H is not None else last
        H = last if since <= hold else None  # lost: hold the last good warp for a blur, black when the camera looks away
        img = cv2.warpPerspective(f, K_of[i] @ T @ H, (cw, ch)) if H is not None else np.zeros((ch, cw, 3), np.uint8)
        ff.stdin.write(img.tobytes())
        if i % 100 == 0:
            say(0.5 + 0.5 * i / max(n, 1), "write")
    ff.stdin.close()
    ff.wait()
    meta = dict(video=str(video), ref_time=ref_time, offset=[ox, oy], canvas=[cw, ch], cuts=cuts, scale=k,
                H=[None if H is None else np.round(K_of[i] @ T @ H @ S, 8).tolist() for i, (H, *_) in enumerate(Hs)],
                how=hows)
    out.with_suffix(".json").write_text(json.dumps(meta))
    return None if corners is None else np.float64(corners) + [ox, oy]


def native_size(video):
    """(w, h) of the camera behind a stabilised video (its source clip, at the scale it was
    stabilised at), or None for a normal one."""
    meta = Path(video).with_suffix(".json")
    if not meta.exists():
        return None
    try:
        m = json.loads(meta.read_text())
        if "canvas" not in m:
            return None
        w, h = probe(m["video"])[:2]
        k = m.get("scale", 1.0)  # the lens in the stabilised video's pixels
        return (w * k, h * k)
    except (OSError, ValueError, KeyError, SystemExit):
        return None


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
