"""HUD video: game-style overlay on the original footage.

Top bar (rally time, shots, shot speed, score), a name tag with stamina bar over each
player, a perspective ring under their feet, a MOVE / ARRIVED target for our side, a
LOCK ON popup when a shot's speed is known, player panels (stamina + MOVE meter) and a
small court map.

Light by design: one process, shapes drawn with OpenCV straight onto the frame, text
rendered once per distinct string with PIL and cached, frames piped to ffmpeg's
hardware H.264 encoder (VideoToolbox) at no more than 30 fps.
"""
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from hud import config, court
from hud.court import COURT_L, COURT_W, NET_Y

FONTS = ["/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
         "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
         # Linux (cloud tests): without a CJK font every Chinese label renders as boxes
         "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"]
FONT = next((f for f in FONTS if Path(f).exists()), None)
PIDS = ["P1", "P2", "P3", "P4"]
TEAM = {"P1": "us", "P2": "us", "P3": "opp", "P4": "opp"}
BG = (30, 24, 14)            # panel background (BGR)
EDGE = (120, 100, 60)
WHITE, GREY, RED, GREEN = (245, 245, 245), (170, 170, 160), (80, 80, 255), (90, 225, 160)
TYPE_LABEL = {"smash": "殺球", "clear": "高遠球", "drop": "吊球", "net": "網前", "drive": "平抽", "other": ""}
MAX_OUT_FPS = 30
MAX_OUT_SIDE = 1920
LOCK_S, BANNER_S = 1.0, 1.6
COST_S = 0.9                 # stroke-cost popup on the name tag
TRAIL_N = 8                  # shuttle tail, frames
TRACK_MIN = 5                # shuttle drawn only on runs of at least this many real sightings
POS_SMOOTH_S, BOX_SMOOTH_S = 0.3, 0.4  # display smoothing of feet (rings) and boxes (name tags)
TAG_EASE = 0.35              # name tags move this fraction of the way to their new place per frame
TAG_HOLD_S = 1.0             # ... and stay put this long when the player's box is lost
COST_LABEL = {"defence": "防守", "other": "擊球"}
MOVE_MS, STOP_MS = 2.0, 0.8  # target marker while faster than this; ARRIVED once below STOP_MS


def hex_bgr(h):
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


@lru_cache(maxsize=4096)
def sprite(text, size, color):
    """Text as an RGBA array (cached: most HUD strings repeat frame after frame)."""
    font = ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()
    l, t, r, b = font.getbbox(text)
    img = Image.new("RGBA", (max(1, r - l + 2), max(1, b - t + 2)), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((1 - l, 1 - t), text, font=font, fill=(color[2], color[1], color[0], 255))
    return np.asarray(img)


def put(img, text, size, color, x, y, anchor="lt"):
    """Blit text; anchor: l/c/r + t/m/b. Returns the text width."""
    s = sprite(str(text), int(size), tuple(color))
    h, w = s.shape[:2]
    x = int(x - (w / 2 if anchor[0] == "c" else w if anchor[0] == "r" else 0))
    y = int(y - (h / 2 if anchor[1] == "m" else h if anchor[1] == "b" else 0))
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return w
    s = s[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4].astype(np.float32) / 255
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - a) + s[..., 2::-1] * a).astype(np.uint8)
    return w


def panel(img, x0, y0, x1, y1, accent=None, alpha=0.78):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), W), min(int(y1), H)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha) + np.array(BG) * alpha).astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), EDGE, 1, cv2.LINE_AA)
    if accent is not None:
        cv2.rectangle(img, (x0, y0), (x0 + max(3, (x1 - x0) // 90), y1 - 1), accent, -1)


def bar(img, x0, y0, w, h, frac, color):
    cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), (70, 60, 45), -1)
    if frac > 0:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + w * min(frac, 1)), int(y0 + h)), color, -1)


def floor_poly(Ginv, pts):
    return np.int32(court.to_court(Ginv, pts))


def circle_pts(c, r, n=40):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]


# --- per-frame state --------------------------------------------------------------------

def load(out_dir, F, fps):
    from hud.pipeline import names, paths
    p = paths(out_dir)
    cfg = config.load()
    pl = pd.read_parquet(p["players"])
    pos = np.full((F, 4, 2), np.nan)
    box = np.full((F, 4, 4), np.nan)
    for k, pid in enumerate(PIDS):
        g = pl[(pl.player_id == pid) & (pl.frame < F)]
        pos[g.frame, k] = g[["x_m", "y_m"]].to_numpy()
        box[g.frame, k] = g[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy()
    # fill short gaps so rings don't blink, then smooth what the eye follows: feet over 0.3 s (the
    # positions are already RTS-smoothed), boxes (name tags) over 0.4 s. Centred windows: no lag
    w = max(1, int(0.2 * fps))
    wp, wb = max(1, int(POS_SMOOTH_S * fps)), max(1, int(BOX_SMOOTH_S * fps))
    for k in range(4):
        for d in range(2):
            s = pd.Series(pos[:, k, d]).interpolate(limit=int(fps), limit_area="inside")
            pos[:, k, d] = s.rolling(wp, center=True, min_periods=1).mean().where(s.notna())
        for d in range(4):
            s = pd.Series(box[:, k, d]).interpolate(limit=int(fps // 2), limit_area="inside")
            box[:, k, d] = s.rolling(wb, center=True, min_periods=1).mean().where(s.notna()).to_numpy()
    shots = pd.read_parquet(p["shots"]) if p["shots"].exists() else pd.DataFrame()
    rallies = pd.read_parquet(p["rallies"]) if p["rallies"].exists() else pd.DataFrame()
    sh = pd.read_parquet(p["shuttle"]) if p["shuttle"].exists() else None
    shuttle = shown_track(sh, F, lens_short(out_dir)) if sh is not None else np.full((F, 2), np.nan)

    step = np.r_[np.zeros((1, 4)), np.nan_to_num(np.hypot(*np.diff(pos, axis=0).transpose(2, 0, 1)))]
    step = np.minimum(step, 12.0 / fps)  # > 12 m/s is an identity jump, not running
    speed = np.stack([np.convolve(step[:, k], np.ones(w) / w, "same") * fps for k in range(4)], 1)
    rally_of = np.full(F, -1)
    for i, r in enumerate(rallies.itertuples()):
        rally_of[int(r.start_frame):min(int(r.end_frame) + 1, F)] = i
    shuttle = in_play_only(shuttle, rallies, fps, lens_short(out_dir))
    sc = cfg["move"]
    stamina, costs = stamina_track(step, speed, rally_of, shots, fps, cfg["stamina"], accelerations(pos, fps))
    moving = move_states(speed, fps)
    win = max(1, int(sc["window_s"] * fps))
    move = np.stack([np.convolve(step[:, k], np.ones(win), "full")[:F] for k in range(4)], 1) / sc["full_scale_m"]
    return dict(pos=pos, box=box, shots=shots, rallies=rallies, shuttle=shuttle, speed=speed, stamina=stamina, costs=costs,
                move=np.clip(move, 0, 1), moving=moving, rally_of=rally_of, names=names(out_dir), cfg=cfg,
                colors={pid: hex_bgr(cfg["colors"][pid]) for pid in PIDS})


def lens_short(out_dir):
    """Short side (px) of the camera behind the analysed frames (the lens, for a stabilised canvas)."""
    import json
    from hud.pipeline import paths
    from hud.stabilize import native_size
    p = paths(out_dir)
    f = p["stab_calib"] if p["stab_calib"].exists() else p["calib"]
    try:
        c = json.loads(f.read_text())
        return float(min(native_size(c["video"]) or c["frame_size"]))
    except (OSError, ValueError, KeyError):
        return 1080.0


def shown_track(sh, F, short=1080, min_run=TRACK_MIN, max_gap=2, max_step=0.1):
    """(F, 2) shuttle pixels to draw: real TrackNet sightings only (no InpaintNet guesses, no
    interpolation), and only runs of at least min_run frames that move like one shuttle (a
    step of at most max_step of the camera's short side (px) per frame, gaps of max_gap frames
    filled). A stray sighting on a light or a shoe makes a lone dot that jumps around: hidden."""
    out = np.full((F, 2), np.nan)
    real = sh[sh.visible & ~sh.interpolated & (~sh.inpainted if "inpainted" in sh else True) & (sh.frame < F)]
    if not len(real):
        return out
    fr, P = real.frame.to_numpy(), real[["x_px", "y_px"]].to_numpy(float)
    run = [0]
    for i in range(1, len(fr)):
        gap = fr[i] - fr[i - 1]
        if gap <= max_gap + 1 and np.hypot(*(P[i] - P[i - 1])) <= max_step * short * gap:
            run.append(i)
            continue
        _keep(out, fr, P, run, min_run)
        run = [i]
    _keep(out, fr, P, run, min_run)
    return out


def in_play_only(shuttle, rallies, fps, short=1080, before_s=0.3, after_s=0.6, still_s=0.4, still_px=8):
    """Hide the shuttle where it isn't in play: outside the rallies (picked up, carried, tossed
    to the server: a ball drifting about between points, which the user saw as the tracker
    "floating" at 1:29 and 1:38), and wherever it hangs nearly still for still_s (held in a
    hand, or TrackNet stuck on something: a flying shuttle is never that slow for 0.4 s)."""
    out = shuttle.copy()
    F = len(out)
    play = np.zeros(F, bool)
    for r in rallies.itertuples():
        if getattr(r, "manual", False) is True:
            continue
        play[max(0, int(r.start_frame - before_s * fps)):min(F, int(r.end_frame + after_s * fps) + 1)] = True
    out[~play] = np.nan
    w = max(2, int(still_s * fps))
    tol = still_px * short / 1080
    ok = np.isfinite(out[:, 0])
    hide = np.zeros(F, bool)
    for f in range(F - w + 1):
        seg = out[f:f + w]
        if ok[f:f + w].all() and np.hypot(*(seg - seg.mean(0)).T).max() <= tol:
            hide[f:f + w] = True  # the whole still window, its first and last frames too
    out[hide] = np.nan
    return out


def _keep(out, fr, P, run, min_run):
    if len(run) < min_run:
        return
    f, q = fr[run], P[run]
    full = np.arange(f[0], f[-1] + 1)
    out[full, 0] = np.interp(full, f, q[:, 0])
    out[full, 1] = np.interp(full, f, q[:, 1])


def move_states(speed, fps, on_s=0.3, off_s=0.3, arrived_s=0.6):
    """(F, 4) 0 = nothing, 1 = MOVE, 2 = ARRIVED, with hysteresis: MOVE once faster than MOVE_MS
    for on_s, kept until slower than STOP_MS for off_s, then ARRIVED for arrived_s. Per-frame
    thresholds made the markers blink on and off."""
    F = len(speed)
    out = np.zeros((F, speed.shape[1]), np.int8)
    on, off, arr = int(on_s * fps), int(off_s * fps), int(arrived_s * fps)
    for k in range(speed.shape[1]):
        v = np.nan_to_num(speed[:, k])
        state, fast, slow, left = 0, 0, 0, 0
        for f in range(F):
            fast = fast + 1 if v[f] > MOVE_MS else 0
            slow = slow + 1 if v[f] < STOP_MS else 0
            if state != 1 and fast >= on:
                state = 1
            elif state == 1 and slow >= off:
                state, left = 2, arr
            elif state == 2:
                left -= 1
                state = 2 if left > 0 else 0
            out[f, k] = state
    return out


def smooth_maps(M, cuts, half=3):
    """Per-frame original -> stabilised homographies averaged over +-half frames, never across a
    cut: the frame-to-frame ORB noise (~1 px) shook every ring and tag drawn on the original."""
    M = np.asarray(M, float) / np.asarray(M, float)[:, 2:3, 2:3]
    out = M.copy()
    bounds = [0, *[c for c in cuts if 0 < c < len(M)], len(M)]
    for a, b in zip(bounds, bounds[1:]):
        for f in range(a, b):
            out[f] = M[max(a, f - half):min(b, f + half + 1)].mean(0)
    return out


def stroke_kinds(shots):
    """Shot type per stroke for the stamina cost: a stroke answering an opponent's smash in the
    same rally is 'defence' whatever it became."""
    if not len(shots):
        return []
    sh = shots.sort_values("frame")
    prev = sh.shot_type.shift()
    same = sh.rally_id.eq(sh.rally_id.shift()) if "rally_id" in sh else pd.Series(True, index=sh.index)
    kind = np.where(same & prev.eq("smash") & sh.team.ne(sh.team.shift()), "defence", sh.shot_type.fillna("other"))
    return list(zip(sh.frame.astype(int), sh.player_id, kind))


def accelerations(pos, fps, half_s=0.15):
    """(F, 4) acceleration magnitude (m/s^2) from the smoothed positions: footwork (stops,
    pushes off, direction changes) costs more than running at the same speed."""
    h = max(1, int(half_s * fps))
    v = (np.roll(pos, -h, 0) - np.roll(pos, h, 0)) * fps / (2 * h)
    a = (np.roll(v, -h, 0) - np.roll(v, h, 0)) * fps / (2 * h)
    a[:2 * h] = a[-2 * h:] = np.nan
    return np.nan_to_num(np.hypot(a[..., 0], a[..., 1]))


def stamina_track(step, speed, rally_of, shots, fps, st, accel=None):
    """Per frame and player stamina (see config [stamina]) -> ((F, 4) values, [(frame, k, cost,
    kind)] stroke costs for the HUD popups).
    Drains with effort, in proportion: running above walking pace (per_mps per m/s above
    walk_mps), footwork (per_accel per m/s^2 above accel_free), and each stroke by type, an
    attacking stroke more when the same player keeps attacking (streak). Standing still
    (slower than rest_mps) recovers, slowly during a rally (a server, a partner waiting) and
    faster between rallies, up to a ceiling that falls with fatigue x what was spent."""
    F = len(speed)
    cost_of = st.get("shot_cost", {})
    attack = set(st.get("attack", ["smash", "drive"]))
    hit_at, last_attack = {}, {}
    for f, pid, kind, z in stroke_list(shots):
        if pid not in PIDS or not 0 <= f < F:
            continue
        k = PIDS.index(pid)
        c = float(cost_of.get(kind, cost_of.get("other", 0)))
        if kind == "smash" and np.isfinite(z) and z >= st.get("jump_z_m", 2.6):
            c += st.get("jump_extra", 0.5)
        if kind in attack:  # attacking again within streak_s: each one costs more
            prev_f, n = last_attack.get(k, (-10 ** 9, 0))
            n = n + 1 if f - prev_f <= st.get("streak_s", 3.0) * fps else 0
            last_attack[k] = (f, n)
            c *= 1 + min(n * st.get("streak_step", 0.3), st.get("streak_max", 1.0))
        hit_at.setdefault(f, []).append((k, c, kind))
    start, fat = float(st["start"]), float(st.get("fatigue", 0))
    cur, ceil = np.full(4, start), np.full(4, start)
    out, costs = np.zeros((F, 4)), []
    accel = np.zeros((F, 4)) if accel is None else accel
    v = np.nan_to_num(speed)
    for f in range(F):
        spent = (st["per_mps"] * np.maximum(v[f] - st["walk_mps"], 0)
                 + st["per_accel"] * np.maximum(accel[f] - st["accel_free"], 0)) / fps
        for k, c, kind in hit_at.get(f, ()):
            spent = spent + c * (np.arange(4) == k)
            costs.append((f, k, c, kind))
        ceil = np.maximum(ceil - fat * spent, 0)
        rest = v[f] < st["rest_mps"]
        back = st["recover_rally_per_s"] if rally_of[f] >= 0 else st["recover_per_s"]
        cur = np.clip(cur - spent + rest * back / fps, 0, ceil)
        out[f] = cur
    return out, costs


def stroke_list(shots):
    """[(frame, player, kind, contact height)]: stroke_kinds plus how high it was hit."""
    if not len(shots):
        return []
    sh = shots.sort_values("frame")
    z = sh.contact_z_m.to_numpy(float) if "contact_z_m" in sh else np.full(len(sh), np.nan)
    return [(f, pid, kind, zz) for (f, pid, kind), zz in zip(stroke_kinds(sh), z)]


# --- drawing ----------------------------------------------------------------------------

class Hud:
    def __init__(self, S, H, size, fps):
        self.S, self.fps = S, fps
        self.Ginv = np.linalg.inv(H)
        self.Ginvs = None
        self.tag_at = {}  # pid -> (x, y, frame) of the tag last drawn
        self.W, self.Hh = size
        self.u = min(size) / 1080  # layout unit: 1 px on a 1080-wide portrait frame
        sc = S["cfg"]["score"]
        self.start_score = (sc["us"], sc["opp"])

    def draw(self, img, f):
        S, u = self.S, self.u
        if self.Ginvs is not None:  # stabilised analysis drawn on the moving original: per-frame court
            self.Ginv = self.Ginvs[min(f, len(self.Ginvs) - 1)]
        self.floor(img, f)
        self.trail(img, f)
        self.tags(img, f)
        self.lock_on(img, f)
        self.top_bar(img, f)
        self.banner(img, f)
        self.player_panel(img, f)
        self.court_map(img, f)

    # rings, MOVE targets, landing burst
    def floor(self, img, f):
        S, u = self.S, self.u
        for k, pid in enumerate(PIDS):
            p = S["pos"][f, k]
            if not np.isfinite(p[0]):
                continue
            col = S["colors"][pid]
            ring = floor_poly(self.Ginv, circle_pts(p, 0.55))
            cv2.polylines(img, [ring], True, tuple(int(c * 0.55) for c in col), max(2, int(7 * u)), cv2.LINE_AA)
            cv2.polylines(img, [ring], True, col, max(1, int(3 * u)), cv2.LINE_AA)
            c = court.to_court(self.Ginv, [p])[0]
            put(img, S["names"][pid], 20 * u, col, c[0], c[1], "cm")
            if TEAM[pid] == "us":
                self.move_target(img, f, k, col)
        r = S["rally_of"][max(f - int(BANNER_S * self.fps), 0)] if f else -1
        self.landing_burst(img, f)

    def move_target(self, img, f, k, col):
        S, u, fps = self.S, self.u, self.fps
        state = S["moving"][f, k]
        if state == 1:
            ahead = S["pos"][f:min(f + int(0.5 * fps), len(S["pos"]) - 1) + 1, k]
            ahead = ahead[np.isfinite(ahead[:, 0])]
            dest = ahead[-1] if len(ahead) else S["pos"][f, k]
            label = "MOVE"
        elif state == 2:
            dest, label = S["pos"][f, k], "ARRIVED"
        else:
            return
        if not np.isfinite(dest[0]):
            return
        g = (90, 230, 170)
        for r in (0.35, 0.6, 0.85):
            d = floor_poly(self.Ginv, dest + r * np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]))
            cv2.polylines(img, [d], True, g, max(1, int(3 * u)), cv2.LINE_AA)
        cv2.polylines(img, [floor_poly(self.Ginv, circle_pts(dest, 1.0))], True, g, max(1, int(2 * u)), cv2.LINE_AA)
        tip = court.to_court(self.Ginv, [dest - [0, 1.1]])[0]
        w = sprite(label, int(22 * u), (255, 255, 255)).shape[1]
        panel(img, tip[0] - w / 2 - 8 * u, tip[1], tip[0] + w / 2 + 8 * u, tip[1] + 34 * u, alpha=0.85)
        put(img, label, 22 * u, (140, 255, 200), tip[0], tip[1] + 17 * u, "cm")

    def trail(self, img, f, n=TRAIL_N):
        """The shuttle and a short fading tail of where it just was (tracked frames only)."""
        P = self.S["shuttle"][max(0, f - n):f + 1]
        u = self.u
        pts = [(k, p) for k, p in enumerate(P) if np.isfinite(p[0])]
        for (k0, a), (k1, b) in zip(pts, pts[1:]):
            if k1 - k0 > 2:  # don't join across a gap
                continue
            w = (k1 + 1) / len(P)
            cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), (int(255 * w), int(240 * w), 120), max(1, int(4 * u * w)), cv2.LINE_AA)
        if np.isfinite(P[-1, 0]):
            cv2.circle(img, tuple(np.int32(P[-1])), max(3, int(9 * u)), (255, 255, 255), max(1, int(2 * u)), cv2.LINE_AA)

    def landing_burst(self, img, f):
        """Expanding rings where a scoring shot landed; gold for PERFECT/GREAT placement."""
        S, fps = self.S, self.fps
        for r in S["rallies"].itertuples():
            age = (f - r.end_frame) / fps
            if not 0 <= age <= BANNER_S or not np.isfinite(r.landing_x) or not r.landing_in:
                continue
            good = r.winning_label in ("PERFECT", "GREAT")
            col = (60, 210, 255) if good else (255, 255, 255)
            for i in range(3 if good else 1):
                rad = 0.3 + (age * 1.6 + i * 0.4) % 1.6
                cv2.polylines(img, [floor_poly(self.Ginv, circle_pts((r.landing_x, r.landing_y), rad))], True, col,
                              max(1, int(3 * self.u)), cv2.LINE_AA)

    def tags(self, img, f):
        """Name + stamina over each head, stacked so they never overlap."""
        S, u = self.S, self.u
        tw, th = 250 * u, 58 * u
        items = []
        for k, pid in enumerate(PIDS):
            b = S["box"][f, k]
            if not np.isfinite(b[0]):
                last = self.tag_at.get(pid)  # box lost for a moment: the tag stays put (up to TAG_HOLD_S)
                if last is not None and f - last[2] <= TAG_HOLD_S * self.fps and np.isfinite(S["pos"][f, k, 0]):
                    b = last[3]
                else:
                    continue
            items.append([(b[0] + b[2]) / 2 - tw / 2, b[1] - th - 40 * u, k, pid, (b[0] + b[2]) / 2, b[1], b])
        # stack in a steady order (nearest player first: depth changes slowly, unlike the box
        # tops, whose order flipped back and forth when partners overlapped), then let each tag
        # glide to its place instead of jumping
        depth = lambda it: S["pos"][f, it[2], 1] if np.isfinite(S["pos"][f, it[2], 1]) else 99.0
        items.sort(key=depth)
        placed = []
        for it in items:
            x, y = min(max(it[0], 4), self.W - tw - 4), max(it[1], 120 * u)
            for px, py in placed:
                if abs(px - x) < tw and abs(py - y) < th + 6 * u:
                    y = py - th - 6 * u
            placed.append((x, y))
            k, pid, hx, hy, b = it[2:]
            last = self.tag_at.get(pid)
            if last is not None and f - last[2] <= TAG_HOLD_S * self.fps:
                x, y = last[0] + TAG_EASE * (x - last[0]), last[1] + TAG_EASE * (y - last[1])
            if np.isfinite(S["box"][f, k, 0]):
                self.tag_at[pid] = (x, y, f, b)
            else:
                self.tag_at[pid] = (x, y, last[2], b)  # keep the time it was last really seen
            col = S["colors"][pid]
            cv2.line(img, (int(x + tw / 2), int(y + th)), (int(hx), int(hy - 6 * u)), col, max(1, int(2 * u)), cv2.LINE_AA)
            panel(img, x, y, x + tw, y + th, col)
            stam = S["stamina"][f, k]
            put(img, S["names"][pid], 26 * u, col, x + 14 * u, y + 8 * u)
            put(img, f"{stam:.0f}", 26 * u, RED if stam < 15 else WHITE, x + tw - 12 * u, y + 8 * u, "rt")
            bar(img, x + 14 * u, y + th - 16 * u, tw - 28 * u, 7 * u, stam / 100, col)
            for g, kk, c, kind in self.recent_costs(f, k):  # what that stroke cost, floating up
                age = (f - g) / self.fps
                put(img, f"-{c:.1f} {TYPE_LABEL.get(kind, '') or COST_LABEL.get(kind, '')}".rstrip(), 22 * u,
                    (90, 170, 255), x + tw - 12 * u, y - 4 * u - age * 30 * u, "rb")

    def recent_costs(self, f, k, span=COST_S):
        return [c for c in self.S.get("costs", ()) if c[1] == k and 0 <= f - c[0] <= span * self.fps]

    def current_shot(self, f):
        """Last shot of the rally in progress (or just finished) at frame f, or None."""
        S = self.S
        r = S["rally_of"][f]
        if r < 0 or not len(S["shots"]):
            return None, 0, None
        rid = S["rallies"].rally_id.iat[r]
        sh = S["shots"][(S["shots"].rally_id == rid) & (S["shots"].frame <= f)]
        return (sh.iloc[-1] if len(sh) else None), len(sh), S["rallies"].iloc[r]

    def lock_on(self, img, f):
        S, u = self.S, self.u
        shot, _, _ = self.current_shot(f)
        if shot is None or not pd.notna(shot.speed_kmh_est) or (f - shot.frame) / self.fps > LOCK_S:
            return
        # on the shuttle as it is now, or not at all: pinned to where it was hit, the reticle hung
        # in the air while the camera panned (the user saw it as the tracker floating)
        p = S["shuttle"][f]
        if not np.isfinite(p[0]):
            return
        col = (255, 230, 120)
        cv2.circle(img, (int(p[0]), int(p[1])), int(22 * u), col, max(1, int(2 * u)), cv2.LINE_AA)
        for a in range(4):
            d = np.array([np.cos(a * np.pi / 2), np.sin(a * np.pi / 2)])
            cv2.line(img, tuple(np.int32(p + d * 14 * u)), tuple(np.int32(p + d * 30 * u)), col, max(1, int(2 * u)))
        x, y = p[0] + 40 * u, p[1] - 70 * u
        panel(img, x, y, x + 200 * u, y + 62 * u, col)
        put(img, "LOCK ON", 18 * u, (200, 230, 255), x + 14 * u, y + 6 * u)
        put(img, f"{shot.speed_kmh_est:.0f} km/h", 28 * u, WHITE, x + 14 * u, y + 28 * u)

    def top_bar(self, img, f):
        S, u = self.S, self.u
        m, y, h = 24 * u, 40 * u, 92 * u
        shot, n, rally = self.current_shot(f)
        r = S["rally_of"][f]
        if r < 0:  # between rallies: keep showing the last one's final numbers
            prev = np.flatnonzero(S["rallies"].end_frame.to_numpy() < f) if len(S["rallies"]) else []
            rally = S["rallies"].iloc[prev[-1]] if len(prev) else None
            n = int(rally.shot_count) if rally is not None else 0
            t = rally.duration_s if rally is not None else 0.0
        else:
            t = (f - rally.start_frame) / self.fps
        widths = [235, 235, 330, 200]
        x = m
        heads = ["回合時間", "擊球數", "球速（估）" if shot is not None and pd.notna(shot.speed_kmh_est) else "搜尋羽球", "比分"]
        for i, (wd, head) in enumerate(zip(widths, heads)):
            wd *= u
            panel(img, x, y, x + wd, y + h, (240, 200, 54) if i == 1 else None)
            put(img, head, 20 * u, (240, 200, 54) if i == 1 else GREY, x + 16 * u, y + 10 * u)
            vy = y + 40 * u
            if i == 0:
                put(img, f"{int(t // 60):02d}:{t % 60:05.2f}", 38 * u, WHITE, x + 16 * u, vy)
            elif i == 1:
                put(img, f"{n} 拍", 38 * u, WHITE, x + 16 * u, vy)
            elif i == 2:
                if shot is not None and pd.notna(shot.speed_kmh_est):
                    w = put(img, f"{shot.speed_kmh_est:.0f}", 38 * u, WHITE, x + 16 * u, vy)
                    put(img, "km/h", 20 * u, GREY, x + 26 * u + w, vy + 14 * u)
                    put(img, TYPE_LABEL.get(shot.shot_type, ""), 24 * u, (140, 255, 200), x + wd - 14 * u, vy + 8 * u, "rt")
                else:
                    put(img, "---", 38 * u, GREY, x + 16 * u, vy)
                    put(img, "km/h", 20 * u, GREY, x + 110 * u, vy + 14 * u)
            else:
                us, opp = self.score_at(f)
                w = put(img, f"{us}", 38 * u, S["colors"]["P1"], x + 16 * u, vy)
                w += put(img, " : ", 38 * u, WHITE, x + 16 * u + w, vy)
                put(img, f"{opp}", 38 * u, S["colors"]["P3"], x + 16 * u + w, vy)
            x += wd + 10 * u

    def score_at(self, f):
        R = self.S["rallies"]
        if not len(R) or "score_us" not in R:
            return self.start_score
        done = R[R.end_frame <= f]
        return (int(done.score_us.iat[-1]), int(done.score_opp.iat[-1])) if len(done) else self.start_score

    def banner(self, img, f):
        """'我隊 得分' (+ placement label) right after a rally is decided."""
        S, u = self.S, self.u
        for r in S["rallies"].itertuples():
            age = (f - r.end_frame) / self.fps
            team = getattr(r, "point_team_final", None)
            if not 0 <= age <= BANNER_S or team not in ("us", "opp"):
                continue
            text = ("我隊 得分" if team == "us" else "對手 得分")
            guessed = getattr(r, "point_source", "") == "guess"
            col = S["colors"]["P1" if team == "us" else "P3"]
            cx, cy = self.W / 2, 175 * u
            panel(img, cx - 190 * u, cy, cx + 190 * u, cy + 70 * u, col, alpha=0.85)
            put(img, text, 36 * u, col, cx, cy + 35 * u, "cm")
            if guessed:  # landing not seen: say so (the editor's table fixes it)
                put(img, "推測", 18 * u, GREY, cx + 180 * u, cy + 8 * u, "rt")
            lab = r.winning_label if isinstance(r.winning_label, str) else ""
            if lab:
                put(img, lab, 30 * u, (60, 210, 255) if lab in ("PERFECT", "GREAT") else WHITE, cx, cy + 92 * u, "cm")

    def player_panel(self, img, f):
        S, u = self.S, self.u
        w, rh = 450 * u, 64 * u
        x, y0 = 24 * u, self.Hh - 24 * u - 4 * (rh + 6 * u) - 40 * u
        for k, pid in enumerate(PIDS):
            y = y0 + k * (rh + 6 * u)
            col = S["colors"][pid]
            stam = S["stamina"][f, k]
            panel(img, x, y, x + w, y + rh, col)
            put(img, S["names"][pid], 25 * u, col, x + 16 * u, y + 6 * u)
            put(img, f"體力 {stam:.0f}", 23 * u, RED if stam < 15 else WHITE, x + w - 14 * u, y + 7 * u, "rt")
            bar(img, x + 16 * u, y + 36 * u, w - 30 * u, 7 * u, stam / 100, col)
            put(img, "MOVE", 12 * u, GREY, x + 16 * u, y + 48 * u)
            lit = int(round(S["move"][f, k] * 16))
            for d in range(16):
                dx = x + 62 * u + d * 17 * u
                cv2.rectangle(img, (int(dx), int(y + 50 * u)), (int(dx + 11 * u), int(y + 56 * u)),
                              col if d < lit else (80, 70, 55), -1)

    def court_map(self, img, f):
        S, u = self.S, self.u
        pw, ph = 470 * u, 4 * 70 * u - 6 * u
        x0, y0 = self.W - 24 * u - pw, self.Hh - 24 * u - ph - 40 * u
        panel(img, x0, y0, x0 + pw, y0 + ph, (240, 200, 54))
        put(img, "球場地圖", 22 * u, WHITE, x0 + 16 * u, y0 + 8 * u)
        put(img, "球員位置", 16 * u, GREY, x0 + pw - 14 * u, y0 + 12 * u, "rt")
        cx0, cy0, cw, ch = x0 + 26 * u, y0 + 62 * u, pw - 52 * u, ph - 112 * u
        put(img, "我隊", 18 * u, S["colors"]["P1"], cx0, y0 + 38 * u)
        put(img, "對手", 18 * u, S["colors"]["P3"], cx0 + cw, y0 + 38 * u, "rt")
        to = lambda p: (int(cx0 + cw * p[1] / COURT_L), int(cy0 + ch * (1 - p[0] / COURT_W)))
        line = (200, 190, 160)
        for a, b in court.LINES:
            cv2.line(img, to(a), to(b), line, 1, cv2.LINE_AA)
        nx = to((0, NET_Y))[0]
        for yy in range(int(cy0), int(cy0 + ch), max(2, int(8 * u))):
            cv2.line(img, (nx, yy), (nx, yy + max(1, int(4 * u))), (220, 220, 220), 1)
        for r in S["rallies"].itertuples():  # where points were won
            if r.end_frame <= f and getattr(r, "landing_in", False) and np.isfinite(r.landing_x):
                q = to((r.landing_x, r.landing_y))
                cv2.drawMarker(img, q, (60, 210, 255), cv2.MARKER_TILTED_CROSS, int(10 * u), max(1, int(2 * u)))
        for k, pid in enumerate(PIDS):
            p = S["pos"][f, k]
            if not np.isfinite(p[0]):
                continue
            q = to(np.clip(p, [-1, -1], [COURT_W + 1, COURT_L + 1]))
            col = S["colors"][pid]
            cv2.circle(img, q, int(7 * u), col, -1, cv2.LINE_AA)
            if k == 0:
                cv2.circle(img, q, int(14 * u), WHITE, max(1, int(2 * u)), cv2.LINE_AA)
            put(img, S["names"][pid], 17 * u, col, q[0] + 12 * u, q[1] - 6 * u, "lb")
        put(img, f"{S['names']['P1']} 追蹤中", 15 * u, GREY, x0 + 16 * u, y0 + ph - 10 * u, "lb")


def to_original(box, shuttle, M):
    """Boxes (F, 4, 4) and shuttle points (F, 2) found on the stabilised view -> the original
    frame's pixels (M[f]: original -> stabilised)."""
    box, shuttle = box.copy(), shuttle.copy()
    for f in range(min(len(box), len(M))):
        A = np.linalg.inv(M[f])
        if np.isfinite(shuttle[f, 0]):
            shuttle[f] = court.to_court(A, [shuttle[f]])[0]
        for k in range(box.shape[1]):
            b = box[f, k]
            if np.isfinite(b[0]):
                c = court.to_court(A, [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
                box[f, k] = [*c.min(0), *c.max(0)]
    return box, shuttle


def render(video, out_dir, progress=lambda x: None):
    from hud.pipeline import paths
    from hud.video import frames, probe
    from hud.pipeline import work
    from hud.stabilize import frame_maps
    p = paths(out_dir)
    wv, wc, _ = work(out_dir, video)
    H, calib = court.load(wc)
    w, h, fps, F = probe(video)
    M = frame_maps(wv) if wv != Path(video) else None  # original px -> stabilised px, per frame
    # output: long side <= MAX_OUT_SIDE, <= MAX_OUT_FPS; decoded and shrunk in hardware
    k = min(1.0, MAX_OUT_SIDE / max(w, h))
    W, Hh = int(round(w * k / 2) * 2), int(round(h * k / 2) * 2)
    kx, ky = W / w, Hh / h
    every = max(1, round(fps / MAX_OUT_FPS))  # 60 -> 30 fps; a 31.5 fps (VFR) clip keeps every frame
    S = load(out_dir, F, fps)
    if M is not None:
        from hud.stabilize import cuts_of
        M = smooth_maps(M, cuts_of(wv))
        S["box"], S["shuttle"] = to_original(S["box"], S["shuttle"], M)
    S["box"] = S["box"] * [kx, ky, kx, ky]
    S["shuttle"] = S["shuttle"] * [kx, ky]
    K = np.diag([1 / kx, 1 / ky, 1])
    hud = Hud(S, H @ K, (W, Hh), fps)
    if M is not None:
        hud.Ginvs = np.linalg.inv(H @ M[np.minimum(np.arange(F), len(M) - 1)] @ K)
    tmp = p["hud"].with_suffix(".tmp.mp4")
    enc = ["-c:v", "h264_videotoolbox", "-b:v", "10M"] if _has_videotoolbox() else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{Hh}",
           "-r", f"{fps / every:.4f}", "-i", "-", "-i", str(video), "-map", "0:v", "-map", "1:a?",
           *enc, "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(tmp)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for n, img in enumerate(frames(video, (W, Hh), every=every)):
            f = n * every
            if f >= F:
                break
            img = img.copy()  # ffmpeg's buffer is read-only
            hud.draw(img, f)
            ff.stdin.write(img.tobytes())
            if n % 30 == 0:
                progress(f / max(F, 1))
    finally:
        ff.stdin.close()
        ff.wait()
    if ff.returncode != 0:
        raise RuntimeError("ffmpeg failed to encode the HUD video")
    tmp.replace(p["hud"])
    return p["hud"]


@lru_cache(maxsize=1)
def _has_videotoolbox():
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        return "h264_videotoolbox" in out
    except FileNotFoundError:
        raise SystemExit("ffmpeg not found: brew install ffmpeg")
