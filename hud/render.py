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
    # smooth foot jitter (0.2 s), then fill short gaps so rings don't blink
    w = max(1, int(0.2 * fps))
    for k in range(4):
        for d in range(2):
            s = pd.Series(pos[:, k, d])
            pos[:, k, d] = s.interpolate(limit=int(fps), limit_area="inside").rolling(w, center=True, min_periods=1).mean()
        for d in range(4):
            box[:, k, d] = pd.Series(box[:, k, d]).interpolate(limit=int(fps // 2), limit_area="inside").to_numpy()
    shots = pd.read_parquet(p["shots"]) if p["shots"].exists() else pd.DataFrame()
    rallies = pd.read_parquet(p["rallies"]) if p["rallies"].exists() else pd.DataFrame()
    sh = pd.read_parquet(p["shuttle"]) if p["shuttle"].exists() else None
    shuttle = np.full((F, 2), np.nan)
    if sh is not None:
        g = sh[sh.visible & (sh.frame < F)]
        shuttle[g.frame] = g[["x_px", "y_px"]].to_numpy()

    step = np.r_[np.zeros((1, 4)), np.nan_to_num(np.hypot(*np.diff(pos, axis=0).transpose(2, 0, 1)))]
    step = np.minimum(step, 12.0 / fps)  # > 12 m/s is an identity jump, not running
    speed = np.stack([np.convolve(step[:, k], np.ones(w) / w, "same") * fps for k in range(4)], 1)
    rally_of = np.full(F, -1)
    for i, r in enumerate(rallies.itertuples()):
        rally_of[int(r.start_frame):min(int(r.end_frame) + 1, F)] = i
    st, sc = cfg["stamina"], cfg["move"]
    stamina = np.zeros((F, 4))
    cur = np.full(4, float(st["start"]))
    for f in range(F):
        if rally_of[f] >= 0:
            cur -= st["per_meter"] * step[f] + st["per_sprint_s"] / fps * (speed[f] > st["sprint_speed_mps"])
        else:
            cur += st.get("recover_per_s", 0) / fps
        cur = np.clip(cur, 0, st["start"])
        stamina[f] = cur
    win = max(1, int(sc["window_s"] * fps))
    move = np.stack([np.convolve(step[:, k], np.ones(win), "full")[:F] for k in range(4)], 1) / sc["full_scale_m"]
    return dict(pos=pos, box=box, shots=shots, rallies=rallies, shuttle=shuttle, speed=speed, stamina=stamina,
                move=np.clip(move, 0, 1), rally_of=rally_of, names=names(out_dir), cfg=cfg,
                colors={pid: hex_bgr(cfg["colors"][pid]) for pid in PIDS})


# --- drawing ----------------------------------------------------------------------------

class Hud:
    def __init__(self, S, H, size, fps):
        self.S, self.fps = S, fps
        self.Ginv = np.linalg.inv(H)
        self.W, self.Hh = size
        self.u = min(size) / 1080  # layout unit: 1 px on a 1080-wide portrait frame
        sc = S["cfg"]["score"]
        self.start_score = (sc["us"], sc["opp"])

    def draw(self, img, f):
        S, u = self.S, self.u
        self.floor(img, f)
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
        sp = S["speed"][f, k]
        recent = S["speed"][max(0, f - int(0.6 * fps)):f + 1, k]
        if sp > MOVE_MS:
            dest = S["pos"][min(f + int(0.5 * fps), len(S["pos"]) - 1), k]
            label = "MOVE"
        elif len(recent) and recent.max() > MOVE_MS and sp < STOP_MS:
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
                continue
            items.append([(b[0] + b[2]) / 2 - tw / 2, b[1] - th - 40 * u, k, pid, (b[0] + b[2]) / 2, b[1]])
        items.sort(key=lambda t: t[1])
        placed = []
        for it in items:
            x, y = min(max(it[0], 4), self.W - tw - 4), max(it[1], 120 * u)
            for px, py in placed:
                if abs(px - x) < tw and abs(py - y) < th + 6 * u:
                    y = py - th - 6 * u
            placed.append((x, y))
            k, pid, hx, hy = it[2:]
            col = S["colors"][pid]
            cv2.line(img, (int(x + tw / 2), int(y + th)), (int(hx), int(hy - 6 * u)), col, max(1, int(2 * u)), cv2.LINE_AA)
            panel(img, x, y, x + tw, y + th, col)
            stam = S["stamina"][f, k]
            put(img, S["names"][pid], 26 * u, col, x + 14 * u, y + 8 * u)
            put(img, f"{stam:.0f}", 26 * u, RED if stam < 15 else WHITE, x + tw - 12 * u, y + 8 * u, "rt")
            bar(img, x + 14 * u, y + th - 16 * u, tw - 28 * u, 7 * u, stam / 100, col)

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
        p = S["shuttle"][f] if np.isfinite(S["shuttle"][f, 0]) else S["shuttle"][int(shot.frame)]
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
            col = S["colors"]["P1" if team == "us" else "P3"]
            cx, cy = self.W / 2, 175 * u
            panel(img, cx - 190 * u, cy, cx + 190 * u, cy + 70 * u, col, alpha=0.85)
            put(img, text, 36 * u, col, cx, cy + 35 * u, "cm")
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


def render(video, out_dir, progress=lambda x: None):
    from hud.pipeline import paths
    from hud.video import frames, probe
    p = paths(out_dir)
    H, calib = court.load(p["calib"])
    w, h, fps, F = probe(video)
    # output: long side <= MAX_OUT_SIDE, <= MAX_OUT_FPS; decoded and shrunk in hardware
    k = min(1.0, MAX_OUT_SIDE / max(w, h))
    W, Hh = int(round(w * k / 2) * 2), int(round(h * k / 2) * 2)
    kx, ky = W / w, Hh / h
    every = max(1, round(fps / MAX_OUT_FPS))  # 60 -> 30 fps; a 31.5 fps (VFR) clip keeps every frame
    S = load(out_dir, F, fps)
    S["box"] = S["box"] * [kx, ky, kx, ky]
    S["shuttle"] = S["shuttle"] * [kx, ky]
    hud = Hud(S, H @ np.diag([1 / kx, 1 / ky, 1]), (W, Hh), fps)
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
