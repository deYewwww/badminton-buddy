"""Court-aware identity tracking for 4 doubles players.

Players never cross the net, so each half has exactly 2 identity slots. Every frame,
the detections on a half are matched to that half's slots with the Hungarian algorithm.
The cost is the distance (m) from the slot's constant-velocity prediction plus a small
penalty for torso-colour mismatch (partners usually wear the same kit, so colour only
breaks near-ties). ByteTrack IDs are deliberately NOT in the cost: they swap exactly when
partners cross, which is the case this tracker exists for. They are kept as a column.
Errors left over go in a swaps CSV: `frame,a,b` swaps a<->b from that frame on.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from hud.court import COURT_L, COURT_W, NET_Y, load, to_court
from hud.video import frame_times, frames, probe

# (player_id, team, side): side 0 = near half (camera side, y < NET_Y)
SLOTS = [("P1", "us", 0), ("P2", "us", 0), ("P3", "opp", 1), ("P4", "opp", 1)]
SIDE_MARGIN_M = 0.6  # keep people whose feet are at most this far outside a side line (umpire, line
END_MARGIN_M = 1.3   # judges stand further out) ... or this far behind a baseline (players retreat for clears)
BASE_GATE_M = 0.5   # max jump between consecutive frames (foot-point jitter)...
MAX_SPEED = 8.0     # ...plus this many m/s for every second a slot has been unseen
MIN_SEP_M = 1.0     # a new player must stand at least this far from the one already tracked
DET_FPS = 10        # pose model runs per second of video (see run)
MAX_COAST_S = 1.0   # keep emitting the last ground position this long after losing a player
                    # (occluded by partner, or mid-jump: an airborne foot point projects metres too deep)
HIST_W_M = 0.5


@dataclass
class Slot:
    pid: str
    team: str
    side: int
    pos: np.ndarray | None = None
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    t: float = 0.0
    hist: np.ndarray | None = None

    def predict(self, t):
        return self.pos + self.vel * min(t - self.t, 0.1)  # players don't glide: barely extrapolate


class IdentityTracker:
    def __init__(self):
        self.slots = [Slot(*s) for s in SLOTS]

    def update(self, t, dets):
        """dets: list of dict(xy=court metres, track_id, hist).
        Returns {player_id: (xy, det_index)}; det_index is None while coasting."""
        out = {}
        for side in (0, 1):
            slots = [s for s in self.slots if s.side == side]
            # ponytail: side decided by feet y alone; a jumping near-net player can project
            # past the net. Add hysteresis around NET_Y if this shows up in real footage.
            cand = [i for i, d in enumerate(dets) if (d["xy"][1] >= NET_Y) == side]
            live = [r for r, s in enumerate(slots) if s.pos is not None]
            matched = {}
            if cand and live:
                cost = np.array([[self._cost(slots[r], t, dets[i]) for i in cand] for r in live])
                for a, c in zip(*linear_sum_assignment(cost)):
                    s, d = slots[live[a]], dets[cand[c]]
                    if np.hypot(*(d["xy"] - s.predict(t))) <= BASE_GATE_M + MAX_SPEED * (t - s.t):
                        matched[live[a]] = cand[c]
            # start slots as soon as someone is seen, one at a time: at the start one partner is
            # often hidden behind the other, and waiting for both lost the first second of play
            free = [r for r, s in enumerate(slots) if s.pos is None]
            if free:
                centre = np.array([COURT_W / 2, NET_Y * (0.5 + side)])
                taken = [slots[r].predict(t) for r in live]
                spare = [i for i in cand if i not in matched.values()
                         and all(np.hypot(*(dets[i]["xy"] - q)) > MIN_SEP_M for q in taken)]
                spare = sorted(spare, key=lambda i: np.hypot(*(dets[i]["xy"] - centre)))[:len(free)]
                spare.sort(key=lambda i: dets[i]["xy"][0])  # both new at once: left gets the lower id
                for r, i in zip(free, spare):
                    matched[r] = i
            for r, s in enumerate(slots):
                if r in matched:
                    self._set(s, t, dets[matched[r]])
                    out[s.pid] = (s.pos.copy(), matched[r])
                elif s.pos is not None and t - s.t <= MAX_COAST_S:
                    out[s.pid] = (s.predict(t), None)
        return out

    def pin(self, pid, t, xy):
        """The user clicked this player's feet at time t (court metres). Put the slot there.
        A partner the tracker had at that spot is the mix-up being fixed: it takes over the
        slot's old position and colour history (a swap); otherwise the colour history is
        dropped and re-learnt, since it may have been learnt from the wrong person."""
        s = next(x for x in self.slots if x.pid == pid)
        xy = np.asarray(xy, float)
        if (xy[1] >= NET_Y) != s.side:
            print(f"mark {pid} at {t:.1f}s is on the wrong half of the court, ignored")
            return False
        other = next((o for o in self.slots if o.side == s.side and o is not s and o.pos is not None
                      and np.hypot(*(o.pos - xy)) < MIN_SEP_M), None)
        if other is not None:
            other.pos, other.t, other.hist, s.hist = s.pos, s.t, s.hist, other.hist
            other.vel = np.zeros(2)
        else:
            s.hist = None
        # a pinned slot accepts a detection up to ~1.3 m away (click error on far players)
        s.pos, s.vel, s.t = xy, np.zeros(2), t - 0.1
        return True

    @staticmethod
    def _cost(s, t, d):
        c = np.hypot(*(d["xy"] - s.predict(t)))
        if s.hist is not None and d.get("hist") is not None:
            c += HIST_W_M * (1 - cv2.compareHist(s.hist, d["hist"], cv2.HISTCMP_CORREL))
        return c

    @staticmethod
    def _set(s, t, d):
        dt = t - s.t
        if s.pos is not None and 0 < dt <= 0.3:
            v = 0.6 * s.vel + 0.4 * (d["xy"] - s.pos) / dt
            s.vel = v * min(1.0, MAX_SPEED / (np.hypot(*v) + 1e-9))
        else:
            s.vel = np.zeros(2)
        s.pos, s.t = np.asarray(d["xy"], float), t
        h = d.get("hist")
        if h is not None:
            s.hist = h if s.hist is None else (0.8 * s.hist + 0.2 * h).astype(np.float32)


def torso_hist(frame, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    crop = frame[int(y1 + 0.2 * h):int(y1 + 0.6 * h), int(x1 + 0.2 * w):int(x2 - 0.2 * w)]
    if crop.size == 0:
        return None
    hist = cv2.calcHist([cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, None).astype(np.float32)


def appearance(frame, box):
    """Mean Lab colour of the upper body and of the legs: [top L, a, b, legs L, a, b]."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    lab = []
    for a, b in ((0.2, 0.5), (0.6, 0.85)):
        crop = frame[int(y1 + a * h):int(y1 + b * h), int(x1 + 0.25 * w):int(x2 - 0.25 * w)]
        lab += list(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)) if crop.size else [np.nan] * 3
    return np.round(lab, 1)


APP_COLS = ["app_top_L", "app_top_a", "app_top_b", "app_leg_L", "app_leg_a", "app_leg_b"]


def fill_skipped(df, times, stride):
    """Rows for frames the detector skipped: positions and boxes interpolated between the
    neighbouring detections, 'observed' only if both neighbours were; wrists/colours stay NaN."""
    if stride == 1 or df.empty:
        return df
    frames = np.arange(len(times))
    out = []
    for pid, g in df.groupby("player_id"):
        g = g.set_index("frame").reindex(frames)
        num = ["x_m", "y_m", "box_x1", "box_y1", "box_x2", "box_y2", "foot_px_x", "foot_px_y"]
        g[num] = g[num].interpolate(limit=stride - 1, limit_area="inside")
        obs = g.observed.astype(float)
        g["observed"] = (obs.ffill(limit=stride - 1) == 1) & (obs.bfill(limit=stride - 1) == 1)
        g["player_id"], g["team"] = pid, g.team.ffill().bfill()
        g["bytetrack_id"] = g.bytetrack_id.ffill().fillna(-1).astype(int)
        g["timestamp"] = np.round(times, 4)
        out.append(g[g.x_m.notna()].reset_index(names="frame"))
    return pd.concat(out, ignore_index=True).sort_values(["frame", "player_id"], ignore_index=True)


def apply_swaps(df, path):
    team = dict((p, tm) for p, tm, _ in SLOTS)
    for r in pd.read_csv(path).itertuples():
        if team.get(r.a) != team.get(r.b) or r.a not in team:
            raise SystemExit(f"Bad swap {r.a}<->{r.b}: both must be players of the same team.")
        m = df.frame >= r.frame
        a, b = m & (df.player_id == r.a), m & (df.player_id == r.b)
        df.loc[a, "player_id"], df.loc[b, "player_id"] = r.b, r.a
    return df


def speeds(players, times, half=0.25):
    """(players, frames) speed in m/s from the displacement over +-half s (foot-point jitter
    averages out); NaN where a player isn't tracked. times: timestamp per frame."""
    t = np.asarray(times, float)
    n = len(t)
    a = np.clip(np.searchsorted(t, t - half), 0, n - 1)
    b = np.clip(np.searchsorted(t, t + half), 0, n - 1)
    dt = np.maximum(t[b] - t[a], 1e-6)
    out = []
    for _, g in players.groupby("player_id"):
        g = g[(g.frame >= 0) & (g.frame < n)]
        pos = np.full((n, 2), np.nan)
        pos[g.frame.to_numpy(int)] = g[["x_m", "y_m"]].to_numpy(float)
        out.append(np.hypot(*(pos[b] - pos[a]).T) / dt)
    return np.array(out).reshape(-1, n)


def load_marks(path, times, H):
    """marks.json (editor: player feet clicked at a time) -> list of (frame, pid, court xy)."""
    if not path or not Path(path).exists():
        return []
    import json
    raw = json.loads(Path(path).read_text())
    t = np.asarray(times)
    out = []
    for m in raw:
        f = int(np.clip(np.searchsorted(t, float(m["t"])), 0, len(t) - 1))
        out.append((f, m["pid"], to_court(H, [[m["x"], m["y"]]])[0]))
    return sorted(out, key=lambda r: r[0])


def apply_marks(df, marks):
    """After re-identification, the user's clicks are the final word: at each mark, if the
    partner (not the marked player) stands nearest the clicked spot, swap the two from that
    frame up to the next mark for this pair."""
    if df.empty or not marks:
        return df
    df = df.copy()
    partner = {"P1": "P2", "P2": "P1", "P3": "P4", "P4": "P3"}
    for k, (f, pid, xy) in enumerate(marks):
        pair = {pid, partner[pid]}
        nxt = next((g for g, q, _ in marks[k + 1:] if q in pair and g > f), None)
        here = df[(df.frame == f) & df.player_id.isin(pair)]
        if here.empty:
            continue
        d = np.hypot(here.x_m - xy[0], here.y_m - xy[1])
        near = here.player_id.iat[int(np.argmin(d.to_numpy()))]
        if near != pid and d.min() < MIN_SEP_M:
            m = df.player_id.isin(pair) & (df.frame >= f) & ((df.frame < nxt) if nxt is not None else True)
            df.loc[m, "player_id"] = df.loc[m, "player_id"].map({pid: partner[pid], partner[pid]: pid})
            print(f"mark at frame {f}: {near} was {pid}, swapped up to frame {nxt}")
    return df


def run(video, calib, out, model=None, imgsz=960, swaps=None, max_frames=None, det_fps=DET_FPS, progress=None,
        marks=None):
    """det_fps: run the (slow) pose model this many times per second and interpolate between;
    60 fps footage at 30 halves the time, at 15 ('fast') quarters it.
    marks: marks.json from the editor (player feet clicked at some times), see pin/apply_marks."""
    from hud.detect import PlayerDetector, far_region

    w, h, fps, _ = probe(video)
    size = [w, h]
    times = frame_times(video)  # real timestamps: iPhone 60fps clips are often VFR
    total = len(times)
    stride = max(1, round(fps / det_fps))
    H, _ = load(calib, size)
    # YOLO sees at most 1920 px wide frames (imgsz 960 anyway); the decoder shrinks them in
    # hardware, so a 3K clip costs no more CPU than a 1080p one. Pixels are mapped back below.
    k = min(1.0, 1920 / w)
    sw, sh = int(round(w * k / 2) * 2), int(round(h * k / 2) * 2)
    kx, ky = w / sw, h / sh
    detect, tracker = PlayerDetector(model, imgsz), IdentityTracker()
    far = far_region(H, size)
    detect.far_box = None if far is None else (int(far[0] / kx), int(far[1] / ky), int(far[2] / kx), int(far[3] / ky))
    team = dict((p, tm) for p, tm, _ in SLOTS)
    marks = load_marks(marks, times, H)
    pending = list(marks)  # pinned at the first detection frame at or after each mark
    rows, failed, f = [], 0, 0
    for n, frame in enumerate(frames(video, (sw, sh), every=stride)):
        f = n * stride
        if f >= total or (max_frames is not None and f >= max_frames):
            break
        t = times[f]
        try:
            boxes, ids, feet_px, feet_ok, wrists = detect(frame)
        except Exception as e:  # one bad frame must not kill a whole match; slots coast over it
            failed += 1
            print(f"frame {f}: detection failed ({e})")
            boxes, ids, feet_px, feet_ok, wrists = (np.zeros((0, 4)), np.zeros(0, int), np.zeros((0, 2)),
                                                    np.zeros(0, bool), np.zeros((0, 2, 2)))
        hists = [torso_hist(frame, b) for b in boxes]
        apps = [appearance(frame, b) for b in boxes]
        boxes, feet_px, wrists = boxes * [kx, ky, kx, ky], feet_px * [kx, ky], wrists * [kx, ky]
        feet = to_court(H, feet_px) if len(boxes) else np.zeros((0, 2))
        # feet hidden (ankles not visible) -> position unknown; the slot holds its last ground position
        inside = feet_ok & ((feet[:, 0] > -SIDE_MARGIN_M) & (feet[:, 0] < COURT_W + SIDE_MARGIN_M)
                  & (feet[:, 1] > -END_MARGIN_M) & (feet[:, 1] < COURT_L + END_MARGIN_M))
        dets = [dict(xy=feet[i], track_id=int(ids[i]), hist=hists[i], box=boxes[i], foot=feet_px[i],
                     wrists=wrists[i], app=apps[i])
                for i in np.flatnonzero(inside)]
        while pending and pending[0][0] <= f:
            _, pid, xy = pending.pop(0)
            tracker.pin(pid, t, xy)
        for pid, (xy, i) in tracker.update(t, dets).items():
            box = dets[i]["box"] if i is not None else [np.nan] * 4
            foot = dets[i]["foot"] if i is not None else [np.nan] * 2
            wr = dets[i]["wrists"] if i is not None else np.full((2, 2), np.nan)
            app = dets[i]["app"] if i is not None else [np.nan] * 6
            rows.append(dict(frame=f, timestamp=round(t, 4), player_id=pid, team=team[pid],
                             x_m=round(float(xy[0]), 3), y_m=round(float(xy[1]), 3), observed=i is not None,
                             bytetrack_id=dets[i]["track_id"] if i is not None else -1,
                             box_x1=box[0], box_y1=box[1], box_x2=box[2], box_y2=box[3],
                             foot_px_x=foot[0], foot_px_y=foot[1],
                             wrist_l_x=wr[0, 0], wrist_l_y=wr[0, 1], wrist_r_x=wr[1, 0], wrist_r_y=wr[1, 1],
                             **dict(zip(APP_COLS, app))))
        if n % 20 == 0:
            print(f"{f}/{total} frames")
            if progress:
                progress(f / max(total, 1))
    if not rows:
        raise SystemExit(f"No frames could be read from {video}")
    df = fill_skipped(pd.DataFrame(rows), times, stride)
    from hud.camera import Camera
    from hud.reid import fix_identities
    df = fix_identities(df, Camera(H, size))
    df = apply_marks(df, marks)
    if swaps:  # manual fixes go last, on top of the automatic ones
        df = apply_swaps(df, swaps)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)  # overwrite: rerunning the same video never duplicates rows
    obs = df.groupby("player_id")["observed"].mean().round(3).to_dict() if len(df) else {}
    print(f"Wrote {len(df)} rows ({f} frames, {failed} failed) to {out}. Observed ratio: {obs}")
    return df
