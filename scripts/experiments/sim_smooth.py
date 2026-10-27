"""Synthetic check of hud.tracking.smooth: footwork paths -> pixels (ACE camera) + noise + jumps."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from hud.court.camera import Camera
from hud.tracking import smooth
from hud.court import COURT_W, COURT_L, NET_Y

calib = json.load(open(sys.argv[1]))  # any calib.json (a stabilised one: calib_stab.json)
H = np.array(calib["H"]); cam = Camera(H, calib["frame_size"])
rng = np.random.default_rng(0)
ACCS = [float(a) for a in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["8"])]
NOISE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.015
AMAX = float(sys.argv[4]) if len(sys.argv) > 4 else 6
fps, T = 30, 60
t = np.arange(T * fps) / fps

def path(side):
    # random targets, move with bang-bang accel limited to 6 m/s^2, vmax 4.5
    p = np.array([COURT_W / 2, NET_Y * (0.5 + side)]); v = np.zeros(2); out = []
    tgt = p.copy()
    for i in range(len(t)):
        if np.hypot(*(tgt - p)) < 0.2 or rng.random() < 0.01:
            y0, y1 = (0.5, NET_Y - 0.5) if side == 0 else (NET_Y + 0.5, COURT_L - 0.3)
            tgt = np.array([rng.uniform(0.4, COURT_W - 0.4), rng.uniform(y0, y1)])
        d = tgt - p; dv = d / (np.hypot(*d) + 1e-9) * min(4.5, np.hypot(*d) * 3) - v
        a = dv * fps; a *= min(1, AMAX / (np.hypot(*a) + 1e-9))
        v = v + a / fps; p = p + v / fps; out.append(p.copy())
    return np.array(out)

rows = []
truth = {}
stride = 3  # pose model at 10 fps
for k, (pid, side) in enumerate([("P1", 0), ("P2", 0), ("P3", 1), ("P4", 1)]):
    P = path(side); truth[pid] = P
    for i in range(len(t)):
        meas = i % stride == 0
        X0 = np.r_[P[i], 0.0]
        foot = cam.project(X0); head = cam.project(np.r_[P[i], 1.7])
        h = abs(foot[1] - head[1])
        if meas:
            px = foot + rng.normal(0, max(NOISE * h, 1.5), 2)
            if rng.random() < 0.04:  # jump: feet 0.3 m up
                px = cam.project(np.r_[P[i], 0.3])
            z = (np.c_[px[None], [[1]]] @ H.T); z = z[0, :2] / z[0, 2]
            rows.append(dict(frame=i, timestamp=t[i], player_id=pid, observed=True, x_m=z[0], y_m=z[1],
                             foot_px_x=px[0], foot_px_y=px[1], box_y1=head[1], box_y2=foot[1], app_top_L=1.0))
df = pd.DataFrame(rows)
# old pipeline: interpolate to every frame, rolling mean 0.2 s (render.load)
def old(g):
    s = g.set_index("frame")[["x_m", "y_m"]].reindex(range(len(t))).interpolate(limit_area="inside")
    return s.rolling(6, center=True, min_periods=1).mean().to_numpy()
# new: fill skipped rows like fill_skipped (interpolated, no appearance) then smooth
full = []
for pid, g in df.groupby("player_id"):
    s = g.set_index("frame").reindex(range(len(t)))
    num = ["x_m", "y_m", "foot_px_x", "foot_px_y", "box_y1", "box_y2", "timestamp"]
    s["timestamp"] = t
    s[num] = s[num].interpolate(limit_area="inside")
    s["observed"] = True; s["player_id"] = pid
    full.append(s.reset_index(names="frame"))
full = pd.concat(full)
for acc in ACCS:
    sm = smooth.smooth_players(full, H, accel=acc)
    print(f"accel {acc}")
    for pid in truth:
        P = truth[pid]
        g = sm[sm.player_id == pid].set_index("frame").reindex(range(len(t)))
        new = g[["x_m", "y_m"]].to_numpy(); o = old(df[df.player_id == pid])
        ok = np.isfinite(new[:, 0]) & np.isfinite(o[:, 0])
        err = lambda q: np.sqrt(np.mean(np.sum((q[ok] - P[ok]) ** 2, 1)))
        dist = lambda q: np.nansum(np.hypot(*np.diff(q[ok], axis=0).T))
        p95 = lambda q: np.percentile(np.hypot(*(q[ok] - P[ok]).T), 95)
        print(f"  {pid}: rmse old {err(o):.2f} new {err(new):.2f} m | p95 old {p95(o):.2f} new {p95(new):.2f} | dist true {dist(P):.0f} old {dist(o):.0f} new {dist(new):.0f}")
