"""Can the landing spot of a rally's last shot be predicted from its tracked flight when the
shuttle is lost before it lands? Simulated shots, real camera geometry (ACE: 0.7 m high,
broadcast: 6 m high), pixel noise, random loss before landing."""
import sys, json, time
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from hud.camera import Camera
from hud.hits import flight, DRAG_K
from hud.court import COURT_W, COURT_L, NET_Y
from hud import landing

calib = json.load(open(sys.argv[1]))
cam = Camera(np.array(calib["H"]), calib["frame_size"])
W, Hh = calib["frame_size"]
img_h = float(sys.argv[2]) if len(sys.argv) > 2 else min(W, Hh)
N = int(sys.argv[3]) if len(sys.argv) > 3 else 150
import os
rng = np.random.default_rng(int(os.environ.get("SEED", 1)))
KVAR = float(os.environ.get("KVAR", 0))      # true drag k = DRAG_K * U(1-KVAR, 1+KVAR)
CORNER_PX = float(os.environ.get("CORNER_PX", 0))  # calibration corner error (px) for the estimating camera
OUTL = float(os.environ.get("OUTL", 0))      # fraction of detections that are outliers
NOISE = float(os.environ.get("NOISE", 2.0))
from hud.court import homography, CORNERS_M
cam_true = cam
if CORNER_PX:
    import cv2
    G = np.linalg.inv(np.array(calib["H"]))
    cpx = cv2.perspectiveTransform(np.float64(CORNERS_M).reshape(-1, 1, 2), G).reshape(-1, 2)
    cam = Camera(homography(cpx + rng.normal(0, CORNER_PX, cpx.shape)), calib["frame_size"])
fps = 30.0


def shot():
    side = rng.integers(2)  # hitter side 0 near
    kind = rng.choice(["clear", "smash", "drop", "drive", "lift", "net"])
    yr = lambda a, b: (a + (b - a) * rng.random())
    if kind == "clear":
        y = yr(0.5, 3.0); z = yr(2.2, 2.8); v0 = yr(35, 60); el = yr(25, 50)
    elif kind == "smash":
        y = yr(1.0, 4.5); z = yr(2.3, 3.0); v0 = yr(40, 85); el = yr(-25, -8)
    elif kind == "drop":
        y = yr(0.5, 3.0); z = yr(2.0, 2.7); v0 = yr(12, 22); el = yr(-15, 15)
    elif kind == "drive":
        y = yr(3.5, 5.5); z = yr(1.0, 1.6); v0 = yr(25, 40); el = yr(-3, 8)
    elif kind == "lift":
        y = yr(4.5, 6.0); z = yr(0.3, 0.8); v0 = yr(25, 40); el = yr(35, 60)
    else:
        y = yr(5.3, 6.2); z = yr(0.6, 1.4); v0 = yr(4, 9); el = yr(10, 50)
    x = yr(0.5, COURT_W - 0.5)
    tx = yr(-0.5, COURT_W + 0.5)
    c0 = np.array([x, y if side == 0 else COURT_L - y, z])
    ty = COURT_L - 1 if side == 0 else 1
    az = np.arctan2(ty - c0[1], tx - c0[0]) + rng.normal(0, 0.15)
    e = np.radians(el)
    u = np.array([np.cos(e) * np.cos(az), np.cos(e) * np.sin(az), np.sin(e)])
    taus = np.arange(0, 4, 1 / 300)
    kk = DRAG_K * rng.uniform(1 - KVAR, 1 + KVAR)
    P = c0 + flight(v0, u, taus, k=kk)
    hit = np.flatnonzero(P[:, 2] <= 0)
    if not len(hit):
        return None
    i = hit[0]
    if i < 2:
        return None
    land = P[i - 1] + (P[i] - P[i - 1]) * (P[i - 1, 2] / (P[i - 1, 2] - P[i, 2]))
    t_land = taus[i]
    # net: crossing NET_Y below 1.55 m = into the net (lands on hitter's side): skip, rare
    cross = np.flatnonzero(np.diff(np.sign(P[:i, 1] - NET_Y)))
    if len(cross) and P[cross[0], 2] < 1.55:
        return None
    return kind, side, c0, v0, u, land[:2], t_land, kk


stats = {}
res = []
t0 = time.time()
n = 0
while n < N:
    s = shot()
    if s is None:
        continue
    kind, side, c0, v0, u, land, tl, kk = s
    # frames: hit at t=0 (frame phase random), observed until lost 0.1..0.6 s before landing
    phase = rng.random() / fps
    ts = np.arange(1, 200) / fps - phase
    ts = ts[(ts > 0) & (ts < tl)]
    lost = tl - rng.uniform(0.05, 0.6)
    ts = ts[ts < lost]
    if len(ts) < 5:
        continue
    X = c0 + flight(v0, u, ts, k=kk)
    px, depth = cam_true.project_many(X)
    sig = NOISE * img_h / 1080
    px = px + rng.normal(0, sig, px.shape)
    bad = rng.random(len(px)) < OUTL
    px[bad] += rng.uniform(-60, 60, (bad.sum(), 2))
    vis = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < Hh) & (rng.random(len(ts)) > 0.15)
    if vis.sum() < 5:
        continue
    n += 1
    c0_est = c0 + rng.normal(0, 0.25, 3) * [1, 1, 0.5]
    est = landing.predict(cam, c0_est, ts[vis], px[vis], fps=fps, img_h=img_h)
    in_true = 0 <= land[0] <= COURT_W and 0 <= land[1] <= COURT_L
    far_true = land[1] >= NET_Y
    if est is None:
        res.append((kind, None, None, None, None))
        continue
    xy, sd = est["xy"], est["sd"]
    err = np.hypot(*(xy - land))
    in_est = 0 <= xy[0] <= COURT_W and 0 <= xy[1] <= COURT_L
    res.append((kind, err, (xy[1] >= NET_Y) == far_true, in_est == in_true, sd))

print(f"{N} shots in {time.time() - t0:.0f}s, camera C={cam.C.round(2)}")
for k in ["all", "clear", "smash", "drop", "drive", "lift", "net"]:
    r = [x for x in res if k == "all" or x[0] == k]
    ok = [x for x in r if x[1] is not None]
    if not r:
        continue
    e = np.array([x[1] for x in ok]) if ok else np.array([np.nan])
    print(f"{k:6s} n={len(r):3d} accepted {len(ok) / len(r):.0%}  err median {np.median(e):.2f} m p80 {np.percentile(e, 80):.2f}"
          f"  side ok {np.mean([x[2] for x in ok]) if ok else float('nan'):.0%}  in/out ok {np.mean([x[3] for x in ok]) if ok else float('nan'):.0%}")
