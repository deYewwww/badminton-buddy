"""Synthetic speed test on a real match's camera (docs/speed_audit.md, step 3).

    uv run python scripts/synth_speed.py out/work2 [--n 1500] [--seed 0]

Shots with a known initial speed are flown (drag + gravity) from random contacts on both halves
to random points on the other side and projected through the camera recovered from that match's
calibration. Then the same three estimators hits.py uses are run on what that camera would see:

  fit     hits.fit_speed: the flight model fitted to the observed pixels, end point anchored
  flight  straight distance contact -> end over the flight time, undone for drag
  window  the 0.1 s fallback (point on the contact -> end line, or level flight)

What is made worse on purpose (the real clip's conditions, docs/speed_audit.md R4):
  pixel noise 2 px at 1080p; only OBS_P of frames seen, in runs; the contact estimate off by
  ~0.3 m sideways / 0.2 m up (hitter-vertical guess), the end point off by ~0.4 m; the contact
  anywhere inside its frame; the shuttle's real drag 0.17-0.28 /m while the fit assumes 0.22.

Writes <out>/synth_speed.csv (one row per shot) and prints the tables. This tests the algorithms
and what this camera can resolve. It says nothing about real-world accuracy: real shuttles
tumble, the detector misses more near players, and the drag model is the fit's own.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hud.analysis.hits import DRAG_K, MAX_KMH, fit_speed, flight, from_window_avg  # noqa: E402
from hud.court import COURT_L, COURT_W, NET_Y, load  # noqa: E402
from hud.court.camera import Camera  # noqa: E402

FPS = 30.0
PX_NOISE = 2.0          # px at 1080p (hits.FIT_PX)
OBS_P = 0.6             # share of in-play frames TrackNet saw on the user's clip (in play: 58-62 %)
CONTACT_SD = (0.3, 0.3, 0.2)  # m: x, y, z error of the hitter-vertical contact guess
END_SD = 0.4            # m: next contact / landing estimate (hits.FIT_END_M)
K_RANGE = (0.17, 0.28)  # 1/m: real drag spread (feather vs nylon, speed); the fit uses DRAG_K


def shot(rng):
    """-> contact (3,), target (3,), v0 m/s, k: a hitter on one half playing to the other."""
    near = rng.random() < 0.5
    y = rng.uniform(0.5, NET_Y - 0.6) if near else rng.uniform(NET_Y + 0.6, COURT_L - 0.5)
    c0 = np.array([rng.uniform(0.3, COURT_W - 0.3), y, rng.uniform(0.6, 2.9)])
    ty = rng.uniform(NET_Y + 0.5, COURT_L) if near else rng.uniform(0.0, NET_Y - 0.5)
    target = np.array([rng.uniform(0.0, COURT_W), ty, rng.uniform(0.0, 2.5)])
    return c0, target, rng.uniform(25, 300) / 3.6, rng.uniform(*K_RANGE)


def aim(c0, target, v0, k, rng):
    """Direction that sends the shot towards target, with a random lift: (u, flight times, path)."""
    d = target - c0
    az = np.arctan2(d[1], d[0])
    el = rng.uniform(-0.45, 0.8)  # -26..46 deg: smashes down to clears up
    u = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    taus = np.arange(0, 1.5, 1 / 300)
    P = c0 + flight(v0, u, taus, k=k, h=1 / 1000, max_steps=5000)  # truth: fine steps
    side = np.sign(target[1] - NET_Y)
    past = (P[:, 1] - target[1]) * side >= 0  # reached the receiver's depth
    low = P[:, 2] < 0.05
    stop = np.flatnonzero(past | low)
    i = stop[0] if len(stop) else len(taus) - 1
    return u, taus[:i + 1], P[:i + 1]


def view_angle(cam, c0, u):
    """Angle (deg) between the shot's start direction and the camera's line of sight to the contact:
    0 = straight away from / towards the camera (depth only), 90 = across the view."""
    ray = c0 - cam.C
    ray /= np.linalg.norm(ray)
    return float(np.degrees(np.arccos(abs(ray @ u))))


def observe(cam, c0, v0, u, k, t_end, size, rng):
    """Frames after the hit: (dt, px) the camera saw, contact at a random time inside frame 0."""
    t0 = rng.uniform(0, 1 / FPS)
    ft = np.arange(1, int(t_end * FPS) + 1) / FPS
    ft = ft[ft - t0 > 0]
    X = c0 + flight(v0, u, ft - t0, k=k, h=1 / 1000, max_steps=5000)
    px, depth = cam.project_many(X)
    seen = np.ones(len(ft), bool)
    if OBS_P < 1:  # runs (a seen frame is usually followed by a seen one), OBS_P of frames seen overall
        stay, back = 0.85, OBS_P * 0.15 / (1 - OBS_P)
        on = rng.random() < OBS_P
        for i in range(len(ft)):
            on = rng.random() < (stay if on else back)
            seen[i] = on
    w, h = size
    inside = (depth > 0.1) & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    keep = seen & inside
    noise = rng.normal(0, PX_NOISE * h / 1080, (keep.sum(), 2))
    return ft[keep], px[keep] + noise


def estimate(cam, c0e, end_e, t_end, dt, px, img_h):
    out = {}
    diag = {}
    v = fit_speed(cam, c0e, dt, px, (t_end, end_e), FPS, img_h, diag=diag) if len(dt) else None
    out["fit"] = None if v is None else v * 3.6
    out.update(fit_reason=diag.get("reason"), fit_rel_sd=diag.get("rel_sd"), fit_rms_px=diag.get("rms_px"))
    dist = np.linalg.norm(end_e - c0e)
    avg = dist / t_end * 3.6
    out["flight"] = from_window_avg(dist / t_end, t_end) * 3.6 if 0.1 < t_end < 2.0 and avg <= 100 else None
    js = np.flatnonzero(dt <= 0.1 + 1e-9)
    if len(js):
        j = js[-1]
        c1 = cam.on_line(px[j], c0e, end_e)
        c1 = cam.at_height(px[j], c0e[2]) if c1 is None else c1
        w = dt[j] + 0.5 / FPS
        v = from_window_avg(float(np.linalg.norm(c1 - c0e)) / w, w) * 3.6
        out["window"] = v if 0 < v <= MAX_KMH else None
    else:
        out["window"] = None
    return out


def run(out_dir, n, seed):
    H, calib = load(Path(out_dir) / "calib.json")
    size = calib["frame_size"]
    cam = Camera(H, size)
    img_h = min(size)
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        c0, target, v0, k = shot(rng)
        u, taus, P = aim(c0, target, v0, k, rng)
        t_end = float(taus[-1])
        if t_end < 0.15:
            continue
        dt, px = observe(cam, c0, v0, u, k, t_end, size, rng)
        c0e = c0 + rng.normal(0, CONTACT_SD)
        end_e = P[-1] + rng.normal(0, END_SD, 3)
        est = estimate(cam, c0e, end_e, t_end, dt, px, img_h)
        rows.append(dict(true_kmh=v0 * 3.6, k=k, view_deg=view_angle(cam, c0, u), elev_deg=np.degrees(np.arcsin(u[2])),
                         far=c0[1] > NET_Y, n_obs=len(dt), t_end=t_end,
                         dist_cam=float(np.linalg.norm(c0 - cam.C)), **est))
    df = pd.DataFrame(rows)
    for m in ("fit", "flight", "window"):
        df[f"{m}_err"] = df[m].astype(float) / df.true_kmh - 1
    df.to_csv(Path(out_dir) / "synth_speed.csv", index=False)
    return df, cam


def table(df, by, bins, labels):
    g = pd.cut(df[by], bins, labels=labels)
    rows = []
    for lab, d in df.groupby(g, observed=True):
        r = {by: lab, "n": len(d)}
        for m in ("fit", "flight", "window"):
            e = d[f"{m}_err"].dropna().abs()
            r[f"{m} 有值"] = f"{len(e) / len(d):.0%}"
            r[f"{m} 中位"] = f"{e.median():.0%}" if len(e) else "-"
            r[f"{m} P90"] = f"{e.quantile(0.9):.0%}" if len(e) else "-"
        rows.append(r)
    return pd.DataFrame(rows).to_string(index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["contact", "drag", "seen", "end"], nargs="*", default=[],
                    help="ablation: make these perfect (exact contact, drag = the fit's, every frame seen, exact end)")
    a = ap.parse_args()
    if "contact" in a.only:
        CONTACT_SD = (0.0, 0.0, 0.0)
    if "drag" in a.only:
        K_RANGE = (DRAG_K, DRAG_K)
    if "seen" in a.only:
        OBS_P = 1.0
    if "end" in a.only:
        END_SD = 0.0
    df, cam = run(a.out_dir, a.n, a.seed)
    print(f"camera C={np.round(cam.C, 2).tolist()} f={cam.f:.0f}px | {len(df)} shots | drag fit {DRAG_K}, true {K_RANGE}")
    print("\n== 全部（|誤差| 中位數 / 第 90 百分位）")
    print(table(df, "true_kmh", [0, 1000], ["all"]))
    print("\n== 依「出球方向和視線的夾角」")
    print(table(df, "view_deg", [0, 20, 40, 60, 90], ["0-20", "20-40", "40-60", "60-90"]))
    print("\n== 依看到的點數")
    print(table(df, "n_obs", [-1, 3, 7, 15, 99], ["0-3", "4-7", "8-15", "16+"]))
    print("\n== 依遠近（擊球者在哪半場）")
    print(table(df.assign(side=df.far.astype(int)), "side", [-1, 0.5, 2], ["近", "遠"]))
    print("\n== 依真實初速")
    print(table(df, "true_kmh", [0, 80, 150, 220, 400], ["<80", "80-150", "150-220", "220+"]))
    ok = df.fit.notna()
    print("\n== fit 的相對標準誤能不能預測誤差？")
    print(table(df[ok], "fit_rel_sd", [0, 0.05, 0.1, 0.2, 0.4], ["<5%", "5-10%", "10-20%", "20-40%"]))
    print("\nfit 失敗原因:", json.dumps(df[~ok].fit_reason.value_counts(dropna=False).to_dict(), ensure_ascii=False))
    print("寫入", Path(a.out_dir) / "synth_speed.csv")
