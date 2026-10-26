"""Stabilised analysis, smoothed positions, stamina, landing prediction.
Run: uv run python tests/test_phase3.py"""
import numpy as np
import pandas as pd

from hud.camera import Camera
from hud.court import COURT_L, COURT_W, CORNERS_M, NET_Y, homography
from hud.hits import DRAG_K, flight


def low_camera():
    """A phone 0.7 m high just outside the near corner (like the user's clip), as (cam, H, proj)."""
    C = np.array([6.8, -1.4, 0.7])
    z = np.array([3.05, 6.7, 0.0]) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    R = np.array([x, np.cross(z, x), z])
    f = 530.0
    proj = lambda X: np.array([360, 640]) + f * (R @ (np.asarray(X) - C))[:2] / (R @ (np.asarray(X) - C))[2]
    H = homography([proj([*c, 0]) for c in CORNERS_M])
    return Camera(H, (720, 1280)), H, proj


def test_smoothing_weighs_far_jitter():
    from hud.smooth import smooth_players
    cam, H, proj = low_camera()
    rng = np.random.default_rng(0)
    t = np.arange(300) / 30
    truth = np.c_[1.5 + 3 * t / t[-1], np.full_like(t, 11.0)]  # far player walking sideways
    rows = []
    for i, (x, y) in enumerate(truth):
        foot, head = proj([x, y, 0]), proj([x, y, 1.7])
        px = foot + rng.normal(0, 2.0, 2)
        q = np.r_[px, 1] @ H.T
        rows.append(dict(frame=i, timestamp=t[i], player_id="P3", observed=True, x_m=q[0] / q[2], y_m=q[1] / q[2],
                         foot_px_x=px[0], foot_px_y=px[1], box_y1=head[1], box_y2=foot[1], app_top_L=1.0))
    df = pd.DataFrame(rows)
    sm = smooth_players(df, H)
    err = lambda d: np.sqrt(np.mean(np.sum((d[["x_m", "y_m"]].to_numpy() - truth) ** 2, 1)))
    dist = lambda d: np.hypot(*np.diff(d[["x_m", "y_m"]].to_numpy(), axis=0).T).sum()
    assert err(sm) < 0.4 * err(df), (err(df), err(sm))
    # true distance 3 m; raw jitter (1 px ~ 0.4 m at the far baseline from 0.7 m high) adds ~300 m
    assert dist(sm) < 6.0 and dist(df) > 30 * dist(sm), (dist(df), dist(sm))
    assert {"x_raw", "y_raw"} <= set(sm)


def test_stamina_drains_by_stroke_and_recovers_partly():
    from hud import config
    from hud.render import stamina_track, stroke_kinds
    st = config.load()["stamina"]
    F, fps = 300, 30
    step, speed = np.zeros((F, 4)), np.zeros((F, 4))
    rally_of = np.r_[np.zeros(150, int), np.full(150, -1)]
    shots = pd.DataFrame(dict(frame=[10, 40, 70], player_id=["P1", "P3", "P2"], team=["us", "opp", "us"],
                              shot_type=["smash", "clear", "net"], rally_id=[1, 1, 1]))
    assert [k for _, _, k in stroke_kinds(shots)] == ["smash", "defence", "net"]  # P3 answered a smash
    out, costs = stamina_track(step, speed, rally_of, shots, fps, st)
    c = st["shot_cost"]
    assert np.isclose(out[149, 0], 100 - c["smash"]) and np.isclose(out[149, 2], 100 - c["defence"])
    assert out[149, 3] == 100 and len(costs) == 3
    # rest: recovers, but not all the way (fatigue ceiling)
    assert out[149, 0] < out[-1, 0] < 100


def test_point_from_landing_side():
    from hud.rallies import decisive, in_target
    assert in_target(3.0, 10.0, "us") and not in_target(3.0, 5.0, "us")  # near team into the net: their loss
    assert in_target(3.0, 5.0, "opp") and not in_target(3.0, 14.0, "us")
    assert decisive(3.0, 10.0, "us", 0.3, 1.0) and not decisive(3.0, NET_Y + 0.2, "us", 0.3, 1.0)


def test_landing_predicted_from_lost_flight():
    from hud.landing import predict
    cam, H, proj = low_camera()
    c0 = np.array([2.0, 2.0, 2.5])                      # clear from the near rear court
    u = np.array([0.05, 0.75, 0.66]); u /= np.linalg.norm(u)
    taus = np.arange(0, 4, 1 / 300)
    P = c0 + flight(45.0, u, taus)
    i = np.flatnonzero(P[:, 2] <= 0)[0]
    land, tl = P[i, :2], taus[i]
    ts = np.arange(1, 200) / 30
    ts = ts[ts < tl - 0.4]                             # lost 0.4 s before it lands
    px = np.array([proj(X) for X in c0 + flight(45.0, u, ts)]) + np.random.default_rng(1).normal(0, 1.3, (len(ts), 2))
    est = predict(cam, c0 + [0.2, -0.2, 0.1], ts, px, img_h=720)
    assert est is not None and np.hypot(*(est["xy"] - land)) < 0.5, (est, land)
    assert est["xy"][1] > NET_Y and 0 <= est["xy"][0] <= COURT_W and est["xy"][1] <= COURT_L + 1


def test_landing_settles_despite_tracker_wobble():
    from hud.hits import landed_at
    rng = np.random.default_rng(0)
    fall = np.c_[np.full(10, 590.0), np.linspace(1000, 1180, 10)]
    lying = np.array([590.0, 1180.0]) + rng.uniform(-4, 4, (12, 2))  # 720p TrackNet wobble
    p = np.r_[fall, lying]
    p[14] = np.nan                                                    # and a missed frame
    assert landed_at(p, 0, len(p), still_px=8, still_n=9) in (9, 10)  # the fall reaches 1180 at frame 9
    assert landed_at(p, 0, len(p), still_px=3, still_n=9) == len(p)  # the old 3 px never settled


def test_work_files_default_to_original(tmp=None):
    import tempfile
    from pathlib import Path
    from hud.pipeline import paths, work
    d = Path(tempfile.mkdtemp())
    v, c, m = work(d, "clip.mp4")
    assert v == Path("clip.mp4") and c == paths(d)["calib"] and m == paths(d)["marks"]
    paths(d)["stab"].write_bytes(b"")
    paths(d)["stab_calib"].write_text("{}")
    assert work(d, "clip.mp4")[:2] == (paths(d)["stab"], paths(d)["stab_calib"])


def test_frame_maps_hold_last_good(tmp=None):
    import json
    import tempfile
    from pathlib import Path
    from hud.stabilize import frame_maps
    d = Path(tempfile.mkdtemp())
    A = np.eye(3); A[0, 2] = 5
    (d / "s.json").write_text(json.dumps(dict(H=[None, A.tolist(), None])))
    M = frame_maps(d / "s.mp4")
    assert np.allclose(M[0], np.eye(3)) and np.allclose(M[1], A) and np.allclose(M[2], A)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
