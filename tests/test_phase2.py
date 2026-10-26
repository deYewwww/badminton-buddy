"""Run: uv run python tests/test_phase2.py"""
import numpy as np
import pandas as pd

from hud.camera import Camera
from hud.court import CORNERS_M, homography
from hud.hits import best_per_team, fit_speed, flight, from_window_avg, geo_fit, shot_type, turn_candidates
from hud.shuttle import clean, idle_frames


def synthetic_camera(f=1500.0, C=(3.05, -7.0, 6.0)):
    C = np.array(C)
    z = np.array([3.05, 6.7, 0.0]) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    R = np.array([x, np.cross(z, x), z])
    proj = lambda X: np.array([960, 540]) + f * (R @ (np.asarray(X) - C))[:2] / (R @ (np.asarray(X) - C))[2]
    return proj, f, C


def test_camera_from_homography():
    proj, f, C = synthetic_camera()
    cam = Camera(homography([proj([*c, 0]) for c in CORNERS_M]), (1920, 1080))
    assert abs(cam.f - f) < 1e-3 and np.allclose(cam.C, C, atol=1e-6)
    shuttle = np.array([2.0, 9.0, 2.5])
    assert np.allclose(cam.at_height(proj(shuttle), 2.5), shuttle, atol=1e-6)
    assert np.allclose(cam.near_vertical(proj(shuttle), (2.0, 9.0)), shuttle, atol=1e-6)
    # smash: straight line from far rear to near mid court; recover a point on it from its pixel
    a, b = np.array([3.0, 12.0, 3.0]), np.array([2.5, 3.0, 0.8])
    mid = a + 0.3 * (b - a)
    assert np.allclose(cam.on_line(proj(mid), a, b), mid, atol=1e-6)


def test_clean_interpolates_and_drops_glitches():
    raw = pd.DataFrame({"Frame": range(10), "Visibility": [1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
                        "X": [0, 10, 0, 0, 40, 50, 900, 70, 0, 0], "Y": [0] * 10})
    out = clean(raw, np.arange(10) / 50, fps=50)
    assert out.loc[2:3, "x_px"].tolist() == [20, 30] and out.loc[2:3, "interpolated"].all()
    assert out.loc[6, "x_px"] == 60  # spike at 900 removed, refilled from neighbours
    assert not out.loc[8:9, "visible"].any()  # trailing gap: nothing to interpolate towards
    # 4-frame run locked onto something else, then back on the smooth track
    x = np.arange(30) * 10.0
    x[12:16] = [800, 810, 820, 830]
    raw = pd.DataFrame({"Frame": range(30), "Visibility": 1, "X": x, "Y": 0.0})
    out = clean(raw, np.arange(30) / 50, fps=50)
    assert np.allclose(out.x_px, np.arange(30) * 10.0)


def test_turn_and_speed_jump_candidates():
    t = np.arange(40)
    bounce = np.c_[100 + 5 * t, np.where(t < 20, 10 * t, 400 - 10 * t)].astype(float)
    assert 20 in turn_candidates(bounce, 3)[0]
    smash = np.c_[np.full(40, 500.0), np.where(t < 20, 3 * t, 60 + 15 * (t - 20))].astype(float)
    assert 20 in turn_candidates(smash, 3, min_px=5)[0]
    jitter = np.c_[500 + np.random.default_rng(0).normal(0, 1, 40), np.full(40, 300.0)]
    assert turn_candidates(jitter, 3, min_px=15)[0] == []


def test_hitter_by_nearest_wrist():
    """A near player hits overhead; a far player stands right behind in the image with a
    small box whose centre sits on the shuttle. The near player's wrist is at the shuttle."""
    rows = pd.DataFrame([
        dict(frame=0, player_id="P1", team="us", box_x1=900, box_y1=400, box_x2=1100, box_y2=800,
             wrist_l_x=np.nan, wrist_l_y=np.nan, wrist_r_x=1005, wrist_r_y=385),
        dict(frame=0, player_id="P3", team="opp", box_x1=980, box_y1=340, box_x2=1040, box_y2=460,
             wrist_l_x=990, wrist_l_y=430, wrist_r_x=1030, wrist_r_y=430),
    ])
    best = best_per_team((1010, 380), rows, rows)
    assert best["us"][0] > best["opp"][0] and best["us"][1].player_id == "P1"


def test_geo_fit_dive():
    """Far player dives: racket 1.3 m to the side, 0.3 m off the floor. A near player on the other
    side of the court can't have hit it. (One standing right under the ray could, with a 2.8 m
    jump smash: then geometry can't tell and detect() falls back to the sides alternating.)"""
    from types import SimpleNamespace as R
    proj, f, C = synthetic_camera()
    cam = Camera(homography([proj([*c, 0]) for c in CORNERS_M]), (1920, 1080))
    px = proj([1.7, 10.0, 0.3])
    far, near = R(x_m=3.0, y_m=10.0), R(x_m=4.5, y_m=2.5)
    assert geo_fit(cam, px, far)[0] and not geo_fit(cam, px, near)[0]


def test_initial_speed_under_drag():
    """A 180 km/h clear and a 110 km/h drive, seen at 30 fps by a low 45-degree phone with 2 px
    of noise and a contact point 0.3 m off: the physics fit must land within 8 %; the old 0.1 s
    window average was 37-60 % low on the same shots."""
    proj, f, C = synthetic_camera(C=(7.0, -4.0, 1.5))
    cam = Camera(homography([proj([*c, 0]) for c in CORNERS_M]), (1920, 1080))
    rng = np.random.default_rng(0)
    for c0, kmh, d in (((3.0, 2.0, 2.4), 180, (0.0, 1.0, 0.75)), ((1.5, 4.5, 1.5), 110, (0.3, 1.0, 0.05))):
        u = np.array(d) / np.linalg.norm(d)
        dt = np.arange(1, 25) / 30
        X = np.array(c0) + flight(kmh / 3.6, u, dt - 0.01)  # contact 10 ms after the hit frame
        keep = X[:, 2] > 0
        dt, X = dt[keep], X[keep]
        px = np.array([proj(x) for x in X]) + rng.normal(0, 2, (len(X), 2))
        c0e = cam.near_vertical(proj(c0), np.array(c0[:2]) + [0.3, -0.2])
        v = fit_speed(cam, c0e, dt, px, (dt[-1], X[-1]), 30)
        assert v is not None and abs(v * 3.6 / kmh - 1) < 0.08, (kmh, v and v * 3.6)
    # the fallback's correction: 0.1 s average of a 300 km/h smash back to 300
    t = 0.1
    avg = np.log1p(0.22 * 83.33 * t) / 0.22 / t
    assert abs(from_window_avg(avg, t) - 83.33) < 0.01


def test_shot_rules():
    assert shot_type((3, 11, 2.8), (3, 2.5), 250) == "smash"
    assert shot_type((3, 5.5, 1.0), (3, 7.8), 30) == "net"
    assert shot_type((3, 5.0, 1.0), (3, 12.5), 120) == "clear"
    assert shot_type((3, 12, 2.5), (3, 8.5), 60) == "drop"
    assert shot_type((3, 9.5, 1.4), (3, 3.5), 140) == "drive"


def test_idle_frames():
    """30 fps, 4 players: rally 0-10 s (side steps at 2.4 m/s), standing 10-20 s, rally again
    20-30 s. Only the middle of the standing stretch, minus the pads, is idle."""
    fps, n = 30, 900
    t = np.arange(n) / fps
    cfg = dict(idle_speed_mps=1.8, pad_after_s=2.0, pad_before_s=2.0, min_skip_s=1.0)
    play = (t < 10) | (t >= 20)
    x = np.where(play, 3 + 0.8 * np.sin(3 * t), 3.0)  # peak speed 2.4 m/s
    rows = [dict(frame=f, player_id=p, x_m=x[f] + k, y_m=2.0 + 6 * (k > 1)) for k, p in enumerate(["P1", "P2", "P3", "P4"])
            for f in range(n)]
    idle = idle_frames(pd.DataFrame(rows), t, cfg)
    on = t[idle]
    assert on.min() > 11.5 and on.max() < 19.0 and on.max() - on.min() > 5, (on.min(), on.max())
    assert not idle[play].any()
    # two players lost (dark corner): unknown, never idle
    few = pd.DataFrame([r for r in rows if r["player_id"] in ("P1", "P3")])
    assert not idle_frames(few, t, cfg).any()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
