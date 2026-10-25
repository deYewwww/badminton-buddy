"""Run: uv run python tests/test_phase2.py"""
import numpy as np
import pandas as pd

from hud.camera import Camera
from hud.court import CORNERS_M, homography
from hud.hits import best_per_team, shot_type, turn_candidates
from hud.shuttle import clean


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


def test_shot_rules():
    assert shot_type((3, 11, 2.8), (3, 2.5), 250) == "smash"
    assert shot_type((3, 5.5, 1.0), (3, 7.8), 30) == "net"
    assert shot_type((3, 5.0, 1.0), (3, 12.5), 120) == "clear"
    assert shot_type((3, 12, 2.5), (3, 8.5), 60) == "drop"
    assert shot_type((3, 9.5, 1.4), (3, 3.5), 140) == "drive"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
