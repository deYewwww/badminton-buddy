"""Calibration from any 4+ named points, line snapping, placement ratings.
Run: uv run python tests/test_hud_data.py"""
import cv2
import numpy as np

from hud.court import LINES, auto_detect, homography, snap, to_court
import pandas as pd

from hud import config
from hud.analysis import score
from hud.analysis.rallies import build, rate


def synth(f=1500.0, C=(7.0, -4.0, 1.5)):
    """A low 45-degree phone view: camera off the near-right corner."""
    C = np.array(C)
    z = np.array([3.05, 6.7, 0.0]) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    R = np.array([x, np.cross(z, x), z])
    return lambda X: np.array([960, 540]) + f * (R @ (np.r_[X, 0.0] - C))[:2] / (R @ (np.r_[X, 0.0] - C))[2]


def test_auto_detect_with_corners_off_screen():
    """Low 45-degree view where the near corners fall outside the frame: the court must still
    be found from the painted lines alone, and snap() must polish a rough guess."""
    proj = synth()
    img = np.full((1080, 1920, 3), (180, 90, 40), np.uint8)  # blue-ish floor
    for a, b in LINES:
        cv2.line(img, tuple(proj(a).astype(int)), tuple(proj(b).astype(int)), (255, 255, 255), 5)
    corners = np.array([proj(c) for c in [(0, 0), (6.1, 0), (6.1, 13.4), (0, 13.4)]])
    off = ((corners < 0) | (corners > [1920, 1080])).any(1)
    assert off.sum() >= 1, corners  # the scenario under test
    probe = np.array([proj((x, y)) for x in (0.5, 3, 5.5) for y in (1, 4, 6, 9, 12)])
    truth = np.array([(x, y) for x in (0.5, 3, 5.5) for y in (1, 4, 6, 9, 12)])
    found, frac = auto_detect(img)
    assert found is not None and frac > 0.8, frac
    assert np.abs(to_court(homography(found), probe) - truth).max() < 0.15
    rough = corners + np.random.default_rng(0).normal(0, 15, corners.shape)
    snapped, _ = snap(img, rough)
    before = np.abs(to_court(homography(rough), probe) - truth).max()
    after = np.abs(to_court(homography(snapped), probe) - truth).max()
    assert after < before and after < 0.15, (before, after)


def test_auto_detect_low_resolution():
    """The scene at 854x480, blurred, noisy and JPEG-compressed (cheap phone, heavy compression).
    Detection used to run at native size there: on a real 480p clip it found nothing, here its
    worst point was 0.16 m off. Scaled up to the working size: < 0.12 m."""
    proj = synth()
    img = np.full((1080, 1920, 3), (60, 140, 40), np.uint8)
    for a, b in LINES:
        cv2.line(img, tuple(proj(a).astype(int)), tuple(proj(b).astype(int)), (235, 235, 235), 4)
    small = cv2.GaussianBlur(cv2.resize(img, (854, 480), interpolation=cv2.INTER_AREA), (3, 3), 0.8)
    small = np.clip(small + np.random.default_rng(0).normal(0, 4, small.shape), 0, 255).astype(np.uint8)
    small = cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 35])[1], 1)
    k = 854 / 1920
    probe = np.array([proj((x, y)) for x in (0.5, 3, 5.5) for y in (1, 4, 6, 9, 12)]) * k
    truth = np.array([(x, y) for x in (0.5, 3, 5.5) for y in (1, 4, 6, 9, 12)])
    found, frac = auto_detect(small)
    assert found is not None and frac > 0.8, frac
    assert np.abs(to_court(homography(found), probe) - truth).max() < 0.12


def test_check_flags_squeezed_calibration():
    """The user's real failure: near corners pulled in to the frame edge, court squeezed
    towards the net. Line overlap alone stays high-ish; the camera check must catch it."""
    from hud.analysis.check import calib_fit
    proj = synth()
    img = np.full((1080, 1920, 3), (60, 140, 40), np.uint8)
    for a, b in LINES:
        cv2.line(img, tuple(proj(a).astype(int)), tuple(proj(b).astype(int)), (255, 255, 255), 5)
    good = np.array([proj(c) for c in [(0, 0), (6.1, 0), (6.1, 13.4), (0, 13.4)]])
    bad = good.copy()
    bad[0], bad[1] = (good[0] + good[3]) / 2, (good[1] + good[2]) / 2
    fit, ok = calib_fit(img, good)
    assert fit > 0.9 and ok, (fit, ok)
    assert not calib_fit(img, bad)[1]


def test_venue_align_after_camera_moves():
    """Venue calibrated on one frame; the match is filmed after the phone was turned ~3 degrees
    and zoomed a little (any pure rotation/zoom is a homography of the whole image). Aligned
    corners must land on the true ones; an unrelated scene must be refused."""
    from hud.court import venue
    proj = synth()
    rng = np.random.default_rng(0)
    img = cv2.resize(rng.integers(40, 200, (54, 96, 3), dtype=np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST)
    img[540:] = (60, 140, 40)                                   # textured hall above, plain floor below
    for a, b in LINES:
        cv2.line(img, tuple(proj(a).astype(int)), tuple(proj(b).astype(int)), (255, 255, 255), 5)
    corners = np.array([proj(c) for c in [(0, 0), (6.1, 0), (6.1, 13.4), (0, 13.4)]])
    th = np.radians(3)
    A = np.array([[1.04 * np.cos(th), -1.04 * np.sin(th), 40], [1.04 * np.sin(th), 1.04 * np.cos(th), -60], [0, 0, 1]])
    moved = cv2.warpPerspective(img, A, (1920, 1080))
    truth = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), A).reshape(-1, 2)
    got, rep = venue.align(img, corners, moved)
    assert got is not None, rep
    probe = np.array([proj((x, y)) for x in (0.5, 3, 5.5) for y in (1, 4, 6, 9, 12)])
    probe = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), A).reshape(-1, 2)
    err = np.abs(to_court(homography(got), probe) - to_court(homography(truth), probe)).max()
    assert err < 0.1, (err, rep)
    other = cv2.resize(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8), (1920, 1080))
    assert venue.align(img, corners, other)[0] is None


def test_stabilize_pan(tmp=None):
    """A 'gimbal' pan: frames are the reference view turned by known homographies. After
    stabilising, every frame lines up with the reference (sub-pixel), and the court corners
    come back shifted onto the canvas."""
    import json, tempfile, subprocess
    from pathlib import Path
    from hud.court import stabilize
    rng = np.random.default_rng(1)
    base = cv2.resize(rng.integers(30, 220, (36, 64, 3), dtype=np.uint8), (640, 360), interpolation=cv2.INTER_NEAREST)
    base = cv2.GaussianBlur(base, (3, 3), 0)
    d = Path(tempfile.mkdtemp())
    src = d / "pan.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "640x360", "-r", "30",
                           "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(src)], stdin=subprocess.PIPE)
    for k in range(30):
        dx = 25 * np.sin(k / 6)  # pan left and right
        A = np.array([[1, 0, dx], [0, 1, 0.3 * dx], [0, 0, 1.0]])
        ff.stdin.write(cv2.warpPerspective(base, A, (640, 360)).tobytes())
    ff.stdin.close(); ff.wait()
    corners = [[100, 300], [540, 300], [400, 150], [240, 150]]
    default = stabilize.KEY_EVERY
    # every frame matched: sub-pixel; key frames + interpolation: within the curve of this fast pan
    for every, tol in ((1, 1.0), (default, 3.0)):
        stabilize.KEY_EVERY = every
        try:
            c = stabilize.run(src, d / "stab.mp4", 0.0, corners)
        finally:
            stabilize.KEY_EVERY = default
        meta = json.loads((d / "stab.json").read_text())
        assert np.allclose(c, np.float64(corners) + meta["offset"])
        cap = cv2.VideoCapture(str(d / "stab.mp4"))
        frames = [cap.read()[1] for _ in range(30)]
        ox, oy = meta["offset"]
        ref = cv2.cvtColor(frames[0][oy + 60:oy + 300, ox + 60:ox + 580], cv2.COLOR_BGR2GRAY).astype(np.float32)
        worst = max(np.hypot(*cv2.phaseCorrelate(ref, cv2.cvtColor(f[oy + 60:oy + 300, ox + 60:ox + 580], cv2.COLOR_BGR2GRAY).astype(np.float32))[0])
                    for f in frames[1:])
        assert worst < tol, (every, worst)


def test_evaluate_matching():
    """Rallies matched by overlap, hits by time; winners and side counted on matches only."""
    from hud.analysis import evaluate as E
    pred = pd.DataFrame(dict(start_ts=[1.0, 10.0, 30.0], end_ts=[5.0, 14.0, 31.0], shot_count=[6, 3, 2],
                             point_team_final=["us", "opp", "us"]))
    truth = [dict(start=0.8, end=5.5, winner="us", shots=7), dict(start=9.5, end=15, winner="us", shots=4),
             dict(start=20, end=24, winner="opp", shots=None)]
    r = E.rallies(pred, truth)
    assert (r["found"], r["extra"], r["winner_right"], r["winner_judged"], r["shots_truth"], r["shots_found"]) == (2, 1, 1, 2, 11, 9)
    ph = pd.DataFrame(dict(timestamp=[1.0, 1.9, 3.05, 7.0], team=["us", "opp", "opp", "us"]))
    h = E.hits(ph, [dict(t=1.1, team="us"), dict(t=2.0, team="us"), dict(t=3.0, team="opp"), dict(t=4.0, team="us")], [[0, 5]])
    assert (h["right_side"], h["wrong_side"], h["missed"], h["extra"]) == (2, 1, 1, 0)  # the 7.0 s hit is outside the window


def test_placement():
    assert rate("clear", (0.3, 13.2), "us", [(3, 10)]) >= 85      # deep corner
    assert rate("clear", (3.0, 9.0), "us", [(3, 10)]) < 50         # mid court
    assert rate("drop", (5.9, 7.2), "us", [(3, 11)]) >= 85         # tight, by the side line
    assert rate("smash", (0.2, 10), "us", [(4, 10), (4, 12)]) >= 85
    assert rate("smash", (3.0, 10), "opp", [(1, 3)]) == 0           # landed on own side
    assert rate("net", (3.0, 6.5), "opp", []) >= 70
    assert rate("clear", (6.5, 12), "us", []) == 0                   # out


def test_rallies():
    """Two rallies split by 2 s of shuttle silence; a lone hit later is dropped; a same-team
    double hit collapses to the sharper turn; an unresolved ending is capped by max flight."""
    fps, n = 60, 1200
    real = np.zeros(n, bool)
    real[0:300] = True                 # rally 1 in view
    real[420:1200] = True              # rally 2 stays "in view" forever (tracker noise)
    shuttle = pd.DataFrame({"frame": range(n), "visible": real, "inpainted": False, "interpolated": False})
    rows = [(30, "us", 120), (90, "opp", 130), (150, "opp", 60), (210, "us", 140),   # rally 1, opp hit twice
            (450, "us", 120), (520, "opp", 120),                                     # rally 2
            (1100, "us", 120)]                                                       # lone hit, > max_gap later
    shots = pd.DataFrame([dict(frame=f, timestamp=f / fps, team=t, turn_deg=a, player_id="P1" if t == "us" else "P3",
                               shot_type="drive", landing_x=3.0, landing_y=3.0, landing_kind="intercept",
                               end_frame=f + 60) for f, t, a in rows])
    players = pd.DataFrame(columns=["frame", "player_id", "team", "x_m", "y_m", "observed"]).astype({"observed": bool})
    out, rallies = build(shots, players, config.load(), shuttle)
    assert out.frame.tolist() == [30, 90, 210, 450, 520], out.frame.tolist()
    assert rallies.shot_count.tolist() == [3, 2]
    assert rallies.end_frame.iat[1] <= 520 + 1.5 * fps  # capped: no shot flies longer than 1.5 s


def test_serve_must_be_answered():
    """Server fiddles with the shuttle (a 'hit' at 0 s), serves for real at 2 s, rally goes on."""
    fps = 60
    rows = [(0, "us"), (120, "us"), (160, "opp"), (200, "us")]
    shots = pd.DataFrame([dict(frame=f, timestamp=f / fps, team=t, turn_deg=100, player_id="P1" if t == "us" else "P3",
                               shot_type="drive", landing_x=3.0, landing_y=3.0, landing_kind="intercept",
                               end_frame=f + 40) for f, t in rows])
    players = pd.DataFrame(columns=["frame", "player_id", "team", "x_m", "y_m", "observed"]).astype({"observed": bool})
    out, rallies = build(shots, players, config.load())
    assert out.frame.tolist() == [120, 160, 200] and rallies.start_frame.tolist() == [120]


def test_pickup_toss_back_dropped():
    """Rally 1: serve + fast kill, receiver sprints. Rally 2 (10 s later): two slow underarm
    tosses while everyone walks. Rally 3: same slow shots but a player sprints (a real soft
    exchange). Only rally 2 goes."""
    fps = 30
    rows = [(30, "us", 60), (45, "opp", 250),        # rally 1
            (330, "us", 30), (360, "opp", 25),       # rally 2: toss-backs
            (630, "us", 35), (660, "opp", 30)]       # rally 3: soft, but someone runs
    shots = pd.DataFrame([dict(frame=f, timestamp=f / fps, team=t, turn_deg=100, player_id="P1" if t == "us" else "P3",
                               shot_type="drive", landing_x=3.0, landing_y=3.0, landing_kind="intercept",
                               end_frame=f + 20, speed_kmh_est=float(v)) for f, t, v in rows])
    pl = []
    for f in range(900):
        sprint = 25 <= f < 75 or 625 <= f < 690   # 3 m/s
        for k, pid in enumerate(["P1", "P2", "P3", "P4"]):
            x = 1.0 + 2 * k % 4 + (3.0 * f / fps if (sprint and pid == "P3") else 1.0 * f / fps)  # walking 1 m/s
            pl.append(dict(frame=f, player_id=pid, team="us" if k < 2 else "opp", x_m=x, y_m=3.0 + 6 * (k > 1), observed=True))
    out, rallies = build(shots, pd.DataFrame(pl), config.load())
    assert rallies.start_frame.tolist() == [30, 630], rallies.start_frame.tolist()
    assert sorted(out.rally_id.unique()) == [1, 2]


def test_score():
    cfg = config.load()
    # 20-20 deuce, then us wins 22-20; next game starts 0-0. Rally 3 unknown, rally 2 fixed by hand.
    cfg["score"].update(us=20, opp=20)
    rallies = pd.DataFrame({"rally_id": [1, 2, 3, 4, 5], "point_team": ["us", None, None, "opp", "opp"]})
    fixes = pd.DataFrame({"rally_id": [2], "point_team": ["us"]})
    out = score.apply(rallies, cfg, fixes)
    assert out.point_source.tolist() == ["landing", "fixed", "unknown", "landing", "landing"]
    assert (out.score_us.tolist(), out.score_opp.tolist()) == ([21, 22, 0, 0, 0], [20, 20, 0, 1, 2])
    assert out.game_won.tolist()[1] == "us" and out.games_us.tolist() == [0, 1, 1, 1, 1]
    assert out.server.tolist() == ["us", "us", "us", "opp", "opp"]
    assert score.game_over(29, 29, cfg["score"]) is False and score.game_over(30, 29, cfg["score"])



def test_our_shuttle():
    from hud.visualization.render import our_shuttle
    F = 60
    P = np.full((F, 2), np.nan)
    box = np.full((F, 4, 4), np.nan)
    box[:, 0] = [500, 500, 600, 800]           # one of our players
    P[0:10] = np.c_[np.linspace(560, 900, 10), np.full(10, 450)]    # leaves his racket
    P[12:20] = np.c_[np.linspace(960, 1200, 8), np.full(8, 450)]    # same flight, after a short gap
    P[20:30] = np.c_[np.linspace(50, 300, 10), np.full(10, 400)]    # next court: jumps away, no player near
    P[40:55] = [700, 900]                                           # spare shuttle lying still
    out = our_shuttle(P, box, 30, 1080)
    kept = np.isfinite(out[:, 0])
    assert kept[0:10].all() and kept[12:20].all() and not kept[20:30].any() and not kept[40:55].any()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
