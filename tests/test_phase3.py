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


def test_stamina_follows_effort():
    from hud import config
    from hud.render import stamina_track, stroke_kinds
    st = config.load()["stamina"]
    F, fps = 300, 30
    step = np.zeros((F, 4))
    speed = np.zeros((F, 4))
    speed[:150, 0] = 4.0                         # P1 sprints through the rally, P2 stands, P3/P4 hit
    rally_of = np.r_[np.zeros(150, int), np.full(150, -1)]
    shots = pd.DataFrame(dict(frame=[10, 40, 70, 100], player_id=["P3", "P4", "P3", "P3"],
                              team=["opp", "us", "opp", "opp"], shot_type=["smash", "clear", "smash", "smash"],
                              contact_z_m=[2.0, 1.0, 2.0, 2.0], rally_id=[1, 1, 1, 1]))
    assert [k for _, _, k in stroke_kinds(shots)] == ["smash", "defence", "smash", "smash"]
    out, costs = stamina_track(step, speed, rally_of, shots, fps, st)
    assert out[149, 1] == 100                     # standing still costs nothing
    assert out[149, 0] < 98                       # 5 s of sprinting does
    c = [cost for _, k, cost, _ in costs if k == 2]
    assert c[0] == st["shot_cost"]["smash"] and c[1] > c[0] and c[2] > c[1]  # an attack streak costs more each time
    assert [cost for _, k, cost, _ in costs if k == 3][0] == st["shot_cost"]["defence"]
    assert out[149, 0] < out[-1, 0] < 100         # rest recovers, but not all the way (fatigue)


def test_point_from_landing_side():
    from hud.rallies import decisive, in_target
    assert in_target(3.0, 10.0, "us") and not in_target(3.0, 5.0, "us")  # near team into the net: their loss
    assert in_target(3.0, 5.0, "opp") and not in_target(3.0, 14.0, "us")
    assert decisive(3.0, 10.0, "us", 0.3, 1.0) and not decisive(3.0, NET_Y + 0.2, "us", 0.3, 1.0)


def test_net_error_and_leaning_guess():
    from hud import config, score
    from hud.rallies import build
    fps = 30
    def rally(landing_y, sd):
        shots = pd.DataFrame([dict(frame=f, timestamp=f / fps, team=t, turn_deg=100, player_id="P1" if t == "us" else "P3",
                                   shot_type="drive", landing_x=3.0, landing_y=y, landing_kind=k, landing_sd=s_,
                                   end_frame=f + 20) for f, t, y, k, s_ in
                              [(30, "us", 9.0, "intercept", np.nan), (50, "opp", 3.0, "intercept", np.nan),
                               (70, "us", landing_y, "predicted", sd)]])
        players = pd.DataFrame(columns=["frame", "player_id", "team", "x_m", "y_m", "observed"]).astype({"observed": bool})
        _, r = build(shots, players, config.load())
        return score.apply(r, config.load()).iloc[0]
    net = rally(5.6, 1.0)        # near team's last shot comes down 1.1 m short of the net: their error
    assert net.point_team_final == "opp" and net.point_source == "landing"
    lean = rally(7.2, 1.0)       # just over the net, too close to call: a guess, but the right way
    assert lean.point_team_final == "us" and lean.point_source == "guess"
    lean_out = rally(14.0, 1.0)  # past the far baseline, not by enough to be sure: leans to the receivers
    assert lean_out.point_team_final == "opp" and lean_out.point_source == "guess"


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


def test_far_pass_fills_main_gaps_only():
    from hud.shuttle import far_window, merge_far
    main = pd.DataFrame(dict(Frame=range(4), Visibility=[1, 0, 1, 1], X=[10.0, 0, 30, 40], Y=[5.0, 0, 7, 8],
                             inpainted=[False, False, True, False]))
    far = pd.DataFrame(dict(Frame=range(4), Visibility=[1, 1, 1, 0], X=[11.0, 21, 31, 0], Y=[5.0, 6, 7, 0],
                            inpainted=[False, False, False, False]))
    out, n = merge_far(main, far)
    assert n == 2 and out.X.tolist() == [10, 21, 31, 40] and not out.inpainted.any()
    cam, H, proj = low_camera()
    w = far_window(cam, 720, 1280, (-280, 189, 1280, 720))
    assert w is not None and w[2] < 1280 / 1.6 and w[0] >= 0 and w[1] + w[3] <= 1280


def test_detector_fills_gaps_on_the_path_only():
    from hud.shuttle_yolo import fill, gaps
    assert gaps(np.array([1, 0, 0, 1, 1, 0, 1], bool), 2) == [(0, 3), (4, 6)]
    n = 12
    X = np.linspace(100, 650, n)
    raw = pd.DataFrame(dict(Frame=range(n), Visibility=[1] * 3 + [0] * 6 + [1] * 3, X=np.where(np.arange(n) < 3, X, 0),
                            Y=200.0, inpainted=False))
    raw.loc[9:, "X"] = X[9:]
    imgs = lambda fs: ((f, np.zeros((720, 1280, 3), np.uint8)) for f in fs)
    # tile-relative detections: on the path at frames 4 and 6 (strong, weak); off the path at 5
    hits = {4: (X[4], 200, 0.5), 5: (X[5], 500, 0.9), 6: (X[6] + 5, 202, 0.2)}
    def det(tiles, frames=iter(range(3, 9))):
        out = []
        for _ in tiles:
            f = next(frames)
            if f in hits:
                x, y, c = hits[f]
                x0 = int(np.clip(X[f] - 320, 0, 1280 - 640)); y0 = 0
                out.append(np.array([[x - x0, y - y0, c]]))
            else:
                out.append(np.zeros((0, 3)))
        return out
    out, k = fill(None, raw, 30, 1080, detector=det, read=imgs)
    assert k == 2 and out.Visibility.iloc[4] == 1 and out.Visibility.iloc[6] == 1 and out.Visibility.iloc[5] == 0
    assert abs(out.X.iloc[4] - X[4]) < 1 and not out.inpainted.any()


def test_gap_path_follows_the_arc():
    from hud.shuttle_yolo import gap_path
    f = np.arange(20)
    P = np.c_[20.0 * f, 300 - 40 * f + 2.5 * f ** 2]   # a clear's apex seen from behind
    seen = np.ones(20, bool); seen[8:13] = False
    path = gap_path(np.flatnonzero(seen), P, 7, 13)
    assert np.hypot(*(path(10) - P[10])) < 1, path(10)  # a straight line would be off by ~22 px


def test_detector_finds_the_shuttle_lying_after_the_track_ends():
    from hud.shuttle_yolo import fill
    n = 60
    raw = pd.DataFrame(dict(Frame=range(n), Visibility=[1] * 10 + [0] * 50, X=np.r_[np.full(10, 400.0), np.zeros(50)],
                            Y=np.r_[np.linspace(300, 480, 10), np.zeros(50)], inpainted=False))
    imgs = lambda fs: ((f, np.zeros((720, 1280, 3), np.uint8)) for f in fs)
    lying = np.array([410.0, 520.0])
    def det(tiles, frames=[]):
        return [np.array([[*(lying - [80, 80]), 0.6]]) for _ in tiles]  # tile origin (80, 80)
    out, k = fill(None, raw, 30, 1080, detector=det, read=imgs)
    seen = out[out.Visibility == 1]
    assert k > 10 and np.allclose(out.loc[13, ["X", "Y"]].astype(float), lying)
    assert out.inpainted.iloc[10:12].all() and not out.inpainted.iloc[12:54].any()  # fall guessed, spot seen
    assert (out.Visibility.iloc[10:55] == 1).all() and (out.Visibility.iloc[55:] == 0).all()
    # the same spot already occupied before the shot (a spare shuttle): not a landing
    raw2 = pd.concat([raw.iloc[:0], pd.DataFrame(dict(Frame=range(70), Visibility=0, X=0.0, Y=0.0, inpainted=False))])
    raw2 = raw2.reset_index(drop=True)
    raw2.loc[60:69, "Visibility"], raw2.loc[60:69, "X"], raw2.loc[60:69, "Y"] = 1, 400.0, np.linspace(300, 480, 10)
    raw2 = pd.concat([raw2, pd.DataFrame(dict(Frame=range(70, 130), Visibility=0, X=0.0, Y=0.0, inpainted=False))], ignore_index=True)
    out2, k2 = fill(None, raw2, 30, 1080, detector=det, read=imgs)
    assert (out2.Visibility.iloc[70:] == 0).all(), k2


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
