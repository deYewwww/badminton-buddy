"""Run: uv run python tests/test_phase1.py"""
import numpy as np
import pandas as pd

import pandas as pd

from hud.players import APP_COLS, IdentityTracker, apply_swaps, fill_skipped
from hud.reid import fix_identities


def test_crossing_keeps_ids():
    """Near pair runs through each other; one detection vanishes during the overlap and
    ByteTrack IDs swap at the crossing. Velocity prediction must keep P1/P2 straight."""
    trk, fps = IdentityTracker(), 50
    far = [dict(xy=np.array([2.0, 10.0]), track_id=7, hist=None),
           dict(xy=np.array([4.0, 10.0]), track_id=8, hist=None)]
    for f in range(70):
        t = f / fps
        a, b = np.array([1 + 3 * t, 2.0 + 0.5 * t]), np.array([5 - 3 * t, 2.6 - 0.5 * t])
        crossed = a[0] > b[0]
        near = [dict(xy=a, track_id=2 if crossed else 1, hist=None),
                dict(xy=b, track_id=1 if crossed else 2, hist=None)]
        if np.hypot(*(a - b)) < 0.6:
            near = near[1:]  # merged into one box
        out = trk.update(t, near[::-1] + far)
    assert out["P1"][0][0] > 4 and out["P2"][0][0] < 2, out
    assert out["P3"][0][0] < out["P4"][0][0]


def test_partner_hidden_at_start():
    """Near partner hidden for the first 30 frames: the visible one is tracked from frame 0,
    the other joins when seen, and one person is never split into two slots."""
    trk = IdentityTracker()
    far = [dict(xy=np.array([2.0, 10.0]), track_id=7, hist=None), dict(xy=np.array([4.0, 10.0]), track_id=8, hist=None)]
    for f in range(60):
        near = [dict(xy=np.array([2.0 + 0.002 * f, 2.0]), track_id=1, hist=None)]
        if f >= 30:
            near.append(dict(xy=np.array([4.5, 3.0]), track_id=2, hist=None))
        out = trk.update(f / 50, near + far)
        if f == 0:
            assert "P1" in out and "P2" not in out
    assert out["P1"][1] is not None and out["P2"][1] is not None and out["P2"][0][0] > 4


class _Cam:  # height straight from the box: head pixel y=0 -> 2.0 m, y=400 -> 1.6 m
    def near_vertical(self, px, xy):
        return np.array([xy[0], xy[1], 2.0 - px[1] / 1000])


def _pair(swap_from, tall=(1.75, 1.55), colour=(60, 160)):
    """P1 = tall/dark person, P2 = short/light person, IDs swapped from frame swap_from on."""
    rows = []
    for f in range(300):
        for pid, h, c, x in (("P1", tall[0], colour[0], 1.0), ("P2", tall[1], colour[1], 5.0)):
            if 140 <= f < 160:
                x = 3.0 + (0.2 if pid == "P1" else -0.2)  # partners close together: ambiguous
            who = pid if f < swap_from else {"P1": "P2", "P2": "P1"}[pid]
            rows.append(dict(frame=f, player_id=who, team="us", x_m=x, y_m=2.0, observed=True,
                             box_x1=0.0, box_x2=10.0, box_y1=(2.0 - h) * 1000, box_y2=900.0,
                             **dict(zip(APP_COLS, [c, 128, 128, c, 128, 128]))))
    return pd.DataFrame(rows)


def test_reid_undoes_swap_between_rallies():
    out = fix_identities(_pair(swap_from=150), _Cam())
    tall = out[(out.box_y1 < 300) & ~out.frame.between(140, 159)]  # the 1.75 m person, outside the
    assert (tall.player_id == "P1").all()                              # close-together stretch
    # partners who look alike: nothing changes
    same = _pair(swap_from=150, tall=(1.7, 1.7), colour=(100, 100))
    assert fix_identities(same, _Cam()).player_id.tolist() == same.player_id.tolist()


def test_fill_skipped_frames():
    """Detector ran on even frames only: odd frames get interpolated positions."""
    rows = [dict(frame=f, timestamp=f / 60, player_id="P1", team="us", x_m=f * 0.1, y_m=2.0, observed=True,
                 bytetrack_id=1, box_x1=0.0, box_y1=0.0, box_x2=1.0, box_y2=1.0, foot_px_x=0.0, foot_px_y=0.0)
            for f in range(0, 10, 2)]
    out = fill_skipped(pd.DataFrame(rows), [f / 60 for f in range(9)], 2)
    assert out.frame.tolist() == list(range(9)) and np.allclose(out.x_m, np.arange(9) * 0.1)
    assert out.observed.all() and out.timestamp.iat[3] == round(3 / 60, 4)


def test_swaps():
    df = pd.DataFrame(dict(frame=[0, 0, 5, 5], player_id=["P1", "P2", "P1", "P2"]))
    import io
    out = apply_swaps(df, io.StringIO("frame,a,b\n5,P1,P2\n"))
    assert out.player_id.tolist() == ["P1", "P2", "P2", "P1"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
