"""Rallies and landing-placement ratings.

Rally: consecutive hits until the shuttle settles on the floor, or play goes quiet (no real
shuttle sighting for silence_s). Lone single-hit rallies are dropped as noise. Hits in the
dead time after a landing (picking the shuttle up, tossing it back) are dropped.
Pick-ups: a short "rally" (<= pickup_max_shots) in which no shot left the racket faster than
pickup_max_kmh AND no player moved faster than pickup_max_mps is someone tossing the shuttle
back, not play: dropped. Both signals must agree and both must be measured (players tracked,
speeds known): a short real rally has a fast shot (a kill) or a player sprinting to it.
Swing detection from the pose model was considered and not used: a swing lasts 0.1-0.15 s and
the pose model runs 10 times a second, so it would often see no frame of it.
Point: the last shot landed in the opponents' half -> the hitter's team wins it; in their own half
(into the net) or out -> the other team. The landing is where the shuttle was seen lying, or
where its fitted flight reaches the floor (hits: landing_kind "predicted"); a predicted one
decides only when it is more than landing_sure_sd of its uncertainty from the nearest line.
In/out is ESTIMATED: the landing point carries the calibration error (~5-20 cm), so
line calls are not reliable. Service-court rules are ignored.

Placement rating (0-100) per shot from where it ended (floor or the opponent's contact):
  clear  deep near the baseline, toward a corner     drop   just over the net, toward a side
  net    tight to the net                             smash  near a side line, away from receivers
  drive  deep and away from receivers
Receivers' positions are taken at the moment of the hit, so an intercepted smash that
forced a long reach still rates well.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from hud import config
from hud.court import COURT_L, COURT_W, NET_Y

clip = lambda v: float(np.clip(v, 0, 1))
ENDS = ("floor", "predicted")  # landing kinds that end a rally
NET_SURE_SD = 0.7  # a predicted landing this many sd short of the net, in the hitter's own half: net error


def in_court(x, y):
    return 0 <= x <= COURT_W and 0 <= y <= COURT_L


def in_target(x, y, hit_team):
    """Landed in the opponents' half (a point for the hitter); own half (into the net) or out
    is a point for the other side."""
    return in_court(x, y) and (y >= NET_Y) == (hit_team == "us")


def target_margin(x, y, hit_team):
    """Signed distance (m) to the edge of the opponents' half: > 0 inside."""
    y0, y1 = (NET_Y, COURT_L) if hit_team == "us" else (0.0, NET_Y)
    dx, dy = min(x, COURT_W - x), min(y - y0, y1 - y)
    if dx >= 0 and dy >= 0:
        return min(dx, dy)
    return -float(np.hypot(min(dx, 0), min(dy, 0)))


def decisive(x, y, hit_team, sd, k):
    """A predicted landing decides the point only if it is more than k sd from the edge."""
    return np.isfinite(x) and np.isfinite(sd) and abs(target_margin(x, y, hit_team)) > k * sd


def rate(shot_type, land, hit_team, receivers):
    """0-100, or None if there's nothing to rate. land: (x, y) court m; receivers: [(x, y)]."""
    if not np.isfinite(land[0]) or shot_type == "other":
        return None
    x, y = land
    own_half = (y < NET_Y) == (hit_team == "us")
    if own_half or not in_court(x, y):
        return 0
    d_net, side = abs(y - NET_Y), min(x, COURT_W - x)
    away = min((np.hypot(x - rx, y - ry) for rx, ry in receivers), default=0)
    s = {
        "clear": 70 * clip((d_net - 4.0) / 2.7) + 30 * clip(1 - side / 1.5),
        "drop": 70 * clip((2.5 - d_net) / 2.0) + 30 * clip(1 - side / 1.5),
        "net": 80 * clip((1.5 - d_net) / 1.2) + 20 * clip(1 - side / 1.5),
        "smash": 50 * clip(1 - side / 1.5) + 50 * clip(away / 2.5),
        "drive": 40 * clip((d_net - 2.0) / 3.0) + 60 * clip(away / 2.5),
    }[shot_type]
    return round(s)


def label(score, cfg):
    if score is None:
        return ""
    if score == 0:
        return "OUT"
    p = cfg["placement"]
    return "PERFECT" if score >= p["perfect"] else "GREAT" if score >= p["great"] else "GOOD" if score >= p["good"] else "OK"


def build(shots, players, cfg, shuttle=None, cuts=()):
    r = cfg["rally"]
    if shots.empty:  # no hits found (e.g. shuttle not tracked): no rallies, not a crash
        return shots, pd.DataFrame()
    shots = shots.sort_values("frame").reset_index(drop=True)
    ts = shots.timestamp.to_numpy()
    fps = _fps(shots)
    # real sightings (not InpaintNet guesses): play has stopped once these go quiet
    real = None
    if shuttle is not None and "inpainted" in shuttle:
        real = (shuttle.visible & ~shuttle.inpainted & ~shuttle.interpolated).to_numpy()
    quiet = max(1, round(r["silence_s"] * fps))
    flight_gap = max(1, round(r["flight_gap_s"] * fps))

    def silent_between(a, b):
        if real is None or b <= a:
            return False
        seg = real[a:b]
        return pd.Series(~seg).groupby(seg.cumsum()).sum().max() >= quiet
    keep, rally_ids, rid, dead_until = [], [], 0, -np.inf
    for k, s in shots.iterrows():
        prev = shots.loc[keep[-1]] if keep else None
        cut = prev is not None and any(prev.frame < c <= s.frame for c in cuts)  # edited clip: new moment
        if s.timestamp < dead_until and not cut:
            continue
        if (prev is None or cut or s.timestamp - prev.timestamp > r["max_gap_s"] or prev.landing_kind in ENDS
                or silent_between(int(prev.frame), int(s.frame))):
            rid += 1
        keep.append(k)
        rally_ids.append(rid)
        if s.landing_kind in ENDS:  # rally over once it comes down
            end_t = s.timestamp + max(0, s.end_frame - s.frame) / fps
            dead_until = end_t + r["dead_time_s"]
    shots = shots.loc[keep].copy()
    shots["rally_id"] = rally_ids
    # a rally opener (the serve) must be answered quickly: doubles serves are mostly short. An
    # opener left unanswered is the server fiddling with the shuttle before serving: drop it,
    # the next hit opens the rally (repeat while that holds)
    while True:
        heads = shots.groupby("rally_id").head(2)
        lag = heads.groupby("rally_id").timestamp.agg(lambda t: t.iat[-1] - t.iat[0])  # 0 for a lone hit
        slow = lag[lag > r["serve_answer_s"]].index
        if not len(slow):
            break
        shots = shots.drop(shots[shots.rally_id.isin(slow)].groupby("rally_id").head(1).index)
    # one side never hits twice in a row: of a same-team run inside a rally keep the sharpest
    # turn (the real hit); the rest are tracker noise. Their landings stay as computed.
    run = ((shots.team != shots.team.shift()) | (shots.rally_id != shots.rally_id.shift())).cumsum()
    shots = shots.loc[shots.groupby(run).turn_deg.idxmax()].sort_values("frame")
    # ponytail: a lone "hit" is far more often noise than a served fault; a fault is lost
    shots = shots[shots.groupby("rally_id").frame.transform("size") >= r["min_shots"]]
    shots["rally_id"] = shots.rally_id.rank(method="dense").astype(int)
    shots["shot_no"] = shots.groupby("rally_id").cumcount() + 1

    obs = players[players.observed]
    pos_at = lambda f, team: obs[(obs.frame == f) & (obs.team != team)][["x_m", "y_m"]].to_numpy()
    scores = [rate(s.shot_type, (s.landing_x, s.landing_y), s.team, pos_at(s.frame, s.team))
              for s in shots.itertuples()]
    shots["placement"] = scores
    shots["placement_label"] = [label(v, cfg) for v in scores]

    rows = []
    sure = r.get("landing_sure_sd", 1.0)
    for rid, g in shots.groupby("rally_id"):
        last = g.iloc[-1]
        kind = last.landing_kind
        sd = float(getattr(last, "landing_sd", np.nan)) if kind == "predicted" else 0.0
        ended = kind == "floor" or (kind == "predicted" and decisive(last.landing_x, last.landing_y, last.team, sd, sure))
        inside = ended and in_target(last.landing_x, last.landing_y, last.team)
        other = "opp" if last.team == "us" else "us"
        point = (last.team if inside else other) if ended else None
        lean = None  # not decided: which way the evidence leans (score.apply uses it as the guess)
        if not ended and kind == "predicted" and np.isfinite(last.landing_x) and np.isfinite(sd):
            # the fitted flight came down in the hitter's own half (into the net) or clearly out:
            # the point goes the other way; in the opponents' half: the hitter's
            m = target_margin(last.landing_x, last.landing_y, last.team)
            lean = last.team if m > 0 else other
            own_half = (last.landing_y >= NET_Y) != (last.team == "us")
            if own_half and abs(last.landing_y - NET_Y) > NET_SURE_SD * sd:
                ended, point = True, other  # short of the net, beyond doubt: a net error
        end_frame = int(last.end_frame) if last.end_frame >= 0 else int(last.frame)
        if real is not None and not ended:
            # unresolved ending: the rally ends where the last shot's flight track breaks off
            # (landed or left the view). Later sightings are pick-ups and toss-backs.
            seen_after = np.flatnonzero(real[int(last.frame):])
            gaps = np.flatnonzero(np.diff(seen_after) > flight_gap)
            if len(seen_after):
                end_frame = int(last.frame) + int(seen_after[gaps[0]] if len(gaps) else seen_after[-1])
            end_frame = min(end_frame, int(last.frame + r["max_flight_s"] * fps))  # no shot flies longer
        nxt_cut = [c for c in cuts if c > last.frame]
        if nxt_cut:
            end_frame = min(end_frame, int(nxt_cut[0]) - 1)
        rows.append(dict(rally_id=rid, start_frame=int(g.frame.iat[0]), end_frame=end_frame,
                         start_ts=g.timestamp.iat[0], shot_count=len(g),
                         last_hitter=last.player_id, last_shot_type=last.shot_type, ended_on_floor=ended,
                         landing_x=last.landing_x if ended else np.nan, landing_y=last.landing_y if ended else np.nan,
                         landing_in=inside if ended else None, point_team=point, point_lean=lean,
                         winning_placement=last.placement if inside else None,
                         winning_label=last.placement_label if inside else ""))
    rallies = pd.DataFrame(rows)
    if len(rallies):
        toss = pickups(rallies, shots, players, r, fps)
        if toss:
            print(f"Dropped rallies {toss}: slow shots and nobody running (picking up / tossing back)")
            shots = shots[~shots.rally_id.isin(toss)].copy()
            rallies = rallies[~rallies.rally_id.isin(toss)].copy()
            new = {old: k + 1 for k, old in enumerate(rallies.rally_id)}
            shots["rally_id"], rallies["rally_id"] = shots.rally_id.map(new), rallies.rally_id.map(new)
    if len(rallies):
        rallies["end_ts"] = rallies.start_ts + (rallies.end_frame - rallies.start_frame) / fps
        rallies["duration_s"] = (rallies.end_ts - rallies.start_ts).round(2)
    return shots, rallies


def pickups(rallies, shots, players, r, fps):
    """rally_ids that are someone tossing the shuttle back (see module docstring)."""
    if "speed_kmh_est" not in shots or players is None or players.empty or "pickup_max_kmh" not in r:
        return []
    from hud.players import speeds
    n = int(max(players.frame.max(), rallies.end_frame.max())) + 1
    v = speeds(players, np.arange(n) / fps)
    out = []
    for ral in rallies.itertuples():
        g = shots[shots.rally_id == ral.rally_id]
        if len(g) > r["pickup_max_shots"]:
            continue
        sp = g.speed_kmh_est.astype(float)
        if sp.isna().all():  # no shot speed measured: no evidence either way
            continue
        win = v[:, ral.start_frame:ral.end_frame + 1]
        if win.size == 0 or (np.isfinite(win).sum(0) >= 2).mean() < 0.5:  # players mostly untracked
            continue
        fast_shot = sp.max() >= r["pickup_max_kmh"]
        with np.errstate(invalid="ignore"):
            running = np.nanmax(win) >= r["pickup_max_mps"]
        if not fast_shot and not running:
            out.append(int(ral.rally_id))
    return out


def _fps(shots):
    d = shots[["frame", "timestamp"]].drop_duplicates().sort_values("frame")
    if len(d) < 2 or d.frame.iat[-1] == d.frame.iat[0]:
        return 30.0
    return (d.frame.iat[-1] - d.frame.iat[0]) / (d.timestamp.iat[-1] - d.timestamp.iat[0])


MANUAL_ID = 1000  # manual points get rally ids from here: the detected rallies keep theirs (and their fixes)


def add_manual(rallies, manual, fps):
    """Points the user added at a time (the analysis missed that rally): one-frame pseudo
    rallies, rally_id MANUAL_ID + k, point_source "manual" in score.apply."""
    if manual is None or not len(manual):
        return rallies
    rows = []
    for k, m in enumerate(manual.sort_values("t").itertuples()):
        f = int(round(float(m.t) * fps))
        rows.append(dict(rally_id=MANUAL_ID + k, start_frame=f, end_frame=f, start_ts=float(m.t), end_ts=float(m.t),
                         duration_s=0.0, shot_count=0, ended_on_floor=False, landing_x=np.nan, landing_y=np.nan,
                         landing_in=None, point_team=m.point_team, winning_label="", manual=True))
    out = pd.concat([rallies, pd.DataFrame(rows)], ignore_index=True) if len(rallies) else pd.DataFrame(rows)
    out["manual"] = out.get("manual", False)
    out["manual"] = out.manual.astype("boolean").fillna(False).astype(bool)
    return out.sort_values("start_ts").reset_index(drop=True)


def run(shots_path, players_path, out_shots, out_rallies, cfg_path=config.DEFAULT, shuttle_path=None,
        point_fixes=None, cuts=(), manual_points=None):
    from hud import score
    cfg = config.load(cfg_path)
    shuttle = pd.read_parquet(shuttle_path) if shuttle_path else None
    shots, rallies = build(pd.read_parquet(shots_path), pd.read_parquet(players_path), cfg, shuttle, cuts)
    manual = pd.read_csv(manual_points) if manual_points and Path(manual_points).exists() else None
    if manual is not None and len(manual):
        fps = _fps(shots) if len(shots) else 30.0
        if shuttle is not None and len(shuttle) > 1:
            fps = 1 / np.median(np.diff(shuttle.timestamp))
        rallies = add_manual(rallies, manual, fps)
    if len(rallies):
        fixes = pd.read_csv(point_fixes) if point_fixes and Path(point_fixes).exists() else None
        rallies = score.apply(rallies, cfg, fixes)
    for df, out in ((shots, out_shots), (rallies, out_rallies)):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)  # overwrite: idempotent
    print(f"Wrote {out_shots} ({len(shots)} shots) and {out_rallies} ({len(rallies)} rallies)")
    if len(rallies):
        print(rallies[["rally_id", "start_ts", "duration_s", "shot_count", "last_hitter", "last_shot_type",
                       "landing_in", "point_team_final", "point_source", "score_us", "score_opp",
                       "winning_label"]].to_string(index=False))
        unknown = rallies[rallies.point_source == "unknown"].rally_id.tolist()
        if unknown:
            print(f"No winner seen for rallies {unknown}: score unchanged there. "
                  f"Add them to a point fixes CSV (rally_id,point_team) and rerun with --point-fixes.")
    return shots, rallies
