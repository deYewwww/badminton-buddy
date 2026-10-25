"""Hit detection, hitter, speed estimate and rule-based shot type.

Hit  = the 2-D shuttle path turns sharply (angle between incoming and outgoing velocity
       over a short window), AND a player is within racket reach of the shuttle in the image.
       The hitter is whoever's wrist came nearest the shuttle (best_per_team); when the two
       sides score within TURN_MARGIN, the side whose turn it is. Forcing the sides to
       alternate over a whole exchange was tried and dropped: one stray candidate flipped
       every attribution in it.

SPEED IS AN ESTIMATE. One camera sees the shuttle as a ray, not a 3-D point. We fix it with:
  - contact point: point on the ray nearest the vertical line above the hitter's feet
  - N frames later: the point on the straight 3-D line contact -> end of shot (next contact,
    or the floor) closest to that frame's ray; if the ray runs almost along that line we fall
    back to level flight at contact height
Error sources: real flight curves (a clear climbs well above that line), the end point is
itself estimated,
the level fallback is wrong for rising/falling shots (a smash drops ~0.5 m in 0.1 s), TrackNet's centre wobbles a few px, motion blur stretches the shuttle, the hitter
may reach 0.5-1 m from their feet, focal length/pose come from the court homography only,
and frame timing quantises the 0.1 s window. Expect +-20-30 %. Values > MAX_KMH are discarded.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hud import config
from hud.camera import Camera
from hud.court import COURT_L, COURT_W, NET_Y, load, to_court

ANGLE_DEG = 50        # min turn of the shuttle path at a hit
SPEED_JUMP = 2.0      # ...or the shuttle leaves this many times faster than it came
AIR_MARGIN_M, AIR_TOP_M = 1.5, 9.0  # shuttle must be inside this box over our court
MIN_PX_S = 250        # at 1080p: in or out image speed must reach this, else the 'turn' is jitter
WINDOW_S = 0.06       # half-window for incoming/outgoing velocity
MIN_GAP_S = 0.25      # two hits can't be closer than this
RESET_GONE_S = 0.5    # shuttle unseen this long since the last hit -> new rally, alternation restarts
REACH_W, REACH_UP = 0.6, 0.7  # racket reach as fractions of the player box height
REACH_DOWN = 0.2      # below the feet line too: lunges and dives take the racket that low
SPEED_WINDOW_S = 0.1  # "first N frames after the hit"
TAKEOFF_S = 0.4       # hitter position = median observed stance over this long before the hit
WRIST_WIN_S = 0.07     # nearest wrist to the shuttle within this long of the hit
TURN_MARGIN = 0.15     # wrist scores this close: the side whose turn it is takes the hit
SUPPORT_S, SUPPORT_MIN = 0.17, 0.3  # around a hit, >= 30% of frames must be real detections
MAX_KMH = 450
MAX_GAP_GONE_S = 0.3  # sighting gap longer than this after a rally's last hit = shuttle left play
Z_RANGE = (0.2, 3.5)  # plausible contact heights, m
PICKUP_Z_M = 0.25     # lower than this is a shuttle picked up off the floor, not a stroke

# shot-type thresholds (m from net, km/h, m)
SMASH_KMH, SMASH_MIN_Z = 150, 2.0
SMASH_AVG_KMH = 60    # or: whole-flight average speed, for smashes struck along the view line
NET_ZONE = 2.2        # contact and landing both within this of the net
DEEP_ZONE = 4.3       # landing further than this from the net = clear (incl. lifts)
REAR_ZONE = 3.5       # contact further than this from the net = rear court
DROP_LAND = 2.8
DRIVE_KMH = 70


def turn_candidates(p, k, min_px=0):
    """Hit candidates: frames where the shuttle path turns by >= ANGLE_DEG, or keeps its
    direction but leaves >= SPEED_JUMP times faster (a smash on a falling shuttle looks like
    that from behind). Local maxima of the score only. p: (n,2) px with NaN; min_px: in or
    out displacement over k frames must reach this, else the 'turn' is jitter."""
    n = len(p)
    ang, score = np.zeros(n), np.zeros(n)
    for i in range(k, n - k):
        vin, vout = p[i] - p[i - k], p[i + k] - p[i]
        a, b = np.hypot(*vin), np.hypot(*vout)
        if np.isfinite(a + b) and a > 2 and b > 2 and max(a, b) >= min_px:
            ang[i] = np.degrees(np.arccos(np.clip(vin @ vout / (a * b), -1, 1)))
            jump = b / a / SPEED_JUMP if b >= 2 * min_px else 0
            score[i] = max(ang[i] / ANGLE_DEG, jump)
    return [i for i in range(n) if score[i] >= 1 and score[i] == score[max(0, i - k):i + k + 1].max()], ang


def court_airspace(cam, margin=AIR_MARGIN_M, top=AIR_TOP_M):
    """Convex image polygon of the box over the court (margin around the lines, 0..top m high)."""
    xs, ys = (-margin, COURT_W + margin), (-margin, COURT_L + margin)
    corners = [cam.project((x, y, z)) for x in xs for y in ys for z in (0.0, top)
               if (cam.R @ (np.array([x, y, z]) - cam.C))[2] > 0.1]  # skip points behind the camera
    return cv2.convexHull(np.float32(corners))


def landed_at(p, start, stop, still_px=3, still_n=5):
    """First frame in [start, stop] from which the shuttle sits still (lying on the floor)
    for still_n frames; stop if it never settles in view."""
    for q in range(start, stop - still_n + 1):
        seg = p[q:q + still_n]
        if np.isfinite(seg).all() and np.hypot(*(seg - seg[0]).T).max() <= still_px:
            return q
    return stop


def best_per_team(px, players, around):
    """{team: (score, player row)}: of each team's players whose racket-reach zone contains the
    shuttle pixel, the one whose wrist came nearest the shuttle (in body heights) over the
    frames around the hit. The box centre is no good here: a far player standing behind a
    near one has a small box whose centre lands right on the shuttle above the near head.
    Measured on the oblique clip: nearest wrist picks the right side for 8 of 10 hits."""
    best = {}
    for d, row in in_reach(px, players, with_dist=True):
        g = around[around.player_id == row.player_id]
        w = np.r_[g[["wrist_l_x", "wrist_l_y"]].to_numpy(), g[["wrist_r_x", "wrist_r_y"]].to_numpy()] \
            if "wrist_l_x" in g else np.zeros((0, 2))
        w = w[np.isfinite(w[:, 0])]
        h = row.box_y2 - row.box_y1
        dist = np.hypot(*(w - px).T).min() / h if len(w) else d + 0.3  # no wrist: box centre, penalised
        sc = 1 / (1 + dist)
        if row.team not in best or sc > best[row.team][0]:
            best[row.team] = (sc, row)
    return best


def in_reach(px, players, with_dist=False):
    """Players whose racket-reach zone contains the shuttle pixel, nearest first."""
    out = []
    for r in players.itertuples():
        if np.isnan(r.box_x1):
            continue
        h = r.box_y2 - r.box_y1
        if (r.box_x1 - REACH_W * h <= px[0] <= r.box_x2 + REACH_W * h
                and r.box_y1 - REACH_UP * h <= px[1] <= r.box_y2 + REACH_DOWN * h):
            d = np.hypot(px[0] - (r.box_x1 + r.box_x2) / 2, px[1] - (r.box_y1 + r.box_y2) / 2) / h
            out.append((d, r))
    out.sort(key=lambda x: x[0])
    return out if with_dist else [r for _, r in out]


def shot_type(contact, land, kmh, avg_kmh=None):
    dn_hit, dn_land = abs(contact[1] - NET_Y), abs(land[1] - NET_Y)
    fast = (kmh is not None and kmh >= SMASH_KMH) or (avg_kmh is not None and avg_kmh >= SMASH_AVG_KMH)
    if fast and contact[2] >= SMASH_MIN_Z and dn_land < DEEP_ZONE + 1:
        return "smash"
    if dn_hit < NET_ZONE and dn_land < NET_ZONE:
        return "net"
    if dn_land >= DEEP_ZONE:
        return "clear"
    if dn_hit >= REAR_ZONE and dn_land < DROP_LAND:
        return "drop"
    if kmh is not None and kmh >= DRIVE_KMH:
        return "drive"
    return "other"


def detect(shuttle, players, H, cam, fps, img_h=1080, serve_answer_s=1.25):
    p = shuttle[["x_px", "y_px"]].to_numpy(float, copy=True)
    # multi-court halls: TrackNet happily follows other courts' shuttles; keep only what is
    # inside the image of the air box above our court
    box = court_airspace(cam)
    outside = ~np.array([np.isfinite(q[0]) and cv2.pointPolygonTest(box, (float(q[0]), float(q[1])), False) >= 0
                         for q in p])
    p[outside] = np.nan
    t = shuttle.timestamp.to_numpy()
    k = max(2, round(WINDOW_S * fps))
    # boxes: forward-fill short gaps (occlusion / jump), the moment of a smash is exactly when
    # the foot point is unreliable but the body box is still the right place to look
    pl = players.sort_values("frame").copy()
    box_cols = ["box_x1", "box_y1", "box_x2", "box_y2"]
    pl[box_cols] = pl.groupby("player_id")[box_cols].ffill(limit=round(0.5 * fps))
    by_frame = dict(tuple(pl.groupby("frame")))
    players_obs = players[players.observed]

    cands, ang = turn_candidates(p, k, MIN_PX_S * k / fps * img_h / 1080)
    seen = np.isfinite(p[:, 0])
    # real sightings only: InpaintNet guesses and interpolation can't vouch for a hit
    real = seen & ~shuttle.inpainted.to_numpy() & ~shuttle.interpolated.to_numpy() if "inpainted" in shuttle else seen
    sup = max(1, round(SUPPORT_S * fps))
    support = lambda i: real[max(0, i - sup):i + sup + 1].mean()
    # candidates with both teams' best hitter; teams are decided per exchange below
    found = []
    WRIST_WIN = max(1, round(WRIST_WIN_S * fps))
    for i in cands:
        if support(i) < SUPPORT_MIN:
            continue
        if found and t[i] - t[found[-1][0]] < MIN_GAP_S:
            continue
        around = pl[(pl.frame >= i - WRIST_WIN) & (pl.frame <= i + WRIST_WIN)]
        opts = best_per_team(p[i], by_frame.get(i, pl.iloc[:0]), around)
        if opts:
            found.append((i, opts))
    # exchanges: split where the shuttle vanished for a while or play paused
    groups, cur = [], []
    for i, opts in found:
        if cur:
            j = cur[-1][0]
            gone = pd.Series(~seen[j:i]).groupby(seen[j:i].cumsum()).sum().max()
            if gone / fps >= RESET_GONE_S or t[i] - t[j] > serve_answer_s * 2:
                groups.append(cur)
                cur = []
        cur.append((i, opts))
    groups += [cur] if cur else []
    hits = []
    for g in groups:
        # the opener must be answered quickly, else it was the server fiddling before serving
        while len(g) > 1 and t[g[1][0]] - t[g[0][0]] > serve_answer_s:
            g = g[1:]
        prev = None
        for i, opts in g:
            team = max(opts, key=lambda k: opts[k][0])  # nearest wrist wins; see best_per_team
            # ...unless it's close: then the side whose turn it is (from behind, a shuttle over
            # a near head also sits on the far player's raised wrist)
            other = "opp" if team == "us" else "us"
            if team == prev and other in opts and opts[team][0] - opts[other][0] < TURN_MARGIN:
                team = other
            who = opts[team][1]
            # stance before the swing: at a jump smash the foot point projects metres too deep
            before = players_obs[(players_obs.player_id == who.player_id)
                                 & (players_obs.frame > i - TAKEOFF_S * fps) & (players_obs.frame <= i)]
            stance = before[["x_m", "y_m"]].median().to_numpy() if len(before) else (who.x_m, who.y_m)
            contact = cam.near_vertical(p[i], stance)
            if contact[2] < PICKUP_Z_M:  # scooped off the floor: tossing it back between points
                continue
            contact[2] = np.clip(contact[2], *Z_RANGE)
            prev = team
            hits.append(dict(frame=i, timestamp=t[i], player_id=who.player_id, team=who.team,
                             shuttle_x_px=p[i, 0], shuttle_y_px=p[i, 1], turn_deg=round(ang[i], 1),
                             contact_x_m=contact[0], contact_y_m=contact[1], contact_z_m=contact[2]))

    rows = []
    for n, h in enumerate(hits):
        i = h["frame"]
        c0 = np.array([h["contact_x_m"], h["contact_y_m"], h["contact_z_m"]])
        # landing: next hitter's contact if the rally goes on, else last sighting on the floor
        nxt = hits[n + 1] if n + 1 < len(hits) else None
        if nxt and nxt["team"] != h["team"]:
            land, kind, end = (nxt["contact_x_m"], nxt["contact_y_m"]), "intercept", nxt["frame"]
            end3 = np.array([nxt["contact_x_m"], nxt["contact_y_m"], nxt["contact_z_m"]])
        else:
            stop = nxt["frame"] if nxt else len(p)
            q, last_seen = i + 1, None
            while q < stop and (seen[q] or (last_seen is not None and t[q] - t[last_seen] <= MAX_GAP_GONE_S)):
                last_seen = q if seen[q] else last_seen
                q += 1
            land, kind, end = (np.nan, np.nan), "unknown", last_seen
            if last_seen is not None:
                # where it settled, not where it was last seen: it may roll or get picked up
                settled = landed_at(p, i + 1, last_seen)
                g = to_court(H, [p[settled]])[0]
                if -3 < g[0] < COURT_W + 3 and -3 < g[1] < COURT_L + 3:  # else it left the frame in the air
                    land, kind, end = tuple(g), "floor", settled
                    end3 = np.array([g[0], g[1], 0.0])
        # speed: last detection within SPEED_WINDOW_S, placed on the straight 3-D line from
        # contact to where the shot ended; level-flight fallback when the view runs along it
        j_end = np.searchsorted(t, t[i] + SPEED_WINDOW_S, side="right")
        js = [j for j in range(i + 1, min(j_end, len(p))) if seen[j] and not shuttle.interpolated.iat[j]]
        kmh, method = None, None
        if js:
            j = js[-1]
            c1 = cam.on_line(p[j], c0, end3) if np.isfinite(land[0]) else None
            method = "line" if c1 is not None else "level"
            if c1 is None:
                c1 = cam.at_height(p[j], c0[2])
            kmh = float(np.linalg.norm(c1 - c0) / (t[j] - t[i]) * 3.6)
            if not 0 < kmh <= MAX_KMH:
                kmh, method = None, "rejected"
        # whole-flight average (contact -> interception): a floor for the initial speed, and the
        # only speed cue left when a shot flies straight along the camera's line of sight
        avg = None
        if kind == "intercept":
            avg = float(np.linalg.norm(end3 - c0) / (nxt["timestamp"] - h["timestamp"]) * 3.6)
        rows.append({**h, "speed_kmh_est": None if kmh is None else round(kmh, 1), "speed_method": method,
                     "flight_avg_kmh_est": None if avg is None else round(avg, 1),
                     "shot_type": shot_type(c0, land, kmh, avg) if np.isfinite(land[0]) else "other",
                     "landing_x": land[0], "landing_y": land[1], "landing_kind": kind,
                     "end_frame": -1 if end is None else int(end)})
    df = pd.DataFrame(rows)
    if len(df):
        df.insert(0, "shot_no", np.arange(1, len(df) + 1))
        for c in ("contact_x_m", "contact_y_m", "contact_z_m", "landing_x", "landing_y"):
            df[c] = df[c].astype(float).round(2)
    return df


def run(shuttle_path, players_path, calib_path, out):
    shuttle, players = pd.read_parquet(shuttle_path), pd.read_parquet(players_path)
    H, calib = load(calib_path)
    cam = Camera(H, calib["frame_size"])
    fps = 1 / np.median(np.diff(shuttle.timestamp)) if len(shuttle) > 1 else 30
    df = detect(shuttle, players, H, cam, fps, calib["frame_size"][1], config.load()["rally"]["serve_answer_s"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)  # overwrite: idempotent
    print(f"Wrote {out}: {len(df)} hits")
    if len(df):
        print(df[["shot_no", "frame", "timestamp", "player_id", "contact_z_m", "speed_kmh_est", "flight_avg_kmh_est",
                  "shot_type", "landing_x", "landing_y", "landing_kind"]].to_string(index=False))
    return df
