"""Hit detection, hitter, speed estimate and rule-based shot type.

Hit  = the 2-D shuttle path turns sharply (angle between incoming and outgoing velocity
       over a short window), AND a player is within racket reach of the shuttle in the image.
       Which side hit is decided for a whole exchange at once (assign_sides, Viterbi): each
       hit costs per side by evidence (can that side's player physically reach the shuttle,
       how far the ray passes from them, their nearest wrist), and the same side hitting twice
       in a row costs a penalty, a smaller one where the shuttle was lost between the two
       (a missed hit is likely there). Deciding hit by hit with "sides alternate" as the tie
       break was tried: one missed hit flipped every attribution after it.

SPEED IS AN ESTIMATE. One camera sees the shuttle as a ray, not a 3-D point.
  - contact point: point on the ray nearest the vertical line above the hitter's feet
  - initial speed (fit_speed): a shuttle loses speed fast (quadratic drag, terminal velocity
    ~6.7 m/s), so the average over the first 0.1 s underrates a 300 km/h smash by ~40 %.
    Instead the flight s(t) = ln(1 + k v0 t) / k along a 3-D direction, plus gravity, is fitted
    to the detected pixels of the first ~0.25 s (reprojection error: no 3-D points are ever
    made up from single rays). Unknowns: direction, v0, and the contact time inside its frame.
    k is fixed by physics; fitting it too from 4-7 noisy points trades off against v0.
  - fallback when the fit is unsure (too few frames, shot along the line of sight): the point
    on the straight line contact -> end of shot nearest the ray 0.1 s later; if the ray runs
    almost along that line, level flight at contact height
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
from hud.court.camera import Camera
from hud.court import COURT_L, COURT_W, NET_Y, load, to_court

ANGLE_DEG = 50        # min turn of the shuttle path at a hit
SPEED_JUMP = 2.0      # ...or the shuttle leaves this many times faster than it came
AIR_MARGIN_M, AIR_TOP_M = 1.5, 9.0  # shuttle must be inside this box over our court
MIN_PX_S = 250        # at 1080p: in or out image speed must reach this, else the 'turn' is jitter
WINDOW_S = 0.06       # half-window for incoming/outgoing velocity
MIN_GAP_S = 0.18      # two hits can't be closer than this (a smash returned at the net: 0.23 s)
RESET_GONE_S = 0.5    # shuttle unseen this long since the last hit -> new rally, alternation restarts
REACH_W, REACH_UP = 0.6, 0.7  # racket reach as fractions of the player box height
REACH_DOWN = 0.2      # below the feet line too: lunges and dives take the racket that low
SPEED_WINDOW_S = 0.1  # "first N frames after the hit"
TOP_EDGE = 0.12       # shuttle last/next seen this close to the top of the frame: it flew over it
DRAG_K = 0.22         # 1/m, quadratic drag of a shuttle: g / v_terminal^2, v_terminal ~6.7 m/s
FIT_MAX_S = 1.5       # initial-speed fit: detections of the shot's flight up to this long
FIT_MIN_PTS, FIT_MAX_PTS = 4, 30
FIT_END_M = 0.4       # how well the end of a shot (next contact / landing) is known, m
FIT_MAX_REL_SD = 0.4   # reject a fit whose v0 standard error is larger than this fraction. Simulated smashes
                      # and drops: 0.25 kept 34 % (median error 11 %), 0.4 keeps 84 % (16 %); the fallback
                      # is 20-40 % low
FIT_PX = 2.0          # TrackNet centre noise at 1080p, px (scales the residuals)
FIT_V0 = (3.0, 139.0)  # m/s bounds, 11..500 km/h
REACH_M = 2.0         # a hitter's racket is at most this far (horizontally) from their feet: a full
                      # lunge (step + arm + racket) reaches 1.5-1.8 m; 1.6 lost a far-court retrieve
GEO_Z = (-0.2, 3.6)   # ... at this height: below the floor or above a jump smash = not them
TAKEOFF_S = 0.4       # hitter position = median observed stance over this long before the hit
WRIST_WIN_S = 0.07     # nearest wrist to the shuttle within this long of the hit
GEO_COST = 3.0        # assign_sides: that side's player can't reach the shuttle (geo_fit fails)
MISS_W = 1.5          # ... per metre the ray passes from their vertical
WRIST_W = 2.5         # ... times (1 - nearest-wrist score). Weights: grid search on one real rally in
                      # 3 versions (original, 480p noisy, mirrored+cropped), 63 hits: wrong side 8 -> 1
SAME_COST = 2.0       # the same side hits twice in a row (only possible if a hit was missed)
SAME_LOST_COST = 0.8  # ... where the shuttle went unseen for >= 3 frames between the two
SUPPORT_S, SUPPORT_MIN = 0.17, 0.3  # around a hit, >= 30% of frames must be real detections
MAX_KMH = 450
MISSED_HIT_S = 2.0    # same side hits again within this: not a landing, a hit was missed
DOUBLE_S = 0.6        # ... but within this, the same stroke seen twice (see detect)
FLOOR_PX = 5          # at 1080p: how far a floor landing's pixel may be off (calibration, stabilisation, centre)
LAND_JUMP = 0.12      # x pixel scale: the last step into a landing spot is at most this long
STILL_PX = 12         # at 1080p: a shuttle lying on the floor stays within this of its median (TrackNet's
                      # centre wobbles 5-7 px at 720p; 3 px never let the ACE clip's floor landings settle)
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
    for still_n frames (every sighting within still_px of their median; a missed frame or two
    allowed); stop if it never settles in view."""
    for q in range(start, stop - still_n + 1):
        seg = p[q:q + still_n]
        ok = np.isfinite(seg[:, 0])
        if ok[0] and ok.sum() >= still_n - 2 and np.hypot(*(seg[ok] - np.median(seg[ok], 0)).T).max() <= still_px:
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


def geo_fit(cam, px, row):
    """Could this player have hit a shuttle seen at pixel px? The viewing ray must pass within
    REACH_M of the vertical line above their feet, at a height a racket can reach. Measured in
    metres, so a big near player gets no advantage over a small far one (wrist distances in
    pixels, normalised by body height, credited a far player's diving get to the near player
    on the broadcast test clip). -> (ok, horizontal miss in m)."""
    c = cam.near_vertical(px, (row.x_m, row.y_m))
    miss = float(np.hypot(c[0] - row.x_m, c[1] - row.y_m))
    return miss <= REACH_M and GEO_Z[0] <= c[2] <= GEO_Z[1], miss


def flight(v0, u, taus, k=DRAG_K, h=1 / 150, max_steps=60):
    """Shuttle position (m, relative to contact) at times taus (s, >= 0) after leaving the racket
    at v0 m/s along unit vector u: dv/dt = -k |v| v - g, RK4 with plain floats (fast for the
    few dozen steps of a fit). Drag and gravity are coupled: the 1-D closed form
    s = ln(1 + k v0 t) / k ignores that, and the ~0.1 m it is off by in 0.25 s is 10-20 px
    to a phone 10 m away, enough to pull a smash fit 35 % low."""
    taus = np.clip(np.asarray(taus, float), 0, None)
    h = max(h, taus.max() / max_steps)  # long flights: coarser steps, RK4 stays well under 1 cm...
    h = min(h, 0.3 / (k * max(v0, 1e-6)))  # ...but k*v0*h <= 0.3, or a fast start blows RK4 up
    n = int(taus.max() / h) + 2
    vx, vy, vz = v0 * u[0], v0 * u[1], v0 * u[2]
    x = y = z = 0.0
    P = [(0.0, 0.0, 0.0)]

    def acc(vx, vy, vz):
        m = -k * (vx * vx + vy * vy + vz * vz) ** 0.5
        return m * vx, m * vy, m * vz - 9.81
    for _ in range(n):
        a1 = acc(vx, vy, vz)
        b2 = (vx + h / 2 * a1[0], vy + h / 2 * a1[1], vz + h / 2 * a1[2])
        a2 = acc(*b2)
        b3 = (vx + h / 2 * a2[0], vy + h / 2 * a2[1], vz + h / 2 * a2[2])
        a3 = acc(*b3)
        b4 = (vx + h * a3[0], vy + h * a3[1], vz + h * a3[2])
        a4 = acc(*b4)
        x += h / 6 * (vx + 2 * b2[0] + 2 * b3[0] + b4[0])
        y += h / 6 * (vy + 2 * b2[1] + 2 * b3[1] + b4[1])
        z += h / 6 * (vz + 2 * b2[2] + 2 * b3[2] + b4[2])
        vx += h / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
        vy += h / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
        vz += h / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
        P.append((x, y, z))
    P = np.array(P)
    g = taus / h
    i0 = np.floor(g).astype(int)
    w = (g - i0)[:, None]
    return P[i0] * (1 - w) + P[i0 + 1] * w


def from_window_avg(avg, window, k=DRAG_K):
    """Initial speed (m/s) of a shuttle that averaged `avg` m/s over its first `window` s:
    under quadratic drag the distance is ln(1 + k v0 t) / k, so v0 = (e^(k avg t) - 1) / (k t).
    A 300 km/h smash averages ~150 km/h over 0.1 s; this undoes that."""
    return float(np.expm1(min(k * avg * window, 5.0)) / (k * window))


def fit_speed(cam, c0, dt, px, end=None, fps=30.0, img_h=1080, guess=None):
    """Initial speed (m/s) of a shot, or None if it can't be pinned down.

    dt: (N,) seconds after the hit frame; px: (N,2) detections of this shot's whole flight;
    c0: 3-D contact estimate; end: (dt_end, xyz) where the shot ended (next hitter's contact
    or the floor) or None; guess: a speed to start from (m/s).
    The physical flight c0 + depth * ray + flight(v0, u, dt - t0) is fitted to the pixels by
    robust least squares over direction u, v0, t0 (contact time inside its frame) and the
    contact's depth along its viewing ray (the hitter-vertical guess is ~0.3 m off: tens of px
    to a near camera). The end point anchors the depth: without it a shot flying along the
    line of sight is ambiguous. Whole flight, not the first frames: the physics holds all the
    way, and the deceleration over the flight is what fixes v0.
    Rejected when the reprojection error stays large, v0 ends on a bound, or its standard
    error exceeds 25 %."""
    from scipy.optimize import least_squares
    dt, px = np.asarray(dt, float), np.asarray(px, float)
    if len(dt) > FIT_MAX_PTS:  # evenly thinned: the fit cost grows with points and flight time
        keep = np.unique(np.linspace(0, len(dt) - 1, FIT_MAX_PTS).round().astype(int))
        dt, px = dt[keep], px[keep]
    if len(dt) < FIT_MIN_PTS:
        return None
    sig = FIT_PX * img_h / 1080
    unit = lambda a, e: np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    ray = c0 - cam.C
    ray /= np.linalg.norm(ray)
    t_all = dt if end is None else np.r_[dt, end[0]]

    def path(q):
        return c0 + q[4] * ray + flight(q[2], unit(q[0], q[1]), t_all - q[3])

    def resid(q):
        X = path(q)
        img, depth = cam.project_many(X[:len(dt)])
        if not np.isfinite(img).all():
            return np.full(2 * len(dt) + (3 if end is not None else 0), 1e3)
        r = ((img - px) / sig).ravel()
        r = np.where(np.repeat(depth, 2) > 0.1, r, 1e3)  # behind the camera: reject
        if end is not None:
            r = np.r_[r, (X[-1] - end[1]) / FIT_END_M]
        return r

    if end is not None and np.linalg.norm(end[1] - c0) > 0.5:
        u0 = (end[1] - c0) / np.linalg.norm(end[1] - c0)
    else:  # no end of shot known: level flight through the last sighting
        g = cam.at_height(px[-1], c0[2]) - c0
        u0 = g / (np.linalg.norm(g) + 1e-9)
    a0, e0 = np.arctan2(u0[1], u0[0]), np.arcsin(np.clip(u0[2], -0.97, 0.97))
    lo, hi = [-np.inf, -1.4, FIT_V0[0], -1 / fps, -1.5], [np.inf, 1.4, FIT_V0[1], 1 / fps, 1.5]
    if guess is None and end is not None and end[0] > 0:  # whole-flight average, undone for drag
        guess = from_window_avg(np.linalg.norm(end[1] - c0) / end[0], end[0])
    starts = [guess] if guess else []
    starts += [v for v in (25.0, 70.0) if not guess or abs(v - guess) > 10]

    def judge(sol):
        v0 = sol.x[2]
        img = cam.project_many(path(sol.x)[:len(dt)])[0]
        rms = np.sqrt(np.mean(np.sum((img - px) ** 2, 1))) / sig
        if rms > 2.5 or not (FIT_V0[0] * 1.02 < v0 < FIT_V0[1] * 0.98):
            return None
        try:
            sd = np.sqrt(np.linalg.inv(sol.jac.T @ sol.jac)[2, 2])
        except np.linalg.LinAlgError:
            return None
        return float(v0) if sd < FIT_MAX_REL_SD * v0 else None
    best = None
    for v in starts:  # the first start usually lands; others only when it is rejected
        x0 = [a0, e0, float(np.clip(v, FIT_V0[0] * 1.01, FIT_V0[1] * 0.99)), 0.0, 0.0]
        sol = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=2.0, ftol=1e-4, xtol=1e-4,
                            x_scale=[0.3, 0.3, 10.0, 0.01, 0.3])  # 1e-4: v0 to well under 1 %
        if best is None or sol.cost < best.cost:
            best = sol
            v0 = judge(sol)
            if v0 is not None:
                return v0
    return None


def assign_sides(g, p, cam, real):
    """g: [(frame, {team: (wrist score, player row)})] of one exchange -> team per hit, the
    cheapest sequence under the costs above (Viterbi over the two sides)."""
    teams = ("us", "opp")

    def emit(i, opts, team):
        if team not in opts:
            return np.inf
        ok, miss = geo_fit(cam, p[i], opts[team][1])
        return (0 if ok else GEO_COST) + MISS_W * miss + WRIST_W * (1 - opts[team][0])
    cost = np.array([emit(g[0][0], g[0][1], tm) for tm in teams])
    back = []
    for (j, _), (i, opts) in zip(g, g[1:]):
        lost = (~real[j + 1:i]).sum() >= 3
        same = SAME_LOST_COST if lost else SAME_COST
        trans = np.where(np.eye(2, dtype=bool), same, 0.0)  # same side twice: penalty
        tot = cost[:, None] + trans                          # from (rows) -> to (cols)
        back.append(tot.argmin(0))
        cost = tot.min(0) + np.array([emit(i, opts, tm) for tm in teams])
    k = int(np.argmin(cost))
    path = [k]
    for b in reversed(back):
        k = int(b[k])
        path.append(k)
    return [teams[k] for k in reversed(path)]


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


def detect(shuttle, players, H, cam, fps, img_h=1080, serve_answer_s=1.25, cuts=(), frame_h=None):
    """img_h: the pixel scale (camera short side, 1080 at 1080p); frame_h: the frame's own height."""
    from hud.analysis import landing
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
        here = by_frame.get(i, pl.iloc[:0])
        opts = best_per_team(p[i], here, around)
        # a diving or stretching player's racket is far outside their box: let geometry vouch
        for r in here.itertuples():
            if r.team not in opts and np.isfinite(r.x_m) and geo_fit(cam, p[i], r)[0]:
                opts[r.team] = (0.0, r)
        if opts:
            found.append((i, opts))
    # exchanges: split where the shuttle vanished for a while or play paused. A high clear
    # that leaves through the top of the frame and comes back down is still in play.
    top = TOP_EDGE * (frame_h or img_h)

    def gone_s(j, i):
        longest, q = 0, j
        while q < i:
            if seen[q]:
                q += 1
                continue
            r = q
            while r < i and not seen[r]:
                r += 1
            over_top = q > 0 and r < len(p) and seen[q - 1] and p[q - 1, 1] < top and p[r, 1] < top
            if not over_top:
                longest = max(longest, r - q)
            q = r
        return longest / fps
    groups, cur = [], []
    for i, opts in found:
        if cur:
            j = cur[-1][0]
            gone = gone_s(j, i) * fps
            cut = any(j < c <= i for c in cuts)  # edited clip: the two sides of a cut are different moments
            if cut or gone / fps >= RESET_GONE_S or t[i] - t[j] > serve_answer_s * 2:
                groups.append(cur)
                cur = []
        cur.append((i, opts))
    groups += [cur] if cur else []
    hits = []
    for g in groups:
        # the opener must be answered quickly, else it was the server fiddling before serving
        while len(g) > 1 and t[g[1][0]] - t[g[0][0]] > serve_answer_s:
            g = g[1:]
        def contact_of(i, who):
            # stance before the swing: at a jump smash the foot point projects metres too deep
            before = players_obs[(players_obs.player_id == who.player_id)
                                 & (players_obs.frame > i - TAKEOFF_S * fps) & (players_obs.frame <= i)]
            stance = before[["x_m", "y_m"]].median().to_numpy() if len(before) else (who.x_m, who.y_m)
            return cam.near_vertical(p[i], stance)
        # sides for the whole exchange; a hit scooped off the floor (tossing the shuttle back) is
        # dropped and the sides decided again without it, else it leaves a same-side pair behind
        while g:
            sides = assign_sides(g, p, cam, real)
            low = [k for k, ((i, opts), tm) in enumerate(zip(g, sides)) if contact_of(i, opts[tm][1])[2] < PICKUP_Z_M]
            if not low:
                break
            g = [x for k, x in enumerate(g) if k not in low]
        for (i, opts), team in zip(g, sides if g else []):
            who = opts[team][1]
            contact = contact_of(i, who)
            contact[2] = np.clip(contact[2], *Z_RANGE)
            hit = dict(frame=i, timestamp=t[i], player_id=who.player_id, team=who.team,
                       shuttle_x_px=p[i, 0], shuttle_y_px=p[i, 1], turn_deg=round(ang[i], 1),
                       contact_x_m=contact[0], contact_y_m=contact[1], contact_z_m=contact[2])
            # the other side's player, if they could have hit it too (see the doubles check below)
            other = next((o[1] for tm, o in opts.items() if tm != team), None)
            alt = None
            if other is not None:
                ok, miss = geo_fit(cam, p[i], other)
                if ok:
                    c = contact_of(i, other)
                    c[2] = np.clip(c[2], *Z_RANGE)
                    alt = dict(hit, player_id=other.player_id, team=other.team, contact_x_m=c[0],
                               contact_y_m=c[1], contact_z_m=c[2], miss=miss)
            hits.append(dict(hit, alt=alt))

    # one side twice within DOUBLE_S: the shuttle can't cross the court and come back that fast.
    # If the other side could have hit one of them (in reach), it was a fast exchange with that
    # hit given to the wrong side: flip it (dropping instead cost 2 real hits of a noisy broadcast
    # rally). Otherwise a single stroke whose path the tracker saw turn twice (racket
    # follow-through, a frame of the racket head): keep the sharper turn. On the ACE clip 25 of 28
    # same-side pairs were this, 0.2-0.5 s apart, with the other side 3-7 m out of reach; and
    # each one hid the real last shot's landing (a "same side again soon" means play went on).
    hits = sorted(hits, key=lambda h: h["frame"])
    k = 1
    while k < len(hits):
        a, b = hits[k - 1], hits[k]
        if a["team"] == b["team"] and b["timestamp"] - a["timestamp"] < DOUBLE_S:
            flip = [x for x in (k - 1, k) if hits[x]["alt"] is not None]
            if flip:  # the other side could have hit one of them: a fast exchange with that side wrong
                x = min(flip, key=lambda x: hits[x]["alt"]["miss"])
                hits[x] = dict(hits[x]["alt"], alt=None)
            else:
                hits.pop(k if a["turn_deg"] >= b["turn_deg"] else k - 1)
                continue
        k += 1
    for h in hits:
        h.pop("alt", None)
        h.pop("miss", None)
    rows = []
    for n, h in enumerate(hits):
        i = h["frame"]
        c0 = np.array([h["contact_x_m"], h["contact_y_m"], h["contact_z_m"]])
        # landing: next hitter's contact if the rally goes on, else last sighting on the floor
        nxt = hits[n + 1] if n + 1 < len(hits) else None
        land_sd = np.nan
        if nxt and nxt["team"] != h["team"]:
            land, kind, end = (nxt["contact_x_m"], nxt["contact_y_m"]), "intercept", nxt["frame"]
            end3 = np.array([nxt["contact_x_m"], nxt["contact_y_m"], nxt["contact_z_m"]])
        else:
            stop = nxt["frame"] if nxt else len(p)
            q, last_seen = i + 1, None
            while q < stop and (seen[q] or (last_seen is not None and t[q] - t[last_seen] <= MAX_GAP_GONE_S)):
                last_seen = q if seen[q] else last_seen
                q += 1
            land, kind, end, land_sd = (np.nan, np.nan), "unknown", last_seen, np.nan
            # the same side hits again soon: a hit in between was missed, the shuttle never
            # landed (calling it a landing ended the rally and threw away the hits after it)
            soon = nxt is not None and nxt["timestamp"] - h["timestamp"] < MISSED_HIT_S
            settled, g = None, None
            if last_seen is not None:
                # where it settled, not where it was last seen: it may roll or get picked up.
                # Still for 0.3 s: the top of a high clear seen from behind hangs for ~5 frames.
                # The track must run into the still spot (a jump onto it is TrackNet locking on
                # a shoe or a line). Seen lying there, it landed even if the same side "hits"
                # again soon: that is the pick-up.
                q = landed_at(p, i + 1, last_seen, still_px=STILL_PX * img_h / 1080, still_n=max(5, round(0.3 * fps)))
                before = [j for j in range(max(i + 1, q - 3), q) if seen[j]]
                # ... and look down at the floor: the top of a high serve hangs still against the
                # ceiling for 0.3 s too, and the homography "projects" it behind the camera
                if (q < last_seen and before and np.hypot(*(p[q] - p[before[-1]])) < LAND_JUMP * img_h
                        and cam.ray(p[q])[2] < -0.01):
                    g = to_court(H, [p[q]])[0]
                    settled = q if -3 < g[0] < COURT_W + 3 and -3 < g[1] < COURT_L + 3 else None
            if settled is not None:
                land, kind, end = tuple(g), "floor", settled  # seen lying on the floor
                end3 = np.array([g[0], g[1], 0.0])
                # how far off that floor point may be: FLOOR_PX of pixel error (calibration,
                # stabilisation, TrackNet's centre) through the homography at that pixel. From a
                # 0.7 m phone that is ~0.7 m near the net, so close calls stay undecided
                from hud.tracking.smooth import foot_cov
                land_sd = float(np.sqrt(np.linalg.eigvalsh(foot_cov(H, [p[settled]], [FLOOR_PX * img_h / 1080])[0]).max()))
            elif last_seen is not None and not soon:  # lost in the air: carry the fitted flight to the floor
                pts = [j for j in range(i + 1, last_seen + 1) if real[j]]
                est = landing.predict(cam, c0, t[pts] - t[i], p[pts], fps, img_h) if len(pts) >= 5 else None
                if est is not None and -3 < est["xy"][0] < COURT_W + 3 and -3 < est["xy"][1] < COURT_L + 3:
                    land, kind, land_sd = tuple(est["xy"]), "predicted", est["sd"]
        # speed: last detection within SPEED_WINDOW_S, placed on the straight 3-D line from
        # contact to where the shot ended; level-flight fallback when the view runs along it
        j_end = np.searchsorted(t, t[i] + SPEED_WINDOW_S, side="right")
        js = [j for j in range(i + 1, min(j_end, len(p))) if seen[j] and not shuttle.interpolated.iat[j]]
        kmh, method = None, None
        stop = len(p) if end is None else min(len(p), int(end))  # up to, not including, the next contact
        fit_js = [j for j in range(i + 1, min(stop, i + 1 + round(FIT_MAX_S * fps))) if real[j]]
        fin = (t[end] - t[i], end3) if kind in ("intercept", "floor") and end is not None else None
        v0 = fit_speed(cam, c0, t[fit_js] - t[i], p[fit_js], fin, fps, img_h) if len(fit_js) >= FIT_MIN_PTS else None
        if v0 is not None:
            kmh, method = v0 * 3.6, "drag_fit"
        elif js:
            j = js[-1]
            c1 = cam.on_line(p[j], c0, end3) if kind in ("intercept", "floor") else None
            method = "line" if c1 is not None else "level"
            if c1 is None:
                c1 = cam.at_height(p[j], c0[2])
            # window average -> initial speed under drag; the flight began ~half a frame
            # before the hit frame on average
            w = t[j] - t[i] + 0.5 / fps
            kmh = from_window_avg(float(np.linalg.norm(c1 - c0)) / w, w) * 3.6
            method += "+drag"
            if not 0 < kmh <= MAX_KMH:
                kmh, method = None, "rejected"
        # whole-flight average (contact -> interception): a floor for the initial speed, and the
        # only speed cue left when a shot flies straight along the camera's line of sight
        avg = None
        if kind == "intercept":
            avg = float(np.linalg.norm(end3 - c0) / (nxt["timestamp"] - h["timestamp"]) * 3.6)
            if kmh is not None and kmh < 0.9 * avg:  # a shuttle only slows down: the estimate is broken
                kmh, method = None, "rejected"
        rows.append({**h, "speed_kmh_est": None if kmh is None else round(kmh, 1), "speed_method": method,
                     # sure: the physical flight fit held. Only those speeds and shot types are shown in
                     # the HUD: the window fallbacks gave 3-409 km/h on the user's 4K clip (fit: 12-197)
                     # and the types built on them were "basically all wrong". Scoring keeps using all
                     # of them (hiding them there dropped real 2-shot rallies as pick-ups)
                     "sure": method == "drag_fit",
                     "flight_avg_kmh_est": None if avg is None else round(avg, 1),
                     "shot_type": shot_type(c0, land, kmh, avg) if np.isfinite(land[0]) else "other",
                     "landing_x": land[0], "landing_y": land[1], "landing_kind": kind, "landing_sd": land_sd,
                     "end_frame": -1 if end is None else int(end)})
    df = pd.DataFrame(rows)
    if len(df):
        df.insert(0, "shot_no", np.arange(1, len(df) + 1))
        for c in ("contact_x_m", "contact_y_m", "contact_z_m", "landing_x", "landing_y"):
            df[c] = df[c].astype(float).round(2)
    return df


SERVE_GAP_S = 2.5     # a hit after this long without one opens play: look for the serve before it
SERVE_LOOK_S = 2.0    # ... in the flight that came in, up to this long back
SERVE_RUN_MIN = 4     # ... which must have at least this many real sightings
SERVE_LEAD_S = 0.2    # ... and start at least this long before that hit
SERVE_Z_M = 1.1       # a recovered serve's contact: above the server's feet at this height


def add_serves(df, shuttle, players, fps):
    """The serve is mostly missed: on the user's 16 hand-marked rallies the first hit found came
    0.7-0.9 s after the marked serve (the return), so rallies started late and a shot short, and
    the 1-2 shot ones were dropped as noise. The serve is where the flight into that first hit
    began: walk back over the real sightings before it (gaps up to 3 frames) and put a hit by the
    other side at the start of that run (the shuttle shows a frame or so after leaving the racket;
    at contact it is often behind the server). No run, no serve: nothing is invented."""
    if df.empty:
        return df
    real = (shuttle.visible & ~shuttle.interpolated & ~shuttle.inpainted).to_numpy()
    px = shuttle[["x_px", "y_px"]].to_numpy(float)
    ts = shuttle.timestamp.to_numpy(float)
    df = df.sort_values("frame").reset_index(drop=True)
    first = np.r_[True, np.diff(df.timestamp.to_numpy()) > SERVE_GAP_S]
    add = []
    obs = players[players.observed.astype(bool) & players.box_x1.notna()]
    for i in np.flatnonzero(first):
        h = df.iloc[i]
        f1, k, start, gap = int(h.frame), int(h.frame) - 1, None, 0
        while k > 0 and ts[f1] - ts[k] < SERVE_LOOK_S:
            if real[k]:
                start, gap = k, 0
            else:
                gap += 1
                if gap > 3 and start is not None:
                    break
            k -= 1
        if start is None or real[start:f1].sum() < SERVE_RUN_MIN or ts[f1] - ts[start] < SERVE_LEAD_S:
            continue
        fs = max(0, start - 1)
        team = "opp" if h.team == "us" else "us"
        cand = obs[(obs.team == team) & (obs.frame.between(fs - 3, fs + 3))]
        feet = (np.nan, np.nan)
        if len(cand):  # the server: their player nearest where the shuttle first shows
            c = np.c_[(cand.box_x1 + cand.box_x2) / 2, (cand.box_y1 + cand.box_y2) / 2]
            j = int(np.argmin(np.hypot(*(c - px[start]).T)))
            pid, feet = cand.player_id.iat[j], (cand.x_m.iat[j], cand.y_m.iat[j])
        else:
            pid = "P1" if team == "us" else "P3"
        row = {c: np.nan for c in df.columns}
        row.update(contact_x_m=feet[0], contact_y_m=feet[1], contact_z_m=SERVE_Z_M)  # for its rough speed
        row.update(frame=fs, timestamp=round(float(ts[fs]), 4), player_id=pid, team=team,
                   shuttle_x_px=px[start, 0], shuttle_y_px=px[start, 1], shot_type="serve", landing_kind="intercept",
                   end_frame=f1, speed_method="serve", sure=False, serve=True, turn_deg=0.0)
        add.append(row)
    if "serve" not in df:
        df["serve"] = False
    if add:
        df = pd.concat([df, pd.DataFrame(add)], ignore_index=True).sort_values("frame", ignore_index=True)
        df["shot_no"] = np.arange(1, len(df) + 1)
    print(f"Serves recovered before {len(add)} of {int(first.sum())} openings")
    return df


SMASH_VY = 5.0         # shown_type: image speed downwards over the 6 frames after the hit (px/frame per
SMASH_Z_M = 1.4       # 1000 px of frame height) and contact height for a smash. On the user's 58 typed hits
                      # found by the detector: 6 of 10 smashes, 2 others (a lift, a drop) wrongly. The rule
                      # types were right 9 % of the time and a classifier for all 6 types barely beat "always
                      # lift" (60 vs 57 %), so only the smash is shown
LINE_SIDE_M = 0.8     # line_of: hitter and receiver must each stand this far from the centre line
ROUGH_KMH = (15.0, 250.0)  # rough speeds (speed_shown_kmh) are clipped to this (fits on the clip: 12-199)
FLIGHT_AVG_MAX_KMH = 100.0  # a whole flight averaging more is two hits paired wrongly (8-11 m in 0.2-0.3 s
                            # on the user's clip: a false hit or a serve found too late), not a speed


def shown_types(df, shuttle, frame_h):
    """shown_type ("smash" or None), line ("straight" / "cross" / None) and a speed for every hit
    (speed_shown_kmh, speed_src), for the HUD."""
    if df.empty:
        return df
    df = df.sort_values("frame", ignore_index=True)
    real = (shuttle.visible & ~shuttle.interpolated & ~shuttle.inpainted).to_numpy()
    P = np.where(real[:, None], shuttle[["x_px", "y_px"]].to_numpy(float), np.nan)
    vy = []
    for f in df.frame.astype(int):
        seg = P[f + 1:f + 7]
        ok = np.flatnonzero(np.isfinite(seg[:, 0]))
        vy.append((seg[ok[-1], 1] - seg[ok[0], 1]) / (ok[-1] - ok[0]) / frame_h * 1000 if len(ok) >= 2 else np.nan)
    df = df.copy()
    df["vy_out"] = np.round(vy, 2)
    df["shown_type"] = np.where((df.vy_out > SMASH_VY) & (df.contact_z_m > SMASH_Z_M), "smash", None)
    # straight / cross (a guess): where this shot was hit from vs where it was played next (or landed),
    # left / right of the centre line; too near the centre: no call
    nxt_x = df.contact_x_m.shift(-1).where(df.timestamp.shift(-1) - df.timestamp < MISSED_HIT_S)
    tgt = nxt_x.fillna(df.landing_x)
    a, b = df.contact_x_m - COURT_W / 2, tgt - COURT_W / 2
    sure = (a.abs() > LINE_SIDE_M) & (b.abs() > LINE_SIDE_M)
    df["line"] = np.where(sure & (a * b > 0), "straight", np.where(sure & (a * b < 0), "cross", None))
    df.loc[df.serve.fillna(False).astype(bool), "line"] = None  # a serve always goes across
    return rough_speeds(df, 1 / np.median(np.diff(shuttle.timestamp)))


def rough_speeds(df, fps):
    """A speed for every hit (the user: every shot shows one, rough is fine), best source first:
    fit = the physical flight fit (sure); flight = straight distance to the next contact (or the
    landing) over the flight time, undone for drag (on the 21 fitted hits of the user's clip it
    tracked the fit with r 0.65, median ratio 0.92); window = the 0.1 s fallback (3-409 km/h);
    typical = this match's median of the others (smash / not), when nothing at all was seen."""
    c0 = df[["contact_x_m", "contact_y_m", "contact_z_m"]].to_numpy(float)
    dt = (df.timestamp.shift(-1) - df.timestamp).to_numpy(float, copy=True)
    c1 = np.roll(c0, -1, axis=0)
    land = np.c_[df.landing_x, df.landing_y, np.zeros(len(df))]
    t_land = (df.end_frame - df.frame).to_numpy(float) / fps
    last = ~(dt < MISSED_HIT_S)  # no next contact in this flight: to the landing
    c1[last], dt[last] = land[last], t_land[last]
    dist = np.linalg.norm(c1 - c0, axis=1)
    ok = np.isfinite(dist) & (dt > 0.1) & (dt < MISSED_HIT_S)
    ok[ok] = dist[ok] / dt[ok] * 3.6 <= FLIGHT_AVG_MAX_KMH
    flight = np.full(len(df), np.nan)
    flight[ok] = [from_window_avg(d / t, t) * 3.6 for d, t in zip(dist[ok], dt[ok])]
    fit = df.speed_kmh_est.where(df.sure.fillna(False).astype(bool)).to_numpy(float)
    win = df.speed_kmh_est.to_numpy(float, copy=True)
    kmh = np.where(np.isfinite(fit), fit, np.where(np.isfinite(flight), flight, win))
    src = np.where(np.isfinite(fit), "fit", np.where(np.isfinite(flight), "flight", np.where(np.isfinite(win), "window", "typical")))
    kmh = np.clip(kmh, *ROUGH_KMH)
    smash = (df.shown_type == "smash").to_numpy()
    for grp in (smash, ~smash):
        known = np.isfinite(kmh) & grp
        kmh[grp & ~known] = np.median(kmh[known]) if known.any() else np.nanmedian(kmh) if np.isfinite(kmh).any() else 100.0
    df["speed_shown_kmh"], df["speed_src"] = np.round(kmh), src
    return df


def run(shuttle_path, players_path, calib_path, out):
    shuttle, players = pd.read_parquet(shuttle_path), pd.read_parquet(players_path)
    H, calib = load(calib_path)
    cam = Camera(H, calib["frame_size"])
    fps = 1 / np.median(np.diff(shuttle.timestamp)) if len(shuttle) > 1 else 30
    from hud.court.stabilize import cuts_of, native_size
    # pixel thresholds are tuned at 1080p and scale with the camera's short side: the frame's own
    # height over-scaled a portrait clip 1.8x, and a stabilised canvas (2050 high) more, so the far
    # court's small turns never registered as hits
    lens = native_size(calib["video"]) or calib["frame_size"]
    df = detect(shuttle, players, H, cam, fps, min(lens), config.load()["rally"]["serve_answer_s"],
                cuts=cuts_of(calib["video"]), frame_h=calib["frame_size"][1])
    df = shown_types(add_serves(df, shuttle, players, fps), shuttle, calib["frame_size"][1])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)  # overwrite: idempotent
    print(f"Wrote {out}: {len(df)} hits")
    if len(df):
        print(df[["shot_no", "frame", "timestamp", "player_id", "contact_z_m", "speed_kmh_est", "flight_avg_kmh_est",
                  "shot_type", "landing_x", "landing_y", "landing_kind"]].to_string(index=False))
    return df
