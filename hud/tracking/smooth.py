"""Court positions of the players, smoothed the way the camera sees them.

A foot point is a pixel; its court position comes through the ground homography, and how
much a pixel of jitter moves it depends on where the player stands. Seen from behind the
baseline, one pixel at the far baseline is ~10x more metres than at the near one, and almost
all of it along the depth axis (d^2 / (f * camera height)). A plain rolling mean treats every
metre the same, so the far players' positions on the court map jumped around, and the jitter
added up to 3x the distance they really ran (stamina emptied in a minute).

So each foot point gets its own measurement covariance, the pixel noise pushed through the
homography's Jacobian at that pixel, and a constant-velocity Kalman filter + RTS smoother
(forward pass, then backward) weighs every observation by it: near players stay crisp, far
players are smoothed hard in depth and lightly sideways. Observations far outside what the
motion allows (a jump: the feet leave the floor and project metres deep; a wrong match) are
down-weighted, not trusted.
"""
import numpy as np

PX_NOISE = 0.015      # foot point noise as a fraction of the player's box height (shoe soles, ankles)
PX_MIN = 1.5          # ... and at least this many pixels
ACCEL = 12.0          # m/s^2 white-noise acceleration (lunges, start/stop). Simulated footwork on the ACE
                      # camera: 5 -> far rmse 0.5 m and distance -15 %; 12 -> 0.3 m, -5 %; 16 no better
GATE = 9.21           # chi^2 2 dof, 99 %: further than this is an outlier ...
OUTLIER_INFLATE = 25  # ... whose covariance is inflated this much


def foot_cov(H, px, sigma):
    """2x2 court covariance (m^2) of foot pixels px (N,2) with isotropic noise sigma (N,) px."""
    px = np.asarray(px, float)
    out = np.zeros((len(px), 2, 2))

    def to_court(p):
        q = np.c_[p, np.ones(len(p))] @ np.asarray(H).T
        return q[:, :2] / q[:, 2:]
    base = to_court(px)
    J = np.stack([(to_court(px + [1.0, 0]) - base), (to_court(px + [0, 1.0]) - base)], axis=2)  # (N,2,2) m/px
    out[:] = J @ J.transpose(0, 2, 1) * (np.asarray(sigma, float) ** 2)[:, None, None]
    return out


def rts(times, z, R, accel=ACCEL):
    """Constant-velocity Kalman filter + RTS smoother over one player.
    times (N,); z (N,2) measurements, NaN = none; R (N,2,2). -> (N,2) smoothed positions."""
    n = len(times)
    x = np.zeros((n, 4))
    P = np.zeros((n, 4, 4))
    xp, Pp = np.zeros((n, 4)), np.zeros((n, 4, 4))
    Hm = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    first = np.flatnonzero(np.isfinite(z[:, 0]))
    if not len(first):
        return np.full((n, 2), np.nan)
    xc = np.r_[z[first[0]], 0, 0]
    Pc = np.diag([*np.diag(R[first[0]]), 4.0, 4.0])
    F_of = []
    for i in range(n):
        dt = times[i] - times[i - 1] if i else 0.0
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = accel ** 2
        Q1 = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * q
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = Q1
        Q[np.ix_([1, 3], [1, 3])] = Q1
        F_of.append(F)
        if i:
            xc, Pc = F @ xc, F @ Pc @ F.T + Q
        xp[i], Pp[i] = xc, Pc
        if np.isfinite(z[i, 0]):
            Ri = R[i]
            y = z[i] - xc[:2]
            S = Pc[:2, :2] + Ri
            if y @ np.linalg.solve(S, y) > GATE:
                Ri = Ri * OUTLIER_INFLATE
                S = Pc[:2, :2] + Ri
            K = Pc @ Hm.T @ np.linalg.inv(S)
            xc = xc + K @ y
            Pc = (np.eye(4) - K @ Hm) @ Pc
        x[i], P[i] = xc, Pc
    xs = x.copy()
    for i in range(n - 2, -1, -1):
        F = F_of[i + 1]
        C = P[i] @ F.T @ np.linalg.pinv(Pp[i + 1])
        xs[i] = x[i] + C @ (xs[i + 1] - xp[i + 1])
    return xs[:, :2]


def smooth_players(df, H, accel=ACCEL):
    """players.parquet frame -> same frame with x_m, y_m smoothed (raw kept in x_raw, y_raw).
    Measurements: rows where the pose model saw the player (observed with a foot pixel);
    coasting/interpolated rows are filled by the smoother."""
    if df.empty or "foot_px_x" not in df:
        return df
    df = df.sort_values(["player_id", "frame"]).reset_index(drop=True)
    df["x_raw"], df["y_raw"] = df.x_m, df.y_m
    for pid, g in df.groupby("player_id"):
        meas = g.observed.to_numpy(bool) & g.foot_px_x.notna().to_numpy()
        if "app_top_L" in g:  # interpolated rows between detections carry no appearance
            meas &= g.app_top_L.notna().to_numpy()
        if meas.sum() < 2:
            continue
        z = np.where(meas[:, None], g[["x_m", "y_m"]].to_numpy(float), np.nan)
        hgt = (g.box_y2 - g.box_y1).to_numpy(float)
        sigma = np.maximum(PX_NOISE * np.nan_to_num(hgt, nan=100.0), PX_MIN)
        R = np.repeat(np.eye(2)[None] * 1e6, len(g), 0)
        R[meas] = foot_cov(H, g.loc[meas, ["foot_px_x", "foot_px_y"]].to_numpy(float), sigma[meas]) + np.eye(2) * 1e-4
        # each run of frames separately: a player lost for seconds starts afresh
        fr = g.frame.to_numpy()
        breaks = np.flatnonzero(np.diff(fr) > 1) + 1
        out = np.full((len(g), 2), np.nan)
        for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(g)]):
            out[a:b] = rts(g.timestamp.to_numpy(float)[a:b], z[a:b], R[a:b], accel)
        ok = np.isfinite(out[:, 0])
        idx = g.index[ok]
        df.loc[idx, "x_m"] = np.round(out[ok, 0], 3)
        df.loc[idx, "y_m"] = np.round(out[ok, 1], 3)
    return df.sort_values(["frame", "player_id"], ignore_index=True)
