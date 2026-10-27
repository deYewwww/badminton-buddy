"""Where did the last shot of a rally come down, when the shuttle was lost before it landed?

The camera sees the shuttle as a ray per frame; a single ray says nothing about distance,
and projecting the last sighting onto the floor is hopeless from a low phone (at 0.7 m high,
a shuttle 0.5 m above the far court projects ~7 m too deep). But a shuttle's flight is
physics with no free constants: drag (terminal velocity ~6.7 m/s) and gravity. So the
whole tracked flight is fitted (robust least squares on the pixels, as hits.fit_speed) with
the contact point free to move around its estimate, then integrated on until it reaches the
floor. The landing's uncertainty comes from the fit's covariance; callers only use the
prediction where it is decisive (which half, in or out), else the rally keeps "unknown".
"""
import numpy as np

from hud.analysis.hits import DRAG_K, FIT_PX, flight

CONTACT_SD_M = 0.3    # how far the contact estimate (hitter's vertical) may be off, per axis
K_SD = 0.03           # shuttles differ: drag k (1/m) is fitted too, around DRAG_K with this prior sd
V0_BOUNDS = (2.0, 120.0)
MAX_RMS = 2.5         # reprojection rms (in pixel-noise units) above this: the model doesn't fit
MAX_PTS = 40
HORIZON_S = 4.0       # integrate this long at most


def _unit(a, e):
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _land(c, v0, u, k=DRAG_K):
    """First floor crossing of the flight from c (xy) or None."""
    taus = np.linspace(0, HORIZON_S, 241)
    P = c + flight(v0, u, taus, k=k, max_steps=240)
    below = np.flatnonzero(P[:, 2] <= 0)
    if not len(below) or below[0] == 0:
        return None
    i = below[0]
    a, b = P[i - 1], P[i]
    return a[:2] + (b[:2] - a[:2]) * (a[2] / (a[2] - b[2]))


def predict(cam, c0, dt, px, fps=30.0, img_h=1080):
    """c0: contact estimate (3,), dt (N,) s after the hit frame, px (N,2) real detections of the
    shot's flight. -> dict(xy landing (2,), sd metres (largest axis), rms, v0) or None."""
    from scipy.optimize import least_squares
    dt, px = np.asarray(dt, float), np.asarray(px, float)
    if len(dt) > MAX_PTS:
        keep = np.unique(np.linspace(0, len(dt) - 1, MAX_PTS).round().astype(int))
        dt, px = dt[keep], px[keep]
    if len(dt) < 5:
        return None
    sig = FIT_PX * img_h / 1080
    c0 = np.asarray(c0, float)

    def path(q, t):
        return c0 + q[4:7] + flight(q[2], _unit(q[0], q[1]), t - q[3], k=q[7])

    def resid(q):
        img, depth = cam.project_many(path(q, dt))
        if not np.isfinite(img).all():
            return np.full(2 * len(dt) + 4, 1e3)
        r = np.where(np.repeat(depth, 2) > 0.1, ((img - px) / sig).ravel(), 1e3)
        return np.r_[r, q[4:7] / CONTACT_SD_M, (q[7] - DRAG_K) / K_SD]

    # start: heading towards the last sighting at the contact's height, a few speeds/elevations
    g = cam.at_height(px[-1], c0[2]) - c0 if abs(cam.ray(px[-1])[2]) > 1e-3 else np.array([0, 1.0, 0])
    a0 = np.arctan2(g[1], g[0]) if np.isfinite(g).all() else 0.0
    lo = [-np.inf, -1.45, V0_BOUNDS[0], -1 / fps, -1.0, -1.0, -1.0, DRAG_K * 0.5]
    hi = [np.inf, 1.45, V0_BOUNDS[1], 1 / fps, 1.0, 1.0, 1.0, DRAG_K * 1.5]
    best = None
    for v in (12.0, 35.0, 70.0):
        for e in (-0.3, 0.15, 0.7):
            for a in (a0, a0 + np.pi):
                x0 = [a, e, v, 0.0, 0, 0, 0, DRAG_K]
                sol = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=2.0, ftol=1e-4,
                                    xtol=1e-4, x_scale=[0.3, 0.3, 10.0, 0.01, 0.3, 0.3, 0.3, 0.03], max_nfev=200)
                if best is None or sol.cost < best.cost:
                    best = sol
    q = best.x
    img = cam.project_many(path(q, dt))[0]
    e2 = np.sort(np.sum((img - px) ** 2, 1))
    rms = float(np.sqrt(np.mean(e2[:max(5, int(0.9 * len(e2)))])) / sig)  # TrackNet's stray detections aside
    if rms > MAX_RMS:
        return None
    land = _land(c0 + q[4:7], q[2], _unit(q[0], q[1]), q[7])
    if land is None:
        return None
    # landing covariance: parameter covariance pushed through the landing (finite differences)
    try:
        J = best.jac
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return None
    steps = np.array([0.01, 0.01, 0.2, 0.002, 0.02, 0.02, 0.02, 0.005])
    Jl = np.zeros((2, len(q)))
    for k in range(len(q)):
        dq = q.copy()
        dq[k] += steps[k]
        l2 = _land(c0 + dq[4:7], dq[2], _unit(dq[0], dq[1]), dq[7])
        if l2 is None:
            return None
        Jl[:, k] = (l2 - land) / steps[k]
    cl = Jl @ cov @ Jl.T
    sd = float(np.sqrt(max(np.linalg.eigvalsh(cl).max(), 0)))
    return dict(xy=land, sd=sd, rms=rms, v0=float(q[2]), cov=cl)
