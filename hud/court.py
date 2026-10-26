"""Court model and calibration.

Court frame (metres): origin = near-left corner of the doubles court (camera side),
x across (0..6.10), y away from the camera (0..13.40), net at y = 6.70.

A calibration is the 4 court corners in image pixels ("corners", order near-left,
near-right, far-right, far-left). Corners may lie outside the frame: a low or 45-degree
phone shot often cuts one or two off, the homography still holds.

auto_detect() finds the court from the painted lines alone, so it does not need the
corners to be visible:
  1. white top-hat mask -> Hough segments -> merged infinite lines, split in two
     direction families (lines across the court / lines along it);
  2. every pairing of 2+2 image lines with 2+2 court lines is a candidate homography,
     scored in one numpy batch by how many projected court-line samples land on paint;
  3. the best few are polished by snapping the whole court model onto the line mask.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

COURT_W, COURT_L = 6.10, 13.40
NET_Y = COURT_L / 2
SHORT_SVC, LONG_SVC, SINGLES = 1.98, 0.76, 0.46
CORNERS_M = np.float64([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]])

XS = [0.0, SINGLES, COURT_W / 2, COURT_W - SINGLES, COURT_W]                 # lines along the court
YS = [0.0, LONG_SVC, NET_Y - SHORT_SVC, NET_Y + SHORT_SVC, COURT_L - LONG_SVC, COURT_L]  # across
LINES = [((x, 0), (x, COURT_L)) for x in (0, SINGLES, COURT_W - SINGLES, COURT_W)] + \
        [((0, y), (COURT_W, y)) for y in YS] + \
        [((COURT_W / 2, 0), (COURT_W / 2, NET_Y - SHORT_SVC)), ((COURT_W / 2, NET_Y + SHORT_SVC), (COURT_W / 2, COURT_L))]
NET = ((0, NET_Y), (COURT_W, NET_Y))
WORK = 1280  # detection runs with the long side scaled to this
CAM_MIN_H, CAM_MAX_H = 0.6, 8.0  # plausible camera height (m): chair .. balcony
CAM_BEHIND = 1.0                 # camera at least this far behind the near baseline (m)


def homography(corners):
    """image corners (4,2) px -> H (image px -> court m)."""
    return cv2.getPerspectiveTransform(np.float32(corners), np.float32(CORNERS_M))


def save(path, video, frame_size, corners):
    calib = {"video": str(video), "frame_size": list(frame_size),
             "corners": np.round(np.float64(corners), 1).tolist(), "H": homography(corners).tolist()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(calib, indent=1, ensure_ascii=False))
    return calib


def load(path, frame_size=None):
    """-> (H image px -> court m, calib dict)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Court not calibrated: {path} not found.")
    calib = json.loads(path.read_text())
    if frame_size and list(frame_size) != calib["frame_size"]:
        raise SystemExit(f"Calibration {path} is for {calib['frame_size']}, video is {list(frame_size)}.")
    return np.array(calib["H"]), calib


def to_court(H, pts):
    return cv2.perspectiveTransform(np.float64(pts).reshape(-1, 1, 2), H).reshape(-1, 2)


def to_image(H, pts):
    return to_court(np.linalg.inv(H), pts)


def line_mask(frame):
    """Thin bright structures (painted lines), any floor colour. Dim frames get CLAHE first."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v, sat_max = hsv[..., 2], 110
    dark = np.median(v) < 130
    if dark:
        v, sat_max = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(v), 140
    k = max(9, frame.shape[0] // 60) | 1
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    thr = max(12.0, float(np.percentile(tophat, 98)) * 0.5) if dark else 25.0
    return ((tophat > thr) & (hsv[..., 1] < sat_max)).astype(np.uint8) * 255


def _samples(per_m):
    """Points along every painted line, and whether each point's successor is on the same line."""
    out, cont = [], []
    for a, b in LINES:
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) * per_m))
        out.append(np.linspace(a, b, n))
        cont.append(np.r_[np.ones(n - 1, bool), False])
    return np.concatenate(out), np.concatenate(cont)


def _signed_dist(mask):
    # outside a line = distance to it, inside = minus distance to its edge: keeps pulling to the centre
    return cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3) - cv2.distanceTransform(mask, cv2.DIST_L2, 3)


def _project_batch(G, pts):
    """G (N,3,3) court->image, pts (M,2) -> xy (N,M,2), front (N,M)."""
    X = np.einsum("nij,mj->nmi", G, np.c_[pts, np.ones(len(pts))])
    w = X[..., 2]
    front = w > 1e-9
    xy = X[..., :2] / np.where(front, w, 1)[..., None]
    return xy, front


def _score(G, dist, samples, tol, signed=False):
    """Pixel length of court line that lands on paint, minus half the in-frame length that
    misses it (soft, per candidate). Measured in image pixels, so a court shrunk onto one
    painted line cannot score. -> (score, fraction of in-frame samples on paint)."""
    pts, cont = samples
    h, w = dist.shape
    xy, front = _project_batch(G, pts)
    ok = front & (xy[..., 0] >= 0) & (xy[..., 0] < w - 1) & (xy[..., 1] >= 0) & (xy[..., 1] < h - 1)
    step = np.zeros(ok.shape)
    step[:, :-1] = np.hypot(*(xy[:, 1:] - xy[:, :-1]).transpose(2, 0, 1))
    step = np.where(cont, np.minimum(step, 40), 0)
    xi = np.where(ok, xy[..., 0], 0).astype(np.int32)
    yi = np.where(ok, xy[..., 1], 0).astype(np.int32)
    # choosing a court: distance to the nearest line pixel (inside a line = 0), robust.
    # polishing it (signed=True): negative inside a line, lowest at its centre, so the fit
    # settles on line centres instead of edges (worth ~4 px at the far baseline)
    d = np.clip(dist[yi, xi], -tol, tol) if signed else np.clip(dist[yi, xi], 0, tol)
    u = 1 - 1.5 * d / tol
    score = (np.where(ok, u, 0) * step).sum(1)
    on = ok & (d < tol)
    return score, on.sum(1) / np.maximum(ok.sum(1), 1)


def _lines(mask):
    """Merged infinite lines (theta, rho, length) from Hough segments."""
    h, w = mask.shape
    seg = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=50, minLineLength=int(0.05 * np.hypot(h, w)), maxLineGap=12)
    if seg is None:
        return np.zeros((0, 3))
    seg = seg.reshape(-1, 4).astype(np.float64)
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    theta = (np.arctan2(dy, dx) + np.pi / 2) % np.pi           # normal direction, [0, pi)
    rho = seg[:, 0] * np.cos(theta) + seg[:, 1] * np.sin(theta)
    length = np.hypot(dx, dy)
    merged = []
    for i in np.argsort(-length):
        for m in merged:
            dth = abs(m[0] - theta[i])
            if min(dth, np.pi - dth) < np.radians(1.5):
                r = rho[i] if dth < np.pi / 2 else -rho[i]
                if abs(m[1] - r) < 6:
                    m[2] += length[i]
                    break
        else:
            merged.append([theta[i], rho[i], length[i]])
    return np.array(merged)


def _candidates(lines, top=14):
    """Every assignment of 2 image lines to 2 court lines along the court and 2 others to 2
    court lines across it -> (src court quads, dst image quads), (N,4,2) each. No direction
    grouping: under a 45-degree perspective the two line families overlap in angle."""
    L = lines[np.argsort(-lines[:, 2])[:top]]
    L = np.c_[np.cos(L[:, 0]), np.sin(L[:, 0]), -L[:, 1]]
    n = len(L)
    idx = np.array(np.meshgrid(*[np.arange(n)] * 4, indexing="ij")).reshape(4, -1).T
    i, j, k, m = idx.T
    idx = idx[(i < j) & (k != m) & (k != i) & (k != j) & (m != i) & (m != j)]
    i, j, k, m = idx.T
    P = np.stack([np.cross(L[i], L[k]), np.cross(L[j], L[k]), np.cross(L[j], L[m]), np.cross(L[i], L[m])], 1)
    keep = np.all(np.abs(P[..., 2]) > 1e-9, 1)
    img = P[keep, :, :2] / P[keep, :, 2:]
    xpairs = [(a, b) for q, a in enumerate(XS) for b in XS[q + 1:]]
    ypairs = [(a, b) for q, a in enumerate(YS) for b in YS[q + 1:]]
    wld = np.array([[[xa, ya], [xb, ya], [xb, yb], [xa, yb]] for xa, xb in xpairs for ya, yb in ypairs], np.float64)
    src = np.broadcast_to(wld[None], (len(img), *wld.shape)).reshape(-1, 4, 2)
    dst = np.broadcast_to(img[:, None], (len(img), len(wld), 4, 2)).reshape(-1, 4, 2)
    return src, dst


def _batch_persp(src, dst):
    """Batched getPerspectiveTransform: src, dst (N,4,2) -> (N,3,3)."""
    n = len(src)
    A = np.zeros((n, 8, 8))
    b = np.zeros((n, 8))
    for k in range(4):
        x, y = src[:, k, 0], src[:, k, 1]
        u, v = dst[:, k, 0], dst[:, k, 1]
        A[:, 2 * k] = np.stack([x, y, np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n), -u * x, -u * y], 1)
        A[:, 2 * k + 1] = np.stack([np.zeros(n), np.zeros(n), np.zeros(n), x, y, np.ones(n), -v * x, -v * y], 1)
        b[:, 2 * k], b[:, 2 * k + 1] = u, v
    ok = np.abs(np.linalg.det(A)) > 1e-12
    h = np.zeros((n, 8))
    h[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    G = np.c_[h, np.ones(n)].reshape(n, 3, 3)
    return G[ok]


def _plausible(G, size, strict=True):
    """Camera behind the near baseline, court in front of it, centre in frame, sane scale."""
    w, h = size
    pts = np.float64([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L], [COURT_W / 2, NET_Y],
                      [COURT_W / 2, 0], [COURT_W / 2, COURT_L]])
    G = G * np.sign(np.einsum("nj,j->n", G[:, 2], [COURT_W / 2, NET_Y, 1]))[:, None, None]
    xy, front = _project_batch(G, pts)
    c = xy[:, 4]
    ok = front.all(1) & (c[:, 0] > -0.1 * w) & (c[:, 0] < 1.1 * w) & (c[:, 1] > 0) & (c[:, 1] < h)
    ok &= xy[:, 5, 1] > xy[:, 6, 1] + 0.05 * h          # near baseline lower in the image than the far one
    q = xy[:, :4]
    area = 0.5 * (q[..., 0] * np.roll(q[..., 1], -1, 1) - np.roll(q[..., 0], -1, 1) * q[..., 1]).sum(1)
    ok &= area < 0                                       # not mirrored (nl, nr, fr, fl anticlockwise on screen)
    near_w = np.hypot(*(xy[:, 1] - xy[:, 0]).T)
    ok &= (near_w > 0.15 * w) & (near_w < 8 * w)
    return G, (ok & _real_camera(G, size)) if strict else ok


def _real_camera(G, size, f_range=(0.3, 6.0)):
    """Could a pinhole camera (square pixels, centred principal point) standing where people
    put a phone have produced G? Its first two columns, back-projected with focal f, must be
    orthogonal and of equal length (f comes from either condition, the other is the test; a
    straight rear view makes one of them degenerate). Then the camera centre must be
    CAM_MIN_H..CAM_MAX_H high and behind the near baseline: look-alikes built from a
    neighbouring court's lines put the camera on the floor."""
    w, h = size
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]])
    Gc = np.einsum("ij,njk->nik", T, G)
    g1, g2, g3 = Gc[:, :, 0], Gc[:, :, 1], Gc[:, :, 2]
    s = max(w, h)
    good = np.zeros(len(G), bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_orth = -(g1[:, 0] * g2[:, 0] + g1[:, 1] * g2[:, 1]) / (g1[:, 2] * g2[:, 2])
        f_norm = (g1[:, 0] ** 2 + g1[:, 1] ** 2 - g2[:, 0] ** 2 - g2[:, 1] ** 2) / (g2[:, 2] ** 2 - g1[:, 2] ** 2)
        for f2 in (f_orth, f_norm):
            f = np.sqrt(np.where(f2 > 0, f2, np.nan))[:, None]
            k = np.c_[1 / f, 1 / f, np.ones_like(f)]
            r1, r2, t = g1 * k, g2 * k, g3 * k
            n1, n2 = np.linalg.norm(r1, axis=1), np.linalg.norm(r2, axis=1)
            cos = np.abs((r1 * r2).sum(1)) / (n1 * n2)
            lam = 2 / (n1 + n2) * np.sign(t[:, 2])[:, None].ravel()  # court in front of the camera
            r1, r2, t = r1 * lam[:, None], r2 * lam[:, None], t * lam[:, None]
            r3 = np.cross(r1, r2)
            height, depth = -(r3 * t).sum(1), -(r2 * t).sum(1)
            good |= ((f[:, 0] > f_range[0] * s) & (f[:, 0] < f_range[1] * s) & (cos < 0.1) & (n1 / n2 > 0.85)
                     & (n1 / n2 < 1.18) & (height >= CAM_MIN_H) & (height <= CAM_MAX_H) & (depth <= -CAM_BEHIND))
    return good


def _corner_cost(x, dist, samples, tol):
    G = cv2.getPerspectiveTransform(np.float32(CORNERS_M), np.float32(x.reshape(4, 2)))
    return -_score(G[None], dist, samples, tol, signed=True)[0][0]


def snap(frame, corners, max_shift=0.06):
    """Polish corners (image px, may be off-frame) so the court model sits on the painted
    lines. Returns (corners, on_line fraction). Rejected if it moves too far or fits worse."""
    small, s = _small(frame)
    dist = _signed_dist(line_mask(small))
    pts = _samples(25)
    x0 = np.float64(corners).ravel() * s
    x = x0
    for tol in (12, 6, 3):  # coarse to fine
        x = minimize(_corner_cost, x, args=(dist, pts, tol), method="Powell", options={"xtol": 0.05, "maxiter": 3000}).x
    fit = lambda c: _score(cv2.getPerspectiveTransform(np.float32(CORNERS_M), np.float32(c.reshape(4, 2)))[None],
                           dist, pts, 3)
    (b0, f0), (b1, f1) = fit(x0), fit(x)
    if np.abs(x - x0).max() > max_shift * small.shape[1] or b1[0] < b0[0]:
        return np.float64(corners), float(f0[0])
    return x.reshape(4, 2) / s, float(f1[0])


def _small(frame):
    """Scale to WORK px on the long side, up as well as down: every pixel tolerance in here
    (Hough lengths, line merging, scoring) is tuned at that size. A 480p clip used to be left
    as is and auto-detection found nothing; upscaled it lands within 1 px of the 720p result."""
    s = WORK / max(frame.shape[:2])
    if abs(s - 1) < 1e-3:
        return frame, 1.0
    return cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC), s


def auto_detect(frame):
    """-> (corners (4,2) image px, on_line fraction) or (None, 0)."""
    return _detect(frame, strict=True)


def _detect(frame, strict):
    small, s = _small(frame)
    h, w = small.shape[:2]
    mask = line_mask(small)
    lines = _lines(mask)
    if len(lines) < 4:
        return None, 0.0
    dist = _signed_dist(mask)
    coarse = _samples(6)
    best = []
    src, dst = _candidates(lines)
    for i in range(0, len(src), 50000):
        G = _batch_persp(src[i:i + 50000], dst[i:i + 50000])
        G, ok = _plausible(G, (w, h), strict)
        G = G[ok]
        if len(G):
            sc, _ = _score(G, dist, coarse, 4)
            top = np.argsort(-sc)[:20]
            best += list(zip(sc[top], G[top]))
    if not best:
        return None, 0.0
    best.sort(key=lambda t: -t[0])
    pts = _samples(25)
    results = []
    for _, G in best[:12]:
        x = to_court(G, CORNERS_M).ravel()
        for tol in (8, 4, 2):
            x = minimize(_corner_cost, x, args=(dist, pts, tol), method="Powell", options={"xtol": 0.05, "maxiter": 3000}).x
        Gx = cv2.getPerspectiveTransform(np.float32(CORNERS_M), np.float32(x.reshape(4, 2)))[None]
        Gx, ok = _plausible(Gx, (w, h), strict)
        if ok[0]:
            sc, frac = _score(Gx, dist, pts, 3)
            results.append((sc[0], frac[0], x.reshape(4, 2) / s))
    if not results:
        return None, 0.0
    sc, frac, corners = max(results, key=lambda r: r[0])
    G = cv2.getPerspectiveTransform(np.float32(CORNERS_M), np.float32(corners * s))
    for Gv in _reassign(G, dist, pts, (w, h), strict):
        x = to_court(Gv, CORNERS_M).ravel()
        for tol in (4, 2):
            x = minimize(_corner_cost, x, args=(dist, pts, tol), method="Powell", options={"xtol": 0.05, "maxiter": 3000}).x
        Gx, ok = _plausible(cv2.getPerspectiveTransform(np.float32(CORNERS_M), np.float32(x.reshape(4, 2)))[None], (w, h), strict)
        if ok[0]:
            v, f = _score(Gx, dist, pts, 3)
            if v[0] > sc:
                sc, frac, corners = v[0], f[0], x.reshape(4, 2) / s
    return corners, float(frac)


def _reassign(G, dist, samples, size, strict=True, keep=3):
    """The search can lock onto the right lines with the wrong names: a singles side line
    taken for the doubles one, a service line for the baseline. Try every renaming of the
    found side lines (XS pairs) and end lines (YS pairs) in one batch; return the best few
    plausible ones (court->image) for polishing."""
    A = []
    for i, x0 in enumerate(XS):
        for x1 in XS[i + 1:]:
            for j, y0 in enumerate(YS):
                for y1 in YS[j + 1:]:
                    if (x0, x1, y0, y1) == (0, COURT_W, 0, COURT_L):
                        continue
                    sx, sy = COURT_W / (x1 - x0), COURT_L / (y1 - y0)
                    A.append([[sx, 0, -x0 * sx], [0, sy, -y0 * sy], [0, 0, 1]])
    Gv = np.einsum("ij,njk->nik", G, np.array(A))
    Gv, ok = _plausible(Gv, size, strict)
    Gv = Gv[ok]
    if not len(Gv):
        return []
    sc, _ = _score(Gv, dist, samples, 4)
    return list(Gv[np.argsort(-sc)[:keep]])


def draw(frame, H, color=(0, 255, 255), thick=2):
    G = np.linalg.inv(H)
    for a, b in LINES + [NET]:
        pa, pb = to_court(G, [a, b])
        cv2.line(frame, tuple(map(int, pa)), tuple(map(int, pb)), color, thick, cv2.LINE_AA)
    return frame
