"""Venues: a court calibrated once, on a clean frame, reused for every match filmed there.

Film the empty court first (players can't hide the lines; someone can stand a shuttle tube on
the far corners for a moment), calibrate that clip in the editor, save it as a venue. A match
filmed from the same spot then gets its court by aligning the venue's background to the
match's background, never by detecting the court from scratch:

  1. coarse: ORB features matched between the two backgrounds, a RANSAC homography moves the
     venue's corners (tripod nudged, phone re-mounted a little differently);
  2. fine: court.snap() pulls the court model onto the match's painted lines;
  3. the result must still be a camera a person could have set up (court._plausible) and
     sit on the lines at least as well as the unmoved corners do.
Nothing moved at all (same recording, or the phone untouched): step 1 finds the identity.
A big move (other side of the court) fails step 1 and says so: calibrate again.

Stored in data/venues/<name>/ (venue.json + background.jpg), next to the uploads.
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np

from hud import court

ROOT = Path(__file__).resolve().parents[1] / "data" / "venues"
MIN_INLIERS = 25      # feature matches that agree on one homography
MAX_REPROJ_PX = 4.0   # RANSAC tolerance at the working size (1280 px)


def slug(name):
    return re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE).strip("_")[:60] or "venue"


def save(name, background, corners, root=ROOT):
    d = Path(root) / slug(name)
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "background.jpg"), background, [cv2.IMWRITE_JPEG_QUALITY, 92])
    h, w = background.shape[:2]
    (d / "venue.json").write_text(json.dumps(dict(name=name.strip(), frame_size=[w, h],
                                                  corners=np.round(np.float64(corners), 1).tolist()), ensure_ascii=False))
    return d


def names(root=ROOT):
    root = Path(root)
    return sorted(json.loads(p.read_text())["name"] for p in root.glob("*/venue.json")) if root.exists() else []


def load(name, root=ROOT):
    d = Path(root) / slug(name)
    if not (d / "venue.json").exists():
        raise SystemExit(f"No venue called {name!r}. Saved venues: {', '.join(names(root)) or 'none'}")
    v = json.loads((d / "venue.json").read_text())
    return cv2.imread(str(d / "background.jpg")), np.float64(v["corners"])


def _feature_homography(src, dst):
    """Homography src px -> dst px from ORB matches (both at the working size), or (None, 0)."""
    gs, gd = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (src, dst))
    orb = cv2.ORB_create(5000)
    ks, ds = orb.detectAndCompute(gs, None)
    kd, dd = orb.detectAndCompute(gd, None)
    if ds is None or dd is None or len(ks) < MIN_INLIERS or len(kd) < MIN_INLIERS:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ds, dd, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
    if len(good) < MIN_INLIERS:
        return None, len(good)
    a = np.float32([ks[m.queryIdx].pt for m in good])
    b = np.float32([kd[m.trainIdx].pt for m in good])
    A, inl = cv2.findHomography(a, b, cv2.RANSAC, MAX_REPROJ_PX)
    n = int(inl.sum()) if inl is not None else 0
    return (A, n) if A is not None and n >= MIN_INLIERS else (None, n)


def align(ref_bg, ref_corners, bg):
    """Venue (its background + corners) -> corners for a match whose background is bg.
    -> (corners or None, report dict)."""
    rs, ks = court._small(ref_bg)
    ms, km = court._small(bg)
    A, inliers = _feature_homography(rs, ms)
    rep = dict(inliers=inliers)
    if A is None:
        rep["error"] = "畫面和場地影片對不起來（手機位置差太多？），請重新校正"
        return None, rep
    moved = cv2.perspectiveTransform(np.float64(ref_corners).reshape(-1, 1, 2) * ks, A).reshape(-1, 2) / km
    snapped, on_line = court.snap(bg, moved)
    G = cv2.getPerspectiveTransform(np.float32(court.CORNERS_M), np.float32(snapped * km))[None]
    _, ok = court._plausible(G, ms.shape[1::-1])
    rep.update(on_line=round(float(on_line), 2), plausible=bool(ok[0]),
               moved_px=round(float(np.abs(snapped - np.float64(ref_corners)).max()), 1))
    if not ok[0]:
        rep["error"] = "對齊後的球場位置不合理，請重新校正"
        return None, rep
    return snapped, rep
