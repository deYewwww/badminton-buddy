"""Sanity checks on a finished (or half-finished) analysis: is the calibration believable,
were all 4 players seen, was the shuttle tracked, were hits found? Most "everything is wrong"
results trace back to one bad stage, usually the court calibration: every player position,
the far/near split and the hitter come from it. Writes check.json and check.png (court
model and detected feet drawn on the background frame) next to the other outputs.

Run: uv run hud check out/<match>
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hud import court


def calib_fit(frame, corners, tol=8, native=None):
    """Fraction of the court model (inside the frame) that lies within tol px (at the 1280 px
    working size) of a detected painted line, and whether a phone standing behind the near
    baseline could have produced it.

    Only the second is a verdict. The first depends on line_mask() finding the paint, which
    fails on exactly the footage where auto-detection fails (screen recordings, blur, a
    background median spoiled by camera movement), so a correct hand calibration can score
    low: it is shown for information and never overrides the user's points."""
    small, s = court._small(frame)
    h, w = small.shape[:2]
    dist = court._signed_dist(court.line_mask(small))
    G = cv2.getPerspectiveTransform(np.float32(court.CORNERS_M), np.float32(np.float64(corners) * s))[None]
    _, frac = court._score(G, dist, court._samples(25), tol)
    _, ok = court._plausible(G, (w, h), lens_side=max(native) * s if native else None)
    return float(frac[0]), bool(ok[0])


def run(out_dir, video=None, draw=True):
    from hud.pipeline import background, paths, work
    p = paths(out_dir)
    rep, warn = {}, []
    if not p["calib"].exists():
        return dict(warnings=["The court is not calibrated yet"])
    wv, wc, _ = work(out_dir, video or json.loads(p["calib"].read_text())["video"])
    H, calib = court.load(wc)
    video = wv if wc != p["calib"] else (video or calib["video"])
    bg = background(video)
    from hud.court.stabilize import native_size
    fit, plausible = calib_fit(bg, calib["corners"], native=native_size(video))
    rep["calib"] = dict(corners=calib["corners"], on_line=round(fit, 2), camera_plausible=plausible)
    if fit < 0.3 and not plausible:
        warn.append(f"The court calibration may be wrong: only {fit:.0%} of the yellow lines fall on detected white lines.")
    if not plausible:
        warn.append("The calibration gives an implausible camera position (not behind the near baseline, or the wrong height): the corners are probably misplaced.")

    pl = pd.read_parquet(p["players"]) if p["players"].exists() else None
    if pl is not None:
        frames = max(int(pl.frame.max()) + 1, 1)
        seen = (pl[pl.observed].groupby("player_id").frame.nunique() / frames).reindex(["P1", "P2", "P3", "P4"]).fillna(0)
        rep["players_observed"] = seen.round(2).to_dict()
        if seen[["P3", "P4"]].max() < 0.1 and seen[["P1", "P2"]].max() > 0.3:
            warn.append("Almost no opponents (far side) detected. Usually a wrong calibration: far players are placed off court and dropped, not missed by the model.")
        elif seen.min() < 0.3:
            warn.append(f"Some players are untracked most of the time: {', '.join(f'{k} {v:.0%}' for k, v in seen.items())}")
    sh = pd.read_parquet(p["shuttle"]) if p["shuttle"].exists() else None
    if sh is not None:
        det = float((sh.visible & ~sh.interpolated & ~sh.inpainted).mean())
        rep["shuttle_detected"] = round(det, 3)
        if det < 0.05:
            warn.append(f"The shuttle is barely tracked ({det:.1%} of frames).")
    if p["hits"].exists():
        hits = pd.read_parquet(p["hits"])
        rep["hits"] = len(hits)
        if "team" in hits:
            rep["hits_by_team"] = hits.team.value_counts().to_dict()
        if not len(hits):
            warn.append("No hits found: rallies, shots and score will be empty. Fix the problems above first.")
    if p["rallies"].exists():
        rep["rallies"] = len(pd.read_parquet(p["rallies"]))
    rep["warnings"] = warn
    p["check"].write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    if draw:
        img = court.draw(bg.copy(), H, (0, 255, 255), max(2, bg.shape[1] // 900))
        if pl is not None:
            colours = {"P1": (240, 200, 54), "P2": (160, 224, 91), "P3": (255, 140, 180), "P4": (80, 176, 255)}
            for pid, g in pl[pl.observed].groupby("player_id"):
                g = g.iloc[::max(1, len(g) // 400)]
                for x, y in court.to_image(H, g[["x_m", "y_m"]].to_numpy()):
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        cv2.circle(img, (int(x), int(y)), max(3, img.shape[1] // 400), colours.get(pid, (255, 255, 255)), -1)
        cv2.imwrite(str(p["check_png"]), img)
    return rep


def text(rep):
    lines = [f"{k}: {v}" for k, v in rep.items() if k != "warnings"]
    lines += ["⚠ " + w for w in rep.get("warnings", [])] or ["No obvious problems found"]
    return "\n".join(lines)
