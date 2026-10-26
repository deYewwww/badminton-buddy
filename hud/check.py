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


def calib_fit(frame, corners, tol=8):
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
    _, ok = court._plausible(G, (w, h))
    return float(frac[0]), bool(ok[0])


def run(out_dir, video=None, draw=True):
    from hud.pipeline import background, paths
    p = paths(out_dir)
    rep, warn = {}, []
    if not p["calib"].exists():
        return dict(warnings=["還沒校正球場"])
    H, calib = court.load(p["calib"])
    video = video or calib["video"]
    bg = background(video)
    fit, plausible = calib_fit(bg, calib["corners"])
    rep["calib"] = dict(corners=calib["corners"], on_line=round(fit, 2), camera_plausible=plausible)
    if fit < 0.3 and not plausible:
        warn.append(f"球場校正可能錯了：黃線只有 {fit:.0%} 落在偵測到的白線上。")
    if not plausible:
        warn.append("球場校正推算出的相機位置不合理（不在近端底線後方、或高度不對），角點很可能放錯。")

    pl = pd.read_parquet(p["players"]) if p["players"].exists() else None
    if pl is not None:
        frames = max(int(pl.frame.max()) + 1, 1)
        seen = (pl[pl.observed].groupby("player_id").frame.nunique() / frames).reindex(["P1", "P2", "P3", "P4"]).fillna(0)
        rep["players_observed"] = seen.round(2).to_dict()
        if seen[["P3", "P4"]].max() < 0.1 and seen[["P1", "P2"]].max() > 0.3:
            warn.append("幾乎沒偵測到對手（遠端）。通常是校正錯：遠端球員被算到場外而丟掉，並不是模型看不到人。")
        elif seen.min() < 0.3:
            warn.append(f"有球員大部分時間沒追到：{', '.join(f'{k} {v:.0%}' for k, v in seen.items())}")
    sh = pd.read_parquet(p["shuttle"]) if p["shuttle"].exists() else None
    if sh is not None:
        det = float((sh.visible & ~sh.interpolated & ~sh.inpainted).mean())
        rep["shuttle_detected"] = round(det, 3)
        if det < 0.05:
            warn.append(f"羽球幾乎沒追到（{det:.1%} 的幀）。")
    if p["hits"].exists():
        hits = pd.read_parquet(p["hits"])
        rep["hits"] = len(hits)
        if "team" in hits:
            rep["hits_by_team"] = hits.team.value_counts().to_dict()
        if not len(hits):
            warn.append("沒有找到任何擊球：回合、拍數、比分都會是空的。先修好上面的問題。")
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
    lines += ["⚠ " + w for w in rep.get("warnings", [])] or ["沒有發現明顯問題"]
    return "\n".join(lines)
