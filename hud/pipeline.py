"""Video -> calibration -> players -> shuttle -> hits -> rallies/score -> HUD video.

Every stage writes its own file in the match folder and overwrites it on a rerun, so a
fix (court corners, a rally's winner, names) only reruns the stages after it.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from hud import court, hits, players, rallies, shuttle


def paths(out_dir):
    o = Path(out_dir)
    return {k: o / v for k, v in dict(
        calib="calib.json", players="players.parquet", shuttle="shuttle.parquet", hits="hits.parquet",
        shots="shots.parquet", rallies="rallies.parquet", fixes="point_fixes.csv", names="names.json",
        hud="hud.mp4", timing="timing.json", check="check.json", marks="marks.json",
        check_png="check.png").items()}


def frame(video, index=0):
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(index, max(n - 1, 0)))
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read a frame from {video}")
    return f


def background(video, n=15):
    """Median of n frames spread over the video: players and shuttle vanish, lines stay."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        raise ValueError(f"cannot read frames from {video}")
    return np.median(frames, axis=0).astype(np.uint8)


def calibrate(video, out_dir):
    """Auto-detect the court on the background frame and save it. -> (calib, on_line fraction)."""
    f = background(video)
    corners, frac = court.auto_detect(f)
    if corners is None:
        return None, 0.0
    return court.save(paths(out_dir)["calib"], video, f.shape[1::-1], corners), frac


def analyse(video, out_dir, progress=print, start="players"):
    """Run the stages from `start` on. progress(fraction 0..1, message)."""
    p = paths(out_dir)
    stages = ["players", "shuttle", "hits", "rallies", "hud"]
    todo = stages[stages.index(start):]
    t0 = last = time.time()
    took = {}  # seconds per stage, written to timing.json: measure before optimising

    def lap(stage):
        nonlocal last
        took[stage], last = round(time.time() - last, 1), time.time()
    if "players" in todo:
        progress(0.0, "偵測球員…")
        players.run(video, p["calib"], p["players"], progress=lambda x: progress(0.45 * x, "偵測球員…"),
                    marks=p["marks"])
        lap("players")
    if "shuttle" in todo:
        progress(0.45, "追蹤羽球…")
        shuttle.run(video, p["shuttle"], players_path=p["players"])
        lap("shuttle")
    if "hits" in todo:
        progress(0.7, "判斷擊球…")
        hits.run(p["shuttle"], p["players"], p["calib"], p["hits"])
        lap("hits")
    if "rallies" in todo:
        progress(0.75, "切分回合、計分…")
        rallies.run(p["hits"], p["players"], p["shots"], p["rallies"], shuttle_path=p["shuttle"],
                    point_fixes=p["fixes"])
        lap("rallies")
    if "hud" in todo:
        from hud import render
        progress(0.8, "產生 HUD 影片…")
        render.render(video, out_dir, progress=lambda x: progress(0.8 + 0.2 * x, "產生 HUD 影片…"))
        lap("hud")
    total = time.time() - t0
    p["timing"].write_text(json.dumps(dict(took, total=round(total, 1)), ensure_ascii=False))
    print("Stage seconds:", took)
    warn = []
    try:  # catch the usual "everything is empty" causes (bad calibration first) and say so
        from hud import check
        rep = check.run(out_dir, video)
        print(check.text(rep))
        warn = rep.get("warnings", [])
    except Exception as e:
        print(f"check failed: {e}")
    progress(1.0, f"完成（{total:.0f} 秒：" + "、".join(f"{k} {v:.0f}" for k, v in took.items()) + "）"
             + "".join("\n⚠ " + w for w in warn))
    return p


def names(out_dir, new=None):
    """Player display names, stored per match."""
    f = paths(out_dir)["names"]
    cur = {"P1": "我隊1", "P2": "我隊2", "P3": "對手1", "P4": "對手2"}
    if f.exists():
        cur.update(json.loads(f.read_text()))
    if new:
        cur.update({k: v.strip() for k, v in new.items() if v and v.strip()})
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(cur, ensure_ascii=False))
    return cur
