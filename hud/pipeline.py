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
        check_png="check.png", stab="stab.mp4", stab_calib="calib_stab.json", stab_marks="marks_stab.json").items()}


def work(out_dir, video):
    """(video, calibration, marks) the analysis stages run on: the stabilised copy when the
    camera moved (see prepare), else the original."""
    p = paths(out_dir)
    if p["stab"].exists() and p["stab_calib"].exists():
        return p["stab"], p["stab_calib"], p["stab_marks"]
    return Path(video), p["calib"], p["marks"]


def _ref_time(video, calib):
    """The time the corners were set on; for corners set on the background, the frame they fit best."""
    if calib.get("t") is not None:
        return float(calib["t"])
    from hud.check import calib_fit
    from hud.video import probe
    _, _, fps, n = probe(video)
    idx = np.linspace(0, max(n - 1, 0), 15).astype(int)
    fits = [calib_fit(frame(video, int(i)), calib["corners"])[0] for i in idx]
    return float(idx[int(np.argmax(fits))] / fps)


def prepare(video, out_dir, progress=print):
    """A panning (gimbal) or cut-together clip is stabilised once onto the view the court was
    calibrated on, so every stage sees one fixed camera; the HUD is still drawn on the original
    footage (render maps it back frame by frame). A tripod clip is left alone.
    Reused while the calibration is unchanged. -> what work() returns."""
    from hud import stabilize
    from hud.video import probe
    p = paths(out_dir)
    calib = json.loads(p["calib"].read_text())
    key = dict(src_corners=calib["corners"], src_t=calib.get("t"), src_video=str(video))
    if p["stab_calib"].exists() and p["stab"].exists():
        old = json.loads(p["stab_calib"].read_text())
        if all(old.get(k) == v for k, v in key.items()):
            _stab_marks(p)
            return work(out_dir, video)
    for k in ("stab", "stab_calib", "stab_marks"):
        p[k].unlink(missing_ok=True)
    p["stab"].with_suffix(".json").unlink(missing_ok=True)
    t = _ref_time(video, calib)
    fps = probe(video)[2]
    moved, shift, lost = stabilize.moving(video, int(round(t * fps)))
    print(f"Camera {'moves' if moved else 'is fixed'}: view shift up to {shift:.1%} of the width, {lost} samples unmatched")
    if not moved:
        return work(out_dir, video)
    progress("鏡頭有移動，先穩定畫面…")
    corners = stabilize.run(video, p["stab"], t, calib["corners"])
    cw, ch = json.loads(p["stab"].with_suffix(".json").read_text())["canvas"]
    court.save(p["stab_calib"], p["stab"], (cw, ch), corners, t=t, **key)
    _stab_marks(p)
    return work(out_dir, video)


def _stab_marks(p):
    """marks.json (clicked on the original frames) -> marks_stab.json on the stabilised view."""
    from hud.stabilize import frame_maps
    p["stab_marks"].unlink(missing_ok=True)
    if not p["marks"].exists():
        return
    M = frame_maps(p["stab"])
    from hud.video import probe
    fps = probe(p["stab"])[2]
    out = []
    for m in json.loads(p["marks"].read_text()):
        A = M[int(np.clip(round(m["t"] * fps), 0, len(M) - 1))]
        x, y = court.to_court(A, [[m["x"], m["y"]]])[0]
        out.append(dict(m, x=round(float(x), 1), y=round(float(y), 1)))
    p["stab_marks"].write_text(json.dumps(out))


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
        progress(0.0, "檢查鏡頭是否移動…")
        wv, wc, wm = prepare(video, out_dir, progress=lambda m: progress(0.0, m))
        lap("stabilize")
    else:
        wv, wc, wm = work(out_dir, video)
    if "players" in todo:
        progress(0.0, "偵測球員…")
        players.run(wv, wc, p["players"], progress=lambda x: progress(0.45 * x, "偵測球員…"), marks=wm)
        lap("players")
    if "shuttle" in todo:
        progress(0.45, "追蹤羽球…")
        shuttle.run(wv, p["shuttle"], players_path=p["players"], calib_path=wc)
        lap("shuttle")
    if "hits" in todo:
        progress(0.7, "判斷擊球…")
        hits.run(p["shuttle"], p["players"], wc, p["hits"])
        lap("hits")
    if "rallies" in todo:
        progress(0.75, "切分回合、計分…")
        from hud.stabilize import cuts_of
        rallies.run(p["hits"], p["players"], p["shots"], p["rallies"], shuttle_path=p["shuttle"],
                    point_fixes=p["fixes"], cuts=cuts_of(wv))
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
        rep = check.run(out_dir)
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
