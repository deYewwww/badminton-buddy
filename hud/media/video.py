"""Video input. Frames come from ffmpeg using the Mac's hardware decoder and scaler
(VideoToolbox): on a 3K 60 fps clip that is ~2 s of CPU instead of ~60 s for software
decode + resize. Falls back to software ffmpeg where VideoToolbox is unavailable."""
import json
import subprocess
from functools import lru_cache

import cv2
import numpy as np


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    return cap


def probe(path):
    """-> (width, height, fps, frame count) via ffprobe (no decoding)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets",
                          "-of", "json", str(path)], capture_output=True, text=True)
    s = json.loads(out.stdout or "{}").get("streams")
    if not s:
        raise SystemExit(f"Cannot read video: {path}")
    s = s[0]
    num, den = (int(x) for x in s.get("avg_frame_rate", "30/1").split("/"))
    w, h = int(s["width"]), int(s["height"])
    if _turn(str(path)) in (90, 270):  # as displayed (and as OpenCV / frames() return them)
        w, h = h, w
    return w, h, num / den if den else 30.0, int(s.get("nb_read_packets", 0))


def _packets(path):
    """(pts, discarded) of every video packet, in file order. Discarded (flag D): outside the
    edit list, so never decoded: an iPhone clip starts with ~14 of them at negative times."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "packet=pts_time,flags", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    rows = [r.split(",") for r in out.split() if r.strip()]
    rows = [(float(r[0]), "D" in (r[1] if len(r) > 1 else "")) for r in rows if r[0] not in ("", "N/A")]
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows], bool)


def frame_times(path):
    """Per-frame timestamps (s) of the frames the decoder really outputs, correct for
    variable-frame-rate phone clips. From the container's packet times (sorted: B-frames are
    stored out of order), nothing decoded. Discarded packets are left out: counting them put
    every time on the user's iPhone clip 0.43 s late (14 pre-roll packets at -0.9..0 s) and
    gave 15 times too many; without them all 13169 times equal the decoded frames' (ffmpeg
    showinfo, < 1 us)."""
    pts, gone = _packets(path)
    t = np.sort(pts[~gone])
    if not len(t):
        raise SystemExit(f"No frames could be read from {path}")
    return list(t - t[0])


def describe(path):
    """Video index for the check report: what the file says about its timing next to the frames it
    really has, and warnings when they disagree. Measures, never fixes."""
    import hashlib
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,profile,pix_fmt,r_frame_rate,avg_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True)
    s = (json.loads(out.stdout or "{}").get("streams") or [{}])[0]
    rate = lambda r: (lambda a, b: a / b if b else None)(*(int(x) for x in r.split("/"))) if r else None
    pts, gone = _packets(path)
    t = np.array(frame_times(path))
    d = np.diff(t)
    med = float(np.median(d)) if len(d) else float("nan")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    w, ht, fps_tag, _ = probe(path)
    nominal = rate(s.get("r_frame_rate"))
    info = dict(sha256=h.hexdigest(), codec=s.get("codec_name"), profile=s.get("profile"), pix_fmt=s.get("pix_fmt"),
                size=[w, ht], fps_nominal=round(nominal, 3) if nominal else None, fps_avg_tag=round(fps_tag, 3),
                fps_measured=round(1 / med, 3), packets=len(pts), frames=len(t), discarded=int(gone.sum()),
                duration_s=round(float(t[-1]), 3) if len(t) else 0.0, interval_ms=dict(
                    median=round(med * 1000, 2), min=round(float(d.min()) * 1000, 2), max=round(float(d.max()) * 1000, 2)),
                irregular=int((np.abs(d - med) > 0.25 * med).sum()), dropped=int((d > 1.5 * med).sum()))
    warn = []
    if nominal and abs(nominal * med - 1) > 0.02:
        warn.append(f"標示的幀率 {nominal:.2f} 和實際間隔 {1 / med:.2f} fps 不同：時間一律用實際時間戳。")
    if info["irregular"] > 0.01 * len(d):
        warn.append(f"可變幀率：{info['irregular']} 個間隔偏離中位數 25% 以上（已用每格真實時間，不用名義 fps）。")
    if info["dropped"]:
        warn.append(f"{info['dropped']} 處間隔超過 1.5 格，可能掉格：那裡的速度用到的時間差會比較長。")
    if 1 / med >= 100:
        warn.append("高幀率檔（慢動作）：確認時間戳是拍攝時間而不是播放時間，再拿來算速度。")
    info["warnings"] = warn
    return info


@lru_cache(maxsize=32)
def _ten_bit(path):
    """iPhone HDR clips are 10-bit HEVC: VideoToolbox frames then download as p010le, not nv12
    (asking for nv12 fails, and the software fallback decoded 4K at ~60 fps on every core)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return "10" in out or "12" in out


@lru_cache(maxsize=32)
def _turn(path):
    """Clockwise degrees the player turns the picture by (phone rotation metadata). The software
    decoder applies it by itself; the hardware path does not (an upside-down iPhone clip came
    out upside down and YOLO found nobody), so frames() adds the turn there."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream_side_data=rotation", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    r = next((float(x) for x in out.replace(",", " ").split() if x.lstrip("-").replace(".", "").isdigit()), 0.0)
    return int(round(-r)) % 360


@lru_cache(maxsize=1)
def _hw():
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return "scale_vt" in out


def frames(path, size, every=1, rgb=False, pre=None, keys=False):
    """Yield every `every`-th frame resized to size=(w, h), BGR (or RGB) uint8.
    pre: ffmpeg filters applied before the resize (crop/pad); software decode then.
    keys: decode key frames only (every-th of those): a fast spread-out sample."""
    w, h = size
    sel = f"select='not(mod(n\\,{every}))'," if every > 1 else ""
    pix = "rgb24" if rgb else "bgr24"
    variants = []
    skip = ["-skip_frame", "nokey"] if keys else []
    if pre:
        sel += pre + ","
    elif _hw():
        turn = _turn(str(path))
        sw, sh = (h, w) if turn in (90, 270) else (w, h)  # scale before turning
        fix = {90: ",transpose=clock", 180: ",hflip,vflip", 270: ",transpose=cclock"}.get(turn, "")
        variants.append([*skip, "-noautorotate", "-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld",
                         "-i", str(path), "-vf", f"{sel}scale_vt=w={sw}:h={sh},hwdownload,"
                         f"format={'p010le' if _ten_bit(str(path)) else 'nv12'}{fix}"])
    variants.append([*skip, "-i", str(path), "-vf", f"{sel}scale={w}:{h}:flags=area"])
    n = w * h * 3
    for args in variants:
        proc = subprocess.Popen(["ffmpeg", "-v", "error", *args, "-fps_mode", "passthrough",
                                 "-f", "rawvideo", "-pix_fmt", pix, "-"], stdout=subprocess.PIPE)
        got = 0
        try:
            while True:
                buf = proc.stdout.read(n)
                if len(buf) < n:
                    break
                got += 1
                yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        finally:
            proc.stdout.close()
            proc.wait()
        if got:
            return  # else: hardware path produced nothing, try software


def sample(path, size, n, rgb=False, pre=None):
    """About n frames spread over the clip, key frames only: decoding every frame just to keep a
    few (median backgrounds, the player region) was a whole extra pass over a 10-minute clip."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=flags",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    keys = max(1, out.count("K"))
    return list(frames(path, size, every=max(1, keys // n), rgb=rgb, pre=pre, keys=True))[:n]


def prefetch(gen, depth=16):
    """Run a frame generator in a background thread, `depth` items ahead, so decoding and
    resizing (ffmpeg, cv2: both release the GIL) overlap the model instead of alternating
    with it. On a 4K clip the TrackNet stage spent ~15 s on each side."""
    import queue
    import threading
    q, end = queue.Queue(depth), object()

    def work():
        try:
            for x in gen:
                q.put(x)
        except BaseException as e:  # surfaced in the consumer
            q.put(e)
        q.put(end)
    threading.Thread(target=work, daemon=True).start()
    while (x := q.get()) is not end:
        if isinstance(x, BaseException):
            raise x
        yield x
