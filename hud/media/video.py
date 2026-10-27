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


def frame_times(path):
    """Per-frame timestamps (s), correct for variable-frame-rate phone clips. Read from the
    container's packet times (sorted: B-frames are stored out of order), nothing decoded."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "packet=pts_time", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    t = sorted(float(x) for x in out.split() if x.strip() and x.strip() != "N/A")
    if not t:
        raise SystemExit(f"No frames could be read from {path}")
    return list(np.array(t) - t[0])


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
