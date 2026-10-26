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
    return int(s["width"]), int(s["height"]), num / den if den else 30.0, int(s.get("nb_read_packets", 0))


def frame_times(path):
    """Per-frame timestamps (s), correct for variable-frame-rate phone clips. Read from the
    container's packet times (sorted: B-frames are stored out of order), nothing decoded."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "packet=pts_time", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    t = sorted(float(x) for x in out.split() if x.strip() and x.strip() != "N/A")
    if not t:
        raise SystemExit(f"No frames could be read from {path}")
    return list(np.array(t) - t[0])


@lru_cache(maxsize=1)
def _hw():
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return "scale_vt" in out


def frames(path, size, every=1, rgb=False, pre=None):
    """Yield every `every`-th frame resized to size=(w, h), BGR (or RGB) uint8.
    pre: ffmpeg filters applied before the resize (crop/pad); software decode then."""
    w, h = size
    sel = f"select='not(mod(n\\,{every}))'," if every > 1 else ""
    pix = "rgb24" if rgb else "bgr24"
    variants = []
    if pre:
        sel += pre + ","
    elif _hw():
        variants.append(["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld", "-i", str(path),
                         "-vf", f"{sel}scale_vt=w={w}:h={h},hwdownload,format=nv12"])
    variants.append(["-i", str(path), "-vf", f"{sel}scale={w}:{h}:flags=area"])
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
