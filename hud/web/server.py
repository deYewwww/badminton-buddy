"""Local web editor (standard library only): upload, calibrate, analyse, fix, download.

Listens on 127.0.0.1. --lan opens it to the local network and then requires a password
(HTTP basic auth). One analysis runs at a time: on an 8 GB Mac two would just thrash.
"""
import base64
import json
import mimetypes
import re
import secrets
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pandas as pd

from hud import court, pipeline
from hud.court import venue

ROOT = Path(__file__).resolve().parents[2]
UPLOADS, OUT = ROOT / "data" / "uploads", ROOT / "out"
STATIC = Path(__file__).with_name("static")
JOB = {"id": None, "progress": 0.0, "msg": "", "error": None, "running": False}
LOCK = threading.Lock()
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def match_dir(mid):
    if not SAFE_ID.match(mid or ""):
        raise ValueError("bad match id")
    return OUT / mid


def video_of(mid):
    meta = json.loads((match_dir(mid) / "match.json").read_text())
    return Path(meta["video"])


def bg_frame(mid):
    """Median background frame (players removed), cached in the match folder."""
    f = match_dir(mid) / "background.jpg"
    if not f.exists():
        cv2.imwrite(str(f), pipeline.background(video_of(mid)), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return cv2.imread(str(f))


def calib_frame(mid, t):
    return bg_frame(mid) if t is None or float(t) < 0 else \
        pipeline.frame(video_of(mid), int(float(t) * info(video_of(mid))["fps"]))


def motion(mid):
    """{moving, shift (fraction of the width), ref_t}, measured once per match and cached."""
    f = match_dir(mid) / "match.json"
    m = json.loads(f.read_text())
    if "motion" not in m:
        from hud.court.stabilize import moving
        i = info(video_of(mid))
        ref = i["frames"] // 2
        mv, shift, lost = moving(video_of(mid), ref)
        m["motion"] = dict(moving=mv, shift=shift, lost=lost, ref_t=round(ref / i["fps"], 1))
        f.write_text(json.dumps(m, ensure_ascii=False))
    return m["motion"]


def info(video):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("無法讀取這個影片檔")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return dict(w=w, h=h, fps=fps, frames=n, duration=n / fps)


def matches():
    out = []
    for d in sorted(OUT.glob("*/match.json"), key=lambda p: -p.stat().st_mtime):
        m = json.loads(d.read_text())
        p = pipeline.paths(d.parent)
        out.append(dict(id=d.parent.name, name=m.get("name", ""), calibrated=p["calib"].exists(),
                        done=p["hud"].exists()))
    return out


def run_job(mid, start):
    def progress(x, msg):
        JOB.update(progress=round(float(x), 3), msg=msg)
    try:
        pipeline.analyse(video_of(mid), match_dir(mid), progress=progress, start=start)
    except BaseException as e:  # SystemExit from a stage included: report, don't kill the server
        traceback.print_exc()
        JOB["error"] = str(e) or e.__class__.__name__
    finally:
        JOB["running"] = False
        LOCK.release()


def start_job(mid, start):
    if not LOCK.acquire(blocking=False):
        raise RuntimeError("已有一個分析在執行中，請等它完成")
    JOB.update(id=mid, progress=0.0, msg="開始…", error=None, running=True)
    threading.Thread(target=run_job, args=(mid, start), daemon=True).start()


def rallies(mid):
    p = pipeline.paths(match_dir(mid))
    if not p["rallies"].exists():
        return []
    r = pd.read_parquet(p["rallies"])
    cols = ["rally_id", "start_ts", "end_ts", "duration_s", "shot_count", "point_team_final", "point_source",
            "score_us", "score_opp", "winning_label", "manual"]
    r = r[[c for c in cols if c in r]]
    return json.loads(r.to_json(orient="records", force_ascii=False))


def set_winner(mid, rally_id, team):
    """Store a manual winner (or remove it with team='auto') in point_fixes.csv."""
    f = pipeline.paths(match_dir(mid))["fixes"]
    fx = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["rally_id", "point_team", "start_ts"])
    fx = fx[fx.rally_id != int(rally_id)]
    if team in ("us", "opp", "none"):  # none: not a real rally, no point
        # with the rally's start time: a re-analysis renumbers rallies (rallies.fixes_by_time)
        r = pd.read_parquet(pipeline.paths(match_dir(mid))["rallies"], columns=["rally_id", "start_ts"])
        ts = r.start_ts[r.rally_id == int(rally_id)]
        fx = pd.concat([fx, pd.DataFrame([dict(rally_id=int(rally_id), point_team=team,
                                               start_ts=round(float(ts.iat[0]), 3) if len(ts) else None)])])
    fx.to_csv(f, index=False)


def hit_marks(mid, t=None, kind=None):
    """Hand-marked serves, hits and rally ends (editor ⑦): the answer key for serves, shot counts and
    rally times (analysis/evaluate.truth_from_marks). kind 'remove' drops the mark at t."""
    f = match_dir(mid) / "hit_marks.csv"
    df = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["t", "kind"])
    if t is not None:
        df = df[(df.t - float(t)).abs() > 0.02]
        if kind in ("serve", "hit", "end"):
            df = pd.concat([df, pd.DataFrame([dict(t=round(float(t), 3), kind=kind)])])
        df = df.sort_values("t")
        df.to_csv(f, index=False)
    return json.loads(df.to_json(orient="records"))


def manual_point(mid, t, team):
    """Add a point at video time t (s) for a rally the analysis missed, or remove the one at t
    (team='remove'). Stored in manual_points.csv; takes effect on the next rallies rerun."""
    f = pipeline.paths(match_dir(mid))["manual"]
    mp = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["t", "point_team"])
    mp = mp[(mp.t - float(t)).abs() > 0.05]
    if team in ("us", "opp"):
        mp = pd.concat([mp, pd.DataFrame([dict(t=round(float(t), 2), point_team=team)])])
    mp.sort_values("t").to_csv(f, index=False)
    return len(mp)


class Handler(BaseHTTPRequestHandler):
    password = None

    def log_message(self, fmt, *a):
        pass

    def _auth(self):
        if not self.password:
            return True
        h = self.headers.get("Authorization", "")
        ok = h.startswith("Basic ") and secrets.compare_digest(
            base64.b64decode(h[6:]).decode(errors="ignore").split(":", 1)[-1], self.password)
        if not ok:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="badminton-hud"')
            self.end_headers()
        return ok

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _file(self, path, ctype=None):
        """Serve a file with HTTP Range support (video seeking needs it)."""
        size = path.stat().st_size
        rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        a, b = 0, size - 1
        if rng and (rng.group(1) or rng.group(2)):
            if rng.group(1):
                a = int(rng.group(1))
                b = int(rng.group(2)) if rng.group(2) else size - 1
            else:
                a = size - int(rng.group(2))
            b = min(b, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(a)
            left = b - a + 1
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def do_GET(self):
        if not self._auth():
            return
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._file(STATIC / "index.html", "text/html; charset=utf-8")
            if u.path == "/api/venues":
                return self._json(venue.names())
            if u.path == "/api/matches":
                return self._json(matches())
            if u.path == "/api/match":
                d = match_dir(q["id"])
                m = json.loads((d / "match.json").read_text())
                p = pipeline.paths(d)
                calib = json.loads(p["calib"].read_text()) if p["calib"].exists() else None
                marks = json.loads(p["marks"].read_text()) if p["marks"].exists() else []
                return self._json(dict(m, id=q["id"], corners=calib and calib["corners"], calib_t=calib and calib.get("t"),
                                       names=pipeline.names(d), marks=marks, done=p["hud"].exists()))
            if u.path == "/api/motion":  # does the camera pan / cut? (cached: ~5-10 s to measure)
                return self._json(motion(q["id"]))
            if u.path == "/api/frame":
                f = calib_frame(q["id"], q.get("t"))
                ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                return self.wfile.write(jpg.tobytes())
            if u.path == "/api/label/list":  # shuttle labelling (⑥): frames to label + what is done
                from hud.detection import finetune
                d = match_dir(q["id"])
                lab = finetune.labels(d)
                return self._json(dict(frames=finetune.todo(d), labels=json.loads(lab.to_json(orient="records"))))
            if u.path == "/api/label/frame":  # one full-resolution frame
                cap = cv2.VideoCapture(str(video_of(q["id"])))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(q["f"]))
                ok, f = cap.read()
                cap.release()
                if not ok:
                    return self._json({"error": "讀不到這一幀"}, 404)
                ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                return self.wfile.write(jpg.tobytes())
            if u.path == "/api/video":  # the original upload (hit marking: no HUD numbers to sway the count)
                return self._file(video_of(q["id"]))
            if u.path == "/api/hitmarks":
                return self._json(hit_marks(q["id"]))
            if u.path == "/api/status":
                return self._json(JOB)
            if u.path == "/api/rallies":
                return self._json(rallies(q["id"]))
            if u.path == "/api/hud":
                p = pipeline.paths(match_dir(q["id"]))["hud"]
                return self._file(p, "video/mp4") if p.exists() else self._json({"error": "還沒有 HUD 影片"}, 404)
            self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._json({"error": str(e)}, 400)

    def do_POST(self):
        if not self._auth():
            return
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/upload":  # raw file body, name in the query string
                name = Path(q.get("name", "video.mp4")).name
                ext = Path(name).suffix.lower() if Path(name).suffix.lower() in (".mp4", ".mov", ".m4v") else ".mp4"
                mid = uuid.uuid4().hex[:10]
                UPLOADS.mkdir(parents=True, exist_ok=True)
                dst = UPLOADS / f"{mid}{ext}"
                left = int(self.headers.get("Content-Length", 0))
                with open(dst, "wb") as fh:
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk:
                            break
                        fh.write(chunk)
                        left -= len(chunk)
                meta = dict(name=name, video=str(dst), **info(dst))
                d = match_dir(mid)
                d.mkdir(parents=True, exist_ok=True)
                (d / "match.json").write_text(json.dumps(meta, ensure_ascii=False))
                return self._json(dict(meta, id=mid))
            b = self._body()
            mid = b.get("id")
            if u.path == "/api/calib/auto":
                f = calib_frame(mid, b.get("t"))  # background (t < 0) has no players in front of the lines
                corners, frac = court.auto_detect(f)
                if corners is None:
                    return self._json({"error": "找不到球場白線"}, 422)
                return self._json(dict(corners=np.round(corners, 1).tolist(), on_line=round(frac, 2)))
            if u.path == "/api/calib/snap":
                f = calib_frame(mid, b.get("t"))  # background (t < 0) has no players in front of the lines
                corners, frac = court.snap(f, np.float64(b["corners"]))
                return self._json(dict(corners=np.round(corners, 1).tolist(), on_line=round(frac, 2)))
            if u.path == "/api/venue/save":  # this match's calibration + background, reusable
                corners = b.get("corners") or json.loads(pipeline.paths(match_dir(mid))["calib"].read_text())["corners"]
                court.homography(np.float64(corners))  # raises on a degenerate quad
                venue.save(b["name"], bg_frame(mid), corners)
                return self._json({"ok": True, "venues": venue.names()})
            if u.path == "/api/venue/apply":  # align a saved venue onto this match
                ref, rc = venue.load(b["name"])
                corners, rep = venue.align(ref, rc, bg_frame(mid))
                if corners is None:
                    return self._json({"error": rep.get("error", "對不起來"), **rep}, 422)
                return self._json(dict(rep, corners=np.round(corners, 1).tolist()))
            if u.path == "/api/calib/save":
                v = video_of(mid)
                i = info(v)
                court.homography(np.float64(b["corners"]))  # raises on a degenerate quad
                court.save(pipeline.paths(match_dir(mid))["calib"], v, (i["w"], i["h"]), b["corners"], t=b.get("t"))
                from hud.analysis.check import calib_fit  # warn before a 20-minute analysis on a wrong court
                fits = [calib_fit(bg_frame(mid), b["corners"])]
                if b.get("t") is not None and float(b["t"]) >= 0:  # the frame the user aligned on
                    fits.append(calib_fit(calib_frame(mid, b["t"]), b["corners"]))
                fit, plausible = max(f[0] for f in fits), fits[0][1]
                return self._json({"ok": True, "on_line": round(fit, 2), "plausible": plausible})
            if u.path == "/api/marks":  # player feet clicked in the editor: [{t, pid, x, y}]
                marks = [dict(t=round(float(m["t"]), 2), pid=m["pid"], x=round(float(m["x"]), 1), y=round(float(m["y"]), 1))
                         for m in b.get("marks", []) if m.get("pid") in ("P1", "P2", "P3", "P4")]
                pipeline.paths(match_dir(mid))["marks"].write_text(json.dumps(marks))
                return self._json({"ok": True, "n": len(marks)})
            if u.path == "/api/names":
                return self._json(pipeline.names(match_dir(mid), b.get("names", {})))
            if u.path == "/api/run":
                if not pipeline.paths(match_dir(mid))["calib"].exists():
                    return self._json({"error": "請先校正球場"}, 400)
                start_job(mid, b.get("start", "players"))
                return self._json({"ok": True})
            if u.path == "/api/winner":
                set_winner(mid, b["rally_id"], b["team"])
                return self._json({"ok": True})
            if u.path == "/api/label":  # {f, x, y, state: shuttle/none/hidden/remove}
                from hud.detection import finetune
                return self._json({"ok": True, "n": finetune.save_label(match_dir(mid), b["f"], b.get("x", 0), b.get("y", 0), b["state"])})
            if u.path == "/api/hitmark":  # {t, kind: serve/hit/end/remove}
                return self._json(hit_marks(mid, b["t"], b["kind"]))
            if u.path == "/api/point":  # {t, team: us/opp/remove}
                return self._json({"ok": True, "n": manual_point(mid, b["t"], b["team"])})
            self._json({"error": "not found"}, 404)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": str(e)}, 400)


def free_port(host, port):
    import socket
    for p in range(port, port + 20):
        with socket.socket() as s:
            if s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", p)) != 0:
                return p
    raise SystemExit("no free port")


def serve(port=7860, lan=False, password=None):
    if lan and not password:
        raise SystemExit("--lan 需要 --password（區網上的其他人也能連進來）")
    host = "0.0.0.0" if lan else "127.0.0.1"
    p = free_port(host, port)
    if p != port:
        print(f"Port {port} 被占用，改用 {p}")
    Handler.password = password
    srv = ThreadingHTTPServer((host, p), Handler)
    print(f"打開 http://127.0.0.1:{p}" + ("（區網：用這台 Mac 的 IP）" if lan else ""), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
