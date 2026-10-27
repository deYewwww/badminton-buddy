"""CLI: `hud web` (the editor) or `hud run VIDEO` (whole pipeline, auto court detection)."""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(prog="hud", description="Badminton HUD video maker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("web", help="open the web editor")
    w.add_argument("--port", type=int, default=7860)
    w.add_argument("--lan", action="store_true", help="let phones on the same network connect (needs --password)")
    w.add_argument("--password")
    r = sub.add_parser("run", help="run the whole pipeline from the command line")
    r.add_argument("video")
    r.add_argument("--out", help="output folder (default out/<video name>)")
    r.add_argument("--venue", help="calibrate from a saved venue (save one with hud venue-save)")
    r.add_argument("--from", dest="start", default="players", choices=["players", "shuttle", "hits", "rallies", "hud"],
                   help="stage to rerun from")
    ev = sub.add_parser("eval", help="compare with hand labels: rally, hit and score accuracy")
    ev.add_argument("out", help="analysis folder")
    ev.add_argument("truth", help="label file truth.json (format: hud/analysis/evaluate.py)")
    st = sub.add_parser("stabilize", help="align a panning (gimbal) video to one fixed view")
    st.add_argument("video")
    st.add_argument("--ref", type=float, required=True, help="time (s) of the reference frame (calibrate on this frame)")
    st.add_argument("--out", required=True, help="output video, e.g. data/stable.mp4")
    vs = sub.add_parser("venue-save", help="save an analysis folder's calibration as a venue (best from an empty-court clip)")
    vs.add_argument("name")
    vs.add_argument("--from", dest="src", required=True, help="analysis folder, e.g. out/empty_court")
    sub.add_parser("export-coreml", help="convert TrackNet and YOLO to CoreML (faster on a Mac, once)")
    c = sub.add_parser("check", help="find what went wrong in an analysis (calibration, players, shuttle, hits)")
    c.add_argument("out", help="analysis folder, e.g. out/3fa9c1b2d4")
    a = ap.parse_args()
    if a.cmd == "eval":
        from hud.analysis import evaluate
        print(evaluate.text(evaluate.run(a.out, a.truth)))
        return
    if a.cmd == "stabilize":
        from hud.court import stabilize
        stabilize.run(a.video, a.out, a.ref)
        print(f"Done: {a.out} (open it in hud web and calibrate at {a.ref} s)")
        return
    if a.cmd == "venue-save":
        from hud import court, pipeline
        from hud.court import venue
        _, calib = court.load(pipeline.paths(a.src)["calib"])
        d = venue.save(a.name, pipeline.background(calib["video"]), calib["corners"])
        print(f"Saved venue \"{a.name}\": {d}")
        return
    if a.cmd == "export-coreml":
        from hud.detection import coreml
        coreml.export()
        return
    if a.cmd == "check":
        from hud.analysis import check
        print(check.text(check.run(a.out)))
        print(f"Picture: {Path(a.out) / 'check.png'} (yellow = calibrated court, dots = detected player feet)")
        return
    if a.cmd == "web":
        from hud.web.server import serve
        serve(a.port, a.lan, a.password)
        return
    from hud import pipeline
    out = Path(a.out or Path("out") / Path(a.video).stem)
    if a.venue:
        from hud import court
        from hud.court import venue
        ref, rc = venue.load(a.venue)
        bg = pipeline.background(a.video)
        corners, rep = venue.align(ref, rc, bg)
        if corners is None:
            raise SystemExit(f"Applying the venue failed: {rep.get('error')} ({rep})")
        court.save(pipeline.paths(out)["calib"], a.video, bg.shape[1::-1], corners)
        print(f"Applied venue \"{a.venue}\": {rep}")
    elif not pipeline.paths(out)["calib"].exists():
        calib, frac = pipeline.calibrate(a.video, out)
        if calib is None:
            raise SystemExit("No court found: calibrate by hand in `hud web`")
        print(f"Court auto-detected: {frac:.0%} on the white lines (adjust in `hud web` if it is off)")
    p = pipeline.analyse(a.video, out, progress=lambda x, m: print(f"[{x:4.0%}] {m}", flush=True), start=a.start)
    print(f"Done: {p['hud']}")


if __name__ == "__main__":
    main()
