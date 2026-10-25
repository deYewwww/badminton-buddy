"""CLI: `hud web` (the editor) or `hud run VIDEO` (whole pipeline, auto court detection)."""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(prog="hud", description="羽球 HUD 影片產生器")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("web", help="打開網頁編輯器")
    w.add_argument("--port", type=int, default=7860)
    w.add_argument("--lan", action="store_true", help="讓同網路的手機也能連（需要 --password）")
    w.add_argument("--password")
    r = sub.add_parser("run", help="命令列跑完整流程")
    r.add_argument("video")
    r.add_argument("--out", help="輸出資料夾（預設 out/<影片檔名>）")
    r.add_argument("--from", dest="start", default="players", choices=["players", "shuttle", "hits", "rallies", "hud"],
                   help="從哪一步開始重跑")
    a = ap.parse_args()
    if a.cmd == "web":
        from hud.server import serve
        serve(a.port, a.lan, a.password)
        return
    from hud import pipeline
    out = Path(a.out or Path("out") / Path(a.video).stem)
    if not pipeline.paths(out)["calib"].exists():
        calib, frac = pipeline.calibrate(a.video, out)
        if calib is None:
            raise SystemExit("找不到球場，請用 `hud web` 手動校正")
        print(f"自動偵測球場：白線吻合 {frac:.0%}（不準的話用 `hud web` 調整）")
    p = pipeline.analyse(a.video, out, progress=lambda x, m: print(f"[{x:4.0%}] {m}", flush=True), start=a.start)
    print(f"完成：{p['hud']}")


if __name__ == "__main__":
    main()
