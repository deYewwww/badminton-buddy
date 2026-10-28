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
    r.add_argument("--venue", help="用存好的場地校正（先用 hud venue-save 存）")
    r.add_argument("--from", dest="start", default="players", choices=["players", "shuttle", "hits", "rallies", "hud"],
                   help="從哪一步開始重跑")
    ev = sub.add_parser("eval", help="和人工標記比對：回合、擊球、比分的準確度")
    ev.add_argument("out", help="分析資料夾")
    ev.add_argument("truth", nargs="?", help="標記檔 truth.json（格式見 hud/analysis/evaluate.py）；不給就用網頁 ⑦ 標的擊球")
    st = sub.add_parser("stabilize", help="把鏡頭有轉動（雲台追蹤）的影片對齊成固定鏡頭")
    st.add_argument("video")
    st.add_argument("--ref", type=float, required=True, help="參考畫面的秒數（在這一幀上校正）")
    st.add_argument("--out", required=True, help="輸出影片，例如 data/穩定.mp4")
    vs = sub.add_parser("venue-save", help="把一個分析資料夾的校正存成場地（建議用空場影片）")
    vs.add_argument("name")
    vs.add_argument("--from", dest="src", required=True, help="分析資料夾，例如 out/空場")
    ft = sub.add_parser("finetune", help="用網頁 ⑥ 標記的羽球微調 TrackNet（比原本準才採用）")
    ft.add_argument("out", help="分析資料夾，例如 out/df4b66dd15")
    ft.add_argument("--epochs", type=int, default=6)
    sub.add_parser("export-coreml", help="把 TrackNet 和 YOLO 轉成 CoreML（Mac 上跑得更快，只要做一次）")
    rp = sub.add_parser("report", help="回歸報告：代理指標、和人工標記比對、抽查清單（report.md / hard_examples.csv）")
    rp.add_argument("out", help="分析資料夾，例如 out/work2")
    c = sub.add_parser("check", help="檢查分析結果哪裡出問題（校正、球員、羽球、擊球）")
    c.add_argument("out", help="分析資料夾，例如 out/3fa9c1b2d4")
    a = ap.parse_args()
    if a.cmd == "eval":
        from hud.analysis import evaluate
        print(evaluate.text(evaluate.run(a.out, a.truth)))
        return
    if a.cmd == "stabilize":
        from hud.court import stabilize
        stabilize.run(a.video, a.out, a.ref)
        print(f"完成：{a.out}（在 hud web 裡打開它，於 {a.ref} 秒校正）")
        return
    if a.cmd == "venue-save":
        from hud import court, pipeline
        from hud.court import venue
        _, calib = court.load(pipeline.paths(a.src)["calib"])
        d = venue.save(a.name, pipeline.background(calib["video"]), calib["corners"])
        print(f"已存成場地「{a.name}」：{d}")
        return
    if a.cmd == "finetune":
        from hud.detection import finetune
        rep = finetune.run(a.out, epochs=a.epochs)
        print(f"保留留出的標記上：微調前 {rep['before']}，微調後 {rep['after']}，"
              + ("已採用（之後重新分析就會用新的 TrackNet；先跑 hud export-coreml 讓它變快）" if rep["kept"] else "沒有比較好，維持原本的 TrackNet"))
        return
    if a.cmd == "export-coreml":
        from hud.detection import coreml
        coreml.export()
        return
    if a.cmd == "report":
        from hud.analysis import report
        report.run(a.out)
        print((Path(a.out) / "report.md").read_text())
        return
    if a.cmd == "check":
        from hud.analysis import check
        print(check.text(check.run(a.out)))
        print(f"圖：{Path(a.out) / 'check.png'}（黃線 = 校正的球場，彩色點 = 偵測到的球員腳位）")
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
            raise SystemExit(f"套用場地失敗：{rep.get('error')}（{rep}）")
        court.save(pipeline.paths(out)["calib"], a.video, bg.shape[1::-1], corners)
        print(f"套用場地「{a.venue}」：{rep}")
    elif not pipeline.paths(out)["calib"].exists():
        calib, frac = pipeline.calibrate(a.video, out)
        if calib is None:
            raise SystemExit("找不到球場，請用 `hud web` 手動校正")
        print(f"自動偵測球場：白線吻合 {frac:.0%}（不準的話用 `hud web` 調整）")
    p = pipeline.analyse(a.video, out, progress=lambda x, m: print(f"[{x:4.0%}] {m}", flush=True), start=a.start)
    print(f"完成：{p['hud']}")


if __name__ == "__main__":
    main()
