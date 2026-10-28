"""Regression report after a run (docs/speed_audit.md, step 4): machine-readable report.json, a
human-readable report.md and hard_examples.csv, a short list of moments worth a look.

    uv run hud report out/<match>

Two kinds of numbers, never mixed:
  proxy   computed from the analysis alone (no labels): how often the shuttle is seen in play,
          how the track breaks and jumps, how many shots get a speed and why not. A proxy can
          improve while the real accuracy does not.
  checked against the user's own marks (editor ⑦ hit marks; ⑧ verdicts on hard examples). The
          marks carry their own error: two passes over the same 16 rallies agreed on 54 % of hit
          times within 0.25 s.
Each run keeps the previous report.json as report_prev.json and lists what changed.
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

JUMP = 0.1          # a step over this share of the short side in one frame between sightings: a jump
LONG_GAP_S = 0.5    # a break in the track this long during play is listed as a hard example
PER_KIND = 8        # hard examples per kind at most (spread over the match)
RANDOM_N = 10       # plus this many random moments of play: the failures are not all we look at


def _play(rallies, F, fps):
    play = np.zeros(F, bool)
    for r in rallies.itertuples():
        play[int(r.start_frame):min(int(r.end_frame) + 1, F)] = True
    return play


def track(sh, play, fps, short):
    """Proxy track quality inside play: seen share, breaks (count, length), jumps."""
    real = (sh.visible & ~sh.interpolated & ~sh.inpainted).to_numpy()[:len(play)]
    t = sh.timestamp.to_numpy(float)[:len(play)]
    play_s = play.sum() / fps
    gaps, f = [], 0
    while f < len(play):  # breaks: unseen runs inside play
        if play[f] and not real[f]:
            g = f
            while g < len(play) and play[g] and not real[g]:
                g += 1
            gaps.append((f, g))
            f = g
        else:
            f += 1
    glen = np.array([(t[min(g, len(t) - 1)] - t[f]) for f, g in gaps]) if gaps else np.zeros(0)
    idx = np.flatnonzero(real & play)
    P = sh[["x_px", "y_px"]].to_numpy(float)
    step = np.hypot(*(P[idx[1:]] - P[idx[:-1]]).T) / np.maximum(idx[1:] - idx[:-1], 1)
    jumps = idx[1:][step > JUMP * short]
    return dict(play_s=round(play_s, 1), seen_in_play=round(float(real[play].mean()) if play.any() else 0.0, 3),
                breaks=len(gaps), breaks_per_min=round(len(gaps) / max(play_s / 60, 1e-9), 1),
                break_median_s=round(float(np.median(glen)), 2) if len(glen) else 0.0,
                break_p90_s=round(float(np.quantile(glen, 0.9)), 2) if len(glen) else 0.0,
                jumps=len(jumps), jumps_per_min=round(len(jumps) / max(play_s / 60, 1e-9), 1)), gaps, glen, jumps


def speeds(shots):
    """Proxy: how many shots get a speed, in which state, and why not."""
    if "speed_status" not in shots:
        return {}
    n = len(shots)
    st = shots.speed_status.value_counts().to_dict()
    est = shots[shots.speed_status == "ESTIMATED"]
    reasons = shots[shots.speed_status != "ESTIMATED"].speed_reason.fillna("—").value_counts().to_dict()
    garbage = shots[(shots.fit_failure == "reprojection_error") & (shots.fit_rms_px >= 20)] if "fit_rms_px" in shots else shots[:0]
    return dict(shots=n, **{s: int(st.get(s, 0)) for s in ("ESTIMATED", "ESTIMATED_LOW", "UNAVAILABLE")},
                with_number=round((n - st.get("UNAVAILABLE", 0)) / max(n, 1), 3), not_estimated_because=reasons,
                estimated_rel_sd_median=round(float(est.fit_rel_sd.median()), 3) if len(est) else None,
                track_not_one_flight=len(garbage))


def hard_examples(sh, hits, shots, rallies, gaps, glen, jumps, fps, seed=0):
    """(frame, t, kind, detail): moments to look at, a few per kind, spread over the match, plus
    random moments of play (fixed seed: the same list for the same results)."""
    t = sh.timestamp.to_numpy(float)
    rows = []

    def spread(items, n=PER_KIND):  # items: (frame, detail) sorted by frame -> n evenly spaced
        if len(items) <= n:
            return items
        return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]
    rows += [(f, "跳點", "追蹤一格內跳了很遠：是不是跳到別的東西（衣服、隔壁場的球）？") for f, _ in spread([(int(f), 0) for f in jumps])]
    long = [(int(f), round(float(s), 2)) for (f, g), s in zip(gaps, glen) if s >= LONG_GAP_S]
    rows += [(f, "斷軌", f"對打中 {s} 秒沒看到球：球被擋住、還是追蹤漏掉了？") for f, s in spread(long)]
    if "fit_failure" in hits:
        bad = hits[(hits.fit_failure == "reprojection_error") & (hits.fit_rms_px >= 20)].sort_values("frame")
        rows += [(f, "軌跡不像一次飛行", f"這拍之後的軌跡對不上物理飛行（殘差 {r:.0f} px）：中間混進了別的點嗎？")
                 for f, r in spread(list(zip(bad.frame.astype(int), bad.fit_rms_px)))]
    if "speed_reason" in shots:
        odd = shots[shots.speed_reason == "out_of_range"].sort_values("frame")
        rows += [(f, "球速不合理", "算出的球速超出 15–250 km/h 被丟掉：這拍是真的擊球嗎？") for f in spread(list(odd.frame.astype(int)))]
    if len(hits):
        h = hits.sort_values("frame")
        same = h[(h.team == h.team.shift()) & (h.timestamp - h.timestamp.shift() < 2.0)]
        rows += [(f, "同邊連打", "同一邊連續擊球兩次（不可能）：漏了一拍，還是有一拍是假的？") for f in spread(list(same.frame.astype(int)))]
    rng = np.random.default_rng(seed)
    inside = [f for r in rallies.itertuples() for f in range(int(r.start_frame), int(r.end_frame) + 1)]
    if inside:
        pick = sorted(rng.choice(inside, size=min(RANDOM_N, len(inside)), replace=False))
        rows += [(int(f), "隨機抽查", "隨機一段對打：追蹤、拍數、球速看起來對嗎？") for f in pick]
    rows = sorted(set(rows))
    return pd.DataFrame([dict(frame=f, t=round(float(t[min(f, len(t) - 1)]), 2), kind=k, ask=a) for f, k, a in rows])


def verdicts(out_dir):
    """The user's ⑧ verdicts (hard_review.csv: t, kind, ok) -> share judged right per kind."""
    f = Path(out_dir) / "hard_review.csv"
    if not f.exists():
        return {}
    v = pd.read_csv(f)
    return {k: dict(judged=len(g), ok=int(g.ok.sum())) for k, g in v.groupby("kind")}


def _flat(d, pre=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[pre + k] = v
    return out


def run(out_dir, seed=0):
    from hud import config, pipeline
    from hud.analysis import evaluate
    from hud.court.stabilize import native_size
    out = Path(out_dir)
    p = pipeline.paths(out)
    sh = pd.read_parquet(p["shuttle"])
    hits = pd.read_parquet(p["hits"]) if p["hits"].exists() else pd.DataFrame()
    shots = pd.read_parquet(p["shots"]) if p["shots"].exists() else pd.DataFrame()
    rallies = pd.read_parquet(p["rallies"]) if p["rallies"].exists() else pd.DataFrame()
    fps = 1 / np.median(np.diff(sh.timestamp))
    calib = json.loads(p["calib"].read_text())
    short = min(native_size(calib["video"]) or calib["frame_size"])
    play = _play(rallies, len(sh), fps)
    tr, gaps, glen, jumps = track(sh, play, fps, short)
    check = json.loads(p["check"].read_text()) if p["check"].exists() else {}
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    rep = dict(made=time.strftime("%Y-%m-%d %H:%M"), code=git, config=config.DEFAULT.name,
               video={k: check.get("video", {}).get(k) for k in ("sha256", "fps_measured", "frames", "discarded", "warnings")},
               timing_s=json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {},
               proxy=dict(track=tr, speed=speeds(shots), rallies=len(rallies), hits=len(hits)))
    if (out / "hit_marks.csv").exists():
        ev = evaluate.run(out)
        rep["checked"] = dict(source="hit_marks.csv（人工標記；兩次標記同 16 回合，擊球時間 0.25 s 內一致 54 %）",
                              rallies=ev["rallies"], hits={k: v for k, v in ev["hits"].items() if "side" not in k},
                              marking_lag_s=ev["marking_lag_s"])
    hv = verdicts(out)
    if hv:
        rep.setdefault("checked", {})["hard_review"] = hv
    hx = hard_examples(sh, hits, shots, rallies, gaps, glen, jumps, fps, seed)
    hx.to_csv(out / "hard_examples.csv", index=False)
    rep["hard_examples"] = hx.kind.value_counts().to_dict()
    prev = out / "report.json"
    if prev.exists():
        old = json.loads(prev.read_text())
        a, b = _flat(old.get("proxy")) | _flat(old.get("checked"), "checked."), _flat(rep["proxy"]) | _flat(rep.get("checked"), "checked.")
        rep["changed"] = {k: [a[k], b[k]] for k in sorted(set(a) & set(b)) if a[k] != b[k]}
        rep["previous"] = dict(made=old.get("made"), code=old.get("code"))
        prev.replace(out / "report_prev.json")
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    (out / "report.md").write_text(markdown(rep, hx))
    return rep


def markdown(rep, hx):
    tr, sp = rep["proxy"]["track"], rep["proxy"]["speed"]
    v = rep["video"]
    L = [f"# 分析報告 {rep['made']}", "",
         f"程式 `{rep['code']}`｜設定 `{rep['config']}`｜影片 sha256 `{(v.get('sha256') or '')[:12]}`、"
         f"{v.get('frames')} 格、實測 {v.get('fps_measured')} fps、丟棄封包 {v.get('discarded')}", ""]
    for w in v.get("warnings") or []:
        L.append(f"- ⚠ {w}")
    L += ["", "## 代理指標（沒有標準答案，只說明程式自己的狀態）", "",
          f"- 對打 {tr['play_s']} 秒，看到球的格 {tr['seen_in_play']:.0%}",
          f"- 斷軌 {tr['breaks']} 次（每分鐘 {tr['breaks_per_min']}），長度中位 {tr['break_median_s']} 秒、P90 {tr['break_p90_s']} 秒",
          f"- 跳點 {tr['jumps']} 次（每分鐘 {tr['jumps_per_min']}）"]
    if sp:
        L += [f"- 球速：{sp['shots']} 拍中，估 {sp['ESTIMATED']}、約 {sp['ESTIMATED_LOW']}、不可用 {sp['UNAVAILABLE']}"
              f"（有數字 {sp['with_number']:.0%}）；估的相對標準誤中位 {sp['estimated_rel_sd_median']}",
              f"- 不是「估」的原因：" + "、".join(f"{k} {n}" for k, n in sp["not_estimated_because"].items()),
              f"- 軌跡不像一次飛行（殘差 ≥ 20 px）：{sp['track_not_one_flight']} 拍",
              "- **球速從來沒有和獨立測速比對過**：「估」只代表擬合收斂、標準誤小，不代表準。"]
    c = rep.get("checked")
    if c:
        L += ["", "## 和人工標記比對（真的驗證，但標記本身也有誤差）", ""]
        if "rallies" in c:
            r, h = c["rallies"], c["hits"]
            L += [f"- 來源：{c['source']}；標記延遲 {c['marking_lag_s']:+.2f} 秒已扣掉",
                  f"- 回合：標記 {r['truth']}、找到 {r['found']}、多出 {r['extra']}；起點誤差 {r.get('start_err_s')} 秒、終點 {r.get('end_err_s')} 秒；"
                  f"每回合拍數差 {r.get('shots_abs_err')}",
                  f"- 擊球（0.25 秒內）：標記 {h['truth']}、對上 {h['found']}、漏 {h['missed']}、多 {h['extra']}"]
        for k, g in (c.get("hard_review") or {}).items():
            L.append(f"- ⑧ 抽查「{k}」：看了 {g['judged']} 個，{g['ok']} 個沒問題")
    if rep.get("changed"):
        L += ["", f"## 和上一次比（{rep['previous']['made']}，程式 `{rep['previous']['code']}`）", ""]
        L += [f"- {k}: {a} → {b}" for k, (a, b) in rep["changed"].items()]
    L += ["", f"## 抽查清單（hard_examples.csv，共 {len(hx)} 個；在編輯器 ⑧ 逐一看）", ""]
    L += [f"- {k}：{n}" for k, n in rep["hard_examples"].items()]
    if rep.get("timing_s"):
        L += ["", "## 時間", "", "、".join(f"{k} {s} 秒" for k, s in rep["timing_s"].items())]
    return "\n".join(L) + "\n"
