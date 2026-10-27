"""Score an analysis against hand labels: rallies, hits and the score.

Truth file (JSON), all times in seconds of the analysed video:
  {"rallies": [{"start": 4.0, "end": 9.5, "winner": "us", "shots": 7}, ...],
   "hits": [{"t": 4.1, "team": "us"}, ...],
   "hit_windows": [[4.0, 9.5], ...]}
- winner / shots may be null when you couldn't tell; those rallies still count for finding.
- hits only need labelling inside hit_windows (labelling every hit of a match is slow):
  detected hits outside them are not counted as false.
Matching: a detected rally matches the truth rally it overlaps most (IoU >= 0.3, each used
once); a detected hit matches an unused truth hit within 0.25 s.

Consistency checks need no winners or hits labelled, only the play stretches (the rallies'
start/end): shuttle found during play vs dead time, hits per second of play (amateur doubles
~0.8-1.2), the same side hitting twice in a row (impossible: a pure error rate).

Run: uv run hud eval out/<match> truth.json   (writes out/<match>/eval.json)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HIT_TOL_S = 0.25
RALLY_IOU = 0.3


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def rallies(pred, truth):
    """pred: rallies.parquet frame (start_ts, end_ts, shot_count, point_team_final)."""
    used, pairs = set(), []
    for k, t in enumerate(truth):
        best, bi = 0.0, None
        for i, p in enumerate(pred.itertuples()):
            if i in used:
                continue
            v = _iou((t["start"], t["end"]), (p.start_ts, p.end_ts))
            if v > best:
                best, bi = v, i
        if bi is not None and best >= RALLY_IOU:
            used.add(bi)
            pairs.append((k, bi))
    out = dict(truth=len(truth), found=len(pairs), extra=len(pred) - len(pairs),
               recall=round(len(pairs) / max(len(truth), 1), 3), precision=round(len(pairs) / max(len(pred), 1), 3))
    if pairs:
        d0 = [abs(truth[k]["start"] - pred.start_ts.iat[i]) for k, i in pairs]
        d1 = [abs(truth[k]["end"] - pred.end_ts.iat[i]) for k, i in pairs]
        out.update(start_err_s=round(float(np.median(d0)), 2), end_err_s=round(float(np.median(d1)), 2))
        sh = [(truth[k]["shots"], int(pred.shot_count.iat[i])) for k, i in pairs if truth[k].get("shots")]
        if sh:
            out.update(shots_abs_err=round(float(np.mean([abs(a - b) for a, b in sh])), 2),
                       shots_truth=sum(a for a, _ in sh), shots_found=sum(b for _, b in sh))
        win = [(truth[k]["winner"], pred.point_team_final.iat[i]) for k, i in pairs if truth[k].get("winner")]
        if win:
            out.update(winner_right=sum(a == b for a, b in win), winner_judged=len(win),
                       winner_acc=round(sum(a == b for a, b in win) / len(win), 3))
    out["pairs"] = [(truth[k]["start"], round(float(pred.start_ts.iat[i]), 2)) for k, i in pairs]
    return out


def hits(pred, truth, windows):
    """pred: hits.parquet frame (timestamp, team)."""
    inside = lambda t: any(a <= t <= b for a, b in windows)
    P = pred[[inside(t) for t in pred.timestamp]]
    used, right, wrong = set(), 0, 0
    for t in truth:
        d = np.abs(P.timestamp.to_numpy() - t["t"])
        order = [j for j in np.argsort(d) if d[j] <= HIT_TOL_S and j not in used]
        if order:
            j = order[0]
            used.add(j)
            right += P.team.iat[j] == t["team"]
            wrong += P.team.iat[j] != t["team"]
    n = len(truth)
    return dict(truth=n, found=right + wrong, right_side=right, wrong_side=wrong, missed=n - right - wrong,
                extra=len(P) - len(used), recall=round((right + wrong) / max(n, 1), 3),
                precision=round((right + wrong) / max(len(P), 1), 3),
                side_acc=round(right / max(right + wrong, 1), 3))


def score(pred, truth):
    """Final score from the labelled winners vs the analysis' (rallies with a known winner)."""
    tw = [t["winner"] for t in truth if t.get("winner")]
    pw = list(pred.point_team_final) if len(pred) else []
    return dict(truth_final=f"{tw.count('us')}:{tw.count('opp')}",
                found_final=f"{pw.count('us')}:{pw.count('opp')}")


def consistency(shuttle, hits, windows):
    if shuttle is None or not len(windows):
        return {}
    t = shuttle.timestamp.to_numpy()
    play = np.zeros(len(t), bool)
    for a, b in windows:
        play |= (t >= a) & (t <= b)
    real = (shuttle.visible & ~shuttle.interpolated & ~shuttle.inpainted).to_numpy() if "inpainted" in shuttle else shuttle.visible.to_numpy()
    secs = float(sum(b - a for a, b in windows))
    inside = hits[[any(a <= x <= b for a, b in windows) for x in hits.timestamp]] if len(hits) else hits
    same = int(((inside.team == inside.team.shift()) & (inside.timestamp.diff() < 2.0)).sum()) if len(inside) > 1 else 0
    return dict(shuttle_in_play=round(float(real[play].mean()), 3) if play.any() else None,
                shuttle_dead_time=round(float(real[~play].mean()), 3) if (~play).any() else None,
                hits_in_play=len(inside), hits_per_s=round(len(inside) / max(secs, 1e-6), 2),
                same_side_in_a_row=same, same_side_rate=round(same / max(len(inside) - 1, 1), 3))


def run(out_dir, truth_path):
    out_dir = Path(out_dir)
    truth = json.loads(Path(truth_path).read_text())
    rp = out_dir / "rallies.parquet"
    pr = pd.read_parquet(rp) if rp.exists() else pd.DataFrame()
    if pr.empty:
        pr = pd.DataFrame(columns=["start_ts", "end_ts", "shot_count", "point_team_final"])
    hp = out_dir / "hits.parquet"
    ph = pd.read_parquet(hp) if hp.exists() else pd.DataFrame(columns=["timestamp", "team"])
    if ph.empty:
        ph = pd.DataFrame(columns=["timestamp", "team"])
    sp = out_dir / "shuttle.parquet"
    windows = [(r["start"], r["end"]) for r in truth.get("rallies", [])]
    rep = dict(rallies=rallies(pr, truth.get("rallies", [])),
               hits=hits(ph, truth.get("hits", []), truth.get("hit_windows", [])),
               score=score(pr, truth.get("rallies", [])),
               consistency=consistency(pd.read_parquet(sp) if sp.exists() else None, ph, windows))
    (out_dir / "eval.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    return rep


def text(rep):
    r, h, s = rep["rallies"], rep["hits"], rep["score"]
    lines = [f"Rallies: labelled {r['truth']}, found {r['found']}, extra {r['extra']} (recall {r['recall']:.0%}, precision {r['precision']:.0%})"]
    if "start_err_s" in r:
        lines.append(f"  median start error {r['start_err_s']} s, end {r['end_err_s']} s")
    if "shots_abs_err" in r:
        lines.append(f"  shots: labelled {r['shots_truth']}, found {r['shots_found']} (mean error {r['shots_abs_err']} per rally)")
    if "winner_acc" in r:
        lines.append(f"  winner: {r['winner_right']}/{r['winner_judged']} right ({r['winner_acc']:.0%})")
    if h["truth"]:
        lines.append(f"Hits (in labelled range): labelled {h['truth']}, found {h['found']}, missed {h['missed']}, extra {h['extra']}; "
                     f"right side {h['right_side']}/{h['found']} ({h['side_acc']:.0%})")
    if s["truth_final"] != "0:0":
        lines.append(f"Score: labelled {s['truth_final']}, found {s['found_final']}")
    c = rep.get("consistency") or {}
    if c:
        lines.append(f"Consistency: shuttle seen in play {c['shuttle_in_play']:.0%} / dead time {c['shuttle_dead_time']:.0%}; "
                     f"{c['hits_per_s']} hits/s in play (amateur doubles ~0.8-1.2); same side twice in a row {c['same_side_in_a_row']} ({c['same_side_rate']:.0%})")
    return "\n".join(lines)
