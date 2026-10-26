"""Re-run hits + rallies on saved analyses and score them (broadcast rally as truth for its
degraded copies; ACE against its play-stretch labels)."""
import sys, io, contextlib
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import pandas as pd
from hud import hits, rallies, pipeline, evaluate
from hud.evaluate import hits as ev
from hud.stabilize import cuts_of
def redo(o, video=None):
    p = pipeline.paths(o); wv, wc, _ = pipeline.work(o, video or "x.mp4")
    with contextlib.redirect_stdout(io.StringIO()):
        hits.run(p["shuttle"], p["players"], wc, p["hits"])
        rallies.run(p["hits"], p["players"], p["shots"], p["rallies"], shuttle_path=p["shuttle"], cuts=cuts_of(wv))
    return pd.read_parquet(p["hits"]), pd.read_parquet(p["rallies"])
T, _ = redo("out/reg_final_broadcast_singles")
truth = [dict(t=float(a), team=b) for a, b in zip(T.timestamp, T.team)]
print("broadcast hits", len(T), "same-side", int((T.team == T.team.shift()).sum()))
for d in ("final_r2_lowq", "final_r3_cropped"):
    h, r = redo("out/reg_" + d); e = ev(h, truth, [(0, 100)])
    print(d, f"right {e['right_side']} wrong {e['wrong_side']} missed {e['missed']} extra {e['extra']}")
h, r = redo("out/ace_v3", "data/user/v60070.mp4")
print("ACE", evaluate.text(evaluate.run("out/ace_v3", "docs/ace_truth.json")))
print("ACE landing kinds", h.landing_kind.value_counts().to_dict(), "point sources", r.point_source.value_counts().to_dict())
