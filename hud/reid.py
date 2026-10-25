"""Keep each partner's ID across rallies.

The tracker follows people by motion, so IDs survive running and crossing inside a rally.
But between points partners swap sides, stand close and hide each other, and motion alone
can then hand an ID to the wrong person (seen on the oblique clip: P1/P2 swapped at a
side-switch). Here each team's track is cut at every such ambiguous moment into chunks;
inside a chunk the two IDs are trustworthy. Each chunk is then labelled by appearance:
standing height (camera model: head pixel straight above the feet) and upper/lower-body
colour, against two per-person prototypes learnt from the clip itself (2-means over chunks).
If the partners don't look different enough (same kit, same build) nothing is changed.
"""
import numpy as np
import pandas as pd

from hud.players import APP_COLS

PAIRS = (("P1", "P2"), ("P3", "P4"))
CLOSE_M = 1.0      # partners closer than this: IDs may have swapped -> chunk boundary
MIN_FRAMES = 15    # chunk needs this many frames with both partners seen to be judged
HEIGHT_W = 2.0     # weight of standing height vs each colour channel (all standardised)
MIN_SEP = 1.0      # prototypes closer than this: partners look alike, leave the team alone
MIN_MARGIN = 1.5   # a chunk is swapped only if swapping fits this much better. Oblique clip: the real
                   # swap fitted by 4.2; an unverifiable one between near-identical far players by 0.8


def _features(df, cam):
    height = [cam.near_vertical(((r.box_x1 + r.box_x2) / 2, r.box_y1), (r.x_m, r.y_m))[2]
              if r.observed and np.isfinite(r.box_y1) else np.nan for r in df.itertuples()]
    f = df[APP_COLS].astype(float).copy()
    f.insert(0, "height", height)
    return f


def _chunk_votes(fa, fb, ok):
    """Per chunk of ok frames: (frames, median features of each partner)."""
    chunk = (~ok).cumsum()
    out = []
    for _, fr in ok[ok].groupby(chunk[ok]):
        fr = fr.index
        if len(fr) >= MIN_FRAMES:
            out.append((fr, fa.loc[fr].median().to_numpy(), fb.loc[fr].median().to_numpy()))
    return out


def fix_identities(df, cam):
    if not all(c in df for c in APP_COLS) or df.empty:
        return df
    df = df.copy()
    feats = _features(df, cam)
    for a, b in PAIRS:
        team = df.player_id.isin([a, b])
        f = feats[team]
        scale = f.std().replace(0, 1).to_numpy() / np.r_[HEIGHT_W, np.ones(len(APP_COLS))]
        fa = f[df.player_id[team] == a].set_axis(df.frame[team & (df.player_id == a)]) / scale
        fb = f[df.player_id[team] == b].set_axis(df.frame[team & (df.player_id == b)]) / scale
        A, B = df[df.player_id == a].set_index("frame"), df[df.player_id == b].set_index("frame")
        frames = A.index.intersection(B.index)
        if len(frames) == 0:
            continue
        both = A.loc[frames, "observed"] & B.loc[frames, "observed"]
        apart = np.hypot(A.loc[frames, "x_m"] - B.loc[frames, "x_m"], A.loc[frames, "y_m"] - B.loc[frames, "y_m"]) > CLOSE_M
        ok = both & apart & fa.loc[frames].notna().all(axis=1) & fb.loc[frames].notna().all(axis=1)
        votes = _chunk_votes(fa, fb, ok)
        if len(votes) < 2:
            continue
        # 2-means over chunks: prototypes start from the longest chunk
        _, pa, pb = max(votes, key=lambda v: len(v[0]))
        for _ in range(5):
            dec = [np.linalg.norm(ma - pb) + np.linalg.norm(mb - pa) + MIN_MARGIN
                   < np.linalg.norm(ma - pa) + np.linalg.norm(mb - pb) for _, ma, mb in votes]
            w = np.array([len(v[0]) for v in votes], float)
            pa = np.average([mb if s else ma for s, (_, ma, mb) in zip(dec, votes)], axis=0, weights=w)
            pb = np.average([ma if s else mb for s, (_, ma, mb) in zip(dec, votes)], axis=0, weights=w)
        if np.linalg.norm(pa - pb) < MIN_SEP or not any(dec):
            continue
        # per-frame decision: judged chunks, ambiguous stretches take the next chunk's verdict
        swap = pd.Series(np.nan, index=sorted(set(A.index) | set(B.index)))
        for (fr, _, _), s in zip(votes, dec):
            swap.loc[fr] = float(s)
        swap = swap.bfill().ffill().astype(bool)
        flip = df.frame.map(swap).fillna(False).astype(bool) & team
        df.loc[flip, "player_id"] = df.loc[flip, "player_id"].map({a: b, b: a})
        gain = [np.linalg.norm(ma - pa) + np.linalg.norm(mb - pb) - np.linalg.norm(ma - pb) - np.linalg.norm(mb - pa)
                for _, ma, mb in votes]  # > 0: swapping fits better
        spans = [(int(fr[0]), int(fr[-1]), round(float(gn), 2)) for (fr, _, _), s, gn in zip(votes, dec, gain) if s]
        print(f"Re-identified {a}/{b}: swapped {int(sum(dec))} of {len(dec)} stretches {spans} "
              f"(prototype separation {np.linalg.norm(pa - pb):.1f})")
    return df
