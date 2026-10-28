"""Fine-tune TrackNet on the user's own footage.

The pretrained TrackNetV3 (broadcast rear views) found the shuttle in 58 % of in-play frames of the
user's 4K iPhone clip: it loses it against the hall's ceiling lights and banner, and locks onto
players' shirts. A few hundred frames labelled in the editor (⑥: click the shuttle, or "no shuttle")
teach it this hall.

  1. todo(): which frames to label: in play (bursts of hits), half where TrackNet saw something
     (to confirm or correct it), half where it saw nothing. Fixed once written.
  2. the editor writes shuttle_labels.csv (frame, x, y, state: shuttle / none / hidden).
  3. run(): every labelled frame becomes one 8-frame TrackNet window with that frame at a random
     place in it; the loss is taken on that frame's heatmap only (the others are unlabelled).
     "none" frames are negatives: they teach it to leave shirts and lights alone. "hidden" (behind
     a player) is left out. add_negatives() adds frames outside play (half where TrackNet "saw"
     something there): the first run had 3 negatives in 300 labels, and the adopted net fired
     outside play in 31 % of frames (pretrained 11.5 %), doubling the hits (159 -> 318). A fixed 20 % of the labels is held out; the fine-tuned weights are kept
     only if they beat the pretrained ones there (models/tracknetv3/TrackNet_ft.pt, used by
     shuttle._load_models when present).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hud.detection import shuttle as S

N_LABELS = 300
HOLDOUT = 0.2
TOL_PX = 4          # at 512x288: a detection this close to the label is a hit (TrackNetV3's tolerance)
RADIUS = 2.5        # target disc radius, 512x288 px (TrackNetV3's heatmap)
FT = S.CKPT_DIR / "TrackNet_ft.pt"


def paths(out_dir):
    o = Path(out_dir)
    # X (the windows, ~4 MB each) in a plain .npy read from disk as needed: loaded whole, 383 samples
    # (1.5 GB) on an 8 GB Mac swapped the training to a standstill at epoch 4
    return dict(todo=o / "label_frames.json", labels=o / "shuttle_labels.csv", samples=o / "ft_samples.npz",
                X=o / "ft_X.npy", report=o / "finetune.json")


def todo(out_dir, n=N_LABELS, seed=0):
    """[{f, t, hint: [x, y] or None}] frames to label, chosen once and kept."""
    p = paths(out_dir)
    if p["todo"].exists():
        return json.loads(p["todo"].read_text())
    sh, play, real = _play(out_dir)
    rng = np.random.default_rng(seed)
    pick = sorted(_spread(np.flatnonzero(play & real), n // 2, rng) + _spread(np.flatnonzero(play & ~real), n // 2, rng))
    out = _entries(sh, real, pick)
    p["todo"].write_text(json.dumps(out))
    return out


def _play(out_dir):
    from hud.pipeline import paths as mp
    from hud.visualization.render import hit_groups
    m = mp(out_dir)
    sh, hits = pd.read_parquet(m["shuttle"]), pd.read_parquet(m["hits"])
    fps = 1 / np.median(np.diff(sh.timestamp))
    play = np.zeros(len(sh), bool)
    for fr in hit_groups(hits, fps):
        play[max(0, int(fr[0] - 0.3 * fps)):min(len(sh), int(fr[-1] + 1.5 * fps))] = True
    return sh, play, (sh.visible & ~sh.interpolated & ~sh.inpainted).to_numpy()


def _spread(pool, n, rng, gap=5):
    """Up to n frames of pool, no two within gap frames (near-duplicates teach nothing)."""
    got = []
    for f in rng.permutation(pool):
        if all(abs(f - g) > gap for g in got):
            got.append(int(f))
        if len(got) == n:
            break
    return got


def _entries(sh, real, pick, **extra):
    return [dict(f=f, t=round(float(sh.timestamp.iat[f]), 3), **extra,
                 hint=[round(float(sh.x_px.iat[f]), 1), round(float(sh.y_px.iat[f]), 1)] if real[f] else None)
            for f in pick]


def add_negatives(out_dir, n=150, seed=1):
    """Append n frames from outside play to the frames to label (once): half where TrackNet saw
    something (its usual false alarms: shirts, lights, a shuttle carried about), half anywhere.
    Most are answered with N in the editor. -> how many were added."""
    p = paths(out_dir)
    cur = todo(out_dir)
    if any(e.get("neg") for e in cur):
        return 0
    sh, play, real = _play(out_dir)
    have = {e["f"] for e in cur}
    rng = np.random.default_rng(seed)
    free = lambda m: np.array([f for f in np.flatnonzero(m) if f not in have], int)
    pick = sorted(_spread(free(~play & real), n // 2, rng) + _spread(free(~play & ~real), n - n // 2, rng))
    cur += _entries(sh, real, pick, neg=True)
    p["todo"].write_text(json.dumps(cur))
    return len(pick)


def labels(out_dir):
    f = paths(out_dir)["labels"]
    return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["frame", "x", "y", "state"])


def save_label(out_dir, frame, x, y, state):
    """state: shuttle (x, y in frame pixels) / none / hidden / remove."""
    df = labels(out_dir)
    df = df[df.frame != int(frame)]
    if state in ("shuttle", "none", "hidden"):
        row = dict(frame=int(frame), x=round(float(x), 1) if state == "shuttle" else np.nan,
                   y=round(float(y), 1) if state == "shuttle" else np.nan, state=state)
        df = pd.concat([df, pd.DataFrame([row])])
    df.sort_values("frame").to_csv(paths(out_dir)["labels"], index=False)
    return len(df)


def build(out_dir, seed=0):
    """Labelled frames -> 8-frame windows (main view, plus the far-court view where it applies),
    heatmap targets and holdout flags. Saved to ft_samples.npz (~5 MB per 100 samples)."""
    from hud.pipeline import paths as mp
    from hud.media.video import probe
    m = mp(out_dir)
    calib = json.loads(m["calib"].read_text())
    video = calib["video"]
    fw, fh, _, n = probe(video)
    lab = labels(out_dir)
    lab = lab[lab.state.isin(["shuttle", "none"])].reset_index(drop=True)
    if len(lab) < 30:
        raise SystemExit(f"Only {len(lab)} usable labels: label at least 30 frames (300 is the aim) in the editor's ⑥.")
    wins = [None]
    try:
        from hud.court.camera import Camera
        from hud.court import load
        H, c = load(m["calib"])
        cam = Camera(H, c["frame_size"])
        main = S.window(fw, fh, cam)
        far = S.far_window(cam, fw, fh, main)
        wins = [main] + ([far] if far is not None else [])
    except (ValueError, KeyError):
        pass
    medians = [np.moveaxis(S._median(video, n, w, fw), -1, 0) for w in wins]
    rng = np.random.default_rng(seed)
    hold = rng.random(len(lab)) < HOLDOUT
    X, Y, K, HO, LBL = [], [], [], [], []
    cap = cv2.VideoCapture(str(video))
    for i, r in lab.iterrows():
        k = int(rng.integers(0, 8))  # the labelled frame's place in the window
        f0 = int(r.frame) - k
        if f0 < 0 or f0 + 8 > n:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        frames = []
        for _ in range(8):
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if len(frames) < 8:
            continue
        for v, (win, med) in enumerate(zip(wins, medians)):
            x0, y0, ww, hh = (0, 0, fw, fh) if win is None else win
            if r.state == "shuttle":
                q = ((r.x - x0) / ww * S.W, (r.y - y0) / hh * S.H)
                if not (0 <= q[0] < S.W and 0 <= q[1] < S.H):
                    continue  # the shuttle is outside this view
            elif v > 0:
                continue  # negatives: the main view is enough
            else:
                q = None
            crop = [_cut(fr, win, fw, fh) for fr in frames]
            X.append(np.concatenate([med] + [np.moveaxis(c, -1, 0) for c in crop]).astype(np.uint8))
            Y.append(q if q is not None else (np.nan, np.nan))
            K.append(k)
            HO.append(hold[i])
            LBL.append(i)
    cap.release()
    out = paths(out_dir)["samples"]
    np.save(paths(out_dir)["X"], np.stack(X))
    np.savez(out, Y=np.array(Y, np.float32), K=np.array(K), hold=np.array(HO), label=np.array(LBL))
    print(f"Built {len(X)} samples from {len(lab)} labels ({int(np.sum(HO))} held out) -> {out}")
    return out


def _cut(fr, win, fw, fh):
    if win is None:
        return cv2.resize(fr, (S.W, S.H), interpolation=cv2.INTER_AREA)
    x0, y0, w, h = win
    c = fr[max(y0, 0):min(y0 + h, fh), max(x0, 0):min(x0 + w, fw)]
    c = cv2.copyMakeBorder(c, max(-y0, 0), max(y0 + h - fh, 0), max(-x0, 0), max(x0 + w - fw, 0), cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(c, (S.W, S.H), interpolation=cv2.INTER_AREA)


def _target(q):
    t = np.zeros((S.H, S.W), np.float32)
    if q is not None and np.isfinite(q[0]):
        cv2.circle(t, (int(round(q[0])), int(round(q[1]))), int(np.ceil(RADIUS)), 1.0, -1)
    return t


def score(pred, Y, K):
    """pred (N, 8, H, W) heatmaps -> dict(tp, fn, fp, tn, recall, precision) on the labelled frames."""
    tp = fn = fp = tn = fp_empty = 0
    for hm, q, k in zip(pred, Y, K):
        c = S._locate((hm[k] > 0.5).astype(np.uint8))
        if np.isfinite(q[0]):
            if c is not None and np.hypot(c[0] - q[0], c[1] - q[1]) <= TOL_PX:
                tp += 1
            else:
                fn += 1
                fp += c is not None
        else:
            fp += c is not None
            fp_empty += c is not None
            tn += c is None
    # fp_empty: fired on a frame with no shuttle (the failure that doubled the hits)
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, fp_empty=fp_empty, recall=round(tp / max(tp + fn, 1), 3),
                precision=round(tp / max(tp + fp, 1), 3))


def _predict(net, X, idx, device, bs=4):
    """Heatmaps for samples idx of X (a memory-mapped array: only bs windows in memory at a time)."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            x = torch.from_numpy(np.ascontiguousarray(X[idx[i:i + bs]])).to(device).float().div(255)
            out.append(net(x).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def _samples(p):
    """-> X (memory-mapped), Y, K, hold. Converts an older all-in-one .npz once."""
    d = np.load(p["samples"])
    if "X" in d.files and not p["X"].exists():
        np.save(p["X"], d["X"])
        np.savez(p["samples"], **{k: d[k] for k in d.files if k != "X"})
        d = np.load(p["samples"])
    return np.load(p["X"], mmap_mode="r"), d["Y"], d["K"], d["hold"]


def run(out_dir, epochs=6, lr=1e-4, bs=2, seed=0):
    """Build the samples (if needed), fine-tune, compare with the pretrained net on the held-out
    labels and keep the best epoch's weights, only if they do better. -> report dict (also
    finetune.json). On the user's 203 labels most of the gain came in epochs 1-3 (recall 0.34 ->
    0.54); later epochs risk learning the labels by heart."""
    import torch
    p = paths(out_dir)
    if not p["samples"].exists() or p["labels"].stat().st_mtime > p["samples"].stat().st_mtime:
        build(out_dir, seed)  # new labels since the samples were built
    X, Y, K, hold = _samples(p)
    ho = np.flatnonzero(hold)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    (net, L), _ = S._load_models(device, finetuned=False)
    before = score(_predict(net.eval(), X, ho, device), Y[ho], K[ho])
    best, best_w = before, None
    print(f"Pretrained on the held-out labels: {before}")
    tr = np.flatnonzero(~hold)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    for ep in range(epochs):
        net.train()
        for mod in net.modules():  # BatchNorm statistics stay as pretrained: 2 samples a batch would wreck them
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()
        tot = 0.0
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // bs)):
            b = np.sort(b)  # memory-mapped reads in file order
            x = torch.from_numpy(np.ascontiguousarray(X[b])).to(device).float().div(255)
            tgt = torch.from_numpy(np.stack([_target(Y[i]) for i in b])).to(device)
            if rng.random() < 0.5:  # mirror: the hall is not symmetric, the flight is
                x, tgt = x.flip(-1), tgt.flip(-1)
            y = net(x)[torch.arange(len(b)), torch.from_numpy(K[b]).to(device)].clamp(1e-6, 1 - 1e-6)
            # TrackNetV3's weighted BCE: confident mistakes cost most
            loss = -((1 - y) ** 2 * tgt * torch.log(y) + y ** 2 * (1 - tgt) * torch.log(1 - y)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        net.eval()
        now = score(_predict(net, X, ho, device), Y[ho], K[ho])
        good = (now["recall"] > best["recall"] and now["precision"] >= before["precision"] - 0.02
                and now["fp_empty"] <= before["fp_empty"])  # never more false alarms on empty frames
        if good:  # the best epoch so far, kept on the CPU
            best, best_w = now, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"epoch {ep + 1}/{epochs}: loss {tot / len(tr):.5f}  held-out {now}" + ("  <- best so far" if good else ""))
    after, better = best, best_w is not None
    rep = dict(before=before, after=after, kept=bool(better), samples=int(len(X)), held_out=int(hold.sum()), epochs=epochs)
    if better:
        ck = torch.load(S._joined(S.CKPT_DIR / "TrackNet_best.pt"), map_location="cpu", weights_only=False)
        torch.save(dict(model=best_w, param_dict=ck["param_dict"],
                        finetuned_on=str(out_dir), report=rep), FT)
        print(f"Kept: {FT}")
    else:
        print("Not better on the held-out labels: the pretrained TrackNet stays in use.")
    p["report"].write_text(json.dumps(rep, indent=1))
    return rep
