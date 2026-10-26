"""Train the frame-level shuttle detector that fills TrackNet's gaps (hud/shuttle_yolo.py).

Run on the Mac (Apple GPU via MPS), once:

    export ROBOFLOW_API_KEY=...          # roboflow.com -> Settings -> API Keys (free account works)
    uv run --with roboflow python scripts/train_shuttle.py

  1. downloads public Roboflow Universe shuttlecock datasets (YOLO format, CC BY 4.0) into
     data/shuttle_ds/<name>/  (~3-4 GB, skipped when already there)
  2. merges them into one single-class dataset (data/shuttle_ds/merged.yaml): every class whose
     name looks like a shuttlecock becomes class 0, rackets/players/nets are dropped
  3. trains YOLO11n from the COCO checkpoint at 640 px and copies the best weights to
     models/shuttle_yolo.pt, which hud/shuttle.py then uses automatically

Time on an M2 8 GB: roughly 10-20 min per epoch for the default datasets; --epochs 15 (the
default) is an evening. --quick (3 epochs, 25 % of the images) checks that everything works in
under an hour. Close other apps: training uses most of the memory.

Why a frame detector at all: TrackNet looks at 3 frames + a background and finds the shuttle
by its motion, which is right for a tiny blurred shuttle but misses some that are plainly
visible (against a banner, near the net). A detector trained on still images sees those.
It only fills TrackNet's gaps, and only where the detection sits on the shuttle's path.
"""
import argparse
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "shuttle_ds"
OUT = ROOT / "models" / "shuttle_yolo.pt"
# (workspace, project, version): checked with the Roboflow MCP on 2026-10-27
DATASETS = [
    ("solely-sports", "my-first-project-i8fww", 2),  # 21k images (3x augmented), its hosted model found a
                                                     # shuttle TrackNet missed on the user's ACE clip
    ("diwenne", "smashspeed", 8),                    # 13k images, shuttlecock only
]
SHUTTLE = re.compile(r"shuttle|cock|volant|caulong|badminton.?ball", re.I)


def download(api_key):
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    out = []
    for ws, proj, ver in DATASETS:
        dest = DS / proj
        if not (dest / "data.yaml").exists():
            print(f"downloading {ws}/{proj} v{ver} ...")
            rf.workspace(ws).project(proj).version(ver).download("yolov8", location=str(dest), overwrite=True)
        out.append(dest)
    return out


def merge(dirs, fraction=1.0):
    """One single-class dataset from several YOLO-format ones (files linked, labels rewritten)."""
    import random

    import yaml
    merged = DS / "merged"
    if merged.exists():
        shutil.rmtree(merged)
    counts = {}
    rnd = random.Random(0)
    for d in dirs:
        names = yaml.safe_load((d / "data.yaml").read_text())["names"]
        names = names if isinstance(names, list) else [names[k] for k in sorted(names)]
        keep = {i for i, n in enumerate(names) if SHUTTLE.search(str(n))}
        if not keep:
            print(f"{d.name}: no shuttle class in {names}, skipped")
            continue
        for split in ("train", "valid", "test"):
            imgs = sorted((d / split / "images").glob("*")) if (d / split / "images").exists() else []
            if split == "train" and fraction < 1:
                imgs = [p for p in imgs if rnd.random() < fraction]
            dst_i, dst_l = merged / split / "images", merged / split / "labels"
            dst_i.mkdir(parents=True, exist_ok=True)
            dst_l.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                lab = d / split / "labels" / (img.stem + ".txt")
                lines = []
                if lab.exists():
                    for ln in lab.read_text().splitlines():
                        parts = ln.split()
                        if not parts or int(parts[0]) not in keep:
                            continue
                        v = [float(x) for x in parts[1:]]
                        if len(v) > 4:  # a segmentation polygon: its bounding box
                            xs, ys = v[0::2], v[1::2]
                            v = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
                        lines.append("0 " + " ".join(f"{x:.6f}" for x in v[:4]))
                name = f"{d.name}_{img.name}"
                target = dst_i / name
                try:
                    os.link(img, target)
                except OSError:
                    shutil.copy2(img, target)
                (dst_l / (Path(name).stem + ".txt")).write_text("\n".join(lines))
                counts[split] = counts.get(split, 0) + 1
    val = "valid" if (merged / "valid" / "images").exists() else "test"
    yml = DS / "merged.yaml"
    yml.write_text(f"path: {merged}\ntrain: train/images\nval: {val}/images\nnames:\n  0: shuttlecock\n")
    print(f"merged dataset: {counts}")
    return yml


def train(yml, epochs, imgsz, batch, fraction):
    import torch
    from ultralytics import YOLO
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("yolo11n.pt")  # COCO weights, downloaded by Ultralytics on first use
    # small objects: no mosaic shrinking in the last epochs, mild augmentation (the datasets are augmented)
    r = model.train(data=str(yml), epochs=epochs, imgsz=imgsz, batch=batch, device=device, fraction=fraction,
                    project=str(DS / "runs"), name="shuttle", exist_ok=True, patience=5, close_mosaic=3,
                    workers=2, cache=False, plots=False)
    best = Path(r.save_dir) / "weights" / "best.pt"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, OUT)
    print(f"Wrote {OUT}. hud/shuttle.py uses it from the next analysis on.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="lower it (8) if the Mac runs out of memory")
    ap.add_argument("--quick", action="store_true", help="3 epochs on 25 %% of the images: a smoke test")
    a = ap.parse_args()
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY first (roboflow.com -> Settings -> API Keys).")
    fraction = 0.25 if a.quick else 1.0
    yml = merge(download(key), fraction)
    train(yml, 3 if a.quick else a.epochs, a.imgsz, a.batch, 1.0)


if __name__ == "__main__":
    main()
