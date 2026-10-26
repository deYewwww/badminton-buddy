# badminton-buddy (羽球 HUD 編輯器)

Fixed-camera doubles video -> game-style HUD video (Chinese UI). Read README.md first.

- Run: `uv run hud web` (editor, port 7860) or `uv run hud run VIDEO`.
- Tests: `for f in tests/test_*.py; do uv run python $f; done` (no videos needed).
- Pipeline: hud/pipeline.py: court (auto) -> players -> shuttle -> hits -> rallies/score -> render.
  Each stage writes out/<match>/... and overwrites on rerun.
- The user writes in Chinese; reply in Traditional Chinese.
- Target machine: 8 GB Apple Silicon Mac. Keep it light: hardware decode (VideoToolbox),
  nano YOLO, no multiprocess rendering. On Linux the code falls back to software ffmpeg/CPU.
- Videos are NOT in the repo (data/, out/ are ignored). To test end-to-end in a cloud
  session, ask the user for a clip or use any fixed-camera doubles video.
- The 136 MB TrackNet checkpoint is stored as models/tracknetv3/TrackNet_best.pt.part*
  (GitHub 100 MB limit); hud/shuttle.py joins them on first use.

## Status / known issues (2026-10-26)
- Auto court detection: correct on the 45° test clip, a 3K MCC hall recording and a broadcast
  rear view. Fails on screen recordings of zoomed/moving playback; the editor's 4 control
  points + top-down mini map are the manual fallback. The 4 points default to the court corners and
  may be dragged into the grey margin around the frame (view grows up to 60% past each edge).
- Per-stage seconds go to out/<match>/timing.json. TrackNet (every frame, ~28 GFLOPs/frame) dominates.
- TrackNet skips dead time (shuttle.idle_frames from players.parquet, [shuttle] in config.toml).
  Not yet validated on real footage: if rallies go missing, check skip_idle first.
- `hud check out/<match>` (also run at the end of analyse): calibration fit + camera plausibility,
  players seen per side, shuttle %, hits. A wrong calibration empties everything downstream
  (far players dropped as off-court -> no hits -> no rallies/score).
- Player marks (editor step 3 -> marks.json: feet clicked at a time): players.run pins the tracker
  slot there (swapping with the partner it was confused with), and apply_marks overrides reid
  from each mark to the next one for that pair.
- Hitter side: assign_sides (Viterbi over an exchange: geo_fit reach + ray miss + wrist vs a
  same-side penalty). Speed: fit_speed, RK4 drag+gravity fitted to the whole flight's pixels;
  fallback = 0.1 s window average converted with from_window_avg. Weights/thresholds were tuned
  on ONE broadcast singles rally (3 versions) -> re-check on real doubles phone footage.
- TrackNet on CoreML when models/tracknetv3/TrackNet.mlpackage exists (`hud export-coreml`, Mac);
  vouched for against PyTorch per video. YOLO CoreML export needs numpy <= 2.3.5
  (`uv run --with numpy==2.3.5 hud export-coreml`; with numpy 2.4 its attention block fails).
- Venues (hud/venue.py, data/venues/): calibrate an empty-court clip once, save; a match from the
  same spot aligns to it (ORB homography -> court.snap -> plausibility). On the user's panning
  portrait clip it aligned 4/4 camera positions (~900 inliers, 0.3 s) where auto-detect got 2/4.
  The same alignment per segment is the way to support panning/cut videos (not built yet).
- Portrait clips: TrackNet gets a 16:9 window (shuttle.window), not the squeezed frame.
- User's own footage (2026-10-26): phone ~0.7 m high, portrait 0.5x, gimbal panning, edited cuts.
- hud stabilize (stabilize.py): panning/cut clips -> one reference view (ORB homography per frame,
  chained fallback, per-segment floor snap); cuts from image jumps + view-direction jumps, used as
  hard boundaries by hits/rallies. Pixel thresholds scale with the LENS short side (native_size),
  not the canvas. hud eval (evaluate.py): truth JSON -> rally/hit/score metrics + label-free checks.
- ACE clip (user, 115 s, 0.7 m portrait, gimbal + cuts), stabilised, 5 rounds: rallies found
  33% -> 67-75%, precision 50% -> 80%, shuttle in play 38% -> 63%, hits/s 0.54 -> 1.04;
  same side twice in a row still 27% (all near-near): side assignment is weak at low height.
  Winners/hits could not be hand-labelled from that footage.
- The analysis assumes ONE fixed camera for the whole video; panning/zooming clips break it.
- Shuttle (TrackNetV3) loses the shuttle against dark, crowded backgrounds -> rallies cut short.
- Oblique test clip: 3/3 rallies found + 2 false ones (pick-ups), hitter side 7/7.
