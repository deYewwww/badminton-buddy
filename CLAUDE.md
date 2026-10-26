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

## Status / known issues (2026-10-27)
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
- Panning/cut clips are stabilised automatically (pipeline.prepare -> stab.mp4 + calib_stab.json,
  marks_stab.json; pipeline.work() says which video/calib the stages use). calib.json records t,
  the frame the corners were set on (= the stabilisation reference). render maps the HUD back onto
  the original frames per frame (stabilize.frame_maps). Zooming still breaks it.
- Player positions: smooth.py (per-detection covariance from the homography Jacobian + RTS).
  Tracker gate/cost use the same covariance. YOLO runs on detect.player_region (crop where the
  players are; ACE: all 4 seen 34% -> 50% of frames).
- Last-shot landing: floor if seen lying still (STILL_PX, ray must point down: a serve's apex
  hangs against the ceiling), else landing.predict (drag+gravity fit carried to z=0, decisive
  only beyond landing_sure_sd). Same side twice within DOUBLE_S: flip one to the other side if
  it was in reach, else drop the weaker turn (ACE same-side 26% -> 14%; noisy broadcast copy
  wrong sides 2 -> 1). Tried and dropped: telling the hitter side by which contact gives the
  better physical flight fit (both fit equally, 67% on the broadcast rally); by wrist speed from
  30 fps pose on per-player crops around each hit (67% on the broadcast rally, ACE same-side
  20 -> 38: a swing spans 1-2 frames, blurred wrists drop out, a 60 px far player's keypoint
  jitter is as big as the swing). Next idea: the racket's sound (near hits ~10 dB louder from
  3 m vs 12 m, sharp onsets). ACE can't test it: its audio track is music (72% of energy < 200 Hz).
  Needs a raw clip with the hall's own sound.
- Stamina: render.stamina_track: effort-based (speed above walking, footwork = acceleration,
  per-stroke cost, attack streaks cost more, defence least), recovers when standing (slowly in a
  rally); ACE ends at 69-89 (user: the old per-metre model fell far too fast).
- HUD display smoothing (user: rings/tags shook): feet 0.3 s, boxes 0.4 s, stabilisation maps
  averaged +-3 frames within a cut, tags stacked by depth + eased + held 1 s when the box drops,
  MOVE/ARRIVED with hysteresis (move_states). Shuttle drawn only on runs of >= 5 real sightings
  (shown_track): the user saw the trail "floating" on InpaintNet guesses and stray detections.
- Score guesses (point_source "guess") are marked 推測 in the HUD and the editor table. A predicted
  landing too close to a line still sets point_lean (the guess); one well short of the net in
  the hitter's own half is a net error (NET_SURE_SD). The user's ACE complaints (net shot given
  to the hitter) were all guesses: the montage cuts most rally endings.
- Manual points (editor "補一分", manual_points.csv: t, point_team): rallies.add_manual inserts
  one-frame pseudo rallies (rally_id 1000+, manual=True, point_source "manual"); score.apply
  orders by start_ts so detected rally ids, and the user's point_fixes keyed on them, stay put.
  The user's run of ACE missed 2 of our points (~85 s, ~97 s) that a run with the reference
  calibration finds; their calib keeps failing camera_plausible (far corners ~50 px high).
- Roboflow/HF/Google Drive downloads are blocked by this environment's network policy (the
  Roboflow MCP can search/fork/train but export links are on app.roboflow.com). GitHub release
  assets (e.g. ultralytics yolo11n.pt) and PyPI do download.
- Frame-level shuttle detector (hud/shuttle_yolo.py, optional): models/shuttle_yolo.pt trained on
  the Mac by scripts/train_shuttle.py (Roboflow Universe datasets, merged to 1 class). Fills
  TrackNet gaps <= 1 s along a quadratic path, and searches for the shuttle lying still after a
  track ends (rejected if a spare already lay there before the shot). A hosted Universe model
  (my-first-project-i8fww/2) found a plainly visible shuttle TrackNet missed (conf 0.45) but not a
  faint one (0.10). Not validated end-to-end yet: needs the trained weights.
- Shuttle (TrackNetV3) loses the shuttle against dark, crowded backgrounds -> rallies cut short.
- Oblique test clip: 3/3 rallies found + 2 false ones (pick-ups), hitter side 7/7.
