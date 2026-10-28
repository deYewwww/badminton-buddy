# badminton-buddy (羽球 HUD 編輯器)

Fixed-camera doubles video -> game-style HUD video (Chinese UI). Read README.md first, then
docs/HANDOFF.md (where the last cloud session left off and what is next).

- Run: `uv run hud web` (editor, port 7860) or `uv run hud run VIDEO`.
- Tests: `for f in tests/test_*.py; do uv run python $f; done` (no videos needed).
- Pipeline: hud/pipeline.py: court (auto) -> players -> shuttle -> hits -> rallies/score -> render.
  Each stage writes out/<match>/... and overwrites on rerun.
- Layout (Good-Badminton style packages): hud/media (video decode), hud/court (geometry in
  __init__, camera, venue, stabilize), hud/detection (detect = players' YOLO pose, shuttle =
  TrackNet, shuttle_yolo, coreml, tracknet/), hud/tracking (players, reid, smooth), hud/analysis
  (hits, landing, rallies, score, check, evaluate), hud/visualization (render), hud/web (server,
  static/). `from hud import court/config/pipeline`; everything else `from hud.<pkg> import x`.
- The user writes in Chinese; reply in Traditional Chinese.
- Target machine: 8 GB Apple Silicon Mac. Keep it light: hardware decode (VideoToolbox),
  nano YOLO, no multiprocess rendering. On Linux the code falls back to software ffmpeg/CPU.
- Videos are NOT in the repo (data/, out/ are ignored). To test end-to-end in a cloud
  session, ask the user for a clip or use any fixed-camera doubles video.
- The 136 MB TrackNet checkpoint is stored as models/tracknetv3/TrackNet_best.pt.part*
  (GitHub 100 MB limit); hud/detection/shuttle.py joins them on first use.

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
- Venues (hud/court/venue.py, data/venues/): calibrate an empty-court clip once, save; a match from the
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
  Winners: the user's own corrections, in docs/ace_truth.json (data/ is not in git).
- Panning/cut clips are stabilised automatically (pipeline.prepare -> stab.mp4 + calib_stab.json,
  marks_stab.json; pipeline.work() says which video/calib the stages use). calib.json records t,
  the frame the corners were set on (= the stabilisation reference). render maps the HUD back onto
  the original frames per frame (stabilize.frame_maps). Zooming still breaks it.
- Stabilisation speed (user: a 5-min clip sat on 穩定畫面 for ages, no progress shown): frames
  are stabilised at <= MAX_SIDE 1920 (stab.json "scale"; its H maps ORIGINAL px -> stabilised
  px, native_size returns the lens in stabilised px), decoded/scaled by ffmpeg (VideoToolbox),
  written with h264_videotoolbox on the Mac, and report % for both passes. ORB features on a
  FEAT_SIDE 640 image whatever the size (on half of a 1920 frame RANSAC took wrong homographies
  before a cut).
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
  Then still floating at 1:29/1:38 (between points: the shuttle carried / tossed) -> drawn only in
  play and never while hanging still (in_play_only); LOCK ON only on the shuttle as seen now (it
  stayed pinned to the hit pixel while the camera panned). With the user's own calib reproduced
  (out/user_cal), every drawn point checked at 1:45 sat on the real shuttle.
- Score guesses (point_source "guess") are marked 推測 in the HUD and the editor table. A predicted
  landing too close to a line still sets point_lean (the guess); one well short of the net in
  the hitter's own half is a net error (NET_SURE_SD). The user's ACE complaints (net shot given
  to the hitter) were all guesses: the montage cuts most rally endings.
- Winner guesses, from the user's corrections of their ACE run (data/user/ace_truth.json now has
  winners for 10 of 12 stretches): the last detected hitter LOST 6 of 8 ([score] guess =
  last_hitter_loses), except after a smash (guess_kill). A floor landing calls in/out only
  floor_sure_sd clear of the lines (hits.FLOOR_PX through the homography: ~0.9 m near the net
  from 0.7 m high; one 0.9 m call was wrong). ACE winners 3/8 -> 6/8, final score 5:5 = truth.
  Tuned and checked on the same 8 rallies: re-check on the next clip before trusting it.
- Rally split when everyone (>= 3 tracked) stands still >= [rally] still_s: in the user's run
  three rallies (83-114 s) merged into one and two points vanished; every such stretch on ACE
  was a serve being set up.
- Manual points (editor "補一分", manual_points.csv: t, point_team): rallies.add_manual inserts
  one-frame pseudo rallies (rally_id 1000+, manual=True, point_source "manual"); score.apply
  orders by start_ts so detected rally ids, and the user's point_fixes keyed on them, stay put.
  The user's run of ACE missed 2 of our points (~85 s, ~97 s) that a run with the reference
  calibration finds; their calib keeps failing camera_plausible (far corners ~50 px high).
- Roboflow/HF/Google Drive downloads are blocked by this environment's network policy (the
  Roboflow MCP can search/fork/train but export links are on app.roboflow.com). GitHub release
  assets (e.g. ultralytics yolo11n.pt) and PyPI do download.
- Frame-level shuttle detector (hud/detection/shuttle_yolo.py, optional): models/shuttle_yolo.pt trained on
  the Mac by scripts/train_shuttle.py (Roboflow Universe datasets, merged to 1 class). Fills
  TrackNet gaps <= 1 s along a quadratic path, and searches for the shuttle lying still after a
  track ends (rejected if a spare already lay there before the shot). A hosted Universe model
  (my-first-project-i8fww/2) found a plainly visible shuttle TrackNet missed (conf 0.45) but not a
  faint one (0.10). Not validated end-to-end yet: needs the trained weights.
- Shuttle (TrackNetV3) loses the shuttle against dark, crowded backgrounds -> rallies cut short.
- Oblique test clip: 3/3 rallies found + 2 false ones (pick-ups), hitter side 7/7.
- Speed + DT clip (2026-10-27, local Mac): the user's 10-min 2940x1912 45 fps screen recording never
  finished. Measured on a 60 s excerpt then the whole clip: VideoToolbox decodes that 3K clip at only
  ~100-120 fps (wall-bound, low CPU), so every full pass over the source costs ~4.5 min. Fixes:
  stabilisation decodes at feature size (640) instead of resizing 1920 frames in Python; matches
  every KEY_EVERY=4th frame and interpolates (frame-by-frame warps jittered ~16 px there); warps that
  can't be a phone turning (stabilize.sane) are "lost" instead of chained (the camera turned to the
  bench: chaining drifted into garbage, 88 false cuts in 60 s, 0 rallies); lost > 0.5 s -> black
  frame; the canvas is capped at MAX_SIDE too (was 3074 px). Median / player-region samples come from
  key frames (video.sample) instead of decoding the whole clip. Full clip: ~45 min (est.) -> 23 min.
  ACE (KEY_EVERY 1 vs 4): rallies found 10 vs 9 of 12, winners 5/9 vs 4/8, same side 21% vs 14%.
- TrackNet window (shuttle.window) is also kept high enough to hold the air AIR_M over the net/far
  court: on DT (landscape, camera 1.35 m) the floor-anchored window started below the far baseline:
  shuttle 48% -> 80% of frames, hits 212 (196 near : 16 far) -> 639 (390 : 249), rallies 10 -> 45.
  ACE's window is unchanged. Winners on DT still unverified (no truth yet).
- Speed on the user's iPhone clip (2026-10-27, IMG_9840: 4K HEVC 10-bit HLG, 30 fps, 7.3 min,
  rotation -180): 12 min 11 s end to end (stabilize 14, players 221, shuttle 278, hits 38, hud 164 s).
  Fixed: (1) VideoToolbox frames of 10-bit clips must download as p010le (nv12 failed -> silent
  software decode, ~60 fps on every core); (2) the hardware path ignores rotation metadata ->
  video._turn adds hflip/vflip/transpose and probe() returns the displayed size (frames came out
  upside down: YOLO found nobody); (3) render survives the encoder closing early (-shortest).
  TrackNet's main + far-court windows now share one hardware decode (shuttle._frames_multi,
  decoded in a prefetch thread); the far window used to go through software ffmpeg crop.
  CoreML exports (models/*.mlpackage, not in git) halve TrackNet time; on a 60 s excerpt hits
  21 -> 17-18 vs PyTorch (fp16 noise, still vouched per video). YOLO pose: CoreML 22 ms vs MPS
  19 ms per call, no gain. Good-Badminton's yolo11s-ball (tested as a TrackNet replacement):
  mostly fires on a shuttle lying on the floor, misses the flying one, ~34 fps -> not used.
- User's IMG_9840 review (2026-10-28): (1) passers-by right in front of the camera became 我隊1: near
  baseline margin 1.3 -> 0.5 m, side 0.6 -> 0.3 m (players there in 0.15 % / 0.07 % of detections),
  a slot lost > 1 s re-acquires only inside the court (REACQ_M). Clothing colour did not separate
  them (the passer's light blue ~ our blue shirt). (2) the smoother extrapolated an unseen player
  4 m across the net: hold before/after sightings + keep_on_side clamp. (3) speeds/types shown only
  when hits.sure (drag_fit; others 3-409 km/h); scoring still uses all of them (hiding them dropped
  real rallies as pick-ups). (4) point_fixes carry start_ts and are matched by time
  (rallies.fixes_by_time; old files backfilled from the rallies.parquet they were made on).
- Tried and dropped for rally/serve detection on IMG_9840: next rally's first hit = previous winner
  (19/34, the serve is usually missed); player motion (walking between points ~ short rallies);
  racket sound (neighbouring courts: 5 vs 3 onsets per 4 s, near/far hits equally loud).
  The editor's Quick review (stop after each rally: arrows / 1-2 for missed points) covers it.
- UI and HUD: back to Traditional Chinese (user tried English, did not like it). Foot rings: one thin
  solid ring + two turning arcs (a glow / fill / white highlight was "too bright, ugly"). The shuttle is
  drawn as a reticle + 0.5 s tail, in bursts of hits (render.hit_groups), not only inside detected
  rallies (a 2-hit exchange at 2:42 the rally rules dropped was never drawn); render.fill_flights bridges <= 0.8 s gaps between a hit's contact,
  the real sightings and the next contact (PCHIP), only on flights with >= 2 sightings.
- TrackNet misses on IMG_9840 (2026-10-28): in play it sees the shuttle in 58 % of frames (29 % over the
  whole clip). 12 sampled short gaps: 4 hidden behind a near player, 4 lost against the ceiling lights /
  YONEX banner (visible to a person), 3 already on the floor. Long gaps (> 10 frames) often start and end
  on sightings that sit on players' shirts/shorts: TrackNet locked onto clothing, not the shuttle.

- Shot types (2026-10-28, user's 58 typed hits on IMG_9840): the rule types (hits.shot_type) were right
  9 %, a classifier for all 6 types 60 % vs 57 % "always lift". The HUD now shows only 殺球
  (hits.shown_types: image downward speed after the hit > SMASH_VY and contact > 1.4 m: 6/10 found,
  2 wrong) plus 直線/斜線（推測） from the hitter's vs the next contact's side of the centre line
  (unchecked: no truth). shot_type stays for scoring/stamina.
- Every shot shows a speed now (user, 2026-10-28): hits.rough_speeds -> speed_shown_kmh / speed_src: fit (sure),
  flight (straight distance to the next contact or landing over the flight time, drag undone; r 0.65 vs
  the fit, flights averaging > 100 km/h rejected as mis-paired hits), window, typical (match median,
  ~40 of 194 hits: nothing seen after the hit). Clipped 15-250. HUD shows "≈" + 球速（約） unless fitted.
  Recovered serves get the server's feet as contact (z 1.1 m).
- HUD polish (user: shuttle effect not flashy, turning like the foot rings but not the same): shuttle =
  white dot + 3 turning arcs (upright circle, 1.2 s a turn) + 0.25 s tail, drawn over the name tags;
  speed beside it as plain shadowed text for 1 s (was a LOCK ON box); name tags 132x34 (name + thin
  stamina bar, number only in the player panel); MOVE target = one dashed floor circle + small label.
