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
- The analysis assumes ONE fixed camera for the whole video; panning/zooming clips break it.
- Shuttle (TrackNetV3) loses the shuttle against dark, crowded backgrounds -> rallies cut short.
- Oblique test clip: 3/3 rallies found + 2 false ones (pick-ups), hitter side 7/7.
