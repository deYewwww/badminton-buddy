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
- The analysis assumes ONE fixed camera for the whole video; panning/zooming clips break it.
- Shuttle (TrackNetV3) loses the shuttle against dark, crowded backgrounds -> rallies cut short.
- Oblique test clip: 3/3 rallies found + 2 false ones (pick-ups), hitter side 7/7.
