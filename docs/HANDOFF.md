# 交接：從雲端 session 回到本機 Claude Code（2026-10-27）

給接手的 Claude Code：先讀 `CLAUDE.md`（狀態與已知問題）和 `README.md`（使用方式），再讀這份。
這份只寫「做到哪、為什麼這樣做、下一步」。使用者用中文，**回覆一律用繁體中文**。

## 使用者與工作方式

- 使用者在自己的 8 GB M2 Mac 上用 `uv run hud web` 測試，把 `check.json` / `timing.json` /
  HUD 影片 / 修正過的比分貼回來；我們依回饋修。每一批改進都開 PR 合併到 `main`
  （使用者會明確說「开PR合并到main」）。
- 使用者偏好：先講原因再講做法，用表格整理，誠實講限制（哪些驗證過、哪些沒有）。
- 使用者的球場與影片：手機約 0.7 m 高、直拍 0.5x、常用雲台轉動、有剪接（ACE 精華片段）。
  最新的影片是 DT 場館、橫拍、5 分鐘以上（還沒分析完，見「下一步」）。

## 這個 session 合併的 PR（都在 main）

| PR | 內容 |
|---|---|
| #3 | 鏡頭轉動自動穩定化（`pipeline.prepare`），HUD 逐幀畫回原畫面；腳位依相機幾何 RTS 平滑（`smooth.py`）；YOLO 只偵測球員區域；體力依球種；物理軌跡推算落點（`landing.py`）；同邊連擊修正；遠端 TrackNet 放大窗口；選用的單幀羽球偵測器（`shuttle_yolo.py` + `scripts/train_shuttle.py`，沒有權重就不啟用） |
| #4 | HUD 圈/名牌/MOVE 不再抖動；羽球只畫連續真實偵測；體力改成依跑動速度、腳步（加速度）、球種、連續進攻 |
| #5 | 網頁「補一分」（`manual_points.csv`）；推測改成「最後擊球者失分（殺球除外）」；地上落點離線太近不判；全員靜止 1 秒切回合 |
| #6 | 羽球只在回合中畫、半空靜止不畫；LOCK ON 只跟著當下看得到的球 |
| #7 | 穩定化：顯示進度、最多 1920 px 處理、Mac 硬體解碼/編碼、特徵點固定在 640 px 上找 |

## 驗證資料與方法

- 單元測試：`for f in tests/test_*.py; do uv run python $f; done`（48 個，全部通過）。
- ACE 標準答案：`docs/ace_truth.json`（12 段回合，其中 10 段有使用者修正過的得分方）。
  影片本身不在 git 裡（`data/` 被忽略），在使用者 Mac 上。用法：
  `uv run hud eval out/<資料夾> docs/ace_truth.json`
- 上次結果（ACE，參考校正）：回合找到率 75%、正確率 90%；得分方 6/8；比分 5:5 = 標準答案。
  ⚠ 推測規則是從同一批 8 個回合調出來的，下一支影片要重新驗證。
- 用使用者自己的校正重跑（`camera_plausible: false` 的那份）：13 個回合（多 3 個），得分方 6/9。
- 模擬驗證腳本（`scripts/experiments/`）：
  - `sim_landing.py CALIB IMG_H N`：落點預測的物理模擬（可用環境變數 KVAR / CORNER_PX / OUTL / NOISE 加壓力）
  - `sim_smooth.py CALIB ACCELS NOISE AMAX`：腳位平滑的模擬
  - `regress.py`：重跑 hits + rallies 並評分（需要 `out/reg_final_*`、`out/ace_v3` 這些本機輸出）

## 已知問題（依影響排序）

1. **擊球方判斷**（低機位最弱的一環）：ACE 同邊連擊還有 14%。試過但無效：物理軌跡擬合、
   30 fps 手腕速度（都寫在 CLAUDE.md）。**下一個想法：擊球聲**（近端比遠端大聲約 10 dB），
   需要一支**沒剪輯、保留現場原聲**的影片（ACE 的音軌是配樂）。
2. **比分**：沒看到落地就只能推測。推測規則需要更多標記過的回合驗證。使用者可以在網頁表格改、
   或用「補一分」。
3. **使用者的校正**一直出現「相機位置不合理」（遠端兩角比參考校正高約 50 px）→ 多出/漏掉回合。
   建議他重新校正遠端兩角。
4. **羽球偵測器**（`scripts/train_shuttle.py`）寫好了但沒訓練：使用者決定先不訓練（要在 Mac 上跑
   幾小時）。Roboflow / Hugging Face / Google Drive 在雲端被擋，在本機沒問題。
5. TrackNet 在深色、人多的背景會丟球；遠端放大窗口只補回約 6 個百分點。

## 下一步（建議順序）

1. 使用者正在用新版（PR #7）重跑 DT 那支 5 分鐘影片。拿到 `timing.json` 後看每一步耗時，
   特別是 stabilize / players / shuttle；必要時：60 fps 影片穩定化可以降到 1280、或只在有移動的
   段落做穩定化。
2. 用 DT 影片驗證比分推測規則（請使用者在表格修正得分方，把 `point_fixes.csv` 和 rallies 表貼回來，
   像這次一樣轉成 truth JSON 再 `hud eval`）。
3. 如果有原聲影片：做擊球聲的實驗（onset + 音量判斷近端/遠端），有效再接進 `hits.assign_sides`。
4. 如果使用者訓練了 `models/shuttle_yolo.pt`：比較加上偵測器前後的「對打中羽球偵測率、擊球數、同邊連擊」。

## 給本機 Claude Code 的第一句話（可以直接貼）

> 讀 CLAUDE.md、README.md 和 docs/HANDOFF.md，接續雲端 session 的工作。我正在用最新的 main 分析
> DT 那支 5 分鐘的影片，跑完會把 timing.json 給你。回覆用繁體中文。
