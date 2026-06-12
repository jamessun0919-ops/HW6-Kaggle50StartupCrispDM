# 📊 Kaggle 50 Startups 簡報影片生成與優化工作報告

本報告紀錄了將 `Startup_Profit_Equation.pptx` 轉換為專業 1080p 解說影片（MP4）的完整技術流程、字幕/進度條修復細節，以及最終的部署成果。

---

## 📋 專案基本資訊

* **輸出解析度**：1920x1080 (16:9 橫向)
* **影格率 (FPS)**：30 fps
* **旁白語言**：繁體中文（Mandarin Chinese）
* **語音模型**：Kokoro TTS 引擎 (`zm_yunxi` 中文男聲模型)
* **影音框架**：HyperFrames (結合 GSAP 動畫與 Puppeteer 截圖編譯)
* **最終影片檔案**：[output.mp4](../output.mp4) (20.5 MB，長度為 4 分 9 秒)

---

## 🛠️ 技術工作細節

### 1. 語音旁白生成與修補 (TTS Calibration)
* **生成腳本**：[generate_audio.py](../generate_audio.py)
* **修補細節**：修正了 `kokoro-onnx` 分詞器在 Windows 環境下處理英文專有名詞（如 "Kaggle", "Startups"）時，因 `espeak-ng` 產生語言標記（如 `(en)`）導致語音碎裂與發音錯誤的問題。透過在 `tokenizer.py` 中以 Regex 清理語言標記，使發音完全流暢清晰。
* **輸出結果**：於 [audio/](../audio) 目錄下生成了 20 張投影片的中文語音檔（`slide_01.wav` 至 `slide_20.wav`）。

### 2. 字幕時序精確對齊 (Subtitle Alignment)
* **對齊腳本**：[align_subtitles.py](../align_subtitles.py)
* **對齊機制**：讀取每張投影片語音檔的實際物理長度，將旁白文字進行標點符號分句與字數加權，線性分配各語句的 `start` 與 `end` 時間軸。
* **輸出檔**：產出精準對齊檔 [subtitles.json](../audio/subtitles.json)。

### 3. CSS 巢狀語法修復與網頁重建
* **構建指令**：`python scratch/build_index_html.py`
* **偵測問題**：原本在隱藏投影片場景時使用了 CSS 巢狀寫法（CSS Nesting）：
  ```css
  #scene2 { #scene3 { #scene4 { ... } } }
  ```
  該語法在部分渲染核心（如 Puppeteer 的 Chromium 實例）解析時會造成語法樹崩潰，使該行之後的所有 CSS（包含 `#subtitle-container` 的絕對定位、層級、黑色背板樣式，以及 HUD 模組）完全失效，導致字幕掉出畫布之外無法顯示。
* **修正方案**：將 CSS 生成邏輯改為標準的逗號分隔選擇器：
  ```css
  #scene2, #scene3, #scene4, ..., #scene20 {
    opacity: 0;
  }
  ```
  修正後經 `npx hyperframes inspect` 檢驗為 **0 個佈局異常**。

### 4. 影片渲染與輸出
* **渲染指令**：
  ```powershell
  $env:PATH = "C:\Users\User\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin;" + $env:PATH;
  npx hyperframes render -o output.mp4 --fps 30
  ```
  以 Chrome  headless shell 在背景對網頁逐影格捕獲，並利用 FFmpeg 將畫面與音軌合成為最終影片。

---

## 🎬 畫面視覺呈現

經由 `hyperframes snapshot` 所擷取的網頁畫面，驗證了字幕與 HUD 模組均已完美燒入影片：

* **Slide 1 封面（概述階段）**：字幕顯示「CRISP-DM」，HUD 指示器啟動「Overview」並標記為藍色，底端進度條亮起。
* **Slide 2 介紹（概述階段）**：字幕顯示「Machine」，進度條線性前進，且所有排版字體皆套用 `Outfit` 與 `Inter` 無襯線黑體字型。

---

## 🚀 部署與線上展示

我們已將所有異動推播至 GitHub 儲存庫：
* **專案網址**：[https://github.com/jamessun0919-ops/HW6-Kaggle50StartupCrispDM](https://github.com/jamessun0919-ops/HW6-Kaggle50StartupCrispDM)
* **Demo 展示網頁**：[demo.html](../demo.html) 
  * 網頁頂部已嵌入影片播放器直接展示最新產出的影片成果。
  * 點擊 README 上的 **🎥 Live Demo** 即可在線上直接觀看影片：[前往展示網頁](https://jamessun0919-ops.github.io/HW6-Kaggle50StartupCrispDM/demo.html)
