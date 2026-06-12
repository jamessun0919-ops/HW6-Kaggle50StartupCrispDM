import os
import espeakng_loader
import soundfile as sf
from kokoro_onnx import Kokoro

# Setup espeak environment variables for Windows
os.environ['ESPEAK_DATA_PATH'] = espeakng_loader.get_data_path()
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = espeakng_loader.get_library_path()

script_data = [
    "歡迎收看 Kaggle 50 Startups 資料分析專案簡報。本專案將基於 CRISP-DM 與 Scikit-Learn，為您呈現完整的機器學習分析與預測流程。",
    "本專案名為 Predicting Startup Profit Using Machine Learning，使用 Kaggle 的 50 家初創企業數據集，建立一個可解釋的獲利預測模型，以解答關鍵的商業決策問題。",
    "專案採用 CRISP-DM 框架，從商業理解開始，經歷數據理解、數據準備、建模、評估，最終進行部署並提取商業洞察。",
    "第一階段是商業理解。企業管理者最關心的是如何提升利潤、如何分配有限預算，以及哪些投資最值得投入。我們將預測未來獲利並最佳化預算。",
    "我們提出五個商業問題與假說：研發支出與利潤是否呈強烈正相關？行銷支出影響如何？行政支出是否只是成本中心？不同州別是否對利潤有實質影響？",
    "專案的商業成功標準在於找出關鍵獲利因子並提供配置建議。技術成功標準則是模型決定係數 R² 大於百分之九十、最小化均方根誤差，並確保模型的可解釋性。",
    "第二階段是數據理解。數據集包含 50 筆記錄，特徵包括研發支出、行政支出、行銷支出、州別以及目標變數利潤。我們首先進行數據加載與基本統計量檢查。",
    "接著進行數據質量檢查，確認無缺失值與重複記錄。單變數分析顯示，利潤大致呈常態分佈，且無明顯的異常離群值。我們也分析了研發、行銷和行政支出的分佈特徵。",
    "雙變數分析中，我們繪製散佈圖與相關係數矩陣。可以發現，研發支出與利潤呈現極高的正相關，行銷支出呈中度正相關，而行政支出與利潤的相關性則非常低。",
    "對州別進行箱形圖分析與單因子變異數分析，結果顯示不同州別之間的利潤差異在統計上並不顯著。異常值分析也確認數據集中僅有一筆極低獲利的離群值。",
    "第三階段是數據準備。我們進行特徵工程，新增了總支出、投資報酬率 ROI、行銷支出比例與研發支出比例等衍生特徵，以幫助模型捕捉更多商業維度的資訊。",
    "我們將類別變數州別進行獨熱編碼，並以八比二的比例切分訓練集與測試集。對於套用正規化的模型，如 Lasso 和 Ridge，我們也使用了 StandardScaler 進行特徵縮放。",
    "在第四階段，我們進行共線性分析。計算變異數膨脹因子 VIF，確認研發支出與行銷支出等主要特徵之間是否存在嚴重的共線性問題，以評估特徵與係數的穩定性。",
    "第五階段是特徵選擇。我們使用 Lasso 迴歸交叉驗證與遞迴特徵消除 RFE 方法，自動選擇最重要的特徵組合，並對原始特徵 and 衍生特徵進行比較排名。",
    "第六階段是建模。我們建立了線性迴歸作為基準模型，並訓練了 Ridge、Lasso、隨機森林和梯度提升迴歸等候選模型，同時也考慮了 XGBoost 與 LightGBM。",
    "第七階段是模型評估。我們使用 R²、平均絕對誤差 MAE 和均方根誤差 RMSE 作為評估指標，並進行五折交叉驗證。梯度提升迴歸在測試集上取得了最優的預測表現。",
    "第八階段是可解釋性 AI。我們使用 SHAP 分析進行全局與局部解釋，結果證實研發支出是主導獲利預測的最關鍵因子，行銷支出次之，行政支出與州別影響極小。",
    "第九階段為商業分析。進行假設模擬：研發支出增加百分之十，利潤將提升約百分之十二。我們進一步分析了高低獲利企業的預算配置差異，並使用 K-Means 進行集群分析。",
    "第十階段提取商業洞察與最終建議。強烈建議企業應增加研發投資，優化行銷效率，控制行政成本，並建立利潤預測系統以支援預算決策。",
    "最後是部署階段。我們推薦使用 Streamlit 建立交互式決策儀表板，提供利潤預測、假設分析與 SHAP 解釋等功能。最終交付成果包括完整報告、代碼庫與 Streamlit 應用程式。"
]

os.makedirs("audio", exist_ok=True)

model_path = os.path.expanduser('~/.cache/hyperframes/tts/models/kokoro-v1.0.onnx')
voices_path = os.path.expanduser('~/.cache/hyperframes/tts/voices/voices-v1.0.bin')

print("Loading Kokoro TTS engine...")
kokoro = Kokoro(model_path, voices_path)
print("TTS engine loaded successfully.")

for i, text in enumerate(script_data):
    slide_num = i + 1
    output_file = f"audio/slide_{slide_num:02d}.wav"
    print(f"Generating voiceover for Slide {slide_num:02d} ({len(text)} chars)...")
    
    # Use 'cmn' (Mandarin Chinese) which is supported by the Windows espeak backend
    samples, sample_rate = kokoro.create(text, voice="zm_yunxi", speed=1.0, lang="cmn")
    
    sf.write(output_file, samples, sample_rate)
    print(f"Saved Slide {slide_num:02d} to {output_file} successfully (duration: {len(samples)/sample_rate:.2f}s).")

print("All 20 slides audio files generated successfully!")
