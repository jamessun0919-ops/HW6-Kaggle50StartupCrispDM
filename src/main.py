"""
Kaggle 50 Startups Analysis - CRISP-DM + Scikit-Learn Machine Learning Project

根據 YAML.txt 定義的完整 CRISP-DM 流程實作：
  1. Business Understanding (背景說明)
  2. Data Understanding (EDA)
  3. Data Preparation (特徵工程、編碼、分割、縮放)
  4. Multicollinearity Analysis (VIF)
  5. Feature Selection (LassoCV、RFE)
  6. Modeling (LinearRegression、Ridge、Lasso、RF、GBR、XGBoost、LightGBM)
  7. Evaluation (R²、MAE、RMSE、交叉驗證)
  8. Explainable AI (SHAP、Permutation Importance)
  9. Business Analytics (What-if、預算配置、ROI、K-Means)
  10. Business Insights & Recommendations
  11. Deployment (Streamlit)
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 100

from src.config import REPORTS_DIR, MODELS_DIR
from src.data_understanding import run_data_understanding
from src.data_preparation import run_data_preparation
from src.multicollinearity import run_vif_analysis
from src.feature_selection import run_feature_selection
from src.modeling import train_all_models
from src.evaluation import evaluate_all_models, find_best_model
from src.explainable_ai import run_explainability
from src.business_analytics import run_business_analytics

import joblib
import os


REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print_section("Kaggle 50 Startups Analysis - CRISP-DM Pipeline")
    print("目標: 建立可解釋的獲利預測模型，驗證 R&D 為主要獲利驅動因子")

    print_section("1. Business Understanding")
    print("""
商業問題:
  - 哪些因素最影響 Profit?
  - R&D 是否比 Marketing 更重要?
  - Administration 是否只是成本中心?
  - State 是否影響 Profit?
  - 如何最佳化資源配置?

假設:
  H1: R&D Spend 與 Profit 高度正相關
  H2: Marketing Spend 與 Profit 中度正相關
  H3: Administration 對 Profit 影響有限
  H4: State 對 Profit 影響不顯著

成功標準:
  - Business: 找出主要獲利因子、提供預算配置建議
  - Technical: R^2 >= 0.90
""")

    print_section("2. Data Understanding (EDA)")
    df = run_data_understanding()

    print_section("3. Data Preparation")
    print("[3a] 使用原始特徵 (無衍生特徵)")
    X_train, X_test, y_train, y_test, scaler, encoder, X_train_orig, X_test_orig = run_data_preparation(df, use_engineered=False)

    print_section("4. Multicollinearity Analysis (VIF)")
    run_vif_analysis(X_train)

    print_section("5. Feature Selection")
    run_feature_selection(X_train, y_train)

    print_section("6. Modeling")
    print("訓練模型中...")
    trained_models = train_all_models(X_train, y_train, include_advanced=True)

    print_section("7. Model Evaluation")
    result_df = evaluate_all_models(trained_models, X_train, y_train, X_test, y_test)
    print("\n[模型比較表]")
    print(result_df.to_string(index=False))

    best_model_name = find_best_model(result_df)
    best_model = trained_models[best_model_name]

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(encoder, MODELS_DIR / "encoder.pkl")
    joblib.dump(X_train.columns.tolist(), MODELS_DIR / "feature_columns.pkl")
    print(f"\n模型已儲存至 {MODELS_DIR}/")

    print_section("8. Explainable AI")
    run_explainability(best_model, X_train, X_test, y_test, best_model_name)

    print_section("9. Business Analytics")
    run_business_analytics(best_model, X_test, X_train.columns.tolist(), scaler, df, X_test_orig)

    print_section("10. Business Insights & Recommendations")
    print("""
預期發現:
  1. R&D 為最重要獲利因子
  2. Marketing 為次要獲利因子
  3. Administration 影響有限
  4. State 不一定具有統計顯著性
  5. 高獲利企業具有特殊投資結構

建議:
  1. 增加 R&D 投資
  2. 提高 Marketing 效率
  3. 控制 Administration 成本
  4. 建立 Profit 預測模型
  5. 建立預算配置決策系統
""")

    print_section("Pipeline Complete")
    print(f"報表已儲存至 {REPORTS_DIR}/")
    print(f"模型已儲存至 {MODELS_DIR}/")
    print("執行 Streamlit Dashboard: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
