import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_understanding import (
    load_data, quality_checks, univariate_analysis,
    bivariate_analysis, correlation_analysis, state_analysis, outlier_analysis
)
from src.data_preparation import feature_engineering, prepare_features, split_data, scale_features
from src.multicollinearity import run_vif_analysis
from src.feature_selection import run_feature_selection
from src.modeling import train_all_models
from src.evaluation import evaluate_all_models, find_best_model
from src.business_analytics import what_if_analysis, budget_analysis, roi_analysis, cluster_analysis
from src.config import MODELS_DIR, REPORTS_DIR

st.set_page_config(page_title="50 Startups Profit Prediction", layout="wide")
st.title("📊 Kaggle 50 Startups 獲利預測分析")
st.markdown("基於 CRISP-DM 框架的完整機器學習分析流程")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "資料探索 (EDA)", "資料準備", "特徵選擇",
    "模型訓練", "模型評估", "商業分析", "Profit 預測"
])

with tab1:
    st.header("資料探索與分析 (EDA)")

    df = load_data()
    st.subheader("原始資料")
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("基本統計")
        st.dataframe(df.describe().round(2), use_container_width=True)
    with col2:
        st.subheader("資料型態")
        buf = []
        df.info(buf=lambda x: buf.append(x + "\n"))
        st.text("".join(buf))

    st.subheader("缺失值檢查")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0] if missing.sum() > 0 else pd.DataFrame({"訊息": ["無缺失值"]}), use_container_width=True)
    st.write(f"重複記錄: {df.duplicated().sum()}")

    st.subheader("單變數分析")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, col in enumerate(["Profit", "R&D Spend", "Marketing Spend", "Administration"]):
        ax = axes[i // 2, i % 2]
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("雙變數分析")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [("R&D Spend", "Profit"), ("Marketing Spend", "Profit"), ("Administration", "Profit")]
    for i, (x, y) in enumerate(pairs):
        sns.scatterplot(data=df, x=x, y=y, ax=axes[i])
        axes[i].set_title(f"Profit vs {x}")
        corr = np.corrcoef(df[x], df[y])[0, 1]
        axes[i].text(0.05, 0.9, f"Corr: {corr:.3f}", transform=axes[i].transAxes,
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("相關性矩陣")
    corr = df[["R&D Spend", "Administration", "Marketing Spend", "Profit"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".3f", square=True, ax=ax)
    st.pyplot(fig)

    st.subheader("State 分析")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="State", y="Profit", ax=ax)
    ax.set_title("Profit by State")
    st.pyplot(fig)

    from scipy.stats import f_oneway
    groups = [g["Profit"].values for _, g in df.groupby("State")]
    f_stat, p_val = f_oneway(*groups)
    st.write(f"**ANOVA 結果**: F={f_stat:.4f}, p-value={p_val:.6f}")
    st.write(f"結論: {'State 對 Profit 有顯著影響' if p_val < 0.05 else 'State 對 Profit 無顯著影響'}")

with tab2:
    st.header("資料準備")

    use_engineered = st.checkbox("使用衍生特徵 (TotalSpend, ROI, RDRatio, MarketingRatio)", value=True)

    if use_engineered:
        df_fe = feature_engineering(df)
        st.subheader("衍生特徵")
        st.dataframe(df_fe[["TotalSpend", "ROI", "RDRatio", "MarketingRatio"]].head(), use_container_width=True)
    else:
        df_fe = df.copy()

    st.subheader("One-Hot Encoding (State)")
    df_enc = pd.get_dummies(df_fe, columns=["State"], drop_first=True)
    st.dataframe(df_enc.head(), use_container_width=True)

    st.subheader("Train/Test Split (80/20)")
    y = df_enc["Profit"]
    X = df_enc.drop(columns=["Profit"])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write(f"訓練集: {X_train.shape}, 測試集: {X_test.shape}")

    st.subheader("StandardScaler 縮放")
    scale_cols = ["R&D Spend", "Administration", "Marketing Spend"]
    scale_cols = [c for c in scale_cols if c in X_train.columns]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])
    st.write("縮放完成")

    if st.button("儲存處理後的資料"):
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
        joblib.dump(X_train_scaled.columns.tolist(), MODELS_DIR / "feature_columns.pkl")
        st.success("已儲存!")

    st.session_state["X_train"] = X_train_scaled
    st.session_state["X_test"] = X_test_scaled
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test

with tab3:
    st.header("共線性與特徵選擇")

    if "X_train" not in st.session_state:
        st.warning("請先在「資料準備」分頁處理資料")
        st.stop()

    X_train = st.session_state["X_train"]
    y_train = st.session_state["y_train"]

    st.subheader("VIF 分析")
    from src.multicollinearity import compute_vif
    vif_df = compute_vif(X_train.select_dtypes(include=[np.number]), X_train.select_dtypes(include=[np.number]).columns.tolist())
    st.dataframe(vif_df, use_container_width=True)

    severe = vif_df[vif_df["VIF"] >= 10]
    if not severe.empty:
        st.warning(f"⚠ 以下特徵 VIF≥10，可能存在共線性: {', '.join(severe['feature'].tolist())}")

    st.subheader("LassoCV 特徵選擇")
    from sklearn.linear_model import LassoCV
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    coef_df = pd.DataFrame({"feature": X_train.columns, "coefficient": lasso.coef_})
    coef_df["selected"] = np.abs(coef_df["coefficient"]) > 1e-6
    coef_df = coef_df.sort_values("coefficient", key=abs, ascending=False)
    st.dataframe(coef_df, use_container_width=True)
    selected = coef_df[coef_df["selected"]]["feature"].tolist()
    st.success(f"LassoCV 選擇 {len(selected)} 個特徵: {selected}")

    st.subheader("RFE 特徵選擇")
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    rfe = RFE(LinearRegression(), n_features_to_select=max(1, X_train.shape[1] // 2))
    rfe.fit(X_train, y_train)
    rfe_df = pd.DataFrame({"feature": X_train.columns, "selected": rfe.support_, "rank": rfe.ranking_}).sort_values("rank")
    st.dataframe(rfe_df, use_container_width=True)
    rfe_selected = rfe_df[rfe_df["selected"]]["feature"].tolist()
    st.success(f"RFE 選擇 {len(rfe_selected)} 個特徵: {rfe_selected}")

with tab4:
    st.header("模型訓練")

    if "X_train" not in st.session_state:
        st.warning("請先在「資料準備」分頁處理資料")
        st.stop()

    X_train = st.session_state["X_train"]
    y_train = st.session_state["y_train"]

    include_advanced = st.checkbox("包含進階模型 (XGBoost, LightGBM)", value=True)

    if st.button("訓練所有模型", type="primary"):
        with st.spinner("訓練中..."):
            trained = train_all_models(X_train, y_train, include_advanced)
            st.session_state["trained_models"] = trained
            st.success(f"已訓練 {len(trained)} 個模型!")
            for name in trained:
                st.write(f"  ✓ {name}")

with tab5:
    st.header("模型評估")

    if "trained_models" not in st.session_state:
        st.warning("請先在「模型訓練」分頁訓練模型")
        st.stop()

    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test = st.session_state["y_test"]
    trained = st.session_state["trained_models"]

    result_df = evaluate_all_models(trained, X_train, y_train, X_test, y_test)
    st.subheader("模型比較表")
    st.dataframe(result_df, use_container_width=True)

    best_name = find_best_model(result_df)
    st.success(f"🏆 最佳模型: {best_name}")

    best_model = trained[best_name]

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    st.session_state["best_model"] = best_model
    st.session_state["best_model_name"] = best_name

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pred = best_model.predict(X_test)
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_xlabel("Actual Profit")
    ax.set_ylabel("Predicted Profit")
    ax.set_title(f"{best_name}: Actual vs Predicted")
    st.pyplot(fig)

with tab6:
    st.header("商業分析")

    if "best_model" not in st.session_state:
        st.warning("請先在「模型評估」分頁完成評估")
        st.stop()

    best_model = st.session_state["best_model"]
    X_test = st.session_state["X_test"]

    st.subheader("What-If 情境分析")
    scenario_df = what_if_analysis(best_model, X_test, None, X_test.columns.tolist(), df_original_row=X_test)
    st.dataframe(scenario_df, use_container_width=True)

    st.subheader("預算配置分析")
    budget_df = budget_analysis(df)
    st.dataframe(budget_df, use_container_width=True)

    st.subheader("ROI 分析")
    roi_df = roi_analysis(df)
    st.dataframe(roi_df, use_container_width=True)

    st.subheader("K-Means 集群分析")
    clustered_df, kmeans_model = cluster_analysis(df)
    st.dataframe(clustered_df.groupby("ClusterLabel")[
        ["R&D Spend", "Marketing Spend", "Administration", "Profit"]
    ].mean().round(2), use_container_width=True)

with tab7:
    st.header("Profit 預測")

    if "best_model" not in st.session_state:
        st.warning("請先在「模型評估」分頁完成模型訓練與評估")
        st.stop()

    best_model = st.session_state["best_model"]

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            rd = st.number_input("R&D Spend", min_value=0.0, value=100000.0, step=1000.0)
        with col2:
            admin = st.number_input("Administration", min_value=0.0, value=100000.0, step=1000.0)
        with col3:
            marketing = st.number_input("Marketing Spend", min_value=0.0, value=100000.0, step=1000.0)

        state = st.selectbox("State", ["New York", "California", "Florida"])

        submitted = st.form_submit_button("預測 Profit", type="primary")

    if submitted:
        input_df = pd.DataFrame([[rd, admin, marketing, state]],
                                columns=["R&D Spend", "Administration", "Marketing Spend", "State"])
        input_enc = pd.get_dummies(input_df, columns=["State"], drop_first=True)

        model_features = X_train.columns.tolist() if "X_train" in st.session_state else []
        if model_features:
            for col in model_features:
                if col not in input_enc.columns:
                    input_enc[col] = 0
            input_enc = input_enc[model_features]

        scale_cols = [c for c in ["R&D Spend", "Administration", "Marketing Spend"] if c in input_enc.columns]
        try:
            scaler_loaded = joblib.load(MODELS_DIR / "scaler.pkl")
            input_enc[scale_cols] = scaler_loaded.transform(input_enc[scale_cols])
        except:
            pass

        pred = best_model.predict(input_enc)[0]

        st.subheader("預測結果")
        col1, col2, col3 = st.columns(3)
        col1.metric("預測 Profit", f"${pred:,.2f}")
        col2.metric("R&D Spend", f"${rd:,.0f}")
        col3.metric("Marketing Spend", f"${marketing:,.0f}")

        total = rd + admin + marketing
        st.metric("Total Spend", f"${total:,.0f}")
        st.metric("ROI", f"{pred / total:.2f}" if total > 0 else "N/A")

        st.subheader("投資配置比例")
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["R&D", "Marketing", "Administration"]
        sizes = [rd, marketing, admin]
        colors = ["#ff9999", "#66b3ff", "#99ff99"]
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

st.sidebar.markdown("""
## 專案資訊
- **資料集**: Kaggle 50 Startups
- **目標**: Profit 預測 (Regression)
- **框架**: CRISP-DM
- **R² 目標**: ≥ 0.90

### CRISP-DM 流程
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Multicollinearity Analysis
5. Feature Selection
6. Modeling
7. Evaluation
8. Explainable AI
9. Business Analytics
10. Deployment
""")
