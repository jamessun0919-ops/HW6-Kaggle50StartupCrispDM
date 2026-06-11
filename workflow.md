# 50 Startups Profit Prediction — Workflow

![Workflow](workflow.png)

## Project Structure

```
HW62/
├── 50_Startups.csv              # Raw data
├── YAML.txt                     # Project specification (CRISP-DM)
├── markdown.txt                 # Markdown documentation
│
├── src/                         # Python modules
│   ├── main.py                  # Full CRISP-DM pipeline
│   ├── config.py                # Paths & constants
│   ├── data_understanding.py    # EDA, statistics, visualization
│   ├── data_preparation.py      # Feature engineering, encoding, scaling
│   ├── multicollinearity.py     # VIF analysis
│   ├── feature_selection.py     # LassoCV + RFE
│   ├── modeling.py              # Model definitions & training
│   ├── evaluation.py            # Metrics, CV, comparison
│   ├── explainable_ai.py        # SHAP + Permutation importance
│   ├── business_analytics.py    # What-if, budget, ROI, clustering
│   ├── feature_importance_viz.py # Feature importance comparison charts
│   ├── generate_selection_curve.py # Forward selection R2 chart
│   ├── make_linechart.py        # R2 progression line chart
│   ├── make_rmse_chart.py       # RMSE progression line chart
│   ├── make_rmse_by_method.py   # RMSE vs n_features (5 methods)
│   └── make_r2_by_method.py     # R2 vs n_features (5 methods)
│
├── app/
│   └── streamlit_app.py         # Streamlit dashboard
│
├── notebooks/
│   └── kaggle_50_startups_full_analysis.ipynb  # Jupyter notebook
│
├── models/                      # Saved trained models
├── reports/                     # Generated charts & reports
├── requirements.txt
└── workflow.md                  # This file
```

## Setup

```bash
pip install -r requirements.txt
```

## Execution Steps

### 1. Full CRISP-DM Pipeline

```bash
python -c "from src.main import main; main()"
```

Executes all phases:
| Phase | Description |
|-------|-------------|
| 1 Business Understanding | Objectives, hypotheses, success criteria |
| 2 Data Understanding | EDA, quality checks, correlation, ANOVA |
| 3 Data Preparation | Feature engineering, encoding, split, scaling |
| 4 Multicollinearity | VIF analysis |
| 5 Feature Selection | LassoCV + RFE |
| 6 Modeling | LinearRegression, Ridge, Lasso, RF, GBR, XGBoost, LightGBM |
| 7 Evaluation | R2, MAE, RMSE, 5-fold CV |
| 8 Explainable AI | SHAP + Permutation Importance |
| 9 Business Analytics | What-if, budget, ROI, K-Means |
| 10 Insights & Recommendations | Conclusions |

### 2. Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Interactive tabs: EDA, Data Preparation, Feature Selection, Model Training, Evaluation, Business Analytics, Profit Prediction.

### 3. Generate Charts

#### Forward Feature Selection (R2 progression)

```bash
python src/make_linechart.py     # -> linechart.png
python src/make_rmse_chart.py    # -> rmse_linechart.png
```

#### R2 by Number of Features (5 methods)

```bash
python src/make_r2_by_method.py  # -> r2_by_method.png
```

![R2 by Method](r2_by_method.png)

#### RMSE by Number of Features (5 methods)

```bash
python src/make_rmse_by_method.py # -> rmse_by_method.png
```

![RMSE by Method](rmse_by_method.png)

#### Feature Importance Comparison

```bash
python -c "from src.feature_importance_viz import main; main()"
```

Generates: `correlation_with_regression.png`, `feature_importance_comparison.png`, `feature_importance_heatmap.png`, `shap_detail.png`, `shap_waterfall_detail.png`

#### Feature Selection Final Figure

```bash
python -c "from src.generate_feature_selection_fig import main; main()"
```

-> `feature_selection_final.png`

#### Custom Sequence Selection Curve

```bash
python -c "from src.generate_selection_curve import main; main()"
```

-> `selection_curve.png` (R&D → +Marketing → +State_NY → +Admin → All)

### 4. Jupyter Notebook

Open `notebooks/kaggle_50_startups_full_analysis.ipynb` for an interactive walkthrough.

## Key Findings

| Finding | Evidence |
|---------|----------|
| R&D is dominant driver | R=0.973, SHAP=24978, R&D-only R2=0.9265 |
| Marketing is secondary | R=0.748, SHAP=3839 |
| Administration has limited impact | R=0.201, SHAP=1622 |
| State is not significant | ANOVA p=0.567 |
| Best model | GradientBoosting (R2=0.9160) |
| R&D +10% → Profit | +12.0% |

## Output Files

| File | Description |
|------|-------------|
| `reports/correlation_heatmap.png` | Correlation matrix |
| `reports/correlation_with_regression.png` | Profit vs each feature with regression line |
| `reports/state_boxplot.png` | Profit distribution by state |
| `reports/univariate_analysis.png` | Histograms + boxplots |
| `reports/bivariate_analysis.png` | Scatter plots |
| `reports/outlier_analysis.png` | Outlier boxplots |
| `reports/budget_allocation.png` | High vs Low profit spending comparison |
| `reports/cluster_analysis.png` | K-Means clusters (3 types) |
| `reports/shap_summary_GradientBoosting.png` | SHAP summary plot |
| `reports/shap_waterfall_GradientBoosting.png` | SHAP waterfall (single sample) |
| `reports/permutation_importance_GradientBoosting.png` | Permutation importance |
| `reports/feature_importance_comparison.png` | Multi-method importance comparison |
| `reports/feature_importance_heatmap.png` | Normalized importance heatmap |
| `reports/feature_selection_final.png` | Comprehensive feature selection figure |
| `reports/feature_selection_linechart.png` | Line chart analysis |
| `reports/selection_curve.png` | Custom sequence R2 curve |
| `reports/linechart.png` | Forward selection R2 progression |
| `reports/rmse_linechart.png` | Forward selection RMSE progression |
| `reports/r2_by_method.png` | R2 vs n_features (5 methods) |
| `reports/rmse_by_method.png` | RMSE vs n_features (5 methods) |
| `models/best_model.pkl` | Trained best model (GradientBoosting) |
| `models/scaler.pkl` | Fitted StandardScaler |
| `models/encoder.pkl` | Fitted OneHotEncoder |
| `models/feature_columns.pkl` | Feature column names |
