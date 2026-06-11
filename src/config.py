from pathlib import Path

RAW_DATA_PATH = Path("50_Startups.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
VIF_THRESHOLD_GOOD = 5
VIF_THRESHOLD_MODERATE = 10
ANOVA_ALPHA = 0.05
R2_MINIMUM = 0.90
