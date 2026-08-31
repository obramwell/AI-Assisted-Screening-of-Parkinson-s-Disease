from pathlib import Path

# PROJECT ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# DATA
DATA = PROJECT_ROOT / "data"

RAW_DATA = DATA / "raw"
INTERIM_DATA = DATA / "interim"
PROCESSED_DATA = DATA / "processed"

# MODELS
MODELS = PROJECT_ROOT / "models"

# OUTPUTS
OUTPUTS = PROJECT_ROOT / "outputs"
# General folders
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
METRICS = OUTPUTS / "metrics"

# SHAP
SHAP = OUTPUTS / "shap"
SHAP_FIGURES = SHAP / "figures"
SHAP_TABLES = SHAP / "tables"
SHAP_BAR = SHAP_FIGURES / "Top 20 Global SHAP Features.png"

SHAP_MODALITY = SHAP_FIGURES / "shap_modality_importance.png"
SHAP_TASK = SHAP_FIGURES / "shap_task_importance.png"
SHAP_SENSOR = SHAP_FIGURES / "shap_sensor_importance.png"
SHAP_WRIST = SHAP_FIGURES / "shap_wrist_importance.png"
SHAP_QUESTIONNAIRE = SHAP_FIGURES / "shap_questionnaire_importance.png"

GLOBAL_SHAP = SHAP_TABLES / "global_shap_importance.csv"
MODALITY_TABLE = SHAP_TABLES / "shap_modality_importance.csv"
TASK_TABLE = SHAP_TABLES / "shap_task_importance.csv"
SENSOR_TABLE = SHAP_TABLES / "shap_sensor_importance.csv"
WRIST_TABLE = SHAP_TABLES / "shap_wrist_importance.csv"
QUESTIONNAIRE_TABLE = SHAP_TABLES / "questionnaire_shap_summary.csv"
QUESTIONNAIRE_IMPORTANCE = SHAP_TABLES / "shap_questionnaire_importance.csv"
HEALTHY_TABLE = SHAP_TABLES / "shap_importance_Healthy.csv"
PD_TABLE = SHAP_TABLES / "shap_importance_Parkinsons_Disease.csv"
OMD_TABLE = SHAP_TABLES / "shap_importance_Other_Movement_Disorders.csv"

# Performance
TEST_RESULTS = METRICS / "test_results.csv"
VALIDATION_TEST_COMPARISON = (TABLES / "validation_test_comparison.csv")
CALIBRATION_RESULTS = (METRICS / "calibration_results.csv")
# FULL MULTIMODAL (FINAL MODEL)
FULL_CM = FIGURES / "full_multimodal_xgboost_test_confusion_matrix.png"
FULL_CLASSIFICATION_REPORT = (METRICS/ "full_multimodal_xgboost_test_classification_report.csv")
FULL_CALIBRATION = (FIGURES/ "calibration_full_multimodal.png")
# ASSETS
ASSETS = PROJECT_ROOT / "dashboard" / "assets"
LOGO = ASSETS / "logo.png"
BANNER = ASSETS / "banner.png"
