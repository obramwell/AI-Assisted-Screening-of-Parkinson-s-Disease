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
# Global SHAP
SHAP_BAR = SHAP_FIGURES / "Top 20 Global SHAP Features.png"
# Global Beeswarm
SHAP_BEESWARM_GLOBAL = SHAP_FIGURES / "shap_beeswarm_global.png"
# Class-specific Beeswarm
BEESWARM_HEALTHY = SHAP_FIGURES / "beeswarm_Healthy.png"
BEESWARM_PD = SHAP_FIGURES / "beeswarm_Parkinsons_Disease.png"
BEESWARM_OMD = SHAP_FIGURES / "beeswarm_Other_Movement_Disorders.png"
# Local SHAP
WATERFALL_HEALTHY = ( SHAP_FIGURES /"waterfall_Healthy_patient_160.png")
WATERFALL_PD = (SHAP_FIGURES /"waterfall_Parkinsons_Disease_patient_25.png")
WATERFALL_OMD = (SHAP_FIGURES / "waterfall_Other_Movement_Disorders_patient_66.png")

SHAP_MODALITY = SHAP_FIGURES / "shap_modality_total_corrected.png"
SHAP_TASK = SHAP_FIGURES / "shap_task_total_corrected.png"
SHAP_SENSOR = SHAP_FIGURES / "shap_modality_total_corrected.png"
SHAP_WRIST = SHAP_FIGURES / "shap_wrist_corrected.png"
SHAP_QUESTIONNAIRE = SHAP_FIGURES / "shap_questionnaire_importance.png"

GLOBAL_SHAP = SHAP_TABLES / "global_shap_importance.csv"
MODALITY_TABLE = SHAP_TABLES / "shap_modality_corrected.csv"
TASK_TABLE = SHAP_TABLES / "shap_task_corrected.csv"
SENSOR_TABLE = SHAP_TABLES / "shap_sensor_corrected.csv"
WRIST_TABLE = SHAP_TABLES / "shap_wrist_corrected.csv"
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
# CONFORMAL PREDICTION
CONFORMAL = OUTPUTS / "conformal"
CONFORMAL_FIGURES = CONFORMAL / "figures"
CONFORMAL_TABLES = CONFORMAL / "tables"
CONFORMITY_COMPARISON = (CONFORMAL_TABLES /"conformity_score_comparison.csv")
CONFIDENCE_COMPARISON = (CONFORMAL_TABLES /"confidence_level_comparison.csv")
SELECTED_CONFIGURATION = (CONFORMAL_TABLES /"selected_conformal_configuration.csv")
PREDICTION_SET_SUMMARY = (CONFORMAL_TABLES /"prediction_set_summary.csv")
COVERAGE_CONFIDENCE = (CONFORMAL_FIGURES /"coverage_vs_confidence.png")
SETSIZE_CONFIDENCE = (CONFORMAL_FIGURES /"prediction_set_size_vs_confidence.png")
SINGLETON_CONFIDENCE = (CONFORMAL_FIGURES /"singleton_rate_vs_confidence.png")
AMBIGUOUS_CONFIDENCE = (CONFORMAL_FIGURES /"ambiguous_rate_vs_confidence.png")
TRADEOFF = (CONFORMAL_FIGURES /"conformity_score_tradeoff.png")
