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

# SHAP
SHAP = OUTPUTS / "shap"
SHAP_FIGURES = SHAP / "figures"
SHAP_TABLES = SHAP / "tables"

# Performance
PERFORMANCE = OUTPUTS / "performance"
PERFORMANCE_FIGURES = PERFORMANCE / "figures"
PERFORMANCE_TABLES = PERFORMANCE / "tables"

# ASSETS
ASSETS = PROJECT_ROOT / "dashboard" / "assets"
LOGO = ASSETS / "logo.png"
BANNER = ASSETS / "banner.png"
# CSS
STREAMLIT = PROJECT_ROOT / "dashboard" / ".streamlit"

CSS = STREAMLIT / "style.css"