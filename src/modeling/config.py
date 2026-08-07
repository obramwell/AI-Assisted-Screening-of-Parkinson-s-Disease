"""
This module centralizes the configuration used by all baseline
machine learning experiments to ensure consistent training,
validation, and evaluation across all data modalities.
"""

from pathlib import Path

# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

TABLES_DIR = OUTPUT_DIR / "tables"

FIGURES_DIR = OUTPUT_DIR / "figures"

METRICS_DIR = OUTPUT_DIR / "metrics"

MODELS_DIR = PROJECT_ROOT / "models"

# =============================================================================
# Input datasets
# =============================================================================

FULL_DATASET_FILE = (
    PROCESSED_DIR /
    "integrated_participant_dataset.csv"
)

TRAIN_DATASET_FILE = (
    PROCESSED_DIR /
    "train_participant_dataset.csv"
)

VALIDATION_DATASET_FILE = (
    PROCESSED_DIR /
    "validation_participant_dataset.csv"
)

TEST_DATASET_FILE = (
    PROCESSED_DIR /
    "test_participant_dataset.csv"
)

# =============================================================================
# Dataset columns
# =============================================================================

PARTICIPANT_ID_COLUMN = "patient_id"

TARGET_COLUMN = "Label"

# =============================================================================
# Dataset split
# =============================================================================

TRAIN_SPLIT = 0.70

VALIDATION_SPLIT = 0.15

TEST_SPLIT = 0.15

GROUPED_VALIDATION = True

# =============================================================================
# Reproducibility
# =============================================================================

RANDOM_STATE = 42

# =============================================================================
# Class imbalance
# =============================================================================

USE_CLASS_WEIGHT = True

CLASS_WEIGHT = "balanced"

# =============================================================================
# Baseline model configuration
# =============================================================================

DUMMY_STRATEGY = "most_frequent"

LOGISTIC_MAX_ITER = 1000

TREE_CRITERION = "gini"

TREE_MAX_DEPTH = None

# =============================================================================
# Evaluation metrics
# =============================================================================

PRIMARY_METRIC = "macro_f1"

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "precision_macro",
    "recall_macro",
]