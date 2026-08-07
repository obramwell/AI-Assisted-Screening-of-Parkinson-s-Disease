"""
Configuration settings for the Week 4 modeling framework.

This module centralizes project paths, dataset filenames,
column names, random seeds, and modeling parameters used
throughout the baseline modeling framework.
"""

from pathlib import Path

# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# =============================================================================
# Input datasets
# =============================================================================

TRAIN_DATASET = (
    PROCESSED_DIR /
    "train_participant_dataset.csv"
)

VALIDATION_DATASET = (
    PROCESSED_DIR /
    "validation_participant_dataset.csv"
)

TEST_DATASET = (
    PROCESSED_DIR /
    "test_participant_dataset.csv"
)

# =============================================================================
# Column names
# =============================================================================

PARTICIPANT_ID_COLUMN = "patient_id"

TARGET_COLUMN = "label"

DIAGNOSIS_COLUMNS = [
    "condition_original",
    "condition_group",
]

# =============================================================================
# Train / validation / test split
# =============================================================================

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# =============================================================================
# Reproducibility
# =============================================================================

RANDOM_STATE = 42

# =============================================================================
# Cross-validation
# =============================================================================

N_SPLITS = 5

# =============================================================================
# Evaluation metrics
# =============================================================================

PRIMARY_METRIC = "macro_f1"

CLASSIFICATION_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "precision_macro",
    "recall_macro",
]