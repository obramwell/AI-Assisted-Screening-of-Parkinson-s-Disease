"""
Configuration settings for modeling framework.

This module centralizes project paths, dataset locations,
column names, random seeds, modeling parameters,
and evaluation settings.
"""
from pathlib import Path
# =============================================================================
# Project paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = (
    DATA_DIR /
    "processed"
)
OUTPUTS_DIR = (
    PROJECT_ROOT /
    "outputs"
)
METRICS_DIR = (
    OUTPUTS_DIR /
    "metrics"
)
FIGURES_DIR = (
    OUTPUTS_DIR /
    "figures"
)
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
# Dataset split
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
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "precision_macro",
    "recall_macro",
]
# =============================================================================
# Class imbalance
# =============================================================================
USE_CLASS_WEIGHT = True
CLASS_WEIGHT = "balanced"
# =============================================================================
# Dummy classifier
# =============================================================================
DUMMY_STRATEGY = "most_frequent"
# =============================================================================
# Logistic Regression
# =============================================================================
LOGISTIC_MAX_ITER = 1000
# =============================================================================
# Random Forest
# =============================================================================
RF_N_ESTIMATORS = 100
RF_CRITERION = "gini"
RF_MAX_DEPTH = 5
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 5
RF_MAX_FEATURES = "sqrt"