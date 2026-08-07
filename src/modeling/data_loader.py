"""
Utilities for loading participant-level datasets used during
Week 4 baseline modeling.

All modality-specific notebooks should use these functions to
ensure that identical datasets are loaded across experiments.
"""

import pandas as pd

from src.modeling.config import (
    PROCESSED_DIR,
    TRAIN_DATASET,
    VALIDATION_DATASET,
    TEST_DATASET,
)

# =============================================================================
# Dataset file
# =============================================================================

FULL_DATASET = (
    PROCESSED_DIR /
    "integrated_participant_dataset.csv"
)

# =============================================================================
# Dataset loading
# =============================================================================

def load_full_dataset() -> pd.DataFrame:
    """
    Load the complete participant-level dataset.

    Returns
    -------
    pd.DataFrame
        Complete integrated dataset.
    """

    return pd.read_csv(
        FULL_DATASET,
        dtype={"patient_id": str},
    )


def load_train_dataset() -> pd.DataFrame:
    """
    Load the training dataset.

    Returns
    -------
    pd.DataFrame
        Training dataset.
    """

    return pd.read_csv(
        TRAIN_DATASET,
        dtype={"patient_id": str},
    )


def load_validation_dataset() -> pd.DataFrame:
    """
    Load the validation dataset.

    Returns
    -------
    pd.DataFrame
        Validation dataset.
    """

    return pd.read_csv(
        VALIDATION_DATASET,
        dtype={"patient_id": str},
    )


def load_test_dataset() -> pd.DataFrame:
    """
    Load the testing dataset.

    Returns
    -------
    pd.DataFrame
        Testing dataset.
    """

    return pd.read_csv(
        TEST_DATASET,
        dtype={"patient_id": str},
    )


# =============================================================================
# Convenience loader
# =============================================================================

def load_all_datasets():
    """
    Load all participant-level datasets.

    Returns
    -------
    tuple
        (train_df, validation_df, test_df)
    """

    train_df = load_train_dataset()
    validation_df = load_validation_dataset()
    test_df = load_test_dataset()

    return (
        train_df,
        validation_df,
        test_df,
    )