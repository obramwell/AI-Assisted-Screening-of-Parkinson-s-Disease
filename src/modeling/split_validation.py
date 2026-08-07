"""
Utilities for validating the participant-level train,
validation, and test split.
These functions verify that the predefined dataset partitions
satisfy the project requirements before any model training
begins.
"""

import pandas as pd

from src.modeling.config import (
    PARTICIPANT_ID_COLUMN,
    TARGET_COLUMN,
)


# =============================================================================
# Participant overlap
# =============================================================================

def verify_no_participant_overlap(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> bool:
    """
    Verify that no participant appears in more than one dataset.
    """

    train_ids = set(train_df[PARTICIPANT_ID_COLUMN])

    validation_ids = set(validation_df[PARTICIPANT_ID_COLUMN])

    test_ids = set(test_df[PARTICIPANT_ID_COLUMN])

    return (
        train_ids.isdisjoint(validation_ids)
        and train_ids.isdisjoint(test_ids)
        and validation_ids.isdisjoint(test_ids)
    )


# =============================================================================
# Participant counts
# =============================================================================

def participant_counts(
    train_df,
    validation_df,
    test_df,
):
    """
    Return the number of unique participants in each split.
    """

    return {
        "train": train_df[PARTICIPANT_ID_COLUMN].nunique(),
        "validation": validation_df[PARTICIPANT_ID_COLUMN].nunique(),
        "test": test_df[PARTICIPANT_ID_COLUMN].nunique(),
    }


# =============================================================================
# Class distribution
# =============================================================================

def class_distribution(df):
    """
    Return the class distribution for a dataset.
    """

    return (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )


# =============================================================================
# Missing values
# =============================================================================

def missing_values(df):
    """
    Count missing values in each column.
    """

    return df.isna().sum()


# =============================================================================
# Duplicate participants
# =============================================================================

def duplicate_participants(df):
    """
    Return duplicated participant IDs, if any.
    """

    duplicated = df[
        df[PARTICIPANT_ID_COLUMN]
        .duplicated()
    ]

    return duplicated


# =============================================================================
# Complete validation
# =============================================================================

def validate_split(
    train_df,
    validation_df,
    test_df,
):
    """
    Perform the complete participant-level split validation.

    Returns
    -------
    dict
        Validation summary.
    """

    summary = {}

    summary["no_participant_overlap"] = (
        verify_no_participant_overlap(
            train_df,
            validation_df,
            test_df,
        )
    )

    summary["participant_counts"] = (
        participant_counts(
            train_df,
            validation_df,
            test_df,
        )
    )

    summary["train_class_distribution"] = (
        class_distribution(train_df)
    )

    summary["validation_class_distribution"] = (
        class_distribution(validation_df)
    )

    summary["test_class_distribution"] = (
        class_distribution(test_df)
    )

    return summary