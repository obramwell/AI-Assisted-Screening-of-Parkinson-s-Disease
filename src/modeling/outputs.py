"""
Output utilities for the Week 4 modeling framework.

This module provides reusable functions for saving model
evaluation results, classification reports, and confusion
matrices.
"""

from pathlib import Path

import pandas as pd

from src.modeling.config import (
    METRICS_DIR,
    FIGURES_DIR,
)

# =============================================================================
# Create output directories
# =============================================================================

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Save metrics
# =============================================================================

def save_metrics(
    metrics: dict,
    filename: str,
):
    """
    Save evaluation metrics.

    Parameters
    ----------
    metrics : dict
        Dictionary containing evaluation metrics.

    filename : str
        Output CSV filename.
    """

    output_file = (
        METRICS_DIR /
        filename
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        output_file,
        index=False,
    )


# =============================================================================
# Save classification report
# =============================================================================

def save_classification_report(
    report: pd.DataFrame,
    filename: str,
):
    """
    Save a classification report.

    Parameters
    ----------
    report : pd.DataFrame
        Classification report.

    filename : str
        Output CSV filename.
    """

    output_file = (
        METRICS_DIR /
        filename
    )

    report.to_csv(
        output_file,
        index=True,
    )


# =============================================================================
# Save confusion matrix
# =============================================================================

def save_confusion_matrix(
    confusion_matrix: pd.DataFrame,
    filename: str,
):
    """
    Save a confusion matrix.

    Parameters
    ----------
    confusion_matrix : pd.DataFrame
        Confusion matrix.

    filename : str
        Output CSV filename.
    """

    output_file = (
        METRICS_DIR /
        filename
    )

    confusion_matrix.to_csv(
        output_file,
        index=True,
    )