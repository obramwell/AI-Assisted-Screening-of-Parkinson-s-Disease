"""
Output utilities for the modeling framework.

This module provides reusable functions for saving model
evaluation results, classification reports, confusion
matrices, comparison tables, and figures.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

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
    """

    output_file = METRICS_DIR / filename

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
    """

    output_file = METRICS_DIR / filename

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
    """

    output_file = METRICS_DIR / filename

    confusion_matrix.to_csv(
        output_file,
        index=True,
    )


# =============================================================================
# Save comparison table
# =============================================================================

def save_model_comparison(
    comparison: pd.DataFrame,
    filename: str,
):
    """
    Save a model comparison table.
    """

    output_file = METRICS_DIR / filename

    comparison.to_csv(
        output_file,
        index=True,
    )


# =============================================================================
# Save generic dataframe
# =============================================================================

def save_table(
    table: pd.DataFrame,
    filename: str,
):
    """
    Save a generic dataframe.
    """

    output_file = METRICS_DIR / filename

    table.to_csv(
        output_file,
        index=False,
    )


# =============================================================================
# Save figure
# =============================================================================

def save_figure(
    figure,
    filename: str,
):
    """
    Save a matplotlib figure.
    """

    output_file = FIGURES_DIR / filename

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )