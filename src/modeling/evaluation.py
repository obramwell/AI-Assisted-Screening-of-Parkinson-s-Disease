"""
Model evaluation utilities for the Week 4 modeling framework.
This module provides reusable functions for evaluating baseline
classification models using a common set of performance metrics.
"""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


# =============================================================================
# Metric calculation
# =============================================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Calculate the standard evaluation metrics used throughout
    Week 4.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.

    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
        ),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }

    return metrics


# =============================================================================
# Classification report
# =============================================================================

def get_classification_report(
    y_true,
    y_pred,
):
    """
    Generate the sklearn classification report.

    Returns
    -------
    pd.DataFrame
    """

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report).transpose()


# =============================================================================
# Confusion matrix
# =============================================================================

def get_confusion_matrix(
    y_true,
    y_pred,
):
    """
    Compute the confusion matrix.

    Returns
    -------
    pd.DataFrame
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    return pd.DataFrame(matrix)


# =============================================================================
# Complete evaluation
# =============================================================================

def evaluate_model(
    model,
    X,
    y,
):
    """
    Evaluate a trained classification model.

    Parameters
    ----------
    model
        Trained scikit-learn estimator.

    X : pd.DataFrame
        Predictor variables.

    y : pd.Series
        True labels.

    Returns
    -------
    metrics : dict

    report : pd.DataFrame

    confusion : pd.DataFrame
    """

    predictions = model.predict(X)

    metrics = calculate_metrics(
        y,
        predictions,
    )

    report = get_classification_report(
        y,
        predictions,
    )

    confusion = get_confusion_matrix(
        y,
        predictions,
    )

    return (
        metrics,
        report,
        confusion,
    )