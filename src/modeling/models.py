"""
Baseline machine learning models used throughout Week 4.

This module provides reusable model constructors for all
modality-specific experiments.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.modeling.config import (
    RANDOM_STATE,
    CLASS_WEIGHT,
    USE_CLASS_WEIGHT,
    LOGISTIC_MAX_ITER,
    TREE_CRITERION,
    TREE_MAX_DEPTH,
)


# =============================================================================
# Logistic Regression
# =============================================================================

def get_logistic_regression():
    """
    Create a multinomial Logistic Regression classifier.

    Returns
    -------
    LogisticRegression
        Configured classifier.
    """

    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITER,
        random_state=RANDOM_STATE,
        class_weight=CLASS_WEIGHT if USE_CLASS_WEIGHT else None,
    )

    return model


# =============================================================================
# Decision Tree
# =============================================================================

def get_decision_tree():
    """
    Create a Decision Tree classifier.

    Returns
    -------
    DecisionTreeClassifier
        Configured classifier.
    """

    model = DecisionTreeClassifier(
        criterion=TREE_CRITERION,
        max_depth=TREE_MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight=CLASS_WEIGHT if USE_CLASS_WEIGHT else None,
    )

    return model


# =============================================================================
# Training utilities
# =============================================================================

def train_model(
    model,
    X_train,
    y_train,
):
    """
    Train a machine learning model.

    Parameters
    ----------
    model
        Scikit-learn estimator.

    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training labels.

    Returns
    -------
    estimator
        Trained model.
    """

    model.fit(
        X_train,
        y_train,
    )

    return model