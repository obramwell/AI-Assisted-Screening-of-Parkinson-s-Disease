"""
Baseline machine learning models used throughout Week 4.

This module provides reusable model constructors for all
modality-specific experiments.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.modeling.config import (
    RANDOM_STATE,
    CLASS_WEIGHT,
    USE_CLASS_WEIGHT,
    LOGISTIC_MAX_ITER,
    RF_N_ESTIMATORS,
    RF_CRITERION,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_SPLIT,
    RF_MIN_SAMPLES_LEAF,
    RF_MAX_FEATURES,
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
    solver="lbfgs",
    max_iter=LOGISTIC_MAX_ITER,
    random_state=RANDOM_STATE,
    class_weight=CLASS_WEIGHT if USE_CLASS_WEIGHT else None,
)

    return model
# =============================================================================
# Random Forest
# =============================================================================
def get_random_forest():
    """
    Create a Random Forest classifier.

    Returns
    -------
    RandomForestClassifier
        Configured classifier.
    """

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        criterion=RF_CRITERION,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
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