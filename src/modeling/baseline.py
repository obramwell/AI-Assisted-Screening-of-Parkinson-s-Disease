"""
Baseline classifier utilities.
This module implements the baseline classifier used as the
reference model for all modality-specific experiments.
"""

from sklearn.dummy import DummyClassifier

from src.modeling.config import (
    RANDOM_STATE,
    DUMMY_STRATEGY,
)


# =============================================================================
# Dummy classifier
# =============================================================================

def get_dummy_classifier():
    """
    Create the baseline Dummy Classifier.

    Returns
    -------
    DummyClassifier
        Configured dummy classifier.
    """

    model = DummyClassifier(
        strategy=DUMMY_STRATEGY,
        random_state=RANDOM_STATE,
    )

    return model


# =============================================================================
# Train baseline
# =============================================================================

def train_dummy_classifier(
    X_train,
    y_train,
):
    """
    Train the baseline Dummy Classifier.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training labels.

    Returns
    -------
    DummyClassifier
        Trained classifier.
    """

    model = get_dummy_classifier()

    model.fit(
        X_train,
        y_train,
    )

    return model