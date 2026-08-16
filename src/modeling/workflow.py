"""
Shared participant-level modeling workflow.

This module executes the common preprocessing,
training, validation, and evaluation pipeline used by
all modality-specific experiments.
"""

from sklearn.pipeline import Pipeline

from src.modeling.preprocessing import prepare_dataset
from src.modeling.evaluation import evaluate_model

# =============================================================================
# Shared workflow
# =============================================================================

def run_model(
    model,
    train_df,
    validation_df,
    test_df=None,
):
    """
    Execute the complete participant-level modeling workflow.

    Parameters
    ----------
    model
        Scikit-learn classifier.

    train_df : pd.DataFrame
        Training dataset.

    validation_df : pd.DataFrame
        Validation dataset.

    test_df : pd.DataFrame, optional
        Independent testing dataset.

    Returns
    -------
    dict
        Model, metrics, reports, and confusion matrices.
    """

    # ==========================================================
    # Prepare datasets
    # ==========================================================

    X_train, y_train, preprocessing = prepare_dataset(
        train_df
    )

    X_validation, y_validation, _ = prepare_dataset(
        validation_df
    )

    # ==========================================================
    # Build pipeline
    # ==========================================================

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "classifier",
                model,
            ),
        ]
    )

    # ==========================================================
    # Train
    # ==========================================================

    pipeline.fit(
        X_train,
        y_train,
    )

    # ==========================================================
    # Validation evaluation
    # ==========================================================

    (
        metrics,
        report,
        confusion,
    ) = evaluate_model(
        pipeline,
        X_validation,
        y_validation,
    )

    results = {
        "model": pipeline,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": confusion,
    }

    # ==========================================================
    # Optional test evaluation
    # ==========================================================

    if test_df is not None:

        X_test, y_test, _ = prepare_dataset(
            test_df
        )

        (
            test_metrics,
            test_report,
            test_confusion,
        ) = evaluate_model(
            pipeline,
            X_test,
            y_test,
        )

        results["test_metrics"] = test_metrics
        results["test_report"] = test_report
        results["test_confusion_matrix"] = test_confusion

    return results


# =============================================================================
# Multiple-model workflow
# =============================================================================

def run_models(
    models,
    train_df,
    validation_df,
    test_df=None,
):
    """
    Train and evaluate multiple machine learning models.

    Parameters
    ----------
    models : dict
        Dictionary of model names and estimators.

    train_df : pd.DataFrame
        Training dataset.

    validation_df : pd.DataFrame
        Validation dataset.

    test_df : pd.DataFrame, optional
        Independent testing dataset.

    Returns
    -------
    dict
        Results for every model.
    """

    results = {}

    for model_name, model in models.items():

        results[model_name] = run_model(
            model=model,
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
        )

    return results