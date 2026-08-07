"""
Shared modeling workflow for baseline experiments.
This module executes the complete machine learning workflow,
including preprocessing, model training, prediction, and
evaluation.
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
):
    """
    Execute the complete machine learning workflow for a single model.

    Parameters
    ----------
    model
        Scikit-learn classifier.

    train_df : pd.DataFrame
        Training dataset.

    validation_df : pd.DataFrame
        Validation dataset.

    Returns
    -------
    dict
        Dictionary containing the trained pipeline,
        evaluation metrics, classification report,
        and confusion matrix.
    """

    # ==============================================================
    # Prepare datasets
    # ==============================================================

    X_train, y_train, preprocessing = prepare_dataset(
        train_df
    )

    X_validation, y_validation, _ = prepare_dataset(
        validation_df
    )

    # ==============================================================
    # Build pipeline
    # ==============================================================

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

    # ==============================================================
    # Train model
    # ==============================================================

    pipeline.fit(
        X_train,
        y_train,
    )

    # ==============================================================
    # Evaluate model
    # ==============================================================

    (
        metrics,
        report,
        confusion,
    ) = evaluate_model(
        pipeline,
        X_validation,
        y_validation,
    )

    return {
        "model": pipeline,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": confusion,
    }


# =============================================================================
# Multiple-model workflow
# =============================================================================

def run_models(
    models,
    train_df,
    validation_df,
):
    """
    Train and evaluate multiple machine learning models.

    Parameters
    ----------
    models : dict
        Dictionary where the key is the model name and
        the value is a scikit-learn estimator.

    train_df : pd.DataFrame
        Training dataset.

    validation_df : pd.DataFrame
        Validation dataset.

    Returns
    -------
    dict
        Dictionary containing the results for every model.
    """

    results = {}

    for model_name, model in models.items():

        results[model_name] = run_model(
            model=model,
            train_df=train_df,
            validation_df=validation_df,
        )

    return results