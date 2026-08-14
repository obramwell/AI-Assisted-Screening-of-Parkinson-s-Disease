"""
Hyperparameter tuning utilities used throughout Week 5.

This module provides reusable functions for hyperparameter
optimization, threshold evaluation, and model comparison
across all multimodal datasets.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.modeling.models import (
    get_logistic_regression,
    get_random_forest,
    get_xgboost,
)

from src.modeling.config import (
    RANDOM_STATE,
)

# =============================================================================
# Logistic Regression parameter grid
# =============================================================================

def get_logistic_param_grid():
    """
    Return the parameter grid used to tune
    Logistic Regression.

    Returns
    -------
    dict
        GridSearchCV parameter grid.
    """

    return {

        "feature_selection__k":[10,15,20,25,"all",],
        "classifier__C":[0.01, 0.1, 1,10,],
        "classifier__solver":["lbfgs",],
        "classifier__max_iter":[1000,],
    }


# =============================================================================
# Random Forest parameter grid
# =============================================================================

def get_random_forest_param_grid():
    """
    Return the parameter grid used to tune
    Random Forest.

    Returns
    -------
    dict
        GridSearchCV parameter grid.
    """

    return {
        "feature_selection__k":[10,15,20,25,"all",],
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [
            None,
            10,
            20,
        ],
        "classifier__min_samples_split": [
            2,
            5,
            10,
        ],
        "classifier__min_samples_leaf": [
            1,
            2,
            4,
        ],
        "classifier__max_features": [
            "sqrt",
            "log2",
        ],
    }


# =============================================================================
# XGBoost parameter grid
# =============================================================================

def get_xgboost_param_grid():
    """
    Return the parameter grid used to tune
    XGBoost.

    Returns
    -------
    dict
        GridSearchCV parameter grid.
    """

    return {
        "feature_selection__k":[10,15,20,25,"all",],
        "classifier__n_estimators": [
            100,
            200,
        ],
        "classifier__learning_rate": [
            0.01,
            0.10,
        ],
        "classifier__max_depth": [
            3,
            5,
            7,
        ],
        "classifier__subsample": [
            0.8,
            1.0,
        ],
        "classifier__colsample_bytree": [
            0.8,
            1.0,
        ],
    }


# =============================================================================
# Model factory
# =============================================================================

def get_model_and_grid(
    model_name,
):
    """
    Return a baseline model together with
    its corresponding parameter grid.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    tuple
        (model, parameter_grid)
    """

    model_name = model_name.lower()

    if model_name == "logistic_regression":
        return (
            get_logistic_regression(),
            get_logistic_param_grid(),
        )

    if model_name == "random_forest":
        return (
            get_random_forest(),
            get_random_forest_param_grid(),
        )

    if model_name == "xgboost":
        return (
            get_xgboost(),
            get_xgboost_param_grid(),
        )

    raise ValueError(
        f"Unsupported model: {model_name}"
    )
"""
Hyperparameter tuning utilities.

This module performs exhaustive hyperparameter optimization
using GridSearchCV and grouped cross-validation.
"""

import pandas as pd

from sklearn.model_selection import GridSearchCV


# =============================================================================
# Hyperparameter tuning
# =============================================================================

def tune_model(
    pipeline,
    param_grid,
    X,
    y,
    cv,
):
    """
    Perform hyperparameter tuning using GridSearchCV.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Complete preprocessing and modeling pipeline.

    param_grid : dict
        Hyperparameter search space.

    X : pd.DataFrame
        Predictor variables.

    y : pd.Series
        Target labels.

    cv
        Cross-validation strategy.

    Returns
    -------
    best_model
        Best fitted estimator.

    best_parameters : dict
        Best hyperparameter combination.

    best_metrics : dict
        Cross-validation performance of the selected model.

    cv_results : pd.DataFrame
        Complete GridSearchCV results.
    """

    scoring = {

        "macro_f1": "f1_macro",

        "balanced_accuracy": "balanced_accuracy",

        "accuracy": "accuracy",

        "precision_macro": "precision_macro",

        "recall_macro": "recall_macro",

    }

    grid_search = GridSearchCV(

        estimator=pipeline,

        param_grid=param_grid,

        scoring=scoring,

        refit="macro_f1",

        cv=cv,

        n_jobs=-1,

        return_train_score=True,

    )

    grid_search.fit(
        X,
        y,
    )

    best_index = grid_search.best_index_

    best_model = grid_search.best_estimator_

    best_parameters = grid_search.best_params_

    best_metrics = {

        "macro_f1":
            grid_search.cv_results_[
                "mean_test_macro_f1"
            ][best_index],

        "macro_f1_std":
            grid_search.cv_results_[
                "std_test_macro_f1"
            ][best_index],

        "balanced_accuracy":
            grid_search.cv_results_[
                "mean_test_balanced_accuracy"
            ][best_index],

        "balanced_accuracy_std":
            grid_search.cv_results_[
                "std_test_balanced_accuracy"
            ][best_index],

        "accuracy":
            grid_search.cv_results_[
                "mean_test_accuracy"
            ][best_index],

        "accuracy_std":
            grid_search.cv_results_[
                "std_test_accuracy"
            ][best_index],

        "precision_macro":
            grid_search.cv_results_[
                "mean_test_precision_macro"
            ][best_index],

        "precision_macro_std":
            grid_search.cv_results_[
                "std_test_precision_macro"
            ][best_index],

        "recall_macro":
            grid_search.cv_results_[
                "mean_test_recall_macro"
            ][best_index],

        "recall_macro_std":
            grid_search.cv_results_[
                "std_test_recall_macro"
            ][best_index],

        "mean_train_macro_f1":
            grid_search.cv_results_[
                "mean_train_macro_f1"
            ][best_index],

    }

    cv_results = pd.DataFrame(
        grid_search.cv_results_
    )

    return (

        best_model,

        best_parameters,

        best_metrics,

        cv_results,

    )
# =============================================================================
# Model evaluation
# =============================================================================

def evaluate_model(
    model,
    X,
    y,
):
    """
    Evaluate a trained model.

    Parameters
    ----------
    model
        Trained estimator.

    X : pd.DataFrame

    y : pd.Series

    Returns
    -------
    dict
        Evaluation metrics.
    """

    predictions = model.predict(X)

    results = {
        "accuracy": accuracy_score(
            y,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y,
            predictions,
        ),
        "macro_f1": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "precision_macro": precision_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }

    return results

# =============================================================================
# Baseline vs Tuned Model Comparison
# =============================================================================

def compare_baseline_vs_tuned(
    baseline_metrics,
    tuned_metrics,
):
    """
    Compare baseline and tuned model performance.

    Parameters
    ----------
    baseline_metrics : dict
        Evaluation metrics obtained from the baseline model.

    tuned_metrics : dict
        Evaluation metrics obtained from the tuned model.

    Returns
    -------
    pd.DataFrame
        Comparison table containing baseline performance,
        tuned performance, and absolute improvement.
    """

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "precision_macro",
        "recall_macro",
    ]

    comparison = []

    for metric in metric_names:

        baseline = baseline_metrics.get(metric, np.nan)

        tuned = tuned_metrics.get(metric, np.nan)

        comparison.append({

            "Metric": metric,

            "Baseline": baseline,

            "Tuned": tuned,

            "Improvement": tuned - baseline,

        })

    comparison = pd.DataFrame(comparison)

    comparison[
        ["Baseline", "Tuned", "Improvement"]
    ] = comparison[
        ["Baseline", "Tuned", "Improvement"]
    ].round(4)

    return comparison
# =============================================================================
# Probability threshold evaluation
# =============================================================================

def evaluate_thresholds(
    model,
    X,
    y,
):
    """
    Placeholder for probability-threshold analysis.

    Notes
    -----
    The current project addresses a three-class
    classification problem (Healthy, Parkinson's Disease,
    and Other Movement Disorders).

    Standard probability-threshold adjustment is primarily
    applicable to binary classification. Because predictions
    in the current implementation are generated using the
    highest predicted class probability (argmax), threshold
    optimization is not directly applied.

    Future work may extend this function using a one-vs-rest
    strategy if threshold optimization for a specific class
    is required.

    Returns
    -------
    None
    """

    return None


# =============================================================================
# Final model ranking
# =============================================================================

def rank_models(
    results,
    metric="macro_f1",
):
    """
    Rank candidate models.

    Parameters
    ----------
    results : pd.DataFrame

    metric : str

    Returns
    -------
    pd.DataFrame
    """

    return (
        results
        .sort_values(
            metric,
            ascending=False,
        )
        .reset_index(drop=True)
    )


# =============================================================================
# Best parameter table
# =============================================================================

def extract_best_parameters(
    model_name,
    grid_search,
):
    """
    Convert the best parameter
    combination into a DataFrame.

    Parameters
    ----------
    model_name : str

    grid_search : GridSearchCV

    Returns
    -------
    pd.DataFrame
    """

    parameters = (
        pd.DataFrame(
            [grid_search.best_params_]
        )
    )

    parameters.insert(
        0,
        "model",
        model_name,
    )

    parameters["best_cv_score"] = (
        grid_search.best_score_
    )

    return parameters
# =============================================================================
# Combine tuning results
# =============================================================================

def combine_tuning_results(
    results,
):
    """
    Combine tuning results from
    multiple models.

    Parameters
    ----------
    results : list

    Returns
    -------
    pd.DataFrame
    """

    return (
        pd.DataFrame(results)
        .sort_values(
            "macro_f1",
            ascending=False,
        )
        .reset_index(drop=True)
    )
# =============================================================================
# Final shortlist
# =============================================================================

def select_final_models(
    ranking,
    top_n=3,
):
    """
    Select the strongest candidate models.

    Parameters
    ----------
    ranking : pd.DataFrame

    top_n : int

    Returns
    -------
    pd.DataFrame
    """

    return ranking.head(top_n).copy()