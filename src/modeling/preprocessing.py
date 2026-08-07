"""
Preprocessing utilities for the Week 4 modeling framework.
This module prepares participant-level datasets for machine
learning by separating predictors from the target variable,
removing identifier and diagnosis-related columns to prevent
data leakage, identifying feature types, and constructing a
shared preprocessing pipeline.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.modeling.config import (
    PARTICIPANT_ID_COLUMN,
    TARGET_COLUMN,
)

# =============================================================================
# Columns excluded from model training
# =============================================================================

EXCLUDED_COLUMNS = [
    # Participant identifiers
    PARTICIPANT_ID_COLUMN,
    "study_id",
    "duplicate_patient_id",

    # Target variable
    TARGET_COLUMN,

    # Diagnosis-related variables (prevent label leakage)
    "condition_original",
    "condition_group",

    # Metadata
    "questionnaire_name",
]


# =============================================================================
# Modeling features
# =============================================================================

def get_modeling_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the predictor variables used for model training.

    Identifier columns, metadata, and diagnosis-related
    variables are removed to prevent information leakage.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Participant-level dataset.

    Returns
    -------
    pd.DataFrame
        Predictor variables.
    """

    return dataframe.drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore",
    )


# =============================================================================
# Feature / target separation
# =============================================================================

def split_features_and_target(
    dataframe: pd.DataFrame,
):
    """
    Separate predictor variables and target labels.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Participant-level dataset.

    Returns
    -------
    X : pd.DataFrame
        Predictor variables.

    y : pd.Series
        Target labels.
    """

    X = get_modeling_features(
        dataframe,
    )

    y = dataframe[TARGET_COLUMN]

    return X, y


# =============================================================================
# Feature type identification
# =============================================================================

def identify_feature_types(
    X: pd.DataFrame,
):
    """
    Identify numerical and categorical predictor variables.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor variables.

    Returns
    -------
    numerical_features : list
        Numerical feature names.

    categorical_features : list
        Categorical feature names.
    """

    numerical_features = (
        X
        .select_dtypes(
            include=["number"],
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X
        .select_dtypes(
            exclude=["number"],
        )
        .columns
        .tolist()
    )

    return (
        numerical_features,
        categorical_features,
    )


# =============================================================================
# Shared preprocessing pipeline
# =============================================================================

def build_preprocessing_pipeline(
    X: pd.DataFrame,
):
    """
    Build the shared preprocessing pipeline used by all
    baseline machine learning models.

    Numerical features:
        - Median imputation
        - Standardization

    Categorical features:
        - Most-frequent imputation
        - One-hot encoding

    Parameters
    ----------
    X : pd.DataFrame
        Predictor variables.

    Returns
    -------
    ColumnTransformer
        Configured preprocessing pipeline.
    """

    (
        numerical_features,
        categorical_features,
    ) = identify_feature_types(X)

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessing_pipeline = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessing_pipeline


# =============================================================================
# Complete dataset preparation
# =============================================================================

def prepare_dataset(
    dataframe: pd.DataFrame,
):
    """
    Prepare a participant-level dataset for machine learning.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Participant-level dataset.

    Returns
    -------
    X : pd.DataFrame
        Predictor variables.

    y : pd.Series
        Target labels.

    preprocessing_pipeline : ColumnTransformer
        Shared preprocessing pipeline.
    """

    X, y = split_features_and_target(
        dataframe,
    )

    preprocessing_pipeline = (
        build_preprocessing_pipeline(
            X,
        )
    )

    return (
        X,
        y,
        preprocessing_pipeline,
    )