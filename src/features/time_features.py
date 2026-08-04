"""
Time-domain feature extraction for preprocessed PADS smartwatch signals.

This module extracts descriptive statistics from each accelerometer,
gyroscope, and magnitude signal after preprocessing.

The extracted features are intended for machine-learning model
development and feature-quality assessment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# Signal columns
# ============================================================

ACC_COLUMNS = [
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
]

GYRO_COLUMNS = [
    "Gyroscope_X",
    "Gyroscope_Y",
    "Gyroscope_Z",
]

MAG_COLUMNS = [
    "Acc_Magnitude",
    "Gyro_Magnitude",
]

FEATURE_COLUMNS = (
    ACC_COLUMNS
    + GYRO_COLUMNS
    + MAG_COLUMNS
)


# ============================================================
# Individual feature calculations
# ============================================================

def root_mean_square(signal: np.ndarray) -> float:
    """Compute root mean square."""
    return np.sqrt(np.mean(signal ** 2))


def signal_energy(signal: np.ndarray) -> float:
    """Compute signal energy."""
    return np.sum(signal ** 2)


# ============================================================
# Feature extraction
# ============================================================

def extract_time_features(
    signal: pd.DataFrame,
) -> pd.Series:
    """
    Extract time-domain features from one preprocessed recording.

    Parameters
    ----------
    signal : pd.DataFrame
        Preprocessed smartwatch recording.

    Returns
    -------
    pd.Series
        Time-domain features.
    """

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in signal.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if signal.empty:
        raise ValueError(
            "Signal is empty."
        )

    if signal.isna().any().any():
        raise ValueError(
            "Signal contains missing values."
        )

    if not np.isfinite(
        signal[FEATURE_COLUMNS]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Signal contains non-finite values."
        )

    features = {}

    for column in FEATURE_COLUMNS:

        values = signal[column].to_numpy(dtype=float)

        features[f"{column}_Mean"] = np.mean(values)

        features[f"{column}_Median"] = np.median(values)

        features[f"{column}_Std"] = np.std(
            values,
            ddof=1,
        )

        features[f"{column}_Min"] = np.min(values)

        features[f"{column}_Max"] = np.max(values)

        features[f"{column}_Range"] = (
            np.max(values)
            - np.min(values)
        )

        q75, q25 = np.percentile(
            values,
            [75, 25],
        )

        features[f"{column}_IQR"] = q75 - q25

        features[f"{column}_RMS"] = (
            root_mean_square(values)
        )

        features[f"{column}_Energy"] = (
            signal_energy(values)
        )

    return pd.Series(features)


# ============================================================
# Feature-quality checks
# ============================================================

def evaluate_feature_quality(
    feature_table: pd.DataFrame,
    variance_threshold: float = 1e-6,
) -> dict:
    """
    Evaluate the quality of the extracted feature table.

    Parameters
    ----------
    feature_table : pd.DataFrame
        Table containing one row per recording.
    variance_threshold : float
        Threshold used to identify low-variance features.

    Returns
    -------
    dict
        Dictionary containing feature-quality results.
    """

    missing_values = (
        feature_table.isna()
        .sum()
    )

    infinite_values = (
        np.isinf(feature_table)
        .sum()
    )

    variances = feature_table.var(
        numeric_only=True
    )

    constant_features = variances[
        variances == 0
    ].index.tolist()

    low_variance_features = variances[
        (variances > 0)
        & (variances < variance_threshold)
    ].index.tolist()

    invalid_features = list(
        set(
            missing_values[
                missing_values > 0
            ].index.tolist()
            +
            infinite_values[
                infinite_values > 0
            ].index.tolist()
        )
    )

    summary = pd.DataFrame({

        "Metric": [

            "Total features",

            "Missing values",

            "Infinite values",

            "Constant features",

            "Low variance features",

            "Invalid features",

        ],

        "Value": [

            feature_table.shape[1],

            int(
                (missing_values > 0).sum()
            ),

            int(
                (infinite_values > 0).sum()
            ),

            len(constant_features),

            len(low_variance_features),

            len(invalid_features),

        ],

    })

    return {

        "summary": summary,

        "constant_features": constant_features,

        "low_variance_features": low_variance_features,

        "invalid_features": invalid_features,

        "variance": variances,

        "missing_values": missing_values,

        "infinite_values": infinite_values,

    }