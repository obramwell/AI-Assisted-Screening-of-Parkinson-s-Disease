"""
Preprocessing utilities for raw PADS smartwatch movement recordings.

The preprocessing is based on the original PADS signal-processing
procedure while preserving all neurological assessment tasks for the
current project's task-specific feature analysis.

Implemented steps:
1. Load raw smartwatch time-series data.
2. Remove the first 48 samples associated with the initial
   smartwatch notification vibration.
3. Estimate and remove the slowly varying accelerometer trend
   using L1 trend filtering (lambda = 50).
4. Calculate accelerometer and gyroscope vector magnitudes.

Raw files are never modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.l1_trend_filter import l1_trend_filter
from src.features.magnitude import add_signal_magnitudes


SIGNAL_COLUMNS = [
    "Time",
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
    "Gyroscope_X",
    "Gyroscope_Y",
    "Gyroscope_Z",
]

ACC_COLUMNS = [
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
]

INITIAL_SAMPLES_TO_REMOVE = 48
L1_LAMBDA = 50


def load_signal(file_path: str | Path) -> pd.DataFrame:
    """
    Load one raw PADS smartwatch time-series recording.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Signal file not found: {file_path}")

    signal = pd.read_csv(
        file_path,
        header=None,
        names=SIGNAL_COLUMNS,
    )

    if signal.empty:
        raise ValueError(f"Signal file is empty: {file_path}")

    if signal.isna().any().any():
        raise ValueError(f"Missing values found in signal: {file_path.name}")

    if not np.isfinite(signal.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite values found in signal: {file_path.name}")

    return signal


def remove_initial_vibration(
    signal: pd.DataFrame,
    n_samples: int = INITIAL_SAMPLES_TO_REMOVE,
) -> pd.DataFrame:
    """
    Remove the initial smartwatch notification-vibration period.

    The original PADS preprocessing removes the first 48 samples.
    """
    if len(signal) <= n_samples:
        raise ValueError(
            f"Signal has {len(signal)} samples and cannot remove "
            f"the first {n_samples}."
        )

    output = signal.iloc[n_samples:].copy().reset_index(drop=True)

    # Re-zero timestamps while preserving their original spacing.
    output["Time"] = output["Time"] - output["Time"].iloc[0]

    return output


def remove_accelerometer_trend(
    signal: pd.DataFrame,
    vlambda: float = L1_LAMBDA,
) -> pd.DataFrame:
    """
    Remove the slowly varying component from accelerometer channels.

    A separate L1 trend is estimated for each accelerometer axis and
    subtracted from the original signal, following the original PADS
    preprocessing approach.
    """
    output = signal.copy()

    for column in ACC_COLUMNS:
        values = output[column].to_numpy(dtype=float)

        trend = l1_trend_filter(
            values,
            vlambda=vlambda,
            verbose=False,
        )

        output[column] = values - trend

    return output


def preprocess_signal(
    signal: pd.DataFrame,
    remove_gravity: bool = True,
    remove_vibration: bool = True,
    add_magnitude: bool = True,
) -> pd.DataFrame:
    """Apply the project preprocessing pipeline to one raw recording."""

    output = signal.copy()

    # 1. Remove accelerometer gravitational / slow trend
    if remove_gravity:
        output = remove_accelerometer_trend(output)

    # 2. Remove first 48 samples associated with notification vibration
    if remove_vibration:
        output = remove_initial_vibration(output)

    # 3. Calculate vector magnitudes from the processed signals
    if add_magnitude:
        output = add_signal_magnitudes(output)

    return output