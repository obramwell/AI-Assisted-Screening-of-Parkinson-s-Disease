"""
Preprocessing utilities for raw PADS smartwatch movement recordings.

Pipeline
--------
1. Load and validate the raw smartwatch recording.
2. Remove the slowly varying accelerometer trend using L1 trend filtering.
3. Remove the first 48 samples associated with the initial notification
   vibration.
4. Re-zero the time channel.
5. Calculate accelerometer and gyroscope vector magnitudes.

CLARABEL is used consistently for all L1 trend-filter optimizations.

Raw source files are never modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.magnitude import add_signal_magnitudes
from src.preprocessing.l1_trend_filter import (
    DEFAULT_LAMBDA,
    l1_trend_filter,
)


# ============================================================
# Constants
# ============================================================

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

GYRO_COLUMNS = [
    "Gyroscope_X",
    "Gyroscope_Y",
    "Gyroscope_Z",
]

INITIAL_SAMPLES_TO_REMOVE = 48
L1_LAMBDA = DEFAULT_LAMBDA


# ============================================================
# Load raw signal
# ============================================================

def load_signal(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Load and validate one raw PADS smartwatch recording.

    Parameters
    ----------
    file_path : str or Path
        Path to a raw .txt movement recording.

    Returns
    -------
    pd.DataFrame
        Raw signal with named channels.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Signal file not found: {file_path}"
        )

    signal = pd.read_csv(
        file_path,
        header=None,
    )

    if signal.empty:
        raise ValueError(
            f"Signal file is empty: {file_path.name}"
        )

    if signal.shape[1] != len(SIGNAL_COLUMNS):
        raise ValueError(
            f"Expected {len(SIGNAL_COLUMNS)} columns but found "
            f"{signal.shape[1]} in {file_path.name}."
        )

    signal.columns = SIGNAL_COLUMNS

    for column in SIGNAL_COLUMNS:
        signal[column] = pd.to_numeric(
            signal[column],
            errors="coerce",
        )

    if signal.isna().any().any():
        raise ValueError(
            f"Missing or non-numeric values found in "
            f"{file_path.name}."
        )

    if not np.isfinite(
        signal.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"Non-finite values found in {file_path.name}."
        )

    return signal


# ============================================================
# Accelerometer trend removal
# ============================================================

def remove_accelerometer_trend(
    signal: pd.DataFrame,
    vlambda: float = L1_LAMBDA,
    return_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Remove the slowly varying trend from accelerometer X, Y, and Z.

    A separate L1 optimization is solved for each accelerometer axis.

    Parameters
    ----------
    signal : pd.DataFrame
        Input signal.
    vlambda : float, default=50
        L1 trend-filter regularization parameter.
    return_info : bool, default=False
        Return solver metadata for each axis when True.

    Returns
    -------
    pd.DataFrame
        Signal with accelerometer trends removed.
    """
    missing_columns = [
        column
        for column in ACC_COLUMNS
        if column not in signal.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing accelerometer columns: {missing_columns}"
        )

    output = signal.copy()

    axis_info: dict[str, Any] = {}

    for column in ACC_COLUMNS:

        values = output[
            column
        ].to_numpy(dtype=float)

        trend, info = l1_trend_filter(
            values,
            vlambda=vlambda,
            verbose=False,
            return_info=True,
        )

        output[column] = (
            values - trend
        )

        axis_info[column] = info

    if return_info:
        return output, axis_info

    return output


# ============================================================
# Initial vibration removal
# ============================================================

def remove_initial_vibration(
    signal: pd.DataFrame,
    n_samples: int = INITIAL_SAMPLES_TO_REMOVE,
) -> pd.DataFrame:
    """
    Remove the initial smartwatch notification-vibration period.

    Parameters
    ----------
    signal : pd.DataFrame
        Input signal.
    n_samples : int, default=48
        Number of initial samples removed.

    Returns
    -------
    pd.DataFrame
        Trimmed signal with time reset to zero.
    """
    if n_samples < 0:
        raise ValueError(
            "n_samples must be greater than or equal to zero."
        )

    if len(signal) <= n_samples:
        raise ValueError(
            f"Signal contains {len(signal)} samples and cannot "
            f"remove the first {n_samples}."
        )

    output = (
        signal
        .iloc[n_samples:]
        .copy()
        .reset_index(drop=True)
    )

    output["Time"] = (
        output["Time"]
        - output["Time"].iloc[0]
    )

    return output


# ============================================================
# Complete preprocessing pipeline
# ============================================================

def preprocess_signal(
    signal: pd.DataFrame,
    vlambda: float = L1_LAMBDA,
    return_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply the complete project preprocessing pipeline.

    The same preprocessing procedure and CLARABEL solver are used
    consistently for every recording.

    Parameters
    ----------
    signal : pd.DataFrame
        Raw PADS smartwatch signal.
    vlambda : float, default=50
        L1 trend-filter regularization parameter.
    return_info : bool, default=False
        Return preprocessing metadata when True.

    Returns
    -------
    pd.DataFrame
        Preprocessed signal.
    """
    missing_columns = (
        set(SIGNAL_COLUMNS)
        - set(signal.columns)
    )

    if missing_columns:
        raise ValueError(
            "Signal is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if signal.empty:
        raise ValueError(
            "Cannot preprocess an empty signal."
        )

    if signal.isna().any().any():
        raise ValueError(
            "Input signal contains missing values."
        )

    if not np.isfinite(
        signal[SIGNAL_COLUMNS]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Input signal contains non-finite values."
        )

    raw_samples = len(signal)

    # --------------------------------------------------------
    # 1. L1 accelerometer trend removal
    # --------------------------------------------------------

    output, axis_info = remove_accelerometer_trend(
        signal,
        vlambda=vlambda,
        return_info=True,
    )

    # --------------------------------------------------------
    # 2. Remove first 48 samples
    # --------------------------------------------------------

    output = remove_initial_vibration(
        output,
        n_samples=INITIAL_SAMPLES_TO_REMOVE,
    )

    # --------------------------------------------------------
    # 3. Add accelerometer and gyroscope magnitudes
    # --------------------------------------------------------

    output = add_signal_magnitudes(
        output
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    expected_samples = (
        raw_samples
        - INITIAL_SAMPLES_TO_REMOVE
    )

    if len(output) != expected_samples:
        raise RuntimeError(
            f"Unexpected processed signal length. "
            f"Expected {expected_samples}, found {len(output)}."
        )

    if output.isna().any().any():
        raise RuntimeError(
            "Preprocessing generated missing values."
        )

    if not np.isfinite(
        output
        .select_dtypes(include="number")
        .to_numpy()
    ).all():
        raise RuntimeError(
            "Preprocessing generated non-finite values."
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    info = {
        "raw_samples": raw_samples,
        "processed_samples": len(output),
        "initial_samples_removed":
            INITIAL_SAMPLES_TO_REMOVE,
        "lambda": vlambda,
        "solver": "CLARABEL",
        "axis_solvers": axis_info,
        "processed_columns": list(output.columns),
    }

    if return_info:
        return output, info

    return output