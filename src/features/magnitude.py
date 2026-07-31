"""Utilities for calculating wearable sensor vector magnitudes."""

import numpy as np
import pandas as pd


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


def calculate_vector_magnitude(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    """
    Calculate Euclidean magnitude from three sensor axes.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing sensor channels.
    columns : list[str]
        X, Y, and Z channel names.

    Returns
    -------
    pd.Series
        Vector magnitude for each sample.
    """
    missing = [column for column in columns if column not in data.columns]

    if missing:
        raise ValueError(f"Missing required sensor columns: {missing}")

    values = data[columns].to_numpy(dtype=float)

    magnitude = np.sqrt(np.sum(values**2, axis=1))

    return pd.Series(magnitude, index=data.index)


def add_signal_magnitudes(data: pd.DataFrame) -> pd.DataFrame:
    """Add accelerometer and gyroscope magnitude columns."""

    output = data.copy()

    output["Acc_Magnitude"] = calculate_vector_magnitude(
        output,
        ACC_COLUMNS,
    )

    output["Gyro_Magnitude"] = calculate_vector_magnitude(
        output,
        GYRO_COLUMNS,
    )

    return output