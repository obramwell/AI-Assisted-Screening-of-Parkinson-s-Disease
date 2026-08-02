"""Frequency-domain feature extraction for preprocessed PADS smartwatch signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


ACCELEROMETER_SIGNALS = (
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
    "Acc_Magnitude",
)

GYROSCOPE_SIGNALS = (
    "Gyroscope_X",
    "Gyroscope_Y",
    "Gyroscope_Z",
    "Gyro_Magnitude",
)

FREQUENCY_METRICS = (
    "dominant_frequency",
    "spectral_centroid",
    "spectral_entropy",
    "spectral_power",
)

TARGET_SAMPLING_FREQUENCY = 100.0


@dataclass(frozen=True)
class Spectrum:
    """One-sided frequency spectrum and power spectral density."""

    frequencies: np.ndarray
    power: np.ndarray
    sampling_frequency: float


def _safe_feature_name(signal_name: str) -> str:
    """Convert source column names to stable snake-case feature prefixes."""
    mapping = {
        "Accelerometer_X": "acc_x",
        "Accelerometer_Y": "acc_y",
        "Accelerometer_Z": "acc_z",
        "Gyroscope_X": "gyro_x",
        "Gyroscope_Y": "gyro_y",
        "Gyroscope_Z": "gyro_z",
        "Acc_Magnitude": "acc_magnitude",
        "Gyro_Magnitude": "gyro_magnitude",
    }
    if signal_name not in mapping:
        raise KeyError(f"Unsupported signal column: {signal_name}")
    return mapping[signal_name]


def infer_sampling_frequency(time_values: np.ndarray) -> float:
    """Estimate sampling frequency for quality-assurance reporting only."""
    time_values = np.asarray(time_values, dtype=float)
    finite_time = time_values[np.isfinite(time_values)]

    if finite_time.size < 2:
        raise ValueError("At least two finite time values are required.")

    differences = np.diff(finite_time)
    valid_differences = differences[
        np.isfinite(differences) & (differences > 0)
    ]

    if valid_differences.size == 0:
        raise ValueError("No positive time differences were found.")

    median_interval = float(np.median(valid_differences))
    sampling_frequency = 1.0 / median_interval

    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("Could not infer a valid sampling frequency.")

    return sampling_frequency


def resample_recording_to_uniform_grid(
    recording: np.ndarray,
    columns: Iterable[str],
    target_sampling_frequency: float = TARGET_SAMPLING_FREQUENCY,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a recording onto a uniform 100 Hz grid.

    The existing post-trim number of samples is preserved exactly, so
    recordings with 976 or 2,000 samples retain those lengths.
    """
    if target_sampling_frequency <= 0:
        raise ValueError("target_sampling_frequency must be positive.")

    column_names = [str(column) for column in columns]
    recording = np.asarray(recording, dtype=float)

    if recording.ndim != 2:
        raise ValueError("Each recording must be a two-dimensional array.")

    if recording.shape[1] != len(column_names):
        raise ValueError(
            "Recording column count does not match the supplied columns."
        )

    if "Time" not in column_names:
        raise ValueError("The recording must contain a Time column.")

    time_index = column_names.index("Time")
    original_time = recording[:, time_index]

    if original_time.size < 2:
        raise ValueError("At least two samples are required for resampling.")

    if not np.isfinite(original_time).all():
        raise ValueError("Time values must all be finite before resampling.")

    relative_time = original_time - original_time[0]
    differences = np.diff(relative_time)

    if np.any(differences <= 0):
        raise ValueError(
            "Time values must be strictly increasing before resampling."
        )

    sample_count = recording.shape[0]
    uniform_time = (
        np.arange(sample_count, dtype=float)
        / float(target_sampling_frequency)
    )

    resampled_recording = np.empty_like(recording, dtype=float)
    resampled_recording[:, time_index] = uniform_time

    for column_index in range(recording.shape[1]):
        if column_index == time_index:
            continue

        signal_values = recording[:, column_index]

        if not np.isfinite(signal_values).all():
            raise ValueError(
                f"Signal column {column_names[column_index]} contains "
                "non-finite values."
            )

        resampled_recording[:, column_index] = np.interp(
            uniform_time,
            relative_time,
            signal_values,
        )

    if resampled_recording.shape != recording.shape:
        raise RuntimeError(
            "Resampling unexpectedly changed the recording shape."
        )

    return uniform_time, resampled_recording


def compute_power_spectrum(
    signal: np.ndarray,
    sampling_frequency: float,
) -> Spectrum:
    """Compute a one-sided power spectrum after centering and Hann windowing."""
    values = np.asarray(signal, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 4:
        return Spectrum(
            frequencies=np.array([], dtype=float),
            power=np.array([], dtype=float),
            sampling_frequency=float(sampling_frequency),
        )

    centered = values - np.mean(values)
    window = np.hanning(centered.size)
    windowed = centered * window

    fft_values = np.fft.rfft(windowed)
    frequencies = np.fft.rfftfreq(
        centered.size,
        d=1.0 / sampling_frequency,
    )

    window_energy = float(np.sum(window**2))

    if window_energy <= 0:
        power = np.zeros_like(frequencies, dtype=float)
    else:
        power = (np.abs(fft_values) ** 2) / (
            sampling_frequency * window_energy
        )

        if centered.size % 2 == 0 and power.size > 2:
            power[1:-1] *= 2.0
        elif power.size > 1:
            power[1:] *= 2.0

    return Spectrum(
        frequencies=frequencies,
        power=power,
        sampling_frequency=float(sampling_frequency),
    )


def _integrate_trapezoid(
    values: np.ndarray,
    coordinates: np.ndarray,
) -> float:
    """Integrate with support for both newer and older NumPy versions."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, coordinates))

    return float(np.trapz(values, coordinates))


def extract_frequency_metrics(
    signal: np.ndarray,
    sampling_frequency: float,
    exclude_dc: bool = True,
) -> dict[str, float]:
    """Extract dominant frequency, centroid, entropy, and spectral power."""
    spectrum = compute_power_spectrum(signal, sampling_frequency)
    frequencies = spectrum.frequencies
    power = spectrum.power

    if frequencies.size == 0 or power.size == 0:
        return {metric: np.nan for metric in FREQUENCY_METRICS}

    finite_mask = np.isfinite(frequencies) & np.isfinite(power)
    frequencies = frequencies[finite_mask]
    power = np.maximum(power[finite_mask], 0.0)

    if exclude_dc:
        positive_mask = frequencies > 0
        frequencies = frequencies[positive_mask]
        power = power[positive_mask]

    if frequencies.size == 0 or power.size == 0:
        return {metric: np.nan for metric in FREQUENCY_METRICS}

    total_power_sum = float(np.sum(power))
    spectral_power = _integrate_trapezoid(power, frequencies)

    if total_power_sum <= 0:
        dominant_frequency = 0.0
        spectral_centroid = 0.0
        spectral_entropy = 0.0
    else:
        dominant_frequency = float(frequencies[int(np.argmax(power))])
        spectral_centroid = float(
            np.sum(frequencies * power) / total_power_sum
        )

        probabilities = power / total_power_sum
        positive_probabilities = probabilities[probabilities > 0]

        if probabilities.size <= 1:
            spectral_entropy = 0.0
        else:
            entropy = -np.sum(
                positive_probabilities * np.log2(positive_probabilities)
            )
            spectral_entropy = float(
                entropy / np.log2(probabilities.size)
            )

    return {
        "dominant_frequency": dominant_frequency,
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy,
        "spectral_power": spectral_power,
    }


def split_recording_key(recording_key: str) -> tuple[str, str]:
    """Split a recording key such as CrossArms__LeftWrist."""
    parts = str(recording_key).split("__", maxsplit=1)

    if len(parts) != 2:
        raise ValueError(
            f"Recording key does not follow 'Task__Wrist': {recording_key}"
        )

    return parts[0], parts[1]


def extract_recording_features(
    recording: np.ndarray,
    columns: Iterable[str],
    patient_id: str,
    recording_key: str,
) -> dict[str, object]:
    """Extract frequency features after uniform 100 Hz interpolation."""
    column_names = [str(column) for column in columns]

    required_columns = (
        "Time",
        *ACCELEROMETER_SIGNALS,
        *GYROSCOPE_SIGNALS,
    )

    missing = [
        column for column in required_columns if column not in column_names
    ]

    if missing:
        raise ValueError(f"Recording is missing required columns: {missing}")

    recording = np.asarray(recording, dtype=float)

    if recording.ndim != 2:
        raise ValueError("Each recording must be a two-dimensional array.")

    if recording.shape[1] != len(column_names):
        raise ValueError(
            "Recording column count does not match __columns__ metadata."
        )

    task, wrist = split_recording_key(recording_key)
    time_index = column_names.index("Time")
    original_time = recording[:, time_index]

    original_sampling_frequency = infer_sampling_frequency(original_time)

    uniform_time, resampled_recording = (
        resample_recording_to_uniform_grid(
            recording=recording,
            columns=column_names,
            target_sampling_frequency=TARGET_SAMPLING_FREQUENCY,
        )
    )

    original_sample_count = int(recording.shape[0])
    resampled_sample_count = int(resampled_recording.shape[0])

    if resampled_sample_count != original_sample_count:
        raise RuntimeError(
            "Uniform resampling changed the post-trim sample count."
        )

    row: dict[str, object] = {
        "patient_id": str(patient_id),
        "recording_key": str(recording_key),
        "task": task,
        "wrist": wrist,
        "n_samples": original_sample_count,
        "original_sampling_frequency_hz": original_sampling_frequency,
        "sampling_frequency_hz": TARGET_SAMPLING_FREQUENCY,
        "resampled_n_samples": resampled_sample_count,
        "uniform_time_start_s": float(uniform_time[0]),
        "uniform_time_end_s": float(uniform_time[-1]),
    }

    for signal_name in (*ACCELEROMETER_SIGNALS, *GYROSCOPE_SIGNALS):
        signal_index = column_names.index(signal_name)

        metrics = extract_frequency_metrics(
            resampled_recording[:, signal_index],
            sampling_frequency=TARGET_SAMPLING_FREQUENCY,
        )

        prefix = _safe_feature_name(signal_name)

        for metric_name, metric_value in metrics.items():
            row[f"{prefix}_{metric_name}"] = metric_value

    return row


def extract_participant_features(npz_path: str | Path) -> pd.DataFrame:
    """Extract one row per task and wrist for one participant."""
    npz_path = Path(npz_path)
    patient_id = npz_path.stem.replace("_preprocessed", "")

    with np.load(npz_path, allow_pickle=False) as participant_data:
        required_metadata = {"__columns__", "__recording_keys__"}
        missing_metadata = required_metadata.difference(participant_data.files)

        if missing_metadata:
            raise ValueError(
                f"{npz_path.name} is missing metadata: "
                f"{sorted(missing_metadata)}"
            )

        columns = participant_data["__columns__"].astype(str).tolist()
        recording_keys = (
            participant_data["__recording_keys__"]
            .astype(str)
            .tolist()
        )

        rows = []

        for recording_key in recording_keys:
            if recording_key not in participant_data.files:
                raise KeyError(
                    f"{recording_key} is listed in __recording_keys__ "
                    f"but is absent from {npz_path.name}."
                )

            rows.append(
                extract_recording_features(
                    recording=participant_data[recording_key],
                    columns=columns,
                    patient_id=patient_id,
                    recording_key=recording_key,
                )
            )

    return pd.DataFrame(rows)


def build_frequency_feature_table(
    npz_directory: str | Path,
    pattern: str = "*_preprocessed.npz",
) -> pd.DataFrame:
    """Build the complete frequency-domain feature table."""
    npz_directory = Path(npz_directory)

    if not npz_directory.exists():
        raise FileNotFoundError(
            f"Processed-signal directory not found: {npz_directory}"
        )

    npz_files = sorted(npz_directory.glob(pattern))

    if not npz_files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' were found in {npz_directory}."
        )

    participant_tables = [
        extract_participant_features(npz_path)
        for npz_path in npz_files
    ]

    feature_table = pd.concat(participant_tables, ignore_index=True)
    feature_columns = get_frequency_feature_columns(feature_table)

    feature_table[feature_columns] = (
        feature_table[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    return feature_table


def get_frequency_feature_columns(
    feature_table: pd.DataFrame,
) -> list[str]:
    """Return only frequency-domain wearable-feature columns."""
    suffixes = tuple(f"_{metric}" for metric in FREQUENCY_METRICS)

    return [
        column
        for column in feature_table.columns
        if column.endswith(suffixes)
    ]


def get_accelerometer_feature_columns(
    feature_table: pd.DataFrame,
) -> list[str]:
    """Return the accelerometer frequency-feature group."""
    return [
        column
        for column in get_frequency_feature_columns(feature_table)
        if column.startswith("acc_")
    ]


def get_gyroscope_feature_columns(
    feature_table: pd.DataFrame,
) -> list[str]:
    """Return the gyroscope frequency-feature group."""
    return [
        column
        for column in get_frequency_feature_columns(feature_table)
        if column.startswith("gyro_")
    ]


def compute_feature_correlations(
    feature_table: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlations among wearable frequency features."""
    feature_columns = get_frequency_feature_columns(feature_table)

    if not feature_columns:
        raise ValueError("No frequency-domain feature columns were found.")

    return feature_table[feature_columns].corr(
        method=method,
        min_periods=2,
    )


def summarize_strong_correlations(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Return unique feature pairs above an absolute correlation threshold."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be greater than 0 and at most 1.")

    rows: list[dict[str, object]] = []
    columns = correlation_matrix.columns.tolist()

    for left_index, feature_1 in enumerate(columns):
        for right_index in range(left_index + 1, len(columns)):
            feature_2 = columns[right_index]
            correlation = correlation_matrix.iloc[left_index, right_index]

            if pd.isna(correlation):
                continue

            if abs(float(correlation)) < threshold:
                continue

            group_1 = (
                "Accelerometer"
                if feature_1.startswith("acc_")
                else "Gyroscope"
            )
            group_2 = (
                "Accelerometer"
                if feature_2.startswith("acc_")
                else "Gyroscope"
            )

            relationship_group = (
                group_1 if group_1 == group_2 else "Cross-sensor"
            )

            rows.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": float(correlation),
                    "absolute_correlation": abs(float(correlation)),
                    "direction": (
                        "Positive" if correlation > 0 else "Negative"
                    ),
                    "relationship_group": relationship_group,
                }
            )

    columns_order = [
        "feature_1",
        "feature_2",
        "correlation",
        "absolute_correlation",
        "direction",
        "relationship_group",
    ]

    if not rows:
        return pd.DataFrame(columns=columns_order)

    return (
        pd.DataFrame(rows, columns=columns_order)
        .sort_values("absolute_correlation", ascending=False)
        .reset_index(drop=True)
    )


def build_correlation_summary(
    feature_table: pd.DataFrame,
    strong_correlations: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Create a compact feature and correlation summary."""
    accelerator_columns = get_accelerometer_feature_columns(feature_table)
    gyroscope_columns = get_gyroscope_feature_columns(feature_table)

    relationship_counts: Mapping[str, int]

    if strong_correlations.empty:
        relationship_counts = {}
    else:
        relationship_counts = (
            strong_correlations["relationship_group"]
            .value_counts()
            .to_dict()
        )

    summary_rows = [
        ("Recordings analysed", len(feature_table)),
        ("Participants analysed", feature_table["patient_id"].nunique()),
        ("Motor tasks", feature_table["task"].nunique()),
        ("Wrists", feature_table["wrist"].nunique()),
        ("Accelerometer frequency features", len(accelerator_columns)),
        ("Gyroscope frequency features", len(gyroscope_columns)),
        ("Strong-correlation threshold", threshold),
        ("Strongly correlated feature pairs", len(strong_correlations)),
        (
            "Accelerometer-only strong pairs",
            relationship_counts.get("Accelerometer", 0),
        ),
        (
            "Gyroscope-only strong pairs",
            relationship_counts.get("Gyroscope", 0),
        ),
        (
            "Cross-sensor strong pairs",
            relationship_counts.get("Cross-sensor", 0),
        ),
    ]

    return pd.DataFrame(summary_rows, columns=["metric", "value"])


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    npz_dir = (
        project_root
        / "data"
        / "processed"
        / "preprocessed_signals"
    )

    print("Project root:", project_root)
    print("Processed signal directory:", npz_dir)

    test_table = build_frequency_feature_table(npz_dir)

    print("\nFrequency-feature extraction completed successfully.")
    print("Feature table shape:", test_table.shape)

    length_summary = (
        test_table["resampled_n_samples"]
        .value_counts()
        .sort_index()
    )

    print("\nPost-resampling sample-length summary:")
    print(length_summary)

    unexpected_lengths = set(length_summary.index) - {976, 2000}

    if unexpected_lengths:
        raise AssertionError(
            "Unexpected post-resampling lengths found: "
            f"{sorted(unexpected_lengths)}"
        )

    if not (
        test_table["n_samples"]
        == test_table["resampled_n_samples"]
    ).all():
        raise AssertionError(
            "At least one recording changed length during resampling."
        )

    print(
        "\nConfirmed: all recordings retain post-trim lengths "
        "of 976 or 2,000 samples."
    )
    print(test_table.head())
