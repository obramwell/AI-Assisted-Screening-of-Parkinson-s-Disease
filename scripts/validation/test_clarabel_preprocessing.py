"""
Dataset-sample validation of the CLARABEL preprocessing pipeline.

Five participants are selected reproducibly using a fixed random seed.
All 22 movement recordings for each selected participant are tested:

    5 participants
    x 11 tasks
    x 2 wrists
    = 110 recordings

Both CVXPY OPTIMAL and OPTIMAL_INACCURATE statuses are accepted as
successfully solved preprocessing runs, but they are reported separately
for numerical-quality auditing.
"""

from pathlib import Path
from time import perf_counter
import random

import numpy as np
import pandas as pd

from src.preprocessing.signal_preprocessing import (
    ACC_COLUMNS,
    GYRO_COLUMNS,
    INITIAL_SAMPLES_TO_REMOVE,
    load_signal,
    preprocess_signal,
)


# ============================================================
# Configuration
# ============================================================

TIMESERIES_PATH = Path(
    "data/raw/movement/timeseries"
)

OUTPUT_DIR = Path(
    "data/processed/solver_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "clarabel_110_recording_validation.csv"
)

RANDOM_SEED = 42
N_PARTICIPANTS = 5

EXPECTED_TASKS = 11

EXPECTED_WRISTS = {
    "LeftWrist",
    "RightWrist",
}

EXPECTED_RECORDINGS_PER_PARTICIPANT = (
    EXPECTED_TASKS
    * len(EXPECTED_WRISTS)
)

EXPECTED_COLUMNS = [
    "Time",
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
    "Gyroscope_X",
    "Gyroscope_Y",
    "Gyroscope_Z",
    "Acc_Magnitude",
    "Gyro_Magnitude",
]

ACCEPTED_SOLVER_STATUSES = {
    "optimal",
    "optimal_inaccurate",
}


# ============================================================
# Helper functions
# ============================================================

def parse_file_name(
    file_path: Path,
) -> tuple[str, str, str]:
    """
    Extract participant ID, task, and wrist from a PADS filename.

    Example
    -------
    001_TouchNose_LeftWrist.txt

    Returns
    -------
    tuple
        participant_id, task, wrist
    """
    parts = file_path.stem.split("_")

    if len(parts) < 3:
        raise ValueError(
            f"Unexpected filename structure: {file_path.name}"
        )

    patient_id = parts[0]
    wrist = parts[-1]

    task = "_".join(
        parts[1:-1]
    )

    return patient_id, task, wrist


# ============================================================
# Find available participants
# ============================================================

movement_files = sorted(
    TIMESERIES_PATH.glob("*.txt")
)

if not movement_files:
    raise FileNotFoundError(
        f"No movement files found in {TIMESERIES_PATH}"
    )


participant_files: dict[str, list[Path]] = {}

for file_path in movement_files:

    patient_id, _, _ = parse_file_name(
        file_path
    )

    participant_files.setdefault(
        patient_id,
        [],
    ).append(file_path)


participant_ids = sorted(
    participant_files.keys()
)


print("=" * 80)
print("CLARABEL PREPROCESSING VALIDATION")
print("=" * 80)

print(
    f"Participants available: "
    f"{len(participant_ids)}"
)

print(
    f"Raw recordings available: "
    f"{len(movement_files)}"
)


# ============================================================
# Identify complete participants
# ============================================================

complete_participants = []

for patient_id in participant_ids:

    files = participant_files[
        patient_id
    ]

    parsed = [
        parse_file_name(file)
        for file in files
    ]

    tasks = {
        task
        for _, task, _ in parsed
    }

    wrists = {
        wrist
        for _, _, wrist in parsed
    }

    task_wrist_pairs = {
        (task, wrist)
        for _, task, wrist in parsed
    }

    complete = (
        len(files)
        == EXPECTED_RECORDINGS_PER_PARTICIPANT
        and len(tasks)
        == EXPECTED_TASKS
        and wrists
        == EXPECTED_WRISTS
        and len(task_wrist_pairs)
        == EXPECTED_RECORDINGS_PER_PARTICIPANT
    )

    if complete:
        complete_participants.append(
            patient_id
        )


if len(complete_participants) < N_PARTICIPANTS:
    raise RuntimeError(
        "Not enough complete participants available "
        "for validation."
    )


# ============================================================
# Reproducible participant selection
# ============================================================

rng = random.Random(
    RANDOM_SEED
)

selected_participants = sorted(
    rng.sample(
        complete_participants,
        N_PARTICIPANTS,
    )
)


print()

print(
    f"Random seed: {RANDOM_SEED}"
)

print(
    "Selected participants: "
    + ", ".join(selected_participants)
)

print(
    f"Expected recordings to test: "
    f"{N_PARTICIPANTS * EXPECTED_RECORDINGS_PER_PARTICIPANT}"
)


# ============================================================
# Run preprocessing validation
# ============================================================

results = []

total_start = perf_counter()


for patient_id in selected_participants:

    files = sorted(
        participant_files[
            patient_id
        ]
    )

    print()
    print("#" * 80)

    print(
        f"PARTICIPANT {patient_id} "
        f"({len(files)} recordings)"
    )

    print("#" * 80)


    for file_path in files:

        _, task, wrist = parse_file_name(
            file_path
        )

        print(
            f"{patient_id} | "
            f"{task:<12} | "
            f"{wrist:<10}",
            end="",
        )

        start = perf_counter()

        try:

            # ------------------------------------------------
            # Load raw recording
            # ------------------------------------------------

            raw = load_signal(
                file_path
            )

            raw_samples = len(
                raw
            )

            expected_processed_samples = (
                raw_samples
                - INITIAL_SAMPLES_TO_REMOVE
            )

            # ------------------------------------------------
            # Preprocess
            # ------------------------------------------------

            processed, info = preprocess_signal(
                raw,
                return_info=True,
            )

            elapsed = (
                perf_counter()
                - start
            )

            raw_trimmed = (
                raw
                .iloc[
                    INITIAL_SAMPLES_TO_REMOVE:
                ]
                .reset_index(drop=True)
            )

            # ------------------------------------------------
            # Solver information
            # ------------------------------------------------

            solver_statuses = {
                axis: axis_info["status"]
                for axis, axis_info
                in info["axis_solvers"].items()
            }

            solver_names = {
                axis: axis_info["solver"]
                for axis, axis_info
                in info["axis_solvers"].items()
            }

            solver_warnings = {
                axis: axis_info["warnings"]
                for axis, axis_info
                in info["axis_solvers"].items()
            }

            # All three axes must have an accepted status.
            all_solver_statuses_acceptable = all(
                status in ACCEPTED_SOLVER_STATUSES
                for status
                in solver_statuses.values()
            )

            # Confirm that the same solver was used throughout.
            all_clarabel = all(
                solver == "CLARABEL"
                for solver
                in solver_names.values()
            )

            # Was every axis fully optimal?
            all_axes_optimal = all(
                status == "optimal"
                for status
                in solver_statuses.values()
            )

            # Did at least one axis return reduced accuracy?
            has_optimal_inaccurate = any(
                status == "optimal_inaccurate"
                for status
                in solver_statuses.values()
            )

            # Number of axes returning reduced accuracy.
            inaccurate_axis_count = sum(
                status == "optimal_inaccurate"
                for status
                in solver_statuses.values()
            )

            # Warnings are kept for auditing.
            warning_count = sum(
                len(axis_warnings)
                for axis_warnings
                in solver_warnings.values()
            )

            # ------------------------------------------------
            # Structural checks
            # ------------------------------------------------

            shape_valid = (
                processed.shape
                == (
                    expected_processed_samples,
                    len(EXPECTED_COLUMNS),
                )
            )

            columns_valid = (
                list(processed.columns)
                == EXPECTED_COLUMNS
            )

            no_missing = (
                processed
                .isna()
                .sum()
                .sum()
                == 0
            )

            all_finite = np.isfinite(
                processed
                .select_dtypes(
                    include="number"
                )
                .to_numpy()
            ).all()

            time_reset = np.isclose(
                processed["Time"].iloc[0],
                0.0,
            )

            # ------------------------------------------------
            # Gyroscope validation
            # ------------------------------------------------

            gyro_unchanged = all(
                np.allclose(
                    processed[
                        column
                    ].to_numpy(),
                    raw_trimmed[
                        column
                    ].to_numpy(),
                )
                for column
                in GYRO_COLUMNS
            )

            # ------------------------------------------------
            # Accelerometer change summary
            # ------------------------------------------------

            acc_mean_absolute_changes = {}

            for column in ACC_COLUMNS:

                change = np.abs(
                    processed[
                        column
                    ].to_numpy()
                    - raw_trimmed[
                        column
                    ].to_numpy()
                )

                acc_mean_absolute_changes[
                    column
                ] = change.mean()

            # ------------------------------------------------
            # Magnitude validation
            # ------------------------------------------------

            expected_acc_magnitude = np.sqrt(
                processed["Accelerometer_X"] ** 2
                + processed["Accelerometer_Y"] ** 2
                + processed["Accelerometer_Z"] ** 2
            )

            expected_gyro_magnitude = np.sqrt(
                processed["Gyroscope_X"] ** 2
                + processed["Gyroscope_Y"] ** 2
                + processed["Gyroscope_Z"] ** 2
            )

            acc_magnitude_valid = np.allclose(
                processed[
                    "Acc_Magnitude"
                ],
                expected_acc_magnitude,
            )

            gyro_magnitude_valid = np.allclose(
                processed[
                    "Gyro_Magnitude"
                ],
                expected_gyro_magnitude,
            )

            # ------------------------------------------------
            # True preprocessing validity
            # ------------------------------------------------

            all_checks_passed = all(
                [
                    all_solver_statuses_acceptable,
                    all_clarabel,
                    shape_valid,
                    columns_valid,
                    no_missing,
                    all_finite,
                    time_reset,
                    gyro_unchanged,
                    acc_magnitude_valid,
                    gyro_magnitude_valid,
                ]
            )

            # ------------------------------------------------
            # Assign quality category
            # ------------------------------------------------

            if not all_checks_passed:

                validation_status = (
                    "FAILED"
                )

            elif has_optimal_inaccurate:

                validation_status = (
                    "PASSED_INACCURATE"
                )

            else:

                validation_status = (
                    "PASSED_OPTIMAL"
                )

            print(
                f" | {validation_status:<18} "
                f"| {elapsed:.2f}s"
            )

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            results.append(
                {
                    "patient_id":
                        patient_id,

                    "task":
                        task,

                    "wrist":
                        wrist,

                    "file":
                        file_path.name,

                    "raw_samples":
                        raw_samples,

                    "processed_samples":
                        len(processed),

                    "processing_time_seconds":
                        round(
                            elapsed,
                            4,
                        ),

                    # Solver names
                    "acc_x_solver":
                        solver_names[
                            "Accelerometer_X"
                        ],

                    "acc_y_solver":
                        solver_names[
                            "Accelerometer_Y"
                        ],

                    "acc_z_solver":
                        solver_names[
                            "Accelerometer_Z"
                        ],

                    # Solver statuses
                    "acc_x_status":
                        solver_statuses[
                            "Accelerometer_X"
                        ],

                    "acc_y_status":
                        solver_statuses[
                            "Accelerometer_Y"
                        ],

                    "acc_z_status":
                        solver_statuses[
                            "Accelerometer_Z"
                        ],

                    "all_axes_optimal":
                        all_axes_optimal,

                    "has_optimal_inaccurate":
                        has_optimal_inaccurate,

                    "inaccurate_axis_count":
                        inaccurate_axis_count,

                    "solver_warning_count":
                        warning_count,

                    # Structural validation
                    "shape_valid":
                        shape_valid,

                    "columns_valid":
                        columns_valid,

                    "no_missing":
                        no_missing,

                    "all_finite":
                        all_finite,

                    "time_reset":
                        time_reset,

                    "gyro_unchanged":
                        gyro_unchanged,

                    "acc_magnitude_valid":
                        acc_magnitude_valid,

                    "gyro_magnitude_valid":
                        gyro_magnitude_valid,

                    # Accelerometer changes
                    "acc_x_mean_absolute_change":
                        acc_mean_absolute_changes[
                            "Accelerometer_X"
                        ],

                    "acc_y_mean_absolute_change":
                        acc_mean_absolute_changes[
                            "Accelerometer_Y"
                        ],

                    "acc_z_mean_absolute_change":
                        acc_mean_absolute_changes[
                            "Accelerometer_Z"
                        ],

                    "status":
                        validation_status,

                    "error":
                        "None",
                }
            )

        except Exception as exc:

            elapsed = (
                perf_counter()
                - start
            )

            print(
                f" | FAILED "
                f"| {elapsed:.2f}s"
            )

            print(
                f"    Error: {exc}"
            )

            results.append(
                {
                    "patient_id":
                        patient_id,

                    "task":
                        task,

                    "wrist":
                        wrist,

                    "file":
                        file_path.name,

                    "raw_samples":
                        np.nan,

                    "processed_samples":
                        np.nan,

                    "processing_time_seconds":
                        round(
                            elapsed,
                            4,
                        ),

                    "acc_x_solver":
                        "CLARABEL",

                    "acc_y_solver":
                        "CLARABEL",

                    "acc_z_solver":
                        "CLARABEL",

                    "acc_x_status":
                        "ERROR",

                    "acc_y_status":
                        "ERROR",

                    "acc_z_status":
                        "ERROR",

                    "all_axes_optimal":
                        False,

                    "has_optimal_inaccurate":
                        False,

                    "inaccurate_axis_count":
                        np.nan,

                    "solver_warning_count":
                        np.nan,

                    "shape_valid":
                        False,

                    "columns_valid":
                        False,

                    "no_missing":
                        False,

                    "all_finite":
                        False,

                    "time_reset":
                        False,

                    "gyro_unchanged":
                        False,

                    "acc_magnitude_valid":
                        False,

                    "gyro_magnitude_valid":
                        False,

                    "acc_x_mean_absolute_change":
                        np.nan,

                    "acc_y_mean_absolute_change":
                        np.nan,

                    "acc_z_mean_absolute_change":
                        np.nan,

                    "status":
                        "FAILED",

                    "error":
                        str(exc),
                }
            )


# ============================================================
# Save validation report
# ============================================================

total_elapsed = (
    perf_counter()
    - total_start
)

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# Final summary
# ============================================================

tested = len(
    results_df
)

optimal_recordings = (
    results_df["status"]
    == "PASSED_OPTIMAL"
).sum()

inaccurate_recordings = (
    results_df["status"]
    == "PASSED_INACCURATE"
).sum()

failed_recordings = (
    results_df["status"]
    == "FAILED"
).sum()

successful_recordings = (
    optimal_recordings
    + inaccurate_recordings
)

total_inaccurate_axes = (
    results_df[
        "inaccurate_axis_count"
    ]
    .fillna(0)
    .sum()
)

warnings_total = (
    results_df[
        "solver_warning_count"
    ]
    .fillna(0)
    .sum()
)

mean_time = (
    results_df[
        "processing_time_seconds"
    ]
    .mean()
)

median_time = (
    results_df[
        "processing_time_seconds"
    ]
    .median()
)

max_time = (
    results_df[
        "processing_time_seconds"
    ]
    .max()
)


print()
print("=" * 80)
print("FINAL VALIDATION SUMMARY")
print("=" * 80)

print(
    f"Participants tested:              "
    f"{N_PARTICIPANTS}"
)

print(
    f"Recordings tested:                "
    f"{tested}"
)

print(
    f"Successfully processed:           "
    f"{successful_recordings}"
)

print(
    f"All axes OPTIMAL:                 "
    f"{optimal_recordings}"
)

print(
    f"At least one OPTIMAL_INACCURATE:  "
    f"{inaccurate_recordings}"
)

print(
    f"Total OPTIMAL_INACCURATE axes:    "
    f"{int(total_inaccurate_axes)}"
)

print(
    f"True preprocessing failures:      "
    f"{failed_recordings}"
)

print(
    f"Solver warnings recorded:         "
    f"{int(warnings_total)}"
)

print(
    f"Mean processing time:             "
    f"{mean_time:.2f} s"
)

print(
    f"Median processing time:           "
    f"{median_time:.2f} s"
)

print(
    f"Maximum processing time:          "
    f"{max_time:.2f} s"
)

print(
    f"Total validation time:            "
    f"{total_elapsed:.2f} s"
)

print()

print(
    f"Validation report: "
    f"{OUTPUT_FILE}"
)


# ============================================================
# Task summary
# ============================================================

print()
print("=" * 80)
print("SOLVER STATUS BY TASK")
print("=" * 80)

task_summary = (
    results_df
    .groupby(
        [
            "task",
            "status",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

print(
    task_summary.to_string()
)


# ============================================================
# Final decision
# ============================================================

if failed_recordings == 0:

    print()
    print("=" * 80)
    print("VALIDATION PASSED")
    print("=" * 80)

    print(
        "All sampled recordings were successfully "
        "preprocessed using CLARABEL."
    )

    print(
        "OPTIMAL and OPTIMAL_INACCURATE outcomes were "
        "recorded separately for numerical-quality auditing."
    )

else:

    print()
    print("=" * 80)
    print("VALIDATION REQUIRES REVIEW")
    print("=" * 80)

    print(
        f"{failed_recordings} recording(s) produced "
        "a true preprocessing failure."
    )

    failed_rows = results_df[
        results_df[
            "status"
        ]
        == "FAILED"
    ]

    print()

    print(
        failed_rows[
            [
                "patient_id",
                "task",
                "wrist",
                "file",
                "error",
            ]
        ].to_string(
            index=False
        )
    )

    raise SystemExit(1)