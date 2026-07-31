"""
Dataset-wide preprocessing runner for PADS smartwatch recordings.

This script:
1. Reads all raw movement .txt recordings.
2. Applies the validated signal preprocessing pipeline.
3. Groups the 22 task/wrist recordings for each participant.
4. Saves one compressed .npz file per participant.
5. Creates a recording-level preprocessing quality report.

Expected dataset structure
--------------------------
469 participants
11 motor tasks
2 wrists
22 recordings per participant
10,318 recordings total

Output
------
data/processed/preprocessed_signals/
    001_preprocessed.npz
    002_preprocessed.npz
    ...
    469_preprocessed.npz

data/processed/preprocessing_report.csv

NPZ structure
-------------
Each participant file contains 22 signal arrays with names such as:

    CrossArms__LeftWrist
    CrossArms__RightWrist
    TouchNose__LeftWrist
    ...

Each signal array has columns:

    Time
    Accelerometer_X
    Accelerometer_Y
    Accelerometer_Z
    Gyroscope_X
    Gyroscope_Y
    Gyroscope_Z
    Acc_Magnitude
    Gyro_Magnitude

Metadata arrays are also included:
    __columns__
    __recording_keys__
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.preprocessing.signal_preprocessing import (
    ACC_COLUMNS,
    INITIAL_SAMPLES_TO_REMOVE,
    L1_LAMBDA,
    SIGNAL_COLUMNS,
    load_signal,
    preprocess_signal,
)


# ============================================================
# Configuration
# ============================================================

RAW_TIMESERIES_DIR = Path(
    "data/raw/movement/timeseries"
)

OUTPUT_DIR = Path(
    "data/processed"
)

SIGNAL_OUTPUT_DIR = (
    OUTPUT_DIR
    / "preprocessed_signals"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "preprocessing_report.csv"
)

EXPECTED_PARTICIPANTS = 469
EXPECTED_TASKS = 11

EXPECTED_WRISTS = {
    "LeftWrist",
    "RightWrist",
}

EXPECTED_RECORDINGS_PER_PARTICIPANT = (
    EXPECTED_TASKS
    * len(EXPECTED_WRISTS)
)

EXPECTED_TOTAL_RECORDINGS = (
    EXPECTED_PARTICIPANTS
    * EXPECTED_RECORDINGS_PER_PARTICIPANT
)

PROCESSED_COLUMNS = (
    SIGNAL_COLUMNS
    + [
        "Acc_Magnitude",
        "Gyro_Magnitude",
    ]
)

ACCEPTED_SOLVER_STATUSES = {
    "optimal",
    "optimal_inaccurate",
}

# Set True only when you deliberately want to recreate
# existing participant files.
FORCE_REPROCESS = False

# Save the CSV report every N processed participants.
# This prevents losing the entire run if execution is interrupted.
CHECKPOINT_EVERY = 5


# ============================================================
# Helpers
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
    tuple[str, str, str]
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

    if wrist not in EXPECTED_WRISTS:
        raise ValueError(
            f"Unexpected wrist '{wrist}' in {file_path.name}"
        )

    return (
        patient_id,
        task,
        wrist,
    )


def create_npz_key(
    task: str,
    wrist: str,
) -> str:
    """
    Create the array name stored inside the participant NPZ.

    Example
    -------
    TouchNose + LeftWrist -> TouchNose__LeftWrist
    """
    return f"{task}__{wrist}"


def save_participant_npz(
    patient_id: str,
    signals: dict[str, np.ndarray],
) -> Path:
    """
    Save all 22 processed recordings for one participant.

    The file is first written to a temporary location and then
    renamed, reducing the chance of leaving a corrupted NPZ if
    execution is interrupted during writing.
    """
    output_path = (
        SIGNAL_OUTPUT_DIR
        / f"{patient_id}_preprocessed.npz"
    )

    temp_path = (
        SIGNAL_OUTPUT_DIR
        / f"{patient_id}_preprocessed.tmp"
    )

    recording_keys = sorted(
        signals.keys()
    )

    payload = {
        "__columns__": np.asarray(
            PROCESSED_COLUMNS,
            dtype=str,
        ),
        "__recording_keys__": np.asarray(
            recording_keys,
            dtype=str,
        ),
    }

    payload.update(
        signals
    )

    # Using an open binary file prevents NumPy from
    # automatically appending ".npz" to the temporary name.
    with open(
        temp_path,
        "wb",
    ) as file_handle:

        np.savez_compressed(
            file_handle,
            **payload,
        )

    temp_path.replace(
        output_path
    )

    return output_path


def load_existing_report() -> pd.DataFrame:
    """
    Load an existing preprocessing report for resumable execution.
    """
    if not REPORT_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        REPORT_PATH,
        dtype={
            "patient_id": str,
        },
    )


def participant_already_complete(
    patient_id: str,
    report: pd.DataFrame,
) -> bool:
    """
    Determine whether a participant can safely be skipped.

    A participant is considered complete only when:
    - its NPZ file exists;
    - exactly 22 report rows exist;
    - every recording was successfully processed.
    """
    npz_path = (
        SIGNAL_OUTPUT_DIR
        / f"{patient_id}_preprocessed.npz"
    )

    if not npz_path.exists():
        return False

    if report.empty:
        return False

    if "patient_id" not in report.columns:
        return False

    rows = report[
        report["patient_id"]
        == patient_id
    ]

    if len(rows) != EXPECTED_RECORDINGS_PER_PARTICIPANT:
        return False

    successful_statuses = {
        "PASSED_OPTIMAL",
        "PASSED_INACCURATE",
    }

    return rows[
        "status"
    ].isin(
        successful_statuses
    ).all()


def replace_participant_report_rows(
    report: pd.DataFrame,
    patient_id: str,
    new_rows: list[dict],
) -> pd.DataFrame:
    """
    Replace report rows for one participant.

    Useful when a participant is reprocessed after an
    interrupted or unsuccessful run.
    """
    new_df = pd.DataFrame(
        new_rows
    )

    if report.empty:
        return new_df

    if "patient_id" in report.columns:
        report = report[
            report["patient_id"]
            != patient_id
        ].copy()

    return pd.concat(
        [
            report,
            new_df,
        ],
        ignore_index=True,
    )


def save_report(
    report: pd.DataFrame,
) -> None:
    """
    Sort and save the preprocessing report.
    """
    if report.empty:
        return

    report = report.sort_values(
        [
            "patient_id",
            "task",
            "wrist",
        ]
    ).reset_index(
        drop=True
    )

    report.to_csv(
        REPORT_PATH,
        index=False,
    )


# ============================================================
# Prepare directories
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SIGNAL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Discover raw recordings
# ============================================================

movement_files = sorted(
    RAW_TIMESERIES_DIR.glob(
        "*.txt"
    )
)

if not movement_files:
    raise FileNotFoundError(
        "No raw movement recordings were found in "
        f"{RAW_TIMESERIES_DIR}"
    )


participant_files: dict[
    str,
    list[Path],
] = {}

for file_path in movement_files:

    patient_id, _, _ = (
        parse_file_name(
            file_path
        )
    )

    participant_files.setdefault(
        patient_id,
        [],
    ).append(
        file_path
    )


participant_ids = sorted(
    participant_files.keys()
)


# ============================================================
# Dataset-level structural validation
# ============================================================

print("=" * 80)
print("PADS DATASET-WIDE SIGNAL PREPROCESSING")
print("=" * 80)

print(
    f"Participants discovered:     "
    f"{len(participant_ids)}"
)

print(
    f"Raw recordings discovered:   "
    f"{len(movement_files)}"
)

print(
    f"Expected participants:        "
    f"{EXPECTED_PARTICIPANTS}"
)

print(
    f"Expected recordings:          "
    f"{EXPECTED_TOTAL_RECORDINGS}"
)

print(
    f"L1 lambda:                    "
    f"{L1_LAMBDA}"
)

print(
    "Solver:                       "
    "CLARABEL"
)

print(
    f"Output directory:             "
    f"{SIGNAL_OUTPUT_DIR}"
)

print(
    f"Report:                       "
    f"{REPORT_PATH}"
)


if (
    len(participant_ids)
    != EXPECTED_PARTICIPANTS
):
    raise RuntimeError(
        "Unexpected number of participants. "
        f"Expected {EXPECTED_PARTICIPANTS}, "
        f"found {len(participant_ids)}."
    )


if (
    len(movement_files)
    != EXPECTED_TOTAL_RECORDINGS
):
    raise RuntimeError(
        "Unexpected number of raw recordings. "
        f"Expected {EXPECTED_TOTAL_RECORDINGS}, "
        f"found {len(movement_files)}."
    )


# ============================================================
# Validate task/wrist completeness before starting
# ============================================================

for patient_id in participant_ids:

    files = participant_files[
        patient_id
    ]

    parsed = [
        parse_file_name(
            file_path
        )
        for file_path in files
    ]

    tasks = {
        task
        for _, task, _
        in parsed
    }

    wrists = {
        wrist
        for _, _, wrist
        in parsed
    }

    task_wrist_pairs = {
        (
            task,
            wrist,
        )
        for _, task, wrist
        in parsed
    }

    if (
        len(files)
        != EXPECTED_RECORDINGS_PER_PARTICIPANT
        or len(tasks)
        != EXPECTED_TASKS
        or wrists
        != EXPECTED_WRISTS
        or len(task_wrist_pairs)
        != EXPECTED_RECORDINGS_PER_PARTICIPANT
    ):
        raise RuntimeError(
            f"Participant {patient_id} does not have "
            "the expected 11 tasks x 2 wrists structure."
        )


print()
print(
    "Dataset structure validation passed."
)


# ============================================================
# Load previous report for resumable execution
# ============================================================

report_df = (
    load_existing_report()
)

if FORCE_REPROCESS:

    print()
    print(
        "FORCE_REPROCESS=True: "
        "existing participant outputs will be recreated."
    )

else:

    completed_before_start = sum(
        participant_already_complete(
            patient_id,
            report_df,
        )
        for patient_id
        in participant_ids
    )

    print()
    print(
        f"Completed participants found: "
        f"{completed_before_start}"
    )


# ============================================================
# Dataset-wide processing
# ============================================================

dataset_start = perf_counter()

processed_this_run = 0
skipped_this_run = 0

participants_with_failures = []

processed_since_checkpoint = 0


for participant_number, patient_id in enumerate(
    participant_ids,
    start=1,
):

    participant_npz_path = (
        SIGNAL_OUTPUT_DIR
        / f"{patient_id}_preprocessed.npz"
    )

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if (
        not FORCE_REPROCESS
        and participant_already_complete(
            patient_id,
            report_df,
        )
    ):

        skipped_this_run += 1

        print(
            f"[{participant_number:03d}/"
            f"{len(participant_ids)}] "
            f"Participant {patient_id}: "
            "SKIPPED (already complete)"
        )

        continue


    if (
        FORCE_REPROCESS
        and participant_npz_path.exists()
    ):
        participant_npz_path.unlink()


    participant_start = (
        perf_counter()
    )

    files = sorted(
        participant_files[
            patient_id
        ]
    )

    participant_signals: dict[
        str,
        np.ndarray,
    ] = {}

    participant_rows = []

    participant_failed = False

    participant_optimal = 0
    participant_inaccurate = 0


    # --------------------------------------------------------
    # Process all 22 recordings
    # --------------------------------------------------------

    for file_path in files:

        _, task, wrist = (
            parse_file_name(
                file_path
            )
        )

        npz_key = (
            create_npz_key(
                task,
                wrist,
            )
        )

        recording_start = (
            perf_counter()
        )

        raw_samples = np.nan

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

            # ------------------------------------------------
            # Apply validated preprocessing
            # ------------------------------------------------

            processed, info = (
                preprocess_signal(
                    raw,
                    return_info=True,
                )
            )

            recording_time = (
                perf_counter()
                - recording_start
            )

            # ------------------------------------------------
            # Final output validation
            # ------------------------------------------------

            if (
                list(
                    processed.columns
                )
                != PROCESSED_COLUMNS
            ):
                raise RuntimeError(
                    "Unexpected processed columns."
                )

            if (
                len(processed)
                != raw_samples
                - INITIAL_SAMPLES_TO_REMOVE
            ):
                raise RuntimeError(
                    "Unexpected processed signal length."
                )

            processed_array = (
                processed.to_numpy(
                    dtype=np.float64
                )
            )

            if not np.isfinite(
                processed_array
            ).all():
                raise RuntimeError(
                    "Processed recording contains "
                    "non-finite values."
                )

            # ------------------------------------------------
            # Solver information
            # ------------------------------------------------

            axis_info = (
                info[
                    "axis_solvers"
                ]
            )

            acc_x_info = (
                axis_info[
                    ACC_COLUMNS[0]
                ]
            )

            acc_y_info = (
                axis_info[
                    ACC_COLUMNS[1]
                ]
            )

            acc_z_info = (
                axis_info[
                    ACC_COLUMNS[2]
                ]
            )

            solver_statuses = [
                acc_x_info["status"],
                acc_y_info["status"],
                acc_z_info["status"],
            ]

            if not all(
                status
                in ACCEPTED_SOLVER_STATUSES
                for status
                in solver_statuses
            ):
                raise RuntimeError(
                    "At least one accelerometer axis "
                    "returned an unacceptable solver status: "
                    f"{solver_statuses}"
                )

            has_inaccurate = any(
                status
                == "optimal_inaccurate"
                for status
                in solver_statuses
            )

            if has_inaccurate:

                status = (
                    "PASSED_INACCURATE"
                )

                participant_inaccurate += 1

            else:

                status = (
                    "PASSED_OPTIMAL"
                )

                participant_optimal += 1


            # ------------------------------------------------
            # Store processed signal for participant NPZ
            # ------------------------------------------------

            participant_signals[
                npz_key
            ] = processed_array


            # ------------------------------------------------
            # Warning information
            # ------------------------------------------------

            x_warnings = (
                acc_x_info[
                    "warnings"
                ]
            )

            y_warnings = (
                acc_y_info[
                    "warnings"
                ]
            )

            z_warnings = (
                acc_z_info[
                    "warnings"
                ]
            )

            total_warning_count = (
                len(x_warnings)
                + len(y_warnings)
                + len(z_warnings)
            )


            # ------------------------------------------------
            # Report row
            # ------------------------------------------------

            participant_rows.append(
                {
                    "patient_id":
                        patient_id,

                    "task":
                        task,

                    "wrist":
                        wrist,

                    "input_file":
                        file_path.name,

                    "npz_key":
                        npz_key,

                    "raw_samples":
                        raw_samples,

                    "processed_samples":
                        len(processed),

                    "samples_removed":
                        INITIAL_SAMPLES_TO_REMOVE,

                    "l1_lambda":
                        L1_LAMBDA,

                    "solver":
                        "CLARABEL",

                    "processing_time_seconds":
                        round(
                            recording_time,
                            4,
                        ),

                    # Solver status
                    "acc_x_status":
                        acc_x_info[
                            "status"
                        ],

                    "acc_y_status":
                        acc_y_info[
                            "status"
                        ],

                    "acc_z_status":
                        acc_z_info[
                            "status"
                        ],

                    # Solver time
                    "acc_x_solver_time_seconds":
                        acc_x_info[
                            "solve_time"
                        ],

                    "acc_y_solver_time_seconds":
                        acc_y_info[
                            "solve_time"
                        ],

                    "acc_z_solver_time_seconds":
                        acc_z_info[
                            "solve_time"
                        ],

                    # Solver iterations
                    "acc_x_iterations":
                        acc_x_info[
                            "iterations"
                        ],

                    "acc_y_iterations":
                        acc_y_info[
                            "iterations"
                        ],

                    "acc_z_iterations":
                        acc_z_info[
                            "iterations"
                        ],

                    # Warnings
                    "acc_x_warning_count":
                        len(
                            x_warnings
                        ),

                    "acc_y_warning_count":
                        len(
                            y_warnings
                        ),

                    "acc_z_warning_count":
                        len(
                            z_warnings
                        ),

                    "solver_warning_count":
                        total_warning_count,

                    "acc_x_warnings":
                        " | ".join(
                            x_warnings
                        )
                        if x_warnings
                        else "None",

                    "acc_y_warnings":
                        " | ".join(
                            y_warnings
                        )
                        if y_warnings
                        else "None",

                    "acc_z_warnings":
                        " | ".join(
                            z_warnings
                        )
                        if z_warnings
                        else "None",

                    "has_optimal_inaccurate":
                        has_inaccurate,

                    "status":
                        status,

                    "participant_npz_saved":
                        False,

                    "error":
                        "None",
                }
            )


        except Exception as exc:

            recording_time = (
                perf_counter()
                - recording_start
            )

            participant_failed = True

            print()
            print(
                f"    FAILED: "
                f"{file_path.name}"
            )

            print(
                f"    Error: {exc}"
            )

            participant_rows.append(
                {
                    "patient_id":
                        patient_id,

                    "task":
                        task,

                    "wrist":
                        wrist,

                    "input_file":
                        file_path.name,

                    "npz_key":
                        npz_key,

                    "raw_samples":
                        raw_samples,

                    "processed_samples":
                        np.nan,

                    "samples_removed":
                        INITIAL_SAMPLES_TO_REMOVE,

                    "l1_lambda":
                        L1_LAMBDA,

                    "solver":
                        "CLARABEL",

                    "processing_time_seconds":
                        round(
                            recording_time,
                            4,
                        ),

                    "acc_x_status":
                        "ERROR",

                    "acc_y_status":
                        "ERROR",

                    "acc_z_status":
                        "ERROR",

                    "acc_x_solver_time_seconds":
                        np.nan,

                    "acc_y_solver_time_seconds":
                        np.nan,

                    "acc_z_solver_time_seconds":
                        np.nan,

                    "acc_x_iterations":
                        np.nan,

                    "acc_y_iterations":
                        np.nan,

                    "acc_z_iterations":
                        np.nan,

                    "acc_x_warning_count":
                        np.nan,

                    "acc_y_warning_count":
                        np.nan,

                    "acc_z_warning_count":
                        np.nan,

                    "solver_warning_count":
                        np.nan,

                    "acc_x_warnings":
                        "Unknown",

                    "acc_y_warnings":
                        "Unknown",

                    "acc_z_warnings":
                        "Unknown",

                    "has_optimal_inaccurate":
                        False,

                    "status":
                        "FAILED",

                    "participant_npz_saved":
                        False,

                    "error":
                        str(exc),
                }
            )


    # --------------------------------------------------------
    # Save participant only if all 22 recordings succeeded
    # --------------------------------------------------------

    participant_complete = (
        not participant_failed
        and len(participant_signals)
        == EXPECTED_RECORDINGS_PER_PARTICIPANT
        and len(participant_rows)
        == EXPECTED_RECORDINGS_PER_PARTICIPANT
    )


    if participant_complete:

        saved_path = (
            save_participant_npz(
                patient_id,
                participant_signals,
            )
        )

        for row in participant_rows:
            row[
                "participant_npz_saved"
            ] = True

    else:

        participants_with_failures.append(
            patient_id
        )

        # Prevent an incomplete/stale participant file
        # from being mistaken for valid processed data.
        if participant_npz_path.exists():
            participant_npz_path.unlink()


    # --------------------------------------------------------
    # Update report
    # --------------------------------------------------------

    report_df = (
        replace_participant_report_rows(
            report_df,
            patient_id,
            participant_rows,
        )
    )

    processed_this_run += 1
    processed_since_checkpoint += 1

    participant_time = (
        perf_counter()
        - participant_start
    )


    # --------------------------------------------------------
    # Participant-level progress
    # --------------------------------------------------------

    if participant_complete:

        print(
            f"[{participant_number:03d}/"
            f"{len(participant_ids)}] "
            f"Participant {patient_id}: "
            f"22/22 processed | "
            f"{participant_optimal} optimal | "
            f"{participant_inaccurate} inaccurate | "
            f"{participant_time:.1f}s"
        )

    else:

        failed_count = sum(
            row["status"]
            == "FAILED"
            for row
            in participant_rows
        )

        print(
            f"[{participant_number:03d}/"
            f"{len(participant_ids)}] "
            f"Participant {patient_id}: "
            f"INCOMPLETE | "
            f"{failed_count} failed | "
            f"{participant_time:.1f}s"
        )


    # --------------------------------------------------------
    # Checkpoint report
    # --------------------------------------------------------

    if (
        processed_since_checkpoint
        >= CHECKPOINT_EVERY
        or participant_failed
    ):

        save_report(
            report_df
        )

        processed_since_checkpoint = 0


# ============================================================
# Final report save
# ============================================================

save_report(
    report_df
)

dataset_elapsed = (
    perf_counter()
    - dataset_start
)


# ============================================================
# Dataset-wide summary
# ============================================================

successful_statuses = {
    "PASSED_OPTIMAL",
    "PASSED_INACCURATE",
}

successful_recordings = (
    report_df[
        "status"
    ].isin(
        successful_statuses
    ).sum()
)

optimal_recordings = (
    report_df[
        "status"
    ]
    == "PASSED_OPTIMAL"
).sum()

inaccurate_recordings = (
    report_df[
        "status"
    ]
    == "PASSED_INACCURATE"
).sum()

failed_recordings = (
    report_df[
        "status"
    ]
    == "FAILED"
).sum()

total_solver_warnings = (
    pd.to_numeric(
        report_df[
            "solver_warning_count"
        ],
        errors="coerce",
    )
    .fillna(0)
    .sum()
)

npz_files_created = len(
    list(
        SIGNAL_OUTPUT_DIR.glob(
            "*_preprocessed.npz"
        )
    )
)


print()
print("=" * 80)
print("DATASET-WIDE PREPROCESSING SUMMARY")
print("=" * 80)

print(
    f"Participants in report:           "
    f"{report_df['patient_id'].nunique()}"
)

print(
    f"Participant NPZ files:            "
    f"{npz_files_created}"
)

print(
    f"Recordings in report:             "
    f"{len(report_df)}"
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
    f"True preprocessing failures:      "
    f"{failed_recordings}"
)

print(
    f"Solver warnings recorded:         "
    f"{int(total_solver_warnings)}"
)

print(
    f"Processed during this run:        "
    f"{processed_this_run}"
)

print(
    f"Skipped during this run:          "
    f"{skipped_this_run}"
)

print(
    f"Total runtime:                    "
    f"{dataset_elapsed / 60:.2f} minutes"
)

print()

print(
    f"Signals saved to: "
    f"{SIGNAL_OUTPUT_DIR}"
)

print(
    f"Report saved to: "
    f"{REPORT_PATH}"
)


# ============================================================
# Final integrity decision
# ============================================================

if (
    failed_recordings == 0
    and successful_recordings
    == EXPECTED_TOTAL_RECORDINGS
    and npz_files_created
    == EXPECTED_PARTICIPANTS
):

    print()
    print("=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)

    print(
        "All 10,318 smartwatch recordings were "
        "successfully preprocessed."
    )

    print(
        "All 469 participant-level NPZ cache files "
        "were successfully created."
    )

else:

    print()
    print("=" * 80)
    print("PREPROCESSING REQUIRES REVIEW")
    print("=" * 80)

    if participants_with_failures:

        print(
            "Participants with failures in this run: "
            + ", ".join(
                participants_with_failures
            )
        )

    print(
        "Review preprocessing_report.csv before "
        "continuing to feature extraction."
    )

    raise SystemExit(1)