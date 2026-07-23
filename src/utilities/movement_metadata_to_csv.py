import json
import pandas as pd
from pathlib import Path


# ======================================================
# PATHS
# ======================================================

# Project root:
# AI-Assisted-Screening-of-Parkinson-s-Disease/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input: original PADS movement JSON files
movement_folder = PROJECT_ROOT / "data" / "raw" / "movement"

# Output: intermediate CSV files
output_folder = PROJECT_ROOT / "data" / "interim"

# Create output directory if it does not exist
output_folder.mkdir(parents=True, exist_ok=True)


# ======================================================
# READ MOVEMENT JSON FILES
# ======================================================

rows = []

json_files = sorted(
    movement_folder.glob("observation_*.json")
)

print(f"Found {len(json_files)} movement files.\n")

if len(json_files) == 0:
    raise FileNotFoundError(
        f"No movement observation JSON files were found in:\n"
        f"{movement_folder}"
    )

for json_file in json_files:

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    patient_id = data["subject_id"]
    device = data["device_id"]
    sampling_rate = data["sampling_rate"]

    for session in data["session"]:

        task = session["record_name"]
        samples = session["rows"]

        left_file = None
        right_file = None

        for record in session["records"]:

            if record["device_location"] == "LeftWrist":
                left_file = record["file_name"]

            elif record["device_location"] == "RightWrist":
                right_file = record["file_name"]

        rows.append({
            "patient_id": patient_id,
            "device": device,
            "sampling_rate": sampling_rate,
            "task": task,
            "samples": samples,
            "left_file": left_file,
            "right_file": right_file
        })


# ======================================================
# CREATE DATAFRAME
# ======================================================

df = pd.DataFrame(rows)
df["patient_id"] = df["patient_id"].astype(int)
df = df.sort_values(
    ["patient_id", "task"]
).reset_index(drop=True)
df["patient_id"] = df["patient_id"].apply(
    lambda x: f"{x:03d}"
)


# ======================================================
# BASIC VALIDATION
# ======================================================

expected_participants = 469
expected_tasks = 11
expected_recordings = expected_participants * expected_tasks

participants = df["patient_id"].nunique()
tasks = df["task"].nunique()
duplicate_rows = df.duplicated(
    subset=["patient_id", "task"]
).sum()

missing_left = df["left_file"].isna().sum()
missing_right = df["right_file"].isna().sum()

if participants != expected_participants:
    print(
        f"Warning: Expected {expected_participants} participants, "
        f"but found {participants}."
    )

if tasks != expected_tasks:
    print(
        f"Warning: Expected {expected_tasks} tasks, "
        f"but found {tasks}."
    )

if len(df) != expected_recordings:
    print(
        f"Warning: Expected {expected_recordings} participant-task "
        f"recordings, but found {len(df)}."
    )

if duplicate_rows > 0:
    print(
        f"Warning: Found {duplicate_rows} duplicate "
        f"participant-task combinations."
    )

if missing_left > 0 or missing_right > 0:
    print(
        f"Warning: Missing wrist file references detected. "
        f"Left: {missing_left}, Right: {missing_right}."
    )


# ======================================================
# SAVE CSV
# ======================================================

output_file = output_folder / "movement_metadata.csv"

df.to_csv(output_file, index=False)


# ======================================================
# SUMMARY
# ======================================================

print("=" * 60)
print("MOVEMENT METADATA CREATED SUCCESSFULLY")
print("=" * 60)

print("\nInput folder:")
print(movement_folder)

print("\nOutput file:")
print(output_file)

print(f"\nParticipants: {participants}")
print(f"Tasks: {tasks}")
print(f"Participant-task recordings: {len(df)}")

print(f"\nDuplicate participant-task combinations: {duplicate_rows}")
print(f"Missing left-wrist file references: {missing_left}")
print(f"Missing right-wrist file references: {missing_right}")

print("\nSampling rates:")
print(df["sampling_rate"].value_counts().sort_index())

print("\nSamples per recording:")
print(df["samples"].value_counts().sort_index())

print("\nTasks:")
print(df["task"].value_counts().sort_index())

print("\nFirst 5 rows:")
print(df.head())

print("\nDone!")
