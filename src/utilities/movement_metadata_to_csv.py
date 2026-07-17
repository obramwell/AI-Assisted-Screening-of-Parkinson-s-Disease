import json
import pandas as pd
from pathlib import Path

# ======================================================
# PATHS
# ======================================================

project_root = Path(__file__).parent

dataset_root = Path(
    r"C:\Users\Daniela\Documents\UNF\Summer 2026-Term 5\pads-parkinsons-disease-smartwatch-dataset-1.0.0"
)

movement_folder = dataset_root / "movement"

output_folder = project_root / "data" / "processed"
output_folder.mkdir(parents=True, exist_ok=True)

# ======================================================
# READ JSON FILES
# ======================================================

rows = []

json_files = sorted(movement_folder.glob("observation_*.json"))

print(f"Found {len(json_files)} movement files.\n")

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
# DATAFRAME
# ======================================================

df = pd.DataFrame(rows)

df["patient_id"] = df["patient_id"].astype(int)

df = df.sort_values(["patient_id", "task"])

df["patient_id"] = df["patient_id"].apply(lambda x: f"{x:03d}")

# ======================================================
# SAVE
# ======================================================

output_file = output_folder / "movement_metadata.csv"

df.to_csv(output_file, index=False)

# ======================================================
# SUMMARY
# ======================================================

print("=" * 60)
print("MOVEMENT METADATA CREATED")
print("=" * 60)

print(f"\nOutput: {output_file}")

print(f"\nRows: {len(df)}")

print("\nTasks:")

print(df["task"].unique())

print("\nFirst rows:\n")

print(df.head())

print("\nDone")