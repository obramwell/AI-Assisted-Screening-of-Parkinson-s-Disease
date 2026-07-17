import json
import pandas as pd
from pathlib import Path

# ======================================================
# PATHS
# ======================================================

dataset_root = Path(
    r"C:\Users\Daniela\Documents\UNF\Summer 2026-Term 5\pads-parkinsons-disease-smartwatch-dataset-1.0.0"
)

patients_folder = dataset_root / "patients"

output_folder = Path(
    r"C:\Users\Daniela\Documents\UNF\Summer 2026-Term 5\Capstone\data\processed"
)

# ======================================================
# LABEL ENCODING
# ======================================================

def encode_label(condition):
    """
    Encode diagnosis into numerical labels.

    Healthy = 0
    Parkinson's Disease = 1
    Other Movement Disorders = 2
    """

    if condition == "Healthy":
        return 0

    elif condition == "Parkinson's":
        return 1

    else:
        return 2


# ======================================================
# READ JSON FILES
# ======================================================

patients = []

json_files = sorted(patients_folder.glob("patient_*.json"))

print(f"Found {len(json_files)} patient files.\n")

for json_file in json_files:

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    patients.append({

        "patient_id": data.get("id"),

        "study_id": data.get("study_id"),

        "condition": data.get("condition"),

        "label": encode_label(data.get("condition")),

        "disease_comment": data.get("disease_comment"),

        "age_at_diagnosis": data.get("age_at_diagnosis"),

        "age": data.get("age"),

        "height_cm": data.get("height"),

        "weight_kg": data.get("weight"),

        "gender": data.get("gender"),

        "handedness": data.get("handedness"),

        "appearance_in_kinship":
            data.get("appearance_in_kinship"),

        "appearance_in_first_grade_kinship":
            data.get("appearance_in_first_grade_kinship"),

        "effect_of_alcohol_on_tremor":
            data.get("effect_of_alcohol_on_tremor")

    })

# ======================================================
# CREATE DATAFRAME
# ======================================================

df = pd.DataFrame(patients)

# Sort by patient ID
df["patient_id"] = df["patient_id"].astype(int)

df = df.sort_values("patient_id")

# Restore leading zeros
df["patient_id"] = df["patient_id"].apply(lambda x: f"{x:03d}")

# ======================================================
# SAVE CSV
# ======================================================

output_file = output_folder / "patients.csv"

df.to_csv(output_file, index=False)

# ======================================================
# SUMMARY
# ======================================================

print("=" * 60)
print("PATIENTS DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print(f"\nOutput file:")
print(output_file)

print(f"\nNumber of patients: {len(df)}")

print("\nClass distribution:")

print(df["condition"].value_counts())

print("\nColumns:")

print(df.columns.tolist())

print("\nFirst 5 rows:\n")

print(df.head())

print("\nDone!")