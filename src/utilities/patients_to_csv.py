import json
import pandas as pd
from pathlib import Path


# ======================================================
# PATHS
# ======================================================

# Project root:
# AI-Assisted-Screening-of-Parkinson-s-Disease/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input: original PADS patient JSON files
patients_folder = PROJECT_ROOT / "data" / "raw" / "patients"

# Output: intermediate CSV files
output_folder = PROJECT_ROOT / "data" / "interim"

# Create output directory if it does not exist
output_folder.mkdir(parents=True, exist_ok=True)


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

if len(json_files) == 0:
    raise FileNotFoundError(
        f"No patient JSON files were found in:\n{patients_folder}"
    )

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
df["patient_id"] = df["patient_id"].astype(int)
df = df.sort_values("patient_id").reset_index(drop=True)
df["patient_id"] = df["patient_id"].apply(
    lambda x: f"{x:03d}"
)


# ======================================================
# BASIC VALIDATION
# ======================================================

expected_patients = 469

if len(df) != expected_patients:
    print(
        f"Warning: Expected {expected_patients} participants, "
        f"but found {len(df)}."
    )

duplicate_ids = df["patient_id"].duplicated().sum()

if duplicate_ids > 0:
    print(
        f"Warning: Found {duplicate_ids} duplicate patient IDs."
    )


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

print("\nInput folder:")
print(patients_folder)

print("\nOutput file:")
print(output_file)

print(f"\nNumber of patients: {len(df)}")

print(f"Duplicate patient IDs: {duplicate_ids}")

print("\nClass distribution:")
print(df["condition"].value_counts(dropna=False))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDone!")
