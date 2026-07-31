import json
import pandas as pd
from pathlib import Path


# ======================================================
# PATHS
# ======================================================

# Project root:
# AI-Assisted-Screening-of-Parkinson-s-Disease/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input: original PADS questionnaire JSON files
questionnaire_folder = (
    PROJECT_ROOT / "data" / "raw" / "questionnaire"
)

# Output: intermediate CSV files
output_folder = PROJECT_ROOT / "data" / "interim"

# Create output directory if it does not exist
output_folder.mkdir(parents=True, exist_ok=True)


# ======================================================
# READ QUESTIONNAIRES
# ======================================================

records = []

json_files = sorted(questionnaire_folder.glob("*.json"))

print(f"Found {len(json_files)} questionnaire files.\n")

if len(json_files) == 0:
    raise FileNotFoundError(
        f"No questionnaire JSON files were found in:\n"
        f"{questionnaire_folder}"
    )

for json_file in json_files:

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    row = {
        "patient_id": data["subject_id"],
        "questionnaire_name": data["questionnaire_name"]
    }

    # Extract questionnaire items
    for item in data["item"]:

        question = f"Q{item['link_id']}"

        # Convert True/False -> 1/0
        row[question] = int(item["answer"])

    records.append(row)


# ======================================================
# CREATE DATAFRAME
# ======================================================

df = pd.DataFrame(records)
df["patient_id"] = df["patient_id"].astype(int)
df = df.sort_values("patient_id").reset_index(drop=True)
df["patient_id"] = df["patient_id"].apply(
    lambda x: f"{x:03d}"
)


# ======================================================
# BASIC VALIDATION
# ======================================================

expected_participants = 469
expected_questions = 30

if len(df) != expected_participants:
    print(
        f"Warning: Expected {expected_participants} participants, "
        f"but found {len(df)}."
    )

duplicate_ids = df["patient_id"].duplicated().sum()

if duplicate_ids > 0:
    print(
        f"Warning: Found {duplicate_ids} duplicate patient IDs."
    )

question_columns = [
    col for col in df.columns
    if col.startswith("Q")
]

if len(question_columns) != expected_questions:
    print(
        f"Warning: Expected {expected_questions} questionnaire items, "
        f"but found {len(question_columns)}."
    )


# ======================================================
# SAVE CSV
# ======================================================

output_file = output_folder / "questionnaire.csv"

df.to_csv(output_file, index=False)


# ======================================================
# SUMMARY
# ======================================================

print("=" * 60)
print("QUESTIONNAIRE DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print("\nInput folder:")
print(questionnaire_folder)

print("\nOutput file:")
print(output_file)

print(f"\nParticipants: {len(df)}")

print(f"Questionnaire items: {len(question_columns)}")

print(f"Duplicate patient IDs: {duplicate_ids}")

print(f"Number of columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDone!")
