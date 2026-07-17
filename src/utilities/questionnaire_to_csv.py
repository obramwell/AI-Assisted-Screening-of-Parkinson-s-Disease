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

questionnaire_folder = dataset_root / "questionnaire"

output_folder = project_root / "data" / "processed"
output_folder.mkdir(parents=True, exist_ok=True)

# ======================================================
# READ QUESTIONNAIRES
# ======================================================

records = []

json_files = sorted(questionnaire_folder.glob("*.json"))

print(f"Found {len(json_files)} questionnaire files.\n")

for json_file in json_files:

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    row = {
        "patient_id": data["subject_id"],
        "questionnaire_name": data["questionnaire_name"]
    }

    # Extract all 30 questions
    for item in data["item"]:

        question = f"Q{item['link_id']}"

        # Convert True/False -> 1/0
        row[question] = int(item["answer"])

    records.append(row)

# ======================================================
# DATAFRAME
# ======================================================

df = pd.DataFrame(records)

# Sort by patient ID
df["patient_id"] = df["patient_id"].astype(int)
df = df.sort_values("patient_id")
df["patient_id"] = df["patient_id"].apply(lambda x: f"{x:03d}")

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

print(f"\nOutput file:\n{output_file}")

print(f"\nParticipants: {len(df)}")

print(f"\nNumber of columns: {len(df.columns)}")

print("\nColumns:")

print(df.columns.tolist())

print("\nFirst 5 rows:\n")

print(df.head())

print("\nDone!")