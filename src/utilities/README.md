# Data Preparation Utilities

These scripts were developed to convert the original PADS JSON metadata into structured CSV files for dataset exploration, validation, and downstream preprocessing.

The scripts read the original files from `data/raw/` and save the generated CSV files to `data/interim/`. These CSV files preserve the information extracted from the source metadata in a structured tabular format and should not be considered fully cleaned or model-ready datasets.

## Scripts

- `patients_to_csv.py`  
  Converts participant demographic and clinical metadata into `data/interim/patients.csv`.

- `questionnaire_to_csv.py`  
  Converts the non-motor symptom questionnaire responses into `data/interim/questionnaire.csv`.

- `movement_metadata_to_csv.py`  
  Extracts participant, device, task, sampling, and wrist-file metadata into `data/interim/movement_metadata.csv`.

## Data Flow

`data/raw/` → `src/utilities/` → `data/interim/`

Further cleaning, preprocessing, signal-quality assessment, and feature engineering are performed separately before analysis-ready datasets are stored in `data/processed/`.
