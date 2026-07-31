# Data Dictionary

This document describes the data sources, variables, structures, and project-generated datasets used in the **Explainable AI-Assisted Screening of Parkinson's Disease Using Multimodal Wearable and Clinical Data** project.

The project uses the **Parkinson's Disease Smartwatch (PADS) dataset**, which contains participant demographic and clinical information, a 30-item Non-Motor Symptoms (NMS) questionnaire, smartwatch movement recordings, and author-provided preprocessed files.

The project follows the data structure:

```text
data/raw/
    ↓
data/interim/
    ↓
data/processed/
```

- **Raw data** contain the original PADS files.
- **Interim data** contain structured versions of the original files, such as CSV representations of JSON metadata.
- **Processed data** contain cleaned, feature-engineered, integrated, or model-ready datasets generated during the project.

---

# 1. Patient Information

**Source:** `data/raw/patients/*.json`  
**Project-generated intermediate file:** `data/interim/patients.csv`

The patient files contain demographic and clinical information for each participant. Each participant is identified by a unique ID that can be used to link patient information with questionnaire responses and movement recordings.

| Variable | Type | Description | Example |
|---|---|---|---|
| `resource_type` | String | Type of dataset resource | Patient |
| `id` | String | Unique participant identifier | 001 |
| `study_id` | String | Study identifier | PADS |
| `condition` | Categorical | Recorded clinical condition of the participant | Healthy / Parkinson's Disease / Essential Tremor |
| `disease_comment` | String | Additional information about the participant's clinical diagnosis | IPS mixed type |
| `age_at_diagnosis` | Integer | Recorded age at diagnosis | 63 |
| `age` | Integer | Participant age in years | 56 |
| `height` | Integer | Participant height in centimeters | 173 |
| `weight` | Integer | Participant weight in kilograms | 78 |
| `gender` | Categorical | Participant gender | Male / Female |
| `handedness` | Categorical | Participant's dominant hand | Right / Left |
| `appearance_in_kinship` | Boolean | Indicates reported occurrence of the condition in family history | True / False |
| `appearance_in_first_grade_kinship` | Boolean | Indicates reported occurrence of the condition among first-degree relatives | True / False |
| `effect_of_alcohol_on_tremor` | Categorical | Reported effect of alcohol consumption on tremor | Better / Worse / No Effect / Unknown |

## 1.1 Project Cleaning Considerations

During project preprocessing:

- Participant IDs are standardized for use across modalities.
- Categorical variables such as gender and handedness are standardized.
- Boolean family-history variables may be converted to interpretable categories such as `Yes`, `No`, or `Unknown`.
- Numeric demographic variables are validated for missing or implausible values.
- `age_at_diagnosis` is not applicable to healthy participants.
- `disease_comment` is retained for descriptive reference but excluded from predictive modeling because it may contain diagnosis-specific information and introduce target leakage.

---

# 2. Diagnostic Label

The diagnostic label used as the target variable for machine learning is available in the author-provided:

`data/raw/preprocessed/file_list.csv`

The label can be associated with participants through their unique participant identifier.

| Label | Clinical Class | Description |
|---:|---|---|
| 0 | Healthy | Participants assigned to the Healthy Control class |
| 1 | Parkinson's Disease | Participants assigned to the Parkinson's Disease class |
| 2 | Other Movement Disorders | Participants assigned to the Other Movement Disorders class |

The `condition` variable contains the recorded clinical condition of the participant, while `label` represents the broader three-class grouping used as the target variable for the proposed machine learning analysis.

Specific diagnoses may therefore be grouped under the **Other Movement Disorders (OMD)** class.

For reporting purposes, the three classes may be abbreviated as:

| Abbreviation | Class |
|---|---|
| HC | Healthy Control |
| PD | Parkinson's Disease |
| OMD | Other Movement Disorders |

---

# 3. Non-Motor Symptoms Questionnaire

**Source:** `data/raw/questionnaire/*.json`  
**Project-generated intermediate file:** `data/interim/questionnaire.csv`

The questionnaire files contain responses to the standardized **30-item Non-Motor Symptoms (NMS) questionnaire**.

Each participant's responses are associated with their unique participant identifier.

## 3.1 Raw Questionnaire Structure

| Variable | Type | Description | Example |
|---|---|---|---|
| `resource_type` | String | Type of dataset resource | questionnaire_response |
| `subject_id` | String | Unique participant identifier | 001 |
| `study_id` | String | Study identifier | PADS |
| `id` | String | Questionnaire response identifier | Non-Motor Symptoms |
| `questionnaire_name` | String | Name or abbreviation of the questionnaire | NMS |
| `item` | List | Collection of the 30 questionnaire items and responses | — |
| `link_id` | String | Unique identifier assigned to each questionnaire item | 01 |
| `text` | String | Full text of the questionnaire item | Dribbling of saliva during the daytime |
| `answer` | Boolean | Indicates whether the participant reported the symptom | True / False |

---

## 3.2 Interim Questionnaire Structure

For analytical purposes, the questionnaire responses are consolidated into one row per participant.

The 30 questionnaire responses are represented as individual binary variables.

| Variable | Type | Description |
|---|---|---|
| `patient_id` | String | Unique participant identifier |
| `questionnaire_name` | String | Questionnaire name or abbreviation |
| `Q01`–`Q30` | Binary | Individual NMS questionnaire responses, encoded as 0 = symptom not reported and 1 = symptom reported |

---

## 3.3 NMS Question Mapping

The 30 NMS questionnaire items are organized into ten symptom categories.

| Question | Category | Symptom |
|---|---|---|
| `Q01` | Gastrointestinal tract | Dribbling |
| `Q02` | Distortion of perception | Taste / smelling |
| `Q03` | Gastrointestinal tract | Swallowing |
| `Q04` | Gastrointestinal tract | Vomiting |
| `Q05` | Gastrointestinal tract | Constipation |
| `Q06` | Gastrointestinal tract | Bowel inconsistence |
| `Q07` | Gastrointestinal tract | Bowel emptying incomplete |
| `Q08` | Urinal tract | Urgency |
| `Q09` | Urinal tract | Nocturia |
| `Q10` | Pain | Pains |
| `Q11` | Miscellaneous | Weight |
| `Q12` | Apathy / attention / memory | Remembering |
| `Q13` | Apathy / attention / memory | Loss of interest |
| `Q14` | Distortion of perception | Hallucinations |
| `Q15` | Apathy / attention / memory | Concentrating |
| `Q16` | Depression / anxiety | Sad / blues |
| `Q17` | Depression / anxiety | Anxiety |
| `Q18` | Sexual function | Sex drive |
| `Q19` | Sexual function | Sex difficulty |
| `Q20` | Cardiovascular | Dizzy |
| `Q21` | Cardiovascular | Falling |
| `Q22` | Sleep / fatigue | Daytime sleepiness |
| `Q23` | Sleep / fatigue | Insomnia |
| `Q24` | Sleep / fatigue | Intense vivid dreams |
| `Q25` | Sleep / fatigue | Acting out during dreams |
| `Q26` | Sleep / fatigue | Restless legs |
| `Q27` | Cardiovascular | Swelling |
| `Q28` | Miscellaneous | Sweating |
| `Q29` | Miscellaneous | Diplopia |
| `Q30` | Distortion of perception | Delusions |

---

## 3.4 NMS Symptom Categories

For feature engineering, individual questionnaire responses can be summarized into symptom-domain counts.

| Derived Variable | Questions | Description |
|---|---|---|
| `gastrointestinal_count` | Q01, Q03, Q04, Q05, Q06, Q07 | Number of gastrointestinal symptoms reported |
| `urinal_count` | Q08, Q09 | Number of urinary-tract symptoms reported |
| `pain_count` | Q10 | Number of pain symptoms reported |
| `miscellaneous_count` | Q11, Q28, Q29 | Number of miscellaneous symptoms reported |
| `apathy_attention_memory_count` | Q12, Q13, Q15 | Number of apathy, attention, or memory symptoms reported |
| `distortion_perception_count` | Q02, Q14, Q30 | Number of perception-related symptoms reported |
| `depression_anxiety_count` | Q16, Q17 | Number of depression or anxiety symptoms reported |
| `sexual_function_count` | Q18, Q19 | Number of sexual-function symptoms reported |
| `cardiovascular_count` | Q20, Q21, Q27 | Number of cardiovascular symptoms reported |
| `sleep_fatigue_count` | Q22, Q23, Q24, Q25, Q26 | Number of sleep or fatigue symptoms reported |

An additional participant-level feature may be generated:

| Variable | Type | Description |
|---|---|---|
| `total_symptom_count` | Integer | Total number of reported symptoms across Q01–Q30 |

---

# 4. Movement Assessment Metadata

**Source:** `data/raw/movement/observation_*.json`  
**Project-generated intermediate file:** `data/interim/movement_metadata.csv`

The movement observation files contain metadata describing each participant's neurological assessment sessions and their associated smartwatch time-series recordings.

The observation metadata identify:

- Participant
- Recording device
- Sampling characteristics
- Neurological assessment task
- Device location
- Sensor channels
- Measurement units
- Corresponding raw time-series file

---

## 4.1 Observation-Level Metadata

| Variable | Type | Description | Example |
|---|---|---|---|
| `resource_type` | String | Type of dataset resource | observation |
| `subject_id` | String | Unique participant identifier | 001 |
| `study_id` | String | Study identifier | PADS |
| `device_id` | String | Smartwatch device used for data collection | Apple Watch Series 4 |
| `id` | String | Type of assessment represented by the observation | Neurological Assessment |
| `endianness` | String | Byte order used to represent numerical data | little |
| `sampling_rate` | Integer | Sensor sampling frequency in Hertz | 100 |
| `data_type` | String | Numerical data type used for sensor recordings | float |
| `bits` | Integer | Bit depth used for recorded numerical values | 32 |
| `session` | List | Collection of neurological assessment sessions/tasks | — |

---

## 4.2 Session-Level Metadata

Each element within the `session` structure represents one neurological assessment task.

| Variable | Type | Description | Example |
|---|---|---|---|
| `record_name` | String | Name of neurological assessment task | Relaxed |
| `rows` | Integer | Expected number of time-series observations for the task | 2048 |
| `records` | List | Collection of smartwatch recordings associated with the task | — |

---

## 4.3 Recording-Level Metadata

Each assessment task contains individual records corresponding to the smartwatch device location.

| Variable | Type | Description | Example |
|---|---|---|---|
| `device_location` | Categorical | Wrist on which the smartwatch was worn | LeftWrist / RightWrist |
| `channels` | List | Sensor channels contained in the time-series file | Time, Accelerometer_X, etc. |
| `units` | List | Measurement units corresponding to each sensor channel | s, g, rad/s |
| `file_name` | String | Relative path to the associated raw time-series file | timeseries/001_Relaxed_LeftWrist.txt |

Each participant-task combination contains paired recordings from the left and right wrists.

---

## 4.4 Neurological Assessment Tasks

The movement observation metadata organize smartwatch recordings according to 11 neurological assessment tasks.

| `record_name` | General Description |
|---|---|
| `Relaxed` | Movement recording collected during a relaxed/resting condition |
| `RelaxedTask` | Movement recording collected during the defined relaxed task condition |
| `StretchHold` | Movement recording associated with the stretch-and-hold assessment |
| `LiftHold` | Movement recording associated with the lift-and-hold assessment |
| `HoldWeight` | Movement recording collected while holding a weight |
| `PointFinger` | Movement recording associated with the pointing-finger assessment |
| `DrinkGlas` | Movement recording associated with the drinking-glass assessment |
| `CrossArms` | Movement recording associated with the crossed-arms assessment |
| `TouchIndex` | Movement recording associated with the touch-index assessment |
| `TouchNose` | Movement recording associated with the touch-nose assessment |
| `Entrainment` | Movement recording associated with the entrainment assessment |

> **Note:** The task names preserve the original `record_name` values used by the PADS dataset, including `DrinkGlas`. Exact clinical task procedures should be interpreted using the official PADS documentation rather than inferred solely from task names.

---

## 4.5 Project-Generated Movement Metadata Structure

The original nested movement JSON files are transformed into a structured participant-task table.

| Variable | Type | Description |
|---|---|---|
| `patient_id` | String | Unique participant identifier |
| `device` | Categorical | Smartwatch model used during recording |
| `sampling_rate` | Integer | Nominal sampling frequency in Hz |
| `task` | Categorical | Neurological assessment task |
| `samples` | Integer | Expected number of samples |
| `left_file` | String | Path/reference to the left-wrist recording |
| `right_file` | String | Path/reference to the right-wrist recording |

Additional validation variables may be generated during the data audit, including:

| Variable | Type | Description |
|---|---|---|
| `duration_seconds` | Float | Expected recording duration calculated from sample count and nominal sampling rate |
| `left_exists` | Boolean | Indicates whether the expected left-wrist file exists |
| `right_exists` | Boolean | Indicates whether the expected right-wrist file exists |

The complete dataset contains:

- **469 participants**
- **11 neurological assessment tasks per participant**
- **5,159 participant-task records**
- **10,318 raw wrist-specific time-series recordings**

---

# 5. Raw Smartwatch Time-Series Data

**Source:** `data/raw/movement/timeseries/*.txt`

The raw time-series files contain inertial sensor measurements collected from smartwatches during neurological assessment tasks.

Each raw recording corresponds to a specific:

- Participant
- Neurological assessment task
- Wrist location

For example:

```text
timeseries/001_Relaxed_LeftWrist.txt
```

represents the left-wrist recording for Participant 001 during the `Relaxed` assessment.

---

## 5.1 Time-Series Channels

Each time-series file contains seven columns.

| Variable | Type | Unit | Description |
|---|---|---|---|
| `Time` | Float | seconds (s) | Timestamp associated with each sensor observation |
| `Accelerometer_X` | Float | g | Acceleration measurement along the X-axis |
| `Accelerometer_Y` | Float | g | Acceleration measurement along the Y-axis |
| `Accelerometer_Z` | Float | g | Acceleration measurement along the Z-axis |
| `Gyroscope_X` | Float | rad/s | Angular velocity measurement along the X-axis |
| `Gyroscope_Y` | Float | rad/s | Angular velocity measurement along the Y-axis |
| `Gyroscope_Z` | Float | rad/s | Angular velocity measurement along the Z-axis |

---

## 5.2 Recording Characteristics

The movement data use a nominal sampling frequency of:

```text
100 Hz
```

Two primary recording lengths are present:

| Samples | Expected Duration |
|---:|---:|
| 1024 | 10.24 seconds |
| 2048 | 20.48 seconds |

The longer recordings correspond to:

- `Relaxed`
- `RelaxedTask`
- `Entrainment`

The remaining tasks use the shorter recording length.

Dataset-wide temporal validation is performed using the actual timestamps in each raw signal file rather than assuming exact uniform sampling.

---

# 6. Signal Quality Assessment

Raw movement recordings are evaluated before feature extraction.

Quality-assurance checks include:

| Quality Metric | Assessment |
|---|---|
| Recording length | Actual rows compared with expected rows |
| Missing values | Checked across all signal channels |
| Invalid values | Infinite/non-finite values identified |
| Constant channels | Sensor channels with no variability identified |
| Statistical extreme values | Values with absolute Z-score greater than 3 used as an exploratory indicator |

Statistical extreme values are **not automatically treated as noise**, because high-amplitude observations may represent genuine participant movement.

The outlier percentage is therefore retained as an exploratory signal-quality indicator rather than an automatic exclusion criterion.

---

# 7. Author-Provided Preprocessed Data

**Source:** `data/raw/preprocessed/`

The PADS dataset provides author-generated preprocessed files intended for machine learning applications.

---

## 7.1 Preprocessed Movement Data

**Source:** `data/raw/preprocessed/movement/*_ml.bin`

| Attribute | Description |
|---|---|
| File Format | Binary (`.bin`) |
| Data Type | `float32` |
| Data Modality | Smartwatch movement signals |
| Sensor Modalities | Accelerometer and gyroscope |
| Device Locations | Left and right wrists |
| Sensor Axes | X, Y, Z |
| Intended Use | Machine learning analysis |

The author-provided preprocessing pipeline is used as a reference when defining the project's signal preprocessing procedure.

The original preprocessing includes treatment of acquisition-related signal characteristics such as the initial smartwatch notification vibration and accelerometer gravitational components.

Any preprocessing steps adopted or modified by the project should be documented explicitly to maintain reproducibility.

---

## 7.2 Preprocessed Questionnaire Data

**Source:** `data/raw/preprocessed/questionnaire/*_ml.bin`

| Attribute | Description |
|---|---|
| File Format | Binary (`.bin`) |
| Data Modality | Non-motor symptom questionnaire responses |
| Intended Use | Machine learning analysis |

The author-provided questionnaire representation can be compared with the project-generated questionnaire table to validate the analytical representation.

---

## 7.3 Official Participant Table

**Source:** `data/raw/preprocessed/file_list.csv`

The author-provided `file_list.csv` contains consolidated participant-level demographic, clinical, and diagnostic-label information.

It can be used as a reference to validate:

- Participant IDs
- Clinical conditions
- Diagnostic labels

---

# 8. Project-Generated Intermediate Files

During the initial data preparation stage, selected original PADS JSON files are transformed into structured tabular formats.

These files are stored in:

```text
data/interim/
```

They preserve the source information while making it easier to inspect, validate, and process programmatically.

| File | Source | Unit of Observation | Description |
|---|---|---|---|
| `patients.csv` | `data/raw/patients/*.json` | Participant | Consolidated participant demographic and clinical information |
| `questionnaire.csv` | `data/raw/questionnaire/*.json` | Participant | Consolidated 30-item NMS questionnaire responses |
| `movement_metadata.csv` | `data/raw/movement/observation_*.json` | Participant-task | Consolidated movement assessment and sensor metadata |

These files should not be interpreted as final model-ready data.

---

# 9. Project-Generated Processed Files

Processed files contain datasets that have undergone cleaning, feature engineering, aggregation, integration, or other analytical transformations.

They are stored in:

```text
data/processed/
```

Examples include:

| File | Unit of Observation | Description |
|---|---|---|
| `demographics_clean.csv` | Participant | Cleaned demographic and participant-level clinical variables |
| `questionnaire_cleaned.csv` | Participant | Cleaned questionnaire responses and derived symptom variables |
| `questionnaire_item_summary.csv` | Questionnaire item | Summary statistics for individual questionnaire items |
| `questionnaire_participant_summary.csv` | Participant | Participant-level questionnaire summary |
| `questionnaire_completion_summary.csv` | Dataset | Questionnaire completion and quality summary |

Additional Week 3 outputs will include wearable feature tables and the final integrated participant-level analytical dataset.

---

# 10. Wearable Feature Engineering ***CHANGE AT THE END OF WEEK 3

During Week 3, raw smartwatch time series are transformed into numerical features suitable for conventional machine-learning models.

Wearable features are generated separately according to:

- Participant
- Task
- Wrist
- Sensor modality
- Sensor axis or magnitude
- Feature type

The recommended naming convention is:

```text
Task_Wrist_Sensor_Axis_Feature
```

For example:

```text
TouchNose_Left_Acc_X_mean
TouchNose_Left_Acc_Mag_RMS
Entrainment_Right_Gyro_Z_dominant_frequency
```

---

## 10.1 Signal Magnitude Features

Three-dimensional accelerometer and gyroscope channels may be combined into orientation-independent magnitude signals.

### Accelerometer Magnitude

```text
Acc_Magnitude = sqrt(Accelerometer_X² + Accelerometer_Y² + Accelerometer_Z²)
```

### Gyroscope Magnitude

```text
Gyro_Magnitude = sqrt(Gyroscope_X² + Gyroscope_Y² + Gyroscope_Z²)
```

---

## 10.2 Time-Domain Features

Potential time-domain features include:

| Feature | Description |
|---|---|
| `mean` | Arithmetic mean of the signal |
| `median` | Median signal value |
| `std` | Standard deviation |
| `min` | Minimum signal value |
| `max` | Maximum signal value |
| `range` | Difference between maximum and minimum |
| `iqr` | Interquartile range |
| `rms` | Root mean square |
| `energy` | Signal energy |

The final feature set should be documented after feature-quality assessment.

---

## 10.3 Frequency-Domain Features

Potential frequency-domain features include:

| Feature | Description |
|---|---|
| `dominant_frequency` | Frequency containing the largest spectral component |
| `spectral_centroid` | Weighted center of the signal frequency spectrum |
| `spectral_entropy` | Measure of spectral complexity or dispersion |
| `spectral_power` | Total power represented in the frequency spectrum |

The final frequency-domain feature set should be documented after implementation and validation.

---

## 10.4 Wrist and Task Features

Wearable features retain information identifying:

- Left wrist
- Right wrist
- Neurological assessment task
- Accelerometer or gyroscope modality

Left-right comparisons may also be used to investigate potential movement asymmetry.

Any derived symmetry or asymmetry variables should be documented in the wearable feature dictionary.

---

# 11. Feature Quality Assessment

After wearable feature extraction, feature-level quality checks include:

- Missing values
- Infinite or invalid values
- Constant features
- Low-variance features
- Duplicate features
- Strongly correlated features

Strong correlations should initially be treated as indicators of possible redundancy rather than automatic reasons for feature removal.

Any feature-selection decisions must be documented and implemented in a manner that avoids information leakage from validation or test data.

---

# 12. Participant-Level Data Integration

The final analytical dataset combines:

```text
Demographics
      +
NMS Questionnaire
      +
NMS Symptom Domains
      +
Wearable Features
      ↓
Integrated Participant-Level Dataset
```

The expected analytical unit is:

```text
one row per participant
```

Participant records are linked using the standardized `patient_id`.

The final dataset should contain **469 unique participant records**, subject to verification after multimodal integration.

---

# 13. Data Leakage Prevention

Participant-level separation must be preserved throughout the modeling pipeline.

All observations, questionnaire responses, demographic information, tasks, wrist recordings, and derived wearable features associated with one participant must remain within the same dataset partition.

The project uses separate:

- Training set
- Validation set
- Test set

No participant should appear in more than one partition.

Variables containing direct or indirect diagnostic information that would not normally be available in a screening context should not be used as predictive features.

Examples include:

- `disease_comment`
- Diagnostic `condition`
- Target `label`
- Other diagnosis-derived variables when inappropriate for the intended screening scenario

The target `label` is retained only as the outcome variable for supervised machine learning.

---

# 14. Data Flow

The project data flow can be summarized as:

```text
Original PADS Data
        │
        ├── patients/*.json
        ├── questionnaire/*.json
        └── movement/
              ├── observation_*.json
              └── timeseries/*.txt
        │
        ▼
data/raw/
        │
        ▼
Structured Data Conversion
        │
        ├── patients.csv
        ├── questionnaire.csv
        └── movement_metadata.csv
        │
        ▼
data/interim/
        │
        ▼
Data Quality Assessment
        │
        ├── Demographic validation
        ├── Questionnaire validation
        ├── Movement metadata validation
        └── Raw signal quality assessment
        │
        ▼
Signal Preprocessing
        │
        ▼
Wearable Feature Engineering
        │
        ├── Time-domain features
        ├── Frequency-domain features
        ├── Signal magnitude
        ├── Task-specific features
        └── Wrist-specific / symmetry features
        │
        ▼
Feature Quality Assessment
        │
        ▼
Participant-Level Aggregation
        │
        ▼
Demographics + Questionnaire + Wearable Integration
        │
        ▼
data/processed/
        │
        ▼
Model-Ready Participant-Level Dataset
        │
        ▼
Machine Learning + Explainable AI
```

---

# 15. Data Dictionary Maintenance

This data dictionary should be updated whenever:

- A new processed dataset is created.
- A new wearable feature is introduced.
- A feature is renamed or removed.
- Feature-selection rules are applied.
- New derived variables are created.
- The final participant-level analytical structure changes.

The final wearable feature dictionary should document, at minimum:

| Field | Description |
|---|---|
| Feature Name | Exact model variable name |
| Source | Original source signal or dataset |
| Task | Neurological task, when applicable |
| Wrist | Left / Right, when applicable |
| Sensor | Accelerometer / Gyroscope |
| Axis | X / Y / Z / Magnitude |
| Feature Domain | Time / Frequency / Symmetry / Clinical |
| Description | Interpretation of the variable |
| Unit | Unit where applicable |
| Modeling Use | Whether the variable is retained for modeling |

This ensures that the transformation from the original PADS dataset to the final analytical dataset remains transparent and reproducible.