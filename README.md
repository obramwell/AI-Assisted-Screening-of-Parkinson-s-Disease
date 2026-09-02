<h1 align="center">Explainable AI-Assisted Screening of Parkinson's Disease</h1>

<p align="center">
  <strong>Multimodal Machine Learning Using Wearable, Demographic and Questionnaire Data</strong>
</p>

<p align="center">
  <strong>Explainable AI</strong> •
  <strong>Multimodal Data</strong> •
  <strong>Machine Learning</strong> •
  <strong>Clinical Decision Support</strong>
</p>

<p align="center">
  <img src="dashboard/assets/logo.png"
       alt="AI-Assisted Screening of Parkinson's Disease Logo"
       width="300">
</p>

---

## 🧠 Project Overview

This project develops an Explainable Artificial Intelligence (XAI) prototype to support the screening of Parkinson's disease using multimodal data from the Parkinson's Disease Smartwatch (PADS) dataset.

The machine learning framework integrates wearable sensor data, non-motor symptom questionnaire responses, demographic information, and movement-related data to explore the classification of:

- Healthy Controls
- Parkinson's Disease
- Other Movement Disorders

> **Important:** The project is designed as an analytical and decision-support prototype and is not intended to replace clinical diagnosis.

The project was developed as part of the **Master of Data Analytics Capstone Project** at the **University of Niagara Falls Canada**.

---

## 🎯 Objectives

The objectives of this project are to:

- Develop a reproducible multimodal machine learning pipeline.
- Prepare and analyze demographic, questionnaire, movement, and wearable sensor data.
- Assess signal quality and perform wearable signal preprocessing.
- Extract and evaluate relevant features from the available data.
- Develop baseline models using individual data modalities.
- Develop and evaluate multimodal classification models.
- Apply cross-validation, feature selection, and hyperparameter tuning.
- Evaluate model performance using validation and final test data.
- Assess the reliability of predicted probabilities using calibration analysis.
- Identify and examine incorrect predictions through error analysis.
- Provide global and participant-level explainability using SHAP.
- Develop a what-if simulation component to explore changes in model predictions.
- Develop an interactive Streamlit dashboard to present and explore the project's analytical results.

---

## 📊 Dataset

This project uses the **Parkinson's Disease Smartwatch (PADS)** dataset.

The dataset includes multiple sources of information, including:

- Demographic and clinical information
- Non-Motor Symptoms (NMS) questionnaire responses
- Movement-related metadata
- Smartwatch accelerometer recordings
- Smartwatch gyroscope recordings

> **Note:** The original dataset is not included in this repository.

---

## 🔄 Project Workflow

The project was completed through the following stages:

1. Data audit
2. Questionnaire cleaning and demographic preparation
3. Movement metadata inspection
4. Raw wearable signal inspection
5. Signal quality assessment
6. Participant data splitting
7. Signal preprocessing and validation
8. Correlation and feature quality analysis
9. Task and wrist analysis
10. Task-aware data integration
11. Modality-specific baseline model development
12. Model results comparison
13. Multimodal data integration
14. Cross-validation and feature selection
15. Hyperparameter tuning
16. Validation evaluation
17. Final test evaluation
18. Calibration analysis
19. SHAP-based explainability
20. Error analysis
21. What-if simulation
22. Interactive dashboard development

---

## 🗂️ Repository Structure

```text
AI-Assisted-Screening-of-Parkinson-s-Disease/
│
├── assets/
│   └── logo.png
│
├── dashboard/
│
├── data/
│   ├── raw/
│   ├── interim/
│   │   └── timeseries/
│   ├── processed/
│   └── data_dictionary/
│
├── models/
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_questionnaire_cleaning.ipynb
│   ├── 02a_demographic_preparation.ipynb
│   ├── 03_movement_metadata_inspection.ipynb
│   ├── 04_raw_signal_inspection.ipynb
│   ├── 05_assess_signal_quality.ipynb
│   ├── 06_participant_data_splitting.ipynb
│   ├── 07_signal_preprocessing_validation.ipynb
│   ├── 07a_correlation_analysis.ipynb
│   ├── 07b_feature_quality_analysis.ipynb
│   ├── 08_task_wrist_analysis.ipynb
│   ├── 09_task_aware_data_integration.ipynb
│   ├── 10_modeling_framework_validation.ipynb
│   ├── 10a_demographic_baseline_models.ipynb
│   ├── 10b_questionnaire_baseline_models.ipynb
│   ├── 10c_wearable_baseline_models.ipynb
│   ├── 10d_model_results_comparison.ipynb
│   ├── 11_multimodal_data_integration.ipynb
│   ├── 11a_cross_validation_feature_selection.ipynb
│   ├── 11b_hyperparameter_tuning.ipynb
│   ├── 12a_validation_evaluation.ipynb
│   ├── 12b_final_test_evaluation.ipynb
│   ├── 13_calibration_analysis.ipynb
│   ├── 14_SHAP_Explainability.ipynb
│   ├── 15_error_analysis.ipynb
│   └── 16_What-if_Simulation.ipynb
│
├── outputs/
│   ├── error_analysis/
│   ├── figures/
│   ├── metrics/
│   ├── shap/
│   ├── simulation/
│   └── tables/
│
├── reports/
│
├── scripts/
│
├── src/
│   ├── features/
│   │   ├── frequency_features.py
│   │   ├── magnitude.py
│   │   └── time_features.py
│   │
│   ├── modeling/
│   │   ├── baseline.py
│   │   ├── config.py
│   │   ├── data_loader.py
│   │   ├── evaluation.py
│   │   ├── feature_sets.py
│   │   ├── models.py
│   │   ├── outputs.py
│   │   ├── preprocessing.py
│   │   ├── split_validation.py
│   │   ├── tuning.py
│   │   └── workflow.py
│   │
│   ├── preprocessing/
│   │   ├── l1_trend_filter.py
│   │   ├── preprocessing.py
│   │   ├── run_signal_preprocessing.py
│   │   └── signal_preprocessing.py
│   │
│   ├── utilities/
│   ├── data_loading.py
│   ├── evaluation.py
│   └── feature_engineering.py
│
├── .gitignore
├── README.md
└── requirements.txt

```
---

## 📈 Project Outputs

The project generates analytical outputs throughout the workflow, including:

- Model evaluation metrics
- Visualizations and figures
- SHAP explainability results
- Error analysis outputs
- What-if simulation results
- Summary tables

All project outputs are organized within the `outputs/` directory.

---

## 💻 Dashboard

An interactive Streamlit dashboard was developed to bring together and present the project's analytical components and results.

The dashboard supports exploration of:

- Model results
- Explainability findings
- Error analysis
- What-if simulation

## ▶️ Running the Dashboard

The interactive dashboard was developed using Streamlit.

### Install the required dependencies

From the project root directory, run:

```bash
pip install -r requirements.txt

```

---

### Launch the dashboard

Run the Streamlit application:

```bash
streamlit run dashboard/app.py
```

---

## 📊 Key Results

The multimodal machine learning approach achieved the strongest overall performance by combining demographic, questionnaire, and wearable-derived features.

### Best Performing Model

The **Full Multimodal XGBoost model** achieved the strongest overall performance on the independent test dataset.

- **Accuracy:** 71.83%
- **Balanced Accuracy:** 61.72%
- **Macro F1-Score:** 64.25%

The model outperformed the other selected feature configurations in terms of overall test accuracy and Macro F1-score.

| Feature Configuration | Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| **Full Multimodal** | **XGBoost** | **71.83%** | 61.72% | **64.25%** |
| Wearable + Questionnaire | Random Forest | 66.20% | **63.33%** | 61.57% |
| Demographics + Questionnaire | Logistic Regression | 63.38% | 60.57% | 57.50% |

These findings suggest that combining **demographic, questionnaire, and wearable-derived information** provides stronger predictive performance than the evaluated reduced feature configurations.

The final model showed lower performance on previously unseen participants compared with validation results, highlighting the importance of independent testing when evaluating machine learning models. 

---

## 💡 Key Recommendations

Based on the project findings and identified limitations, the following areas are recommended for future research:

- **Validate the model using independent datasets:** The model should be evaluated using additional and more diverse participant populations to assess whether its performance generalizes beyond the current dataset.

- **Investigate real-world wearable data:** Future research could evaluate data collected during everyday activities to determine whether the model performs consistently outside standardized neurological assessment tasks.

- **Further investigate classification challenges within the OMD group:** Additional research could examine individual movement-disorder subtypes to better understand the impact of within-class heterogeneity on model performance and classification errors.

- **Continue improving model generalization:** Future work could explore additional data, refined feature engineering approaches, and model optimization techniques to improve performance on previously unseen participants.

- **Maintain appropriate clinical boundaries:** The prototype should continue to be treated as an analytical screening-support tool for research and educational purposes and not as a replacement for professional clinical diagnosis.

---

## 📄 Project Report

A detailed report documenting the project's methodology, data preparation, feature engineering, model development, evaluation, explainability analysis, findings, limitations, and recommendations is available as part of the project documentation.

The report provides a more comprehensive discussion of the research process and results presented in this repository.

---

## 🛠️ Technologies

The project uses the following technologies and tools:

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Matplotlib
- Jupyter Notebook
- Streamlit
- Git
- GitHub

---

## 🔁 Reproducibility

The project was organized to support a reproducible and reviewable analytical workflow.

- Sequentially numbered Jupyter notebooks document the main stages of the analysis.
- Reusable Python modules support data preparation, preprocessing, feature extraction, modelling, evaluation, and workflow management.
- Trained models and project outputs are stored separately from the main analytical code.
- The `requirements.txt` file lists the Python packages required to run the project.
- Git and GitHub were used for version control and tracking changes throughout development.

---

## 👥 Team

**Group 4 – Master of Data Analytics**

- Allison Janet Bazan Luna
- Ophelia Sasha-Gay Bramwell
- Daniela Leon Granados
- Kadian Hutchinson
- Priti Khadka

---

## 📄 License

This repository contains project-generated code and documentation.

The original PADS dataset is not included in this repository and remains subject to its respective data access and licensing terms.
