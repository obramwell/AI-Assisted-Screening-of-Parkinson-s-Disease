# Explainable AI-Assisted Screening of Parkinson's Disease Using Multimodal Wearable and Clinical Data

## Project Overview

This project develops an explainable artificial intelligence (XAI) prototype to support Parkinson's disease screening using multimodal data from the Parkinson's Disease Smartwatch (PADS) dataset. The proposed machine learning framework integrates wearable sensor recordings, non-motor symptom questionnaire responses, and demographic information to differentiate Healthy Controls, Parkinson's Disease, and Other Movement Disorders.

The project is being developed as part of the Master of Data Analytics Capstone Project at the University of Niagara Falls Canada.

---

## Objectives

- Develop a reproducible multimodal machine learning pipeline.
- Compare multiple classification models.
- Evaluate the contribution of demographic, questionnaire, and wearable signal features.
- Provide global and participant-level explainability using SHAP.
- Develop an interactive dashboard for model interpretation.

---

## Dataset

This project uses the **Parkinson's Disease Smartwatch (PADS)** dataset available through PhysioNet.

**Dataset modalities include:**

- Patient demographic and clinical information
- Non-Motor Symptoms (NMS) questionnaire
- Smartwatch accelerometer recordings
- Smartwatch gyroscope recordings
- Neurological assessment metadata

The original dataset is **not included** in this repository.

Place the downloaded dataset under:

```text
data/raw/PADS/
```

---

## Repository Structure

```text
group4_parkinsons_project/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── data_dictionary/
│
├── notebooks/
│
├── src/
│
├── models/
│
├── outputs/
│   ├── tables/
│   ├── figures/
│   ├── metrics/
│   └── shap/
│
├── dashboard/
│
├── reports/
│
├── requirements.txt
└── README.md
```

---

## Project Workflow

1. Data audit
2. Data cleaning
3. Signal quality assessment
4. Feature engineering
5. Machine learning model development
6. Model evaluation
7. Explainability analysis
8. Dashboard development

---

## Team

Group 4 – Master of Data Analytics

- Allison Janet Bazan Luna
- Ophelia Sasha-Gay Bramwell
- Daniela Leon Granados
- Kadian Hutchinson
- Priti Khadka

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Matplotlib
- Jupyter Notebook
- Git
- GitHub

---

## License

This repository contains only project-generated code and documentation.

The original PADS dataset is distributed separately under its respective license.
