import streamlit as st
import pandas as pd

from utils.styles import load_css
from utils.loaders import load_csv

from utils.paths import (
    PROCESSED_DATA,
)

DATASET = (
    PROCESSED_DATA /
    "multimodal_full_task_aware.csv"
)


def show():

    load_css()

    st.title("Participant Overview")

    st.write(
        """
This section summarizes the demographic information,
questionnaire responses, neurological assessment tasks,
sensor availability, and data quality indicators for
an individual participant included in the study.
"""
    )

    df = load_csv(DATASET)

    if df is None:

        st.error(
            "Participant dataset could not be loaded."
        )

        return

    participants = sorted(
        df["patient_id"].unique()
    )

    participant = st.selectbox(
        "Select Participant",
        participants,
    )

    participant_data = (
        df[
            df["patient_id"] == participant
        ]
        .iloc[0]
    )

    st.divider()
    # DEMOGRAPHIC INFORMATION
    st.subheader("Demographic Information")

    left, middle, right = st.columns(3)

    with left:

        st.metric(
            "Age",
            int(participant_data["age"]),
        )

        st.metric(
            "Gender",
            participant_data["gender"],
        )

    with middle:

        st.metric(
            "Weight (kg)",
            participant_data["weight_kg"],
        )

        st.metric(
            "Handedness",
            participant_data["handedness"],
        )

    with right:

        st.metric(
            "Family History",
            participant_data["family_history_any"],
        )

        st.metric(
            "Alcohol Effect",
            participant_data["alcohol_effect_on_tremor"],
        )

    st.divider()
    # QUESTIONNAIRE SUMMARY
    st.subheader("Questionnaire Summary")
    st.write(
        """
The participant's responses were grouped into clinically meaningful
non-motor symptom domains derived from the Non-Motor Symptoms
Questionnaire (NMSQuest).
"""
    )

    questionnaire = pd.DataFrame(
        {
            "Domain": [
                "Gastrointestinal",
                "Urinary",
                "Pain",
                "Apathy / Attention / Memory",
                "Distortion / Perception",
                "Depression / Anxiety",
                "Sexual Function",
                "Cardiovascular",
                "Sleep / Fatigue",
            ],
            "Count": [
                participant_data["gastrointestinal_count"],
                participant_data["urinary_count"],
                participant_data["pain_count"],
                participant_data["apathy_attention_memory_count"],
                participant_data["distortion_perception_count"],
                participant_data["depression_anxiety_count"],
                participant_data["sexual_function_count"],
                participant_data["cardiovascular_count"],
                participant_data["sleep_fatigue_count"],
            ],
        }
    )

    st.bar_chart(
        questionnaire,
        x="Domain",
        y="Count",
        use_container_width=True,
    )

    left, center, right = st.columns(3)

    with left:

        st.metric(
            "Questions Answered",
            int(participant_data["questions_answered"]),
        )

    with center:

        st.metric(
            "Missing Questions",
            int(participant_data["questions_missing"]),
        )

    with right:

        st.metric(
            "Questionnaire Status",
            participant_data["questionnaire_status"],
        )

    st.divider()
    # NEUROLOGICAL ASSESSMENT TASKS
    st.subheader("Neurological Assessment Tasks")

    st.write(
        """
The participant completed the standardized neurological assessment
protocol used throughout the study. The extracted wearable movement
features were derived from the following tasks.
"""
    )

    task_col1, task_col2, task_col3 = st.columns(3)

    with task_col1:

        st.success("✓ Relaxed ")
        st.success("✓ Relaxed Task")
        st.success("✓ Stretch Hold")
        st.success("✓ Drink Glass")

    with task_col2:

        st.success("✓ Lift Hold")
        st.success("✓ Hold Weight")
        st.success("✓ Point Finger")
        st.success("✓ Cross Arms")
    with task_col3:
        st.success("✓ Touch Index")
        st.success("✓ Touch Nose")
        st.success("✓ Entrainment")

    st.divider()

    # SENSOR AVAILABILITY

    st.subheader("Sensor Availability")

    st.write(
        """
Movement features were extracted from bilateral smartwatch recordings
using accelerometer and gyroscope sensors.
"""
    )

    sensor_left, sensor_right = st.columns(2)

    with sensor_left:

        st.info("⌚ Left Wrist")
        st.success("✓ Accelerometer")
        st.success("✓ Gyroscope")

    with sensor_right:

        st.info("⌚ Right Wrist")
        st.success("✓ Accelerometer")
        st.success("✓ Gyroscope")

    st.divider()
    # DATA QUALITY INDICATORS

    st.subheader("Data Quality Indicators")

    st.write(
        """
The indicators below summarize the completeness and consistency
of the selected participant's data after preprocessing and quality
control.
"""
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Questions Answered",
            int(participant_data["questions_answered"]),
        )

    with c2:

        st.metric(
            "Missing Questions",
            int(participant_data["questions_missing"]),
        )

    with c3:

        st.metric(
            "Duplicate Participant",
            "Yes"
            if participant_data["duplicate_patient_id"]
            else "No",
        )

    with c4:

        st.metric(
            "Questionnaire Status",
            participant_data["questionnaire_status"],
        )

    st.divider()

    # PARTICIPANT SUMMARY

    st.subheader("Participant Summary")

    true_class = {
        0: "Healthy Controls",
        1: "Parkinson's Disease",
        2: "Other Movement Disorders",
    }

    st.info(
        f"""
**Participant ID:** {participant}

**Diagnostic Group:** {true_class.get(participant_data['label'], 'Unknown')}

This participant contains demographic information, questionnaire
responses, and bilateral smartwatch-derived movement features
used throughout the machine learning pipeline.
"""
    )