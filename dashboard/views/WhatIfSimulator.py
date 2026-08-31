import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from pathlib import Path

from utils.styles import load_css


# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

MODELS = (
    ROOT
    / "models"
)

DATA = (
    ROOT
    / "data"
    / "processed"
)


MODEL_FILE = (
    MODELS
    / "full_multimodal_xgboost.joblib"
)

DATASET_FILE = (
    DATA
    / "multimodal_full_task_aware.csv"
)


# ==========================================================
# CLASS LABELS
# ==========================================================

CLASS_NAMES = {

    0: "Healthy Controls",

    1: "Parkinson's Disease",

    2: "Other Movement Disorders",

}


# ==========================================================
# SCENARIOS
# ==========================================================

SCENARIOS = {

    "Baseline": "baseline",
    "Scenario 1: Non-Motor Symptoms": "non_motor",
    "Scenario 2: Questionnaire": "questionnaire",
    "Scenario 3: Wearable": "wearable",
    "Scenario 4: Combined": "combined",
}
# LOAD MODEL AND DATA
@st.cache_resource
def load_model():
    return joblib.load(
        MODEL_FILE
    )

@st.cache_data
def load_dataset():
    return pd.read_csv(
        DATASET_FILE
    )

model = load_model()
dataset = load_dataset()
feature_names = list(
    model.feature_names_in_
)
# PREDICTION FUNCTION
def predict_patient(
    patient_features,
):

    probabilities = model.predict_proba(
        patient_features
    )[0]

    prediction = int(
        np.argmax(
            probabilities
        )
    )

    confidence = float(
        np.max(
            probabilities
        )
    )

    return (
        prediction,
        confidence,
        probabilities,
    )
    # ==========================================================
# MAIN PAGE
# ==========================================================

def show():

    load_css()

    st.title("🧪 What-if Simulator")

    st.write(
        """
Simulate hypothetical changes in patient characteristics to
explore how the final Full Multimodal XGBoost model responds
under different clinical scenarios.

This tool is intended for educational and research purposes
only and should not be used for clinical diagnosis.
"""
    )

    st.divider()

    # Select participant
    participants = (
        dataset[
            [
                "patient_id",
                "label",
                "age",
                "gender",
            ]
        ]
        .sort_values("patient_id")
        .copy()
    )

    participant = st.selectbox(

        "Select Participant",

        participants["patient_id"],

    )

    patient = (
        dataset
        .loc[
            dataset["patient_id"] == participant
        ]
        .copy()
        .iloc[0]
    )

    # Feature vector
    patient_features = (
        patient[
            feature_names
        ]
        .to_frame()
        .T
    )

    (
        baseline_prediction,
        baseline_confidence,
        baseline_probabilities,
    ) = predict_patient(
        patient_features
    )

    st.subheader("Participant Information")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Age",
            int(patient["age"]),
        )

    with c2:

        st.metric(
            "Gender",
            patient["gender"],
        )

    with c3:

        st.metric(
            "Weight (kg)",
            patient["weight_kg"],
        )

    with c4:

        st.metric(
            "Handedness",
            patient["handedness"],
        )

    st.divider()
    # SIMULATION SCENARIOS
    st.subheader("Simulation Scenario")

    scenario = st.radio(
        "Select a scenario",
        list(SCENARIOS.keys()),
    )

    simulated_patient = patient_features.copy()
    # BASELINE
    if SCENARIOS[scenario] == "baseline":

        simulated_probabilities = baseline_probabilities
    # SCENARIO 1
    elif SCENARIOS[scenario] == "non_motor":
        simulated_patient["sleep_fatigue_count"] = np.clip(
            simulated_patient["sleep_fatigue_count"] + 2,
            0,
            4,
        )
        simulated_patient["gastrointestinal_count"] = np.clip(
            simulated_patient["gastrointestinal_count"] + 1,
            0,
            4,
        )
        simulated_patient["depression_anxiety_count"] = np.clip(
            simulated_patient["depression_anxiety_count"] + 2,
            0,
            4,
        )
        simulated_patient["pain_count"] = np.clip(
            simulated_patient["pain_count"] + 1,
            0,
            4,
        )

        simulated_probabilities = model.predict_proba(
            simulated_patient
        )[0]
    # SCENARIO 2
    elif SCENARIOS[scenario] == "questionnaire":

        questions = [
            "Q25",
            "Q02",
            "Q05",
            "Q01",
            "Q08",
        ]

        for question in questions:

            simulated_patient[question] = np.clip(
                simulated_patient[question] + 1,
                0,
                4,
            )

        simulated_probabilities = model.predict_proba(
            simulated_patient
        )[0]
    # SCENARIO 3
    elif SCENARIOS[scenario] == "wearable":

        motor_features = [
            "HoldWeight_RightWrist_gyro_y_dominant_frequency",
            "CrossArms_RightWrist_gyro_y_spectral_centroid",
            "LiftHold_Left_AccX_IQR",
            "TouchIndex_RightWrist_acc_magnitude_spectral_centroid",
            "CrossArms_Left_GyroMag_Min",
        ]

        for feature in motor_features:

            simulated_patient[feature] += (
                dataset[feature].std()
            )

        simulated_probabilities = model.predict_proba(
            simulated_patient
        )[0]
    # SCENARIO 4
    elif SCENARIOS[scenario] == "combined":

        simulated_patient["sleep_fatigue_count"] = np.clip(
            simulated_patient["sleep_fatigue_count"] + 2,
            0,
            4,
        )

        simulated_patient["gastrointestinal_count"] = np.clip(
            simulated_patient["gastrointestinal_count"] + 1,
            0,
            4,
        )

        simulated_patient["depression_anxiety_count"] = np.clip(
            simulated_patient["depression_anxiety_count"] + 2,
            0,
            4,
        )

        simulated_patient["pain_count"] = np.clip(
            simulated_patient["pain_count"] + 1,
            0,
            4,
        )

        questions = [
            "Q25",
            "Q02",
            "Q05",
            "Q01",
            "Q08",
        ]

        for question in questions:

            simulated_patient[question] = np.clip(
                simulated_patient[question] + 1,
                0,
                4,
            )

        motor_features = [
            "HoldWeight_RightWrist_gyro_y_dominant_frequency",
            "CrossArms_RightWrist_gyro_y_spectral_centroid",
            "LiftHold_Left_AccX_IQR",
            "TouchIndex_RightWrist_acc_magnitude_spectral_centroid",
            "CrossArms_Left_GyroMag_Min",
        ]

        for feature in motor_features:

            simulated_patient[feature] += (
                dataset[feature].std()
            )

        simulated_probabilities = model.predict_proba(
            simulated_patient
        )[0]

    simulated_prediction = int(
        np.argmax(
            simulated_probabilities
        )
    )

    simulated_confidence = float(
        np.max(
            simulated_probabilities
        )
    )

    st.divider()
    # PREDICTION RESULTS
    st.subheader("Prediction Results")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Baseline Prediction",
            CLASS_NAMES[baseline_prediction],
            f"{baseline_confidence:.3f}",
        )

    with c2:

        st.metric(
            "Scenario Prediction",
            CLASS_NAMES[simulated_prediction],
            f"{simulated_confidence:.3f}",
        )

    st.divider()
    # PROBABILITY COMPARISON
    st.subheader("Predicted Class Probabilities")
    comparison = pd.DataFrame({
        "Class": list(CLASS_NAMES.values()),
        "Baseline": baseline_probabilities,
        "Scenario": simulated_probabilities,
    })
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=comparison["Class"],
            y=comparison["Baseline"],
            name="Baseline",
        )
    )
    fig.add_trace(
        go.Bar(
            x=comparison["Class"],
            y=comparison["Scenario"],
            name="Scenario",
        )
    )
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        yaxis_title="Predicted Probability",
        yaxis=dict(range=[0,1]),
        height=450,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
        }
    )
    st.divider()
    # PROBABILITY COMPARISON TABLE
    st.subheader("Probability Comparison")
    comparison_table = pd.DataFrame({
        "Class": list(CLASS_NAMES.values()),
        "Baseline": np.round(
            baseline_probabilities,
            3,
        ),
        "Scenario": np.round(
            simulated_probabilities,
            3,
        ),

    })

    comparison_table["Change"] = np.round(
        comparison_table["Scenario"]
        - comparison_table["Baseline"],
        3,
    )
    st.dataframe(
        comparison_table,
        use_container_width=True,
        hide_index=True,
    )
    st.divider()
    # INTERPRETATION
    st.subheader("Clinical Interpretation")
    predicted_class = CLASS_NAMES[
        simulated_prediction
    ]
    delta = (
        simulated_probabilities[simulated_prediction]
        - baseline_probabilities[simulated_prediction]
    )
    if delta > 0:
        direction = "increased"
    elif delta < 0:
        direction = "decreased"
    else:
        direction = "did not change"

    st.info(
        f"""
The simulated scenario resulted in a predicted diagnosis of
**{predicted_class}** with a confidence of
**{simulated_confidence:.1%}**.

Compared with the baseline prediction, the probability of the
predicted class **{direction}** by
**{abs(delta):.1%}**.

This illustrates how modifying the selected variables can influence
the model output under hypothetical conditions. These simulations are
intended for research and educational purposes only and should not be
interpreted as clinical recommendations.
"""
    )

    with st.expander("View Raw Probabilities"):
        st.write("Baseline")
        st.write(baseline_probabilities)

        st.write("Scenario")
        st.write(simulated_probabilities)