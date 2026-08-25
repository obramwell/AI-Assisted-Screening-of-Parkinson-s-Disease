import streamlit as st

from utils.constants import (
    MODEL_NAME,
    N_CLASSES,
    N_FEATURES,
    N_PARTICIPANTS
)


def show():

    # ======================================================
    # HERO
    # ======================================================

    st.markdown("## Welcome")

    st.write(
        """
This dashboard presents the final multimodal machine learning system developed
for the early screening of Parkinson's disease.

The application combines wearable sensor measurements, non-motor symptom
questionnaire responses and demographic information with Explainable Artificial
Intelligence (SHAP) to provide transparent and clinically interpretable
predictions.
"""
    )

    st.write("")

    # ======================================================
    # KPI
    # ======================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("👥 Participants", N_PARTICIPANTS)
    c2.metric("🧬 Selected Features", N_FEATURES)
    c3.metric("🌲 Final Model", MODEL_NAME)
    c4.metric("🩺 Diagnostic Classes", N_CLASSES)

    st.divider()

    # ======================================================
    # OVERVIEW
    # ======================================================

    left, right = st.columns([1.2, 1])

    with left:

        st.subheader("Project Overview")

        st.write(
            """
The proposed framework integrates multiple complementary sources of
information to improve Parkinson's disease screening.

Wearable sensor measurements capture motor performance during standardized
neurological tasks, while questionnaire responses describe non-motor symptoms.
Demographic variables provide additional clinical context.

After preprocessing and feature selection, a Random Forest classifier performs
multiclass prediction. SHAP is then used to explain both global model behavior
and individual patient predictions.
"""
        )

    with right:

        st.subheader("Dataset")

        st.info(
            """
**Participants:** 469

**Classes**

• Healthy

• Parkinson's Disease

• Other Movement Disorders

**Modalities**

• Wearable sensors

• Questionnaire

• Demographics
"""
        )

    st.divider()

    # ======================================================
    # PIPELINE
    # ======================================================

    st.subheader("Model Pipeline")

    cols = st.columns(7)

    steps = [
        ("📱", "Wearable\nSensors"),
        ("📋", "Questionnaire"),
        ("👤", "Demographics"),
        ("⚙️", "Feature\nEngineering"),
        ("🧬", "Feature\nSelection"),
        ("🌲", "Random\nForest"),
        ("🧠", "SHAP")
    ]

    for col, (icon, label) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
<div style="
text-align:center;
padding:20px;
border:1px solid #E5E7EB;
border-radius:12px;
background:white;
height:170px;
">

<div style="font-size:42px;">
{icon}
</div>

<br>

<b>{label}</b>

</div>
""",
                unsafe_allow_html=True,
            )

    st.write("")

    st.success(
        "Output: Healthy • Parkinson's Disease • Other Movement Disorders"
    )

    st.divider()

    # ======================================================
    # DASHBOARD MODULES
    # ======================================================

    st.subheader("Dashboard Modules")

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.markdown("### 📈 Model Performance")

            st.caption(
                """
Classification metrics

Confusion matrix

ROC analysis

Calibration

Model comparison
"""
            )

        with st.container(border=True):

            st.markdown("### 🔍 Explainable AI")

            st.caption(
                """
Global SHAP

Local SHAP

Feature importance

Task importance

Questionnaire analysis
"""
            )

    with c2:

        with st.container(border=True):

            st.markdown("### 👤 Patient Prediction")

            st.caption(
                """
Individual prediction

Prediction probabilities

Patient explanation

Clinical interpretation
"""
            )

        with st.container(border=True):

            st.markdown("### ℹ️ About")

            st.caption(
                """
Project

Methodology

Dataset

Development team

References
"""
            )

    st.divider()

    # ======================================================
    # OBJECTIVE
    # ======================================================

    st.subheader("Project Objective")

    st.info(
        """
Develop an interpretable multimodal machine learning model capable of
supporting the early screening of Parkinson's disease by integrating wearable
sensor data, non-motor symptom questionnaire responses and demographic
information while providing transparent explanations through SHAP.
"""
    )