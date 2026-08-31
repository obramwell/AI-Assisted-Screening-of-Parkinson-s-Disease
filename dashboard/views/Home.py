import streamlit as st

from utils.constants import (
    MODEL_NAME,
    N_CLASSES,
    N_FEATURES,
    N_PARTICIPANTS,
)

from utils.paths import BANNER
from utils.loaders import load_image


def show():
    banner = load_image(BANNER)

    if banner is not None:
        st.image(
            banner,
            use_container_width=True,
        )

    st.markdown("# AI-Assisted Screening of Parkinson's Disease")

    st.markdown(
    """
    ### Interpretable Multimodal Machine Learning for Early Parkinson's Disease Screening
    """
    )

    st.write(
    """
    This interactive dashboard presents the final XGBoost model developed for the
    early screening of Parkinson's disease using wearable sensor measurements,
    non-motor symptom questionnaires, and demographic information.

    The platform enables both predictive analysis and model interpretation through
    Explainable Artificial Intelligence (SHAP), providing transparent insights into
    individual predictions and global feature importance.
    """
    )

    st.write("")

    # =====================================================
    # HERO METRICS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Participants",
        N_PARTICIPANTS,
    )

    c2.metric(
        "Selected Features",
        N_FEATURES,
    )

    c3.metric(
        "Final Model",
        MODEL_NAME,
    )

    c4.metric(
        "Classes",
        N_CLASSES,
    )

    st.divider()
    # OVERVIEW
    left, right = st.columns([2,1])

    with left:

        st.subheader("Project Overview")

        st.write(
        """
    The proposed framework combines three complementary information sources
    to support Parkinson's disease screening.

    • Wearable sensor signals collected during standardized neurological tasks

    • Non-Motor Symptoms Questionnaire (NMS)

    • Demographic and clinical variables

    After preprocessing, feature engineering, and feature selection,
    the final XGBoost classifier is used to generate multiclass predictions.
    SHAP explainability provides both global and participant-level interpretation
    of model behavior.
    """
        )

    with right:

        st.subheader("Quick Facts")

        st.success("✔ 469 Participants")

        st.success("✔ 500 Selected Features")

        st.success("✔ 3 Diagnostic Classes")

        st.success("✔ Explainable AI (SHAP)")
# PIPELINE

    st.subheader("Framework Pipeline")

    st.markdown(
    """
    <style>

    .pipeline{
    display:flex;
    justify-content:center;
    align-items:center;
    gap:14px;
    flex-wrap:wrap;
    margin-top:15px;
    margin-bottom:20px;
    }

    .box{
    background:#F8FAFC;
    border:1px solid #CBD5E1;
    border-radius:14px;
    padding:18px;
    width:145px;
    height:125px;
    text-align:center;
    box-shadow:0 1px 6px rgba(0,0,0,.05);
    }

    .icon{
    font-size:34px;
    }

    .title{
    font-weight:600;
    margin-top:8px;
    color:#12355B;
    }

    .arrow{
    font-size:28px;
    font-weight:bold;
    color:#3B82F6;
    }

    </style>

    <div class="pipeline">

    <div class="box">
    <div class="icon">📱</div>
    <div class="title">Wearable<br>Sensors</div>
    </div>

    <div class="arrow">➜</div>

    <div class="box">
    <div class="icon">📋</div>
    <div class="title">Questionnaire</div>
    </div>

    <div class="arrow">➜</div>

    <div class="box">
    <div class="icon">👤</div>
    <div class="title">Demographics</div>
    </div>

    <div class="arrow">➜</div>

    <div class="box">
    <div class="icon">⚙️</div>
    <div class="title">Feature<br>Engineering</div>
    </div>

    <div class="arrow">➜</div>

    <div class="box">
    <div class="icon">🧬</div>
    <div class="title">Feature<br>Selection</div>
    </div>

    <div class="arrow">➜</div>

    <div class="box">
    <div class="icon">🌲</div>
    <div class="title">XGBoost</div>
    </div>

    <div class="arrow">➜</div>

    <div class="box">
    <div class="icon">🧠</div>
    <div class="title">SHAP</div>
    </div>

    </div>
    """,
    unsafe_allow_html=True
    )

    st.info(
    "**Output:** Healthy • Parkinson's Disease • Other Movement Disorders"
    )

    st.divider()
    # DASHBOARD MODULES
    st.subheader("Dashboard Modules")

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        with st.container(border=True):
            st.markdown("### 📈 Performance")
            st.write(
                "Evaluate the final XGBoost model using classification metrics, confusion matrices, ROC curves, precision–recall analysis, and probability calibration."
            )

    with r1c2:
        with st.container(border=True):
            st.markdown("### 🔬 Explainability")
            st.write(
                "Explore SHAP global explanations, feature importance, data modalities, neurological tasks, sensors, and participant-level interpretations."
            )

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        with st.container(border=True):
            st.markdown("### 👤 Prediction")
            st.write(
                "Generate predictions for individual participants and inspect prediction probabilities together with local SHAP explanations."
            )

    with r2c2:
        with st.container(border=True):
            st.markdown("### ℹ️ About")
            st.write(
                "Learn about the dataset, methodology, technologies, development team, and project repository."
            )
