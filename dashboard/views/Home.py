import plotly.express as px
import pandas as pd
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
        left, center, right = st.columns([0.5, 8, 0.5])
        with center:
            st.image(
                banner,
                use_container_width=True,
            )
    st.markdown(
    """
    # AI-Assisted Screening of Parkinson's Disease
    ### Clinical Decision Support Dashboard
    """
    )
    st.caption(
    """
    An interactive multimodal machine learning framework for Parkinson's disease
    screening using wearable sensors, clinical questionnaires, and demographic
    information.
    """
    )
    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        with st.container(border=True):
            st.metric(
                "",
                N_PARTICIPANTS,
            )
            st.caption("Participants")
    with m2:
        with st.container(border=True):
            st.metric(
                "",
                N_FEATURES,
            )
            st.caption("Selected Features")
    with m3:
        with st.container(border=True):
            st.metric(
                "",
                MODEL_NAME,
            )
            st.caption("Final Model")
    with m4:
        with st.container(border=True):
            st.metric(
                "",
                N_CLASSES,
            )
            st.caption("Diagnostic Groups")
    st.markdown(
    """
    <div style="text-align:center;
    font-size:15px;
    color:#64748B;
    padding-top:8px;
    padding-bottom:8px;">

    Combining <b>Wearable Sensors</b>,
    <b>Non-Motor Symptoms</b>, and
    <b>Demographic Information</b> to support
    transparent and uncertainty-aware clinical decision making.

    </div>
    """,
    unsafe_allow_html=True,
    )

    st.divider()
# PROJECT AT A GLANCE
    st.subheader("Project at a Glance")
    left, right = st.columns([1.6, 1], gap="large")
# DATASET COMPOSITION
    with left:
        st.markdown("#### Dataset Composition")
        diagnostic_df = pd.DataFrame(
            {
                "Group": [
                    "Parkinson's Disease",
                    "Healthy Controls",
                    "Other Movement Disorders",
                ],
                "Participants": [
                    276,
                    79,
                    114,
                ],
            }
        )
        fig_groups = px.pie(
            diagnostic_df,
            names="Group",
            values="Participants",
            hole=0.72,
            color="Group",
            color_discrete_sequence=[
                "#2563EB",
                "#10B981",
                "#F59E0B",
            ],
        )

        fig_groups.update_traces(
            textinfo="percent",
            textfont_size=16,
            hovertemplate="<b>%{label}</b><br>%{value} participants<extra></extra>",
        )

        fig_groups.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            annotations=[
                dict(
                    text="<b>469</b><br>Participants",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=18),
                )
            ],
        )

        st.plotly_chart(
            fig_groups,
            use_container_width=True,
            config={
                "displaylogo": False,
            },
        )

        l1, l2, l3 = st.columns(3)

        l1.markdown("🔵 **Parkinson's Disease**")
        l2.markdown("🟢 **Healthy Controls**")
        l3.markdown("🟡 **Other Disorders**")
    # FRAMEWORK OVERVIEW
    with right:
        st.markdown("#### Framework Overview")
        with st.container(border=True):
            st.markdown(
            """
    ### 📱 Wearable Sensors
    ⬇
    ### 📋 NMS Questionnaire
    ⬇
    ### 👤 Demographics
    ⬇
    ### 🌳 XGBoost Classifier
    ⬇
    ### 🧠 SHAP Explainability
    ⬇

    ### 🛡 Conformal Prediction
    """
            )

    st.write("")

   # MULTIMODAL DATA SOURCES
    st.subheader("Multimodal Data Sources")

    st.caption(
        "Three complementary sources of information are integrated to generate each participant-level prediction."
    )

    col1, col2, col3 = st.columns(3)
    # WEARABLE
    with col1:
        with st.container(border=True):
            st.markdown("## 📱")
            st.markdown("### Wearable Sensors")
            st.caption("Movement acquisition")
            st.markdown(
            """
    - Accelerometer
    - Gyroscope
    - 11 Standardized Motor Tasks
    - Time & Frequency Features
    """
            )
    # QUESTIONNAIRE
    with col2:
        with st.container(border=True):
            st.markdown("## 📋")
            st.markdown("### Questionnaire")
            st.caption("Clinical assessment")
            st.markdown(
            """
    - Non-Motor Symptoms (NMS)
    - 30 Clinical Questions
    - Patient-Reported Outcomes
    - Symptom Assessment
    """
            )
    # DEMOGRAPHICS
    with col3:
        with st.container(border=True):
            st.markdown("## 👤")
            st.markdown("### Demographics")
            st.caption("Participant characteristics")
            st.markdown(
            """
    - Age
    - Sex
    - Handedness
    - Clinical Variables
    """
            )

    st.caption(
        "The proposed multimodal framework combines wearable sensor signals, questionnaire responses, and demographic information to improve Parkinson's disease screening."
    )
    st.divider()
    # NEUROLOGICAL ASSESSMENT PROTOCOL
    st.subheader("Neurological Assessment Protocol")
    st.caption(
        "Hover over each task to view a brief description."
    )
    st.markdown(
        """
        <style>
        .task-grid{
        display:grid;
        grid-template-columns:repeat(3,1fr);
        gap:20px;
        margin-top:15px;
        margin-bottom:20px;
        }
        .task-card{
        background:#F8FAFC;
        border:1px solid #E2E8F0;
        border-radius:16px;
        padding:20px;
        }
        .task-card h4{
        color:#12355B;
        margin-bottom:15px;
        }
        .chip{
        display:inline-block;
        background:#E8F1FD;
        color:#12355B;
        padding:8px 12px;
        margin:5px;
        border-radius:999px;
        font-size:14px;
        cursor:help;
        transition:.2s;
        }
        .chip:hover{
        background:#D5E8FF;
        transform:translateY(-2px);
        }
        .summary{
        margin-top:15px;
        text-align:center;
        font-size:15px;
        color:#475569;
        }
        </style>
        <div class="task-grid">
        <div class="task-card">
        <h4>🧘 Resting</h4>
        <span class="chip"
        title="Resting with closed eyes while sitting. Duration: 20 seconds.">
        Relaxed
        </span>
        <span class="chip"
        title="Resting while performing serial sevens. Duration: 20 seconds.">
        Relaxed Task
        </span>
        </div>
        <div class="task-card">
        <h4>💪 Postural</h4>
        <span class="chip"
        title="Lift and extend both arms. Duration: 10 seconds.">
        Stretch & Hold
        </span>
        <span class="chip"
        title="Remain both arms lifted. Duration: 10 seconds.">
        Lift Hold
        </span>
        <span class="chip"
        title="Hold a 1 kg weight in each hand. Duration: 10 seconds.">
        Hold Weight
        </span>
        <span class="chip"
        title="Synchronize foot movements with an external rhythm while arms remain extended. Duration: 20 seconds.">
        Entrainment
        </span>
        </div>
        <div class="task-card">
        <h4>👆 Kinetic</h4>
        <span class="chip"
        title="Point the index finger toward the examiner's hand. Duration: 10 seconds.">
        Point Finger
        </span>
        <span class="chip"
        title="Simulate drinking from an empty glass. Duration: 10 seconds.">
        Drink Glass
        </span>
        <span class="chip"
        title="Cross and extend both arms. Duration: 10 seconds.">
        Cross Arms
        </span>
        <span class="chip"
        title="Bring both index fingers together. Duration: 10 seconds.">
        Touch Index
        </span>
        <span class="chip"
        title="Tap the nose with the index finger. Duration: 10 seconds.">
        Touch Nose
        </span>
        </div>
        </div>
        <div class="summary">
        <b>11 standardized neurological tasks</b><br>
        used throughout the wearable signal acquisition protocol.
        </div>
        """,
            unsafe_allow_html=True,
        )
    st.divider()
# EXPLORE THE DASHBOARD
    st.subheader("Explore the Dashboard")
    st.caption(
        "Explore each module to understand the complete machine learning workflow, from data exploration to explainable predictions."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("## 📈 Performance")
            st.caption("Model evaluation")
            st.markdown(
            """
    Evaluate the predictive performance of the final XGBoost model.

    **Includes**

    • Classification Metrics

    • ROC Curves

    • Precision–Recall Analysis

    • Calibration Assessment
    """
            )

    with c2:

        with st.container(border=True):

            st.markdown("## 🧠 Explainability")

            st.caption("Model interpretation")

            st.markdown(
            """
    Understand how the model makes its predictions.

    **Includes**

    • Global SHAP

    • Local SHAP

    • Feature Importance

    • Feature Group Analysis
    """
            )

    with c3:
        with st.container(border=True):
            st.markdown("## 📡 Signal Explorer")
            st.caption("Wearable recordings")
            st.markdown(
            """
    Interactively inspect wearable sensor signals.

    **Includes**

    • Time Domain

    • Frequency Domain

    • Participant Explorer

    • Recording Statistics
    """
            )

    c4, c5, c6 = st.columns(3)
    with c4:
        with st.container(border=True):
            st.markdown("## 👤 Prediction")
            st.caption("Participant inference")
            st.markdown(
            """
    Generate predictions for individual participants.

    **Includes**

    • Predicted Class

    • Prediction Probabilities

    • Local SHAP Explanation

    • Clinical Summary
    """
            )

    with c5:
        with st.container(border=True):
            st.markdown("## 🛡️ Conformal Prediction")
            st.caption("Prediction uncertainty")
            st.markdown(
            """
    Evaluate prediction confidence and uncertainty.

    **Includes**

    • Prediction Sets

    • Coverage

    • Confidence Levels

    • Reliability Analysis
    """
            )

    with c6:
        with st.container(border=True):
            st.markdown("## ℹ️ About")
            st.caption("Project information")
            st.markdown(
            """
    Learn more about the project and methodology.

    **Includes**

    • Dataset

    • Analytical Framework

    • Technologies

    • Development Team
    """
            )

    st.divider()