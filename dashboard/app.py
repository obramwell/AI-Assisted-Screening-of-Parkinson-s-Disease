from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu

from utils.styles import load_css

from views import (
    Home,
    Performance,
    Explainability,
    Prediction,
    ResponsibleUse,
    About
)

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI-Assisted Screening of Parkinson's Disease",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# LOAD CSS
# ==========================================================

load_css()

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).parent

ASSETS_DIR = BASE_DIR / "assets"

LOGO = ASSETS_DIR / "logo.png"

# ==========================================================
# HEADER
# ==========================================================

col_logo, col_title = st.columns([1, 9])

with col_logo:
    st.image(str(LOGO), width=95)

with col_title:

    st.markdown(
        """
# AI-Assisted Screening of Parkinson's Disease

### Clinical Decision Support Dashboard
"""
    )

st.markdown(
"""
This application provides an interactive interface for exploring the multimodal
machine learning model developed for the early screening of Parkinson's disease.
The dashboard combines wearable sensor features, non-motor symptom questionnaire
responses, demographic information, and Explainable Artificial Intelligence
(SHAP) to support transparent clinical decision-making.
"""
)

# ==========================================================
# NAVIGATION
# ==========================================================

selected = option_menu(
    menu_title=None,

    options=[
        "Home",
        "Performance",
        "Explainability",
        "Prediction",
        "Responsible Use",
        "About"
    ],

    icons=[
        "house-fill",
        "graph-up",
        "cpu",
        "person-badge",
        "shield-check",
        "journal-medical"
    ],

    default_index=0,

    orientation="horizontal"
)

st.divider()

# ==========================================================
# ROUTER
# ==========================================================

if selected == "Home":
    Home.show()

elif selected == "Performance":
    Performance.show()

elif selected == "Explainability":
    Explainability.show()

elif selected == "Prediction":
    Prediction.show()
    
elif selected == "Responsible Use":
    ResponsibleUse.show()

elif selected == "About":
    About.show()

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown(
"""
<div class="footer">

© 2026 University of Niagara Falls Canada

Master of Data Analytics • Capstone Project • Summer 2026

</div>
""",
unsafe_allow_html=True
)