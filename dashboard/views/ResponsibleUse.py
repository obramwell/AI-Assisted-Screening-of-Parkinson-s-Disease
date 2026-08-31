import streamlit as st

from utils.styles import load_css


def show():

    load_css()

    st.title("Responsible Use & Clinical Limitations")

    st.write(
        """
This prototype was developed as part of a graduate research project to
demonstrate the application of explainable multimodal machine learning
for Parkinson's disease screening.

The system is intended exclusively for research, education, and
technology demonstration.
"""
    )

    st.divider()

    # ======================================================
    # DISCLAIMER
    # ======================================================

    st.error(
        """
### Clinical Disclaimer

This system was developed for academic research and demonstration.
It is **not a medical device**, has **not undergone prospective clinical
validation**, and **must not be used to diagnose, exclude, or treat
Parkinson's disease or any other medical condition**.

The predictions presented by this prototype should never replace
clinical judgment or specialist evaluation.
"""
    )

    st.divider()

    # ======================================================
    # LIMITATIONS
    # ======================================================

    st.header("Model Limitations")

    c1, c2 = st.columns(2)

    with c1:

        st.info(
            """
### Dataset

- Results are specific to the PhysioNet Parkinson's Smartwatch Dataset.
- External validation has not been completed.
- The prototype was evaluated using retrospective data only.
- Generalization to other clinical populations remains unknown.
"""
        )

        st.info(
            """
### Other Movement Disorders

The Other Movement Disorders (OMD) class includes several heterogeneous
neurological conditions.

Consequently, prediction performance for this class may vary depending
on the underlying disorder represented in the dataset.
"""
        )

    with c2:

        st.info(
            """
### Explainability

SHAP explanations indicate how the model used the available features
to produce a prediction.

These explanations **do not imply clinical causation** and should not
be interpreted as evidence that a variable causes Parkinson's disease.
"""
        )

        st.info(
            """
### Prediction Errors

Like any machine learning model, this prototype may produce:

- False positives
- False negatives
- Low-confidence predictions

Predictions should therefore be interpreted with caution.
"""
        )

    st.divider()

    # ======================================================
    # RESPONSIBLE USE
    # ======================================================

    st.header("Responsible Use")

    st.success(
        """
This prototype may assist researchers in understanding multimodal
machine learning models and their predictions.

It should only be used as a **screening-support research tool**
and never as a substitute for professional neurological assessment.
"""
    )

    st.divider()

    # ======================================================
    # RECOMMENDED ACTIONS
    # ======================================================

    st.header("Prototype Recommendations")

    recommendations = [
        (
            "High Parkinson's disease probability with high confidence",
            "Recommend additional neurological assessment."
        ),
        (
            "Similar probabilities between classes",
            "Prediction is uncertain. Specialist review is recommended."
        ),
        (
            "Low-confidence prediction",
            "Interpret results with caution. Additional information may be required."
        ),
        (
            "Poor or incomplete data quality",
            "Repeat the assessment before interpreting the prediction."
        ),
    ]

    for condition, action in recommendations:

        with st.container(border=True):

            st.markdown(f"**Condition**  \n{condition}")

            st.markdown(f"**Recommended Prototype Action**  \n{action}")

    st.divider()

    # ======================================================
    # ETHICAL CONSIDERATIONS
    # ======================================================

    st.header("Ethical Considerations")

    st.markdown(
        """
- Protection of participant privacy and confidentiality.
- Responsible use of publicly available health datasets.
- Awareness of potential demographic and clinical biases.
- Recognition that model performance may differ across population subgroups.
- Avoidance of automation bias when interpreting predictions.
- Transparent communication of model uncertainty and limitations.
"""
    )

