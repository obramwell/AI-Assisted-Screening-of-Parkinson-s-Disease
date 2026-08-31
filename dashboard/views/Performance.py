import streamlit as st
import matplotlib.pyplot as plt

from utils.styles import load_css

from utils.loaders import (
    load_csv,
    load_image,
)

from utils.paths import (
    TEST_RESULTS,
    VALIDATION_TEST_COMPARISON,
    CALIBRATION_RESULTS,
    FULL_CLASSIFICATION_REPORT,
    FULL_CALIBRATION,
    FULL_CM,
)


def show():

    load_css()

    st.title("Model Performance")

    st.write(
        """
This section summarizes the predictive performance of the final
Full Multimodal XGBoost classifier evaluated on the independent
participant-level test set.

Performance is reported using the evaluation metrics adopted
throughout this study, together with the confusion matrix,
classification report, probability calibration, and comparison
between validation and test performance.
"""
    )

    st.divider()

    # ==========================================================
    # OVERALL PERFORMANCE
    # ==========================================================

    st.subheader("Overall Test Performance")

    metrics = load_csv(TEST_RESULTS)

    if metrics is not None:

        metrics = metrics[
            metrics["dataset"] == "Full Multimodal"
        ].reset_index(drop=True)

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.metric(
                "Accuracy",
                f"{metrics.loc[0,'accuracy']:.3f}",
            )

        with c2:
            st.metric(
                "Balanced Accuracy",
                f"{metrics.loc[0,'balanced_accuracy']:.3f}",
            )

        with c3:
            st.metric(
                "Macro F1",
                f"{metrics.loc[0,'macro_f1']:.3f}",
            )

        with c4:
            st.metric(
                "Precision",
                f"{metrics.loc[0,'precision_macro']:.3f}",
            )

        with c5:
            st.metric(
                "Recall",
                f"{metrics.loc[0,'recall_macro']:.3f}",
            )

    else:

        st.info("Performance metrics not available.")

    st.divider()

    # ==========================================================
    # CONFUSION MATRIX
    # ==========================================================

    st.subheader("Confusion Matrix")

    st.write(
        """
The confusion matrix summarizes the number of correctly and
incorrectly classified participants for each diagnostic class.
"""
    )

    image = load_image(FULL_CM)

    if image is not None:

        st.image(
            image,
            use_container_width=True,
        )

    else:

        st.info("Confusion matrix not available.")

    st.divider()

    # ==========================================================
    # CLASSIFICATION REPORT
    # ==========================================================

    st.subheader("Classification Report")

    st.write(
        """
Class-wise precision, recall, F1-score, and support obtained on
the independent participant-level test set.
"""
    )

    report = load_csv(
        FULL_CLASSIFICATION_REPORT
    )

    if report is not None:

        st.dataframe(
            report,
            use_container_width=True,
            hide_index=False,
        )

    else:

        st.info("Classification report not available.")

    st.divider()

    # ==========================================================
    # CALIBRATION
    # ==========================================================

    st.subheader("Probability Calibration")

    st.write(
        """
Calibration evaluates whether the predicted probabilities
correspond to the observed frequencies for each diagnostic class.
"""
    )

    image = load_image(FULL_CALIBRATION)

    if image is not None:

        st.image(
            image,
            use_container_width=True,
        )

    else:

        st.info("Calibration figure not available.")

    calibration = load_csv(
        CALIBRATION_RESULTS
    )

    if calibration is not None:

        calibration = calibration[
            calibration["Dataset"] == "Full Multimodal"
        ]

        st.dataframe(
            calibration,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ==========================================================
    # VALIDATION VS TEST
    # ==========================================================

    st.subheader("Validation vs Test Comparison")

    st.write(
        """
Comparison between validation and independent test performance
for the candidate machine-learning models evaluated during model
selection.
"""
    )

    comparison = load_csv(
        VALIDATION_TEST_COMPARISON
    )

    if comparison is not None:

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Validation-test comparison not available."
        )