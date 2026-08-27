import streamlit as st
import pandas as pd

from pathlib import Path

# PATHS
ROOT = Path(__file__).resolve().parents[2]

FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"

# PAGE
def show():

    st.title("📈 Model Performance")

    st.write(
        """
Evaluate the predictive performance of the final Full Multimodal
XGBoost classifier using the independent test set.
"""
    )

    st.divider()
    # METRICS
    st.subheader("Overall Performance")

    metrics_file = TABLES / "test_metrics.csv"

    if metrics_file.exists():

        metrics = pd.read_csv(metrics_file)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Accuracy",
                f"{metrics.loc[0,'Accuracy']:.3f}"
            )

        with c2:
            st.metric(
                "Macro F1",
                f"{metrics.loc[0,'Macro F1']:.3f}"
            )

        with c3:
            st.metric(
                "Balanced Accuracy",
                f"{metrics.loc[0,'Balanced Accuracy']:.3f}"
            )

        with c4:
            st.metric(
                "ROC AUC",
                f"{metrics.loc[0,'ROC AUC']:.3f}"
            )

    else:

        st.info("Metrics file not found.")

    st.divider()
    # CONFUSION MATRIX
    st.subheader("Confusion Matrix")

    img = FIGURES / "confusion_matrix.png"

    if img.exists():

        st.image(
            img,
            use_container_width=True
        )

    else:

        st.warning("Confusion Matrix not available.")

    st.divider()

    # ROC
    st.subheader("ROC Curves")

    img = FIGURES / "roc_curves.png"

    if img.exists():

        st.image(
            img,
            use_container_width=True
        )

    else:

        st.warning("ROC Curves not available.")

    st.divider()

    # ===================================================
    # PR
    # ===================================================

    st.subheader("Precision-Recall Curves")

    img = FIGURES / "precision_recall_curves.png"

    if img.exists():

        st.image(
            img,
            use_container_width=True
        )

    else:

        st.warning("Precision-Recall Curves not available.")

    st.divider()

    # ===================================================
    # CALIBRATION
    # ===================================================

    st.subheader("Probability Calibration")

    img = FIGURES / "calibration_curve.png"

    if img.exists():

        st.image(
            img,
            use_container_width=True
        )

    else:

        st.info("Calibration results coming soon.")

    st.divider()

    # ===================================================
    # MODEL COMPARISON
    # ===================================================

    st.subheader("Model Comparison")

    comparison = TABLES / "model_comparison.csv"

    if comparison.exists():

        df = pd.read_csv(comparison)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "model_comparison.csv"
        )

    else:

        st.info("Comparison table not found.")