import streamlit as st

from utils.styles import load_css

from utils.loaders import (
    load_image,
    load_csv,
)

from utils.paths import (
    SHAP_BAR,
    SHAP_MODALITY,
    SHAP_TASK,
    SHAP_SENSOR,
    SHAP_WRIST,
    SHAP_QUESTIONNAIRE,
    GLOBAL_SHAP,
    MODALITY_TABLE,
    TASK_TABLE,
    SENSOR_TABLE,
    WRIST_TABLE,
    QUESTIONNAIRE_TABLE,
    HEALTHY_TABLE,
    PD_TABLE,
    OMD_TABLE,
)

def show():
    load_css()
    st.title("Explainable Artificial Intelligence")
    st.write(
        """
This section presents the SHAP (SHapley Additive Explanations) analysis of the
final Full Multimodal XGBoost classifier.

The visualizations illustrate the global importance of predictors and how
different data modalities, neurological tasks, sensor types, wrists, and
questionnaire domains contributed to the model predictions.
"""
    )
    st.divider()
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🌍 Global",
            "📦 Modalities",
            "🖐️ Tasks",
            "⌚ Sensors",
            "📝 Questionnaire",
            "🧠 Classes",
        ]
    )

    # GLOBAL
    with tab1:
        st.subheader("Global SHAP Feature Importance")
        st.write(
            """
The figure below presents the twenty most influential predictors identified
by SHAP across the complete test set.
"""
        )
        image = load_image(SHAP_BAR)
        if image:
            st.image(
                image,
                use_container_width=True,
            )
        st.markdown("### Global Ranking")
        table = load_csv(GLOBAL_SHAP)
        if table is not None:

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
            )

    # MODALITIES
    with tab2:
        st.subheader("Importance by Data Modality")
        image = load_image(SHAP_MODALITY)
        if image:
            st.image(
                image,
                use_container_width=True,
            )

        st.dataframe(
            load_csv(MODALITY_TABLE),
            use_container_width=True,
        )
   # TASKS
    with tab3:

        st.subheader("Importance by Neurological Task")

        image = load_image(SHAP_TASK)

        if image:

            st.image(
                image,
                use_container_width=True,
            )

        st.dataframe(
            load_csv(TASK_TABLE),
            use_container_width=True,
        )

    # SENSOR + WRIST
    with tab4:
        left, right = st.columns(2)
        with left:
            st.subheader("Sensor Type")
            image = load_image(SHAP_SENSOR)
            if image:
                st.image(
                    image,
                    use_container_width=True,
                )
            st.dataframe(
                load_csv(SENSOR_TABLE),
                use_container_width=True,
            )
        with right:
            st.subheader("Wrist")
            image = load_image(SHAP_WRIST)
            if image:
                st.image(
                    image,
                    use_container_width=True,
                )
            st.dataframe(
                load_csv(WRIST_TABLE),
                use_container_width=True,
            )

    # QUESTIONNAIRE
    with tab5:

        st.subheader("Questionnaire Domains")

        image = load_image(SHAP_QUESTIONNAIRE)

        if image:

            st.image(
                image,
                use_container_width=True,
            )

        st.dataframe(
            load_csv(QUESTIONNAIRE_TABLE),
            use_container_width=True,
        )

    # CLASS SPECIFIC
    with tab6:

        st.subheader("Class-Specific SHAP Importance")

        option = st.selectbox(
            "Select class",
            [
                "Healthy Controls",
                "Parkinson's Disease",
                "Other Movement Disorders",
            ],
        )

        if option == "Healthy Controls":

            table = load_csv(HEALTHY_TABLE)

        elif option == "Parkinson's Disease":

            table = load_csv(PD_TABLE)

        else:

            table = load_csv(OMD_TABLE)

        if table is not None:

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
            )