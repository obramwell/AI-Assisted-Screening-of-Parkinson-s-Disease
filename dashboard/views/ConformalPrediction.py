import streamlit as st

from utils.styles import load_css

from utils.loaders import (
    load_csv,
    load_image,
)

from utils.paths import (
    CONFORMITY_COMPARISON,
    CONFIDENCE_COMPARISON,
    SELECTED_CONFIGURATION,

    TRADEOFF,
    COVERAGE_CONFIDENCE,
    SETSIZE_CONFIDENCE,
    SINGLETON_CONFIDENCE,
    AMBIGUOUS_CONFIDENCE,
)


def show():

    load_css()

    st.title("Conformal Prediction")

    st.write(
        """
Conformal Prediction extends the final Full Multimodal XGBoost classifier by
quantifying predictive uncertainty.

Instead of returning only a single predicted diagnosis, conformal prediction
produces statistically valid prediction sets that indicate when multiple
diagnostic classes remain plausible, providing an additional layer of
transparency for the proposed academic screening-support framework.
"""
    )

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        [
            "⚖️ Method Comparison",
            "📈 Confidence Levels",
            "🛡️ Final Configuration",
        ]
    )

    # =====================================================
    # TAB 1
    # =====================================================

    with tab1:

        st.subheader("Comparison of Conformity Scores")

        st.write(
            """
Five multiclass conformity scores were evaluated using a fixed
95% confidence level to identify the most informative uncertainty
quantification strategy.
"""
        )

        image = load_image(TRADEOFF)

        if image:

            st.image(
                image,
                use_container_width=True,
            )

        table = load_csv(CONFORMITY_COMPARISON)

        if table is not None:

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
            )

        st.success(
            """
**Selected conformity score:** Least Ambiguous Classifier (LAC)

LAC achieved the best trade-off between empirical coverage and
prediction set informativeness. Although APS achieved perfect
coverage, it generated considerably larger prediction sets,
whereas LAC maintained reliable coverage while substantially
reducing prediction ambiguity.
"""
        )

    # =====================================================
    # TAB 2
    # =====================================================

    with tab2:

        st.subheader("Confidence Level Analysis")

        st.write(
            """
The selected LAC conformity score was evaluated using confidence
levels of 80%, 90%, and 95% to investigate the trade-off between
prediction reliability and prediction set size.
"""
        )

        c1, c2 = st.columns(2)

        with c1:

            image = load_image(COVERAGE_CONFIDENCE)

            if image:

                st.image(
                    image,
                    caption="Empirical Coverage",
                    use_container_width=True,
                )

        with c2:

            image = load_image(SETSIZE_CONFIDENCE)

            if image:

                st.image(
                    image,
                    caption="Average Prediction Set Size",
                    use_container_width=True,
                )

        c3, c4 = st.columns(2)

        with c3:

            image = load_image(SINGLETON_CONFIDENCE)

            if image:

                st.image(
                    image,
                    caption="Singleton Prediction Rate",
                    use_container_width=True,
                )

        with c4:

            image = load_image(AMBIGUOUS_CONFIDENCE)

            if image:

                st.image(
                    image,
                    caption="Ambiguous Prediction Rate",
                    use_container_width=True,
                )

        table = load_csv(CONFIDENCE_COMPARISON)

        if table is not None:

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
            )

        st.info(
            """
Increasing the confidence level improves empirical coverage but also
increases prediction ambiguity by enlarging the prediction sets.
A 95% confidence level was retained because reliable coverage is
preferred over overly confident singleton predictions in clinical
decision-support applications.
"""
        )

    # =====================================================
    # TAB 3
    # =====================================================

    with tab3:

        st.subheader("Selected Conformal Prediction Configuration")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Coverage",
            "97.2%",
        )

        col2.metric(
            "Avg. Set Size",
            "1.77",
        )

        col3.metric(
            "Singleton",
            "36.6%",
        )

        col4.metric(
            "Confidence",
            "95%",
        )

        st.divider()

        table = load_csv(SELECTED_CONFIGURATION)

        if table is not None:

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Key Findings")

        st.markdown(
            """
- **Final classifier:** Full Multimodal XGBoost

- **Selected conformity score:** Least Ambiguous Classifier (LAC)

- **Confidence level:** 95%

- **Empirical coverage:** 97.2%

- **Average prediction set size:** 1.77 diagnostic classes

- **Conformal prediction complements traditional performance metrics by explicitly communicating predictive uncertainty rather than replacing the predicted diagnosis.**

- **The proposed framework is intended exclusively as an academic clinical decision-support prototype and should not be interpreted as a standalone diagnostic system.**
"""
        )