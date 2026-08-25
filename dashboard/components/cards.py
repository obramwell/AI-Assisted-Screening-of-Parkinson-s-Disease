import streamlit as st


def metric_card(title, value, subtitle=""):

    st.markdown(
        f"""
        <div style="
            border-radius:18px;
            padding:25px;
            border:1px solid #E5E7EB;
            background:white;
            box-shadow:0 2px 8px rgba(0,0,0,.06);
            text-align:center;
        ">

        <h4 style="margin-bottom:0;color:#6B7280;">
        {title}
        </h4>

        <h2 style="margin-top:8px;color:#12355B;">
        {value}
        </h2>

        <p style="color:#8A94A6;">
        {subtitle}
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )