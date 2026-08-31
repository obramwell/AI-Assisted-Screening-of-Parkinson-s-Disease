import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from utils.styles import load_css
# PATHS
ROOT = Path(__file__).resolve().parents[2]
PREPROCESSED_SIGNALS = (
    ROOT
    / "data"
    / "processed"
    / "preprocessed_signals"
)
# TASKS
TASKS = [
    "Relaxed",
    "RelaxedTask",
    "StretchHold",
    "LiftHold",
    "HoldWeight",
    "PointFinger",
    "DrinkGlas",
    "CrossArms",
    "TouchIndex",
    "TouchNose",
    "Entrainment",
]

TASK_LABELS = {
    "Relaxed": "Relaxed",
    "RelaxedTask": "Relaxed Task",
    "StretchHold": "Stretch & Hold",
    "LiftHold": "Lift & Hold",
    "HoldWeight": "Hold Weight",
    "PointFinger": "Point Finger",
    "DrinkGlas": "Drink Glass",
    "CrossArms": "Cross Arms",
    "TouchIndex": "Touch Index",
    "TouchNose": "Touch Nose",
    "Entrainment": "Entrainment",
}

WRISTS = [
    "LeftWrist",
    "RightWrist",
]
# ==========================================================
# CHANNEL COLORS
# ==========================================================

CHANNEL_COLORS = {

    "Accelerometer_X": "#1f77b4",
    "Accelerometer_Y": "#ff7f0e",
    "Accelerometer_Z": "#2ca02c",

    "Gyroscope_X": "#d62728",
    "Gyroscope_Y": "#9467bd",
    "Gyroscope_Z": "#8c564b",

    "Acc_Magnitude": "#000000",
    "Gyro_Magnitude": "#7f7f7f",

}
# LOAD RECORDING
def load_recording(
    participant,
    task,
    wrist,
):
    npz_path = (
        PREPROCESSED_SIGNALS
        / f"{participant}_preprocessed.npz"
    )

    if not npz_path.exists():

        return None

    with np.load(
        npz_path,
        allow_pickle=True,
    ) as data:

        columns = list(
            data["__columns__"]
        )

        key = f"{task}__{wrist}"

        if key not in data.files:

            return None

        recording = data[key]

    return pd.DataFrame(
        recording,
        columns=columns,
    )
# FFT
def compute_fft(
    signal,
    sampling_frequency=100,
):

    signal = np.asarray(signal)

    n = len(signal)

    frequencies = np.fft.rfftfreq(
        n,
        d=1 / sampling_frequency,
    )

    magnitude = np.abs(
        np.fft.rfft(signal)
    )

    return frequencies, magnitude
# SIGNAL SUMMARY
def signal_summary(
    dataframe,
):

    summary = pd.DataFrame(
        {
            "Mean": dataframe.mean(
                numeric_only=True,
            ),
            "Standard Deviation": dataframe.std(
                numeric_only=True,
            ),
            "Minimum": dataframe.min(
                numeric_only=True,
            ),
            "Maximum": dataframe.max(
                numeric_only=True,
            ),
            "RMS": np.sqrt(
                (
                    dataframe.select_dtypes(
                        include=np.number,
                    )
                    ** 2
                ).mean()
            ),
        }
    )

    return summary.round(4)
# MAIN PAGE
def show():

    load_css()

    st.title("📈 Signal Explorer")

    st.write(
        """
Explore the preprocessed smartwatch recordings used throughout the
wearable feature extraction pipeline.

Select a participant, neurological assessment task, and wrist to
visualize the processed accelerometer and gyroscope signals.
"""
    )

    st.divider()
    # Participant list
    participants = sorted(
        [
            file.stem.replace(
                "_preprocessed",
                ""
            )
            for file in PREPROCESSED_SIGNALS.glob(
                "*_preprocessed.npz"
            )
        ]
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        participant = st.selectbox(
            "Participant",
            participants,
        )

    with col2:

        selected_task = st.selectbox(
            "Motor Task",
            TASKS,
            format_func=lambda x: TASK_LABELS[x],
        )

    with col3:

        selected_wrist = st.selectbox(
            "Wrist",
            WRISTS,
            format_func=lambda x: x.replace(
                "Wrist",
                ""
            ),
        )

    recording = load_recording(
        participant,
        selected_task,
        selected_wrist,
    )

    if recording is None:

        st.error(
            "Recording could not be loaded."
        )

        return

    st.divider()
    # RECORDING INFORMATION
    st.subheader("Recording Information")

    duration = (
        recording["Time"].iloc[-1]
        - recording["Time"].iloc[0]
    )

    sampling_frequency = float(
    1
    / recording["Time"].diff().median()
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Participant", participant)
    c2.metric("Task", TASK_LABELS[selected_task])
    c3.metric("Wrist", selected_wrist.replace("Wrist",""))
    c4.metric("Samples", len(recording))
    c5.metric("Duration", f"{duration:.2f} s")
    st.divider()
        # ======================================================
    # CHANNEL SELECTION
    # ======================================================

    st.subheader("Signal Channels")

    left, right = st.columns(2)

    with left:

        accelerometer = st.checkbox(
            "Accelerometer",
            value=True,
        )

    with right:

        gyroscope = st.checkbox(
            "Gyroscope",
            value=False,
        )

    selected_channels = []

    if accelerometer:

        selected_channels.extend([
            "Accelerometer_X",
            "Accelerometer_Y",
            "Accelerometer_Z",
            "Acc_Magnitude",
        ])

    if gyroscope:

        selected_channels.extend([
            "Gyroscope_X",
            "Gyroscope_Y",
            "Gyroscope_Z",
            "Gyro_Magnitude",
        ])

    if len(selected_channels) == 0:

        st.warning(
            "Please select at least one signal."
        )

        return

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        [
            "📈 Time Series",
            "⚡ Frequency Spectrum",
            "📊 Signal Statistics",
        ]
    )
    # TIME-SERIES VISUALIZATION
    with tab1:

        st.subheader("Time-Series Visualization")

        st.write(
            """
    The figure below displays the selected wearable channels after
    preprocessing.
    """
        )

        figure = go.Figure()

        for channel in selected_channels:

            figure.add_trace(

                go.Scatter(

                    x=recording["Time"],

                    y=recording[channel],

                    mode="lines",

                    name=channel,

                    line=dict(
                        color=CHANNEL_COLORS[channel],
                        width=2,
                    ),

                )

            )

        figure.update_layout(

            template="plotly_white",

            hovermode="x unified",

            height=550,

            xaxis_title="Time (seconds)",

            yaxis_title="Signal",

            legend=dict(

                orientation="h",

                y=1.02,

                x=.5,

                xanchor="center",

            ),

        )

        st.plotly_chart(

            figure,

            use_container_width=True,

            config={

                "displaylogo": False,

                "responsive": True,

            },

        )
    # FREQUENCY SPECTRUM
    with tab2:
        st.subheader("Frequency Spectrum")

    fft_figure = go.Figure()

    for channel in selected_channels:

        frequencies, magnitude = compute_fft(

            recording[channel],

            sampling_frequency,

        )

        fft_figure.add_trace(

            go.Scatter(

                x=frequencies,

                y=magnitude,

                mode="lines",

                name=channel,

                line=dict(

                    color=CHANNEL_COLORS[channel],

                    width=2,

                ),

            )

        )

    fft_figure.update_layout(

        template="plotly_white",

        hovermode="x unified",

        height=500,

        xaxis_title="Frequency (Hz)",

        yaxis_title="Magnitude",

    )

    fft_figure.update_xaxes(

        range=[0,20]

    )

    st.plotly_chart(

        fft_figure,

        use_container_width=True,

        config={

            "displaylogo": False,

            "responsive": True,

        },

    )
    
    # SIGNAL STATISTICS
    with tab3:

        st.subheader("Signal Statistics")
        summary = signal_summary(
            recording.drop(
                columns=["Time"],
                errors="ignore",
            )
        )
        st.dataframe(
            summary,
            use_container_width=True,
        )
        csv = recording.to_csv(
            index=False,
        ).encode("utf-8")

        st.download_button(
            "Download Recording",
            recording.to_csv(index=False),
            file_name=f"{participant}_{selected_task}_{selected_wrist}.csv",
            mime="text/csv",
        )