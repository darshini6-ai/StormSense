import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import h5py
import torch
import streamlit as st
import matplotlib.pyplot as plt
from app.map_utils import (
    create_vil_map,
    find_relative_storm_center
)


# ============================================================
# Configuration
# ============================================================

FILE_PATH = (
    "data/sevir/vil/"
    "SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
)

MODEL_PATH = (
    "models/"
    "stormsense_convlstm_v3_multistep.pth"
)

SEQUENCE_INDEX = 0

INPUT_FRAMES = 12
FUTURE_FRAMES = 12
IMAGE_SIZE = 128


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="StormSense",
    page_icon="🌩️",
    layout="wide"
)


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# Title
# ============================================================

st.title("🌩️ StormSense")

st.markdown(
    """
    **Hyper-local severe-convective-weather nowcasting**

    Explore the current storm field and StormSense's predicted
    evolution over the next 60 minutes.
    """
)


# ============================================================
# Load model
# ============================================================

@st.cache_resource
def load_model():

    model = StormSenseConvLSTMv3(
        input_channels=1,
        hidden_channels=32
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


# ============================================================
# Import model here
# ============================================================

from models.convlstm_v3 import StormSenseConvLSTMv3


# ============================================================
# Load sequence
# ============================================================

@st.cache_data
def load_sequence(sequence_index):

    with h5py.File(FILE_PATH, "r") as f:

        event_id = f["id"][sequence_index]

        vil = f["vil"][
            sequence_index,
            :,
            :,
            :
        ]

    return event_id, vil


# ============================================================
# Prepare data
# ============================================================

@st.cache_data
def prepare_data(sequence_index):

    event_id, vil = load_sequence(sequence_index)

    vil = torch.tensor(
        vil,
        dtype=torch.float32
    ) / 255.0

    # H, W, T → T, 1, H, W
    vil = vil.permute(2, 0, 1)
    vil = vil.unsqueeze(1)

    # Resize
    vil = torch.nn.functional.interpolate(
        vil,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    past = vil[:INPUT_FRAMES]

    actual_future = vil[
        INPUT_FRAMES:
        INPUT_FRAMES + FUTURE_FRAMES
    ]

    return event_id, past, actual_future


# ============================================================
# Generate prediction
# ============================================================

@st.cache_data
def generate_prediction(sequence_index):

    event_id, past, actual_future = prepare_data(
        sequence_index
    )

    model = load_model()

    model_input = past.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        prediction = model(
            model_input,
            future_frames=None,
            future_steps=FUTURE_FRAMES,
            teacher_forcing_ratio=0.0
        )

    prediction = prediction.cpu().squeeze(0)

    return (
        event_id,
        past,
        prediction,
        actual_future
    )


# ============================================================
# Generate
# ============================================================

with st.spinner("Generating StormSense forecast..."):

    event_id, past, prediction, actual_future = (
        generate_prediction(SEQUENCE_INDEX)
    )


# ============================================================
# Event information
# ============================================================

if isinstance(event_id, bytes):
    event_id_display = event_id.decode()
else:
    event_id_display = str(event_id)

st.success(
    f"Forecast ready for SEVIR event {event_id_display}"
)


# ============================================================
# Time slider
# ============================================================

st.subheader("Forecast timeline")

selected_minutes = st.slider(
    "Select forecast time",
    min_value=0,
    max_value=60,
    value=5,
    step=5
)


# Current = 0
if selected_minutes == 0:

    selected_frame = past[-1, 0].numpy()

    frame_label = "Current observed VIL"

else:

    frame_index = (selected_minutes // 5) - 1

    selected_frame = prediction[
        frame_index,
        0
    ].numpy()

    frame_label = (
        f"Predicted VIL — +{selected_minutes} minutes"
    )
# ============================================================
# Interactive Storm Map + Storm Trajectory
# ============================================================

st.subheader("🗺️ Interactive Storm Map")

st.caption(
    "The map shows the selected VIL forecast and the "
    "relative movement of the strongest storm region."
)

# ------------------------------------------------------------
# Build trajectory up to selected forecast time
# ------------------------------------------------------------

trajectory = []

if selected_minutes > 0:

    number_of_frames = selected_minutes // 5

    for i in range(number_of_frames):

        frame = prediction[
            i,
            0
        ].numpy()

        center = find_relative_storm_center(
            frame,
            percentile=90
        )

        if center is not None:
            trajectory.append(center)

# ------------------------------------------------------------
# Create interactive map
# ------------------------------------------------------------

map_fig = create_vil_map(
    selected_frame,
    title=frame_label,
    trajectory=trajectory
)

st.plotly_chart(
    map_fig,
    use_container_width=True
)


# ============================================================
# Main visualization
# ============================================================

col1, col2 = st.columns([2, 1])


with col1:

    st.subheader(frame_label)

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    image = ax.imshow(
        selected_frame,
        cmap="turbo",
        vmin=0,
        vmax=1
    )

    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")

    plt.colorbar(
        image,
        ax=ax,
        label="Normalized VIL"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


with col2:

    st.subheader("Forecast information")

    if selected_minutes == 0:

        st.metric(
            "Forecast time",
            "Current"
        )

    else:

        st.metric(
            "Forecast horizon",
            f"+{selected_minutes} min"
        )

    max_vil = float(
        selected_frame.max()
    )

    mean_vil = float(
        selected_frame.mean()
    )

    st.metric(
        "Maximum VIL",
        f"{max_vil:.3f}"
    )

    st.metric(
        "Mean VIL",
        f"{mean_vil:.3f}"
    )

    st.caption(
        "VIL is a radar-derived field. "
        "Risk thresholds are not yet calibrated "
        "to operational severe-weather criteria."
    )


# ============================================================
# Forecast strip
# ============================================================

st.subheader("Forecast progression")

cols = st.columns(6)

for i, minutes in enumerate(
    [5, 15, 25, 35, 45, 60]
):

    frame_index = (minutes // 5) - 1

    frame = prediction[
        frame_index,
        0
    ].numpy()

    with cols[i]:

        st.image(
            frame,
            caption=f"+{minutes} min",
            clamp=True,
            use_container_width=True
        )


# ============================================================
# Actual comparison
# ============================================================

with st.expander("Compare prediction with actual future"):

    comparison_cols = st.columns(2)

    if selected_minutes == 0:

        st.info(
            "Select a future forecast time to compare "
            "prediction with the corresponding actual frame."
        )

    else:

        frame_index = (selected_minutes // 5) - 1

        predicted_frame = prediction[
            frame_index,
            0
        ].numpy()

        actual_frame = actual_future[
            frame_index,
            0
        ].numpy()

        with comparison_cols[0]:

            st.image(
                predicted_frame,
                caption=(
                    f"StormSense prediction +"
                    f"{selected_minutes} min"
                ),
                clamp=True,
                use_container_width=True
            )

        with comparison_cols[1]:

            st.image(
                actual_frame,
                caption=(
                    f"Observed actual +"
                    f"{selected_minutes} min"
                ),
                clamp=True,
                use_container_width=True
            )
