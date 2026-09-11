import h5py
import torch
import matplotlib.pyplot as plt

from models.convlstm_v3 import StormSenseConvLSTMv3


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
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# ============================================================
# Load SEVIR sequence
# ============================================================

with h5py.File(FILE_PATH, "r") as f:

    event_id = f["id"][SEQUENCE_INDEX]

    vil = f["vil"][
        SEQUENCE_INDEX,
        :,
        :,
        :
    ]

print()
print("Event ID:", event_id)
print("Original sequence shape:", vil.shape)


# ============================================================
# Normalize
# ============================================================

vil = torch.tensor(
    vil,
    dtype=torch.float32
) / 255.0


# ============================================================
# Convert:
# (H, W, T)
# →
# (T, 1, H, W)
# ============================================================

vil = vil.permute(2, 0, 1)

vil = vil.unsqueeze(1)


# ============================================================
# Resize to 128 × 128
# ============================================================

vil = torch.nn.functional.interpolate(
    vil,
    size=(IMAGE_SIZE, IMAGE_SIZE),
    mode="bilinear",
    align_corners=False
)


# ============================================================
# Split observed and future frames
# ============================================================

past = vil[:INPUT_FRAMES]

actual_future = vil[
    INPUT_FRAMES:
    INPUT_FRAMES + FUTURE_FRAMES
]


# ============================================================
# Add batch dimension
# ============================================================

past = past.unsqueeze(0)

actual_future = actual_future.unsqueeze(0)


print("Past shape:", past.shape)
print("Actual future shape:", actual_future.shape)


# ============================================================
# Move to device
# ============================================================

past = past.to(DEVICE)
actual_future = actual_future.to(DEVICE)


# ============================================================
# Create V3 model
# ============================================================

model = StormSenseConvLSTMv3(
    input_channels=1,
    hidden_channels=32
).to(DEVICE)


# ============================================================
# Load trained V3 checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.eval()

print()
print("V3 model loaded successfully.")


# ============================================================
# Generate future predictions
# ============================================================

with torch.no_grad():

    prediction = model(
        past,
        future_frames=None,
        future_steps=FUTURE_FRAMES,
        teacher_forcing_ratio=0.0
    )


print()
print("Prediction completed.")
print("Prediction shape:", prediction.shape)

print(
    "Prediction range:",
    prediction.min().item(),
    "to",
    prediction.max().item()
)


# ============================================================
# Move to CPU
# ============================================================

past_cpu = past.cpu()

prediction_cpu = prediction.cpu()

actual_cpu = actual_future.cpu()


# ============================================================
# Create visualization
# ============================================================

fig, axes = plt.subplots(
    3,
    FUTURE_FRAMES,
    figsize=(20, 6)
)


# ============================================================
# Row 1 — Last observed frame
# ============================================================

last_observed = past_cpu[
    0,
    -1,
    0
].numpy()


for i in range(FUTURE_FRAMES):

    axes[0, i].imshow(
        last_observed,
        cmap="turbo",
        vmin=0,
        vmax=1
    )

    axes[0, i].set_title(
        "Current"
    )

    axes[0, i].axis("off")


# ============================================================
# Row 2 — Predicted future
# ============================================================

for i in range(FUTURE_FRAMES):

    frame = prediction_cpu[
        0,
        i,
        0
    ].numpy()

    axes[1, i].imshow(
        frame,
        cmap="turbo",
        vmin=0,
        vmax=1
    )

    axes[1, i].set_title(
        f"Pred +{(i + 1) * 5} min"
    )

    axes[1, i].axis("off")


# ============================================================
# Row 3 — Actual future
# ============================================================

for i in range(FUTURE_FRAMES):

    frame = actual_cpu[
        0,
        i,
        0
    ].numpy()

    axes[2, i].imshow(
        frame,
        cmap="turbo",
        vmin=0,
        vmax=1
    )

    axes[2, i].set_title(
        f"Actual +{(i + 1) * 5} min"
    )

    axes[2, i].axis("off")


# ============================================================
# Row labels
# ============================================================

axes[0, 0].set_ylabel(
    "Observed",
    fontsize=12
)

axes[1, 0].set_ylabel(
    "Predicted",
    fontsize=12
)

axes[2, 0].set_ylabel(
    "Actual",
    fontsize=12
)


plt.suptitle(
    f"StormSense V3 Forecast — Event {event_id}",
    fontsize=16
)

plt.tight_layout()

plt.show()
