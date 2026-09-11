import torch
import matplotlib.pyplot as plt

from app.preprocessing import create_training_sample
from models.convlstm import StormSenseConvLSTM


# ============================================================
# Configuration
# ============================================================

FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"

MODEL_PATH = "models/stormsense_convlstm_sliding.pth"

INPUT_FRAMES = 12
TARGET_FRAMES = 12
IMAGE_SIZE = 128

SEQUENCE_INDEX = 0


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
# Load original SEVIR sequence
# ============================================================

past_original, future_original, event_id = create_training_sample(
    index=SEQUENCE_INDEX,
    input_frames=INPUT_FRAMES,
    target_frames=TARGET_FRAMES
)

print()
print("Event ID:", event_id)
print("Original past shape:", past_original.shape)
print("Original future shape:", future_original.shape)


# ============================================================
# Convert original 384x384 data to 128x128
# ============================================================

past = torch.tensor(
    past_original,
    dtype=torch.float32
)

future = torch.tensor(
    future_original,
    dtype=torch.float32
)


# Add channel dimension
# (T, H, W)
# →
# (T, 1, H, W)

past = past.unsqueeze(1)
future = future.unsqueeze(1)


# Resize
past = torch.nn.functional.interpolate(
    past,
    size=(IMAGE_SIZE, IMAGE_SIZE),
    mode="bilinear",
    align_corners=False
)

future = torch.nn.functional.interpolate(
    future,
    size=(IMAGE_SIZE, IMAGE_SIZE),
    mode="bilinear",
    align_corners=False
)


# Add batch dimension
# (T, 1, H, W)
# →
# (1, T, 1, H, W)

past = past.unsqueeze(0)
future = future.unsqueeze(0)


past = past.to(DEVICE)
future = future.to(DEVICE)


print("Model input shape:", past.shape)
print("Actual future shape:", future.shape)


# ============================================================
# Create model
# ============================================================

model = StormSenseConvLSTM(
    input_channels=1,
    hidden_channels=16,
    output_channels=1
)

model = model.to(DEVICE)


# ============================================================
# Load trained model
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print()
print("Model loaded successfully.")


# ============================================================
# Generate prediction
# ============================================================

with torch.no_grad():

    prediction = model(
        past,
        future_steps=TARGET_FRAMES
    )


# ============================================================
# Calculate MSE
# ============================================================

mse = torch.mean(
    (prediction - future) ** 2
).item()


print()
print("Prediction completed successfully.")
print("--------------------------------")

print("Prediction shape:", prediction.shape)
print("Actual shape:", future.shape)

print("MSE:", mse)

print(
    "Prediction range:",
    prediction.min().item(),
    "to",
    prediction.max().item()
)


# ============================================================
# Move results to CPU
# ============================================================

prediction_cpu = prediction.detach().cpu()

actual_cpu = future.detach().cpu()


# ============================================================
# Create visualization
# ============================================================

fig, axes = plt.subplots(
    3,
    TARGET_FRAMES,
    figsize=(18, 5)
)


# ------------------------------------------------------------
# Row 1: Last observed frame
# ------------------------------------------------------------

last_past = past[0, -1, 0].detach().cpu().numpy()

for i in range(TARGET_FRAMES):

    axes[0, i].imshow(
        last_past,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0, i].set_title(
        "Last observed"
    )

    axes[0, i].axis("off")


# ------------------------------------------------------------
# Row 2: Actual future
# ------------------------------------------------------------

for i in range(TARGET_FRAMES):

    actual_frame = actual_cpu[
        0,
        i,
        0
    ].numpy()

    axes[1, i].imshow(
        actual_frame,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1, i].set_title(
        f"Actual +{(i + 1) * 5} min"
    )

    axes[1, i].axis("off")


# ------------------------------------------------------------
# Row 3: Predicted future
# ------------------------------------------------------------

for i in range(TARGET_FRAMES):

    predicted_frame = prediction_cpu[
        0,
        i,
        0
    ].numpy()

    axes[2, i].imshow(
        predicted_frame,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2, i].set_title(
        f"Predicted +{(i + 1) * 5} min"
    )

    axes[2, i].axis("off")


# ============================================================
# Row labels
# ============================================================

axes[0, 0].set_ylabel(
    "OBSERVED",
    fontsize=12
)

axes[1, 0].set_ylabel(
    "ACTUAL",
    fontsize=12
)

axes[2, 0].set_ylabel(
    "PREDICTED",
    fontsize=12
)


plt.tight_layout()


# ============================================================
# Save visualization
# ============================================================

OUTPUT_PATH = "data/results/prediction_sliding_test.png"

plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print()
print("Visualization saved:")
print(OUTPUT_PATH)
