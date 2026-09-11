import torch
import numpy as np

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3


# --------------------------------------------------
# Configuration
# --------------------------------------------------

FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
MODEL_FILE = "models/stormsense_convlstm_v3_multistep.pth"

IMAGE_SIZE = 128

INPUT_FRAMES = 12
TARGET_FRAMES = 12

# Use a separate range from the training samples
TRAINING_SAMPLES = 2000
TEST_SAMPLES = 200


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = SEVIRVILDataset(
    FILE,
    input_frames=INPUT_FRAMES,
    target_frames=TARGET_FRAMES,
    image_size=IMAGE_SIZE,
    max_samples=TRAINING_SAMPLES + TEST_SAMPLES
)

print("Total available samples:", len(dataset))

test_indices = range(
    TRAINING_SAMPLES,
    TRAINING_SAMPLES + TEST_SAMPLES
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = StormSenseConvLSTMv3(
    input_channels=1,
    hidden_channels=32
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model.eval()

print("V3 model loaded.")


# --------------------------------------------------
# Storage
# --------------------------------------------------

v3_errors = [[] for _ in range(TARGET_FRAMES)]
persistence_errors = [[] for _ in range(TARGET_FRAMES)]


# --------------------------------------------------
# Backtest
# --------------------------------------------------

with torch.no_grad():

    for index in test_indices:

        past, future = dataset[index]

        past = past.unsqueeze(0).to(device)
        future = future.unsqueeze(0).to(device)

        # V3 forecast
        prediction = model(
            past,
            future_frames=None,
            future_steps=TARGET_FRAMES,
            teacher_forcing_ratio=0.0
        )

        prediction = prediction[0, :, 0]
        actual = future[0, :, 0]

        # Persistence:
        # repeat the last observed frame
        last_frame = past[0, -1, 0]

        persistence = last_frame.unsqueeze(0).repeat(
            TARGET_FRAMES,
            1,
            1
        )

        # Calculate error at every horizon
        for step in range(TARGET_FRAMES):

            v3_mse = torch.mean(
                (prediction[step] - actual[step]) ** 2
            ).item()

            persistence_mse = torch.mean(
                (persistence[step] - actual[step]) ** 2
            ).item()

            v3_errors[step].append(v3_mse)
            persistence_errors[step].append(
                persistence_mse
            )


# --------------------------------------------------
# Calculate averages
# --------------------------------------------------

v3_horizon = np.array([
    np.mean(errors)
    for errors in v3_errors
])

persistence_horizon = np.array([
    np.mean(errors)
    for errors in persistence_errors
])


v3_overall = np.mean(v3_horizon)
persistence_overall = np.mean(
    persistence_horizon
)


overall_improvement = (
    (persistence_overall - v3_overall)
    / persistence_overall
) * 100


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n========================================")
print("STORMSENSE BACKTEST")
print("========================================")

print(f"Test samples: {TEST_SAMPLES}")

print("\nOverall:")
print("----------------------------------------")

print(
    f"V3 MSE:          "
    f"{v3_overall:.6f}"
)

print(
    f"Persistence MSE: "
    f"{persistence_overall:.6f}"
)

print(
    f"Improvement:     "
    f"{overall_improvement:.2f}%"
)


print("\nHorizon-wise results:")
print("----------------------------------------")

for step in range(TARGET_FRAMES):

    minutes = (step + 1) * 5

    v3_value = v3_horizon[step]
    persistence_value = persistence_horizon[step]

    improvement = (
        (persistence_value - v3_value)
        / persistence_value
    ) * 100

    print(
        f"{minutes:2d} min | "
        f"V3={v3_value:.6f} | "
        f"Persistence={persistence_value:.6f} | "
        f"Improvement={improvement:.2f}%"
    )


print("\nBacktest complete.")
