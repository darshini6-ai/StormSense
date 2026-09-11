import torch
import numpy as np
from torch.utils.data import DataLoader

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3


FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
MODEL_FILE = "models/stormsense_convlstm_v5_spatialloss.pth"

IMAGE_SIZE = 128
INPUT_FRAMES = 12
TARGET_FRAMES = 12

TRAINING_SAMPLES = 2000
TEST_SAMPLES = 200


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Unseen test dataset
# --------------------------------------------------

full_dataset = SEVIRVILDataset(
    FILE,
    input_frames=INPUT_FRAMES,
    target_frames=TARGET_FRAMES,
    image_size=IMAGE_SIZE,
    max_samples=TRAINING_SAMPLES + TEST_SAMPLES
)

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
    torch.load(MODEL_FILE, map_location=device)
)

model.eval()

print("Model loaded.")


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

mse_values = []
persistence_values = []

peak_errors = []

horizon_mse = [[] for _ in range(TARGET_FRAMES)]
horizon_persistence = [[] for _ in range(TARGET_FRAMES)]

with torch.no_grad():

    for index in test_indices:

        past, future = full_dataset[index]

        past_input = past.unsqueeze(0).to(device)
        future_input = future.unsqueeze(0).to(device)

        prediction = model(
            past_input,
            future_frames=None,
            future_steps=TARGET_FRAMES,
            teacher_forcing_ratio=0.0
        )

        # Prediction:
        # [1, 12, 1, 128, 128]

        pred = prediction[0, :, 0]
        actual = future_input[0, :, 0]

        # Persistence baseline
        persistence = past_input[0, -1, 0].unsqueeze(0)
        persistence = persistence.repeat(TARGET_FRAMES, 1, 1)

        # Overall MSE
        mse = torch.mean(
            (pred - actual) ** 2
        ).item()

        baseline_mse = torch.mean(
            (persistence - actual) ** 2
        ).item()

        mse_values.append(mse)
        persistence_values.append(baseline_mse)

        # Peak VIL error
        pred_peak = pred.amax(dim=(1, 2))
        actual_peak = actual.amax(dim=(1, 2))

        peak_error = torch.mean(
            torch.abs(pred_peak - actual_peak)
        ).item()

        peak_errors.append(peak_error)

        # Horizon-wise MSE
        for step in range(TARGET_FRAMES):

            step_mse = torch.mean(
                (pred[step] - actual[step]) ** 2
            ).item()

            step_baseline = torch.mean(
                (persistence[step] - actual[step]) ** 2
            ).item()

            horizon_mse[step].append(step_mse)
            horizon_persistence[step].append(step_baseline)


# --------------------------------------------------
# Results
# --------------------------------------------------

avg_mse = np.mean(mse_values)
avg_persistence = np.mean(persistence_values)

improvement = (
    (avg_persistence - avg_mse)
    / avg_persistence
) * 100

avg_peak_error = np.mean(peak_errors)


print("\n==============================")
print("V5 UNSEEN TEST RESULTS")
print("==============================")

print(f"Test samples: {TEST_SAMPLES}")

print(f"\nV5 MSE:         {avg_mse:.6f}")
print(f"Persistence:    {avg_persistence:.6f}")
print(f"Improvement:    {improvement:.2f}%")

print(f"\nAverage peak VIL error: {avg_peak_error:.6f}")


print("\nHorizon-wise results:")
print("------------------------------")

for step in range(TARGET_FRAMES):

    v5 = np.mean(horizon_mse[step])
    base = np.mean(horizon_persistence[step])

    improvement_h = (
        (base - v5) / base
    ) * 100

    minutes = (step + 1) * 5

    print(
        f"{minutes:2d} min | "
        f"V5={v5:.6f} | "
        f"Persistence={base:.6f} | "
        f"Improvement={improvement_h:.2f}%"
    )


print("\nV5 evaluation complete.")
