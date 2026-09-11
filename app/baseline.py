import torch
import torch.nn as nn

from app.sevir_dataset import SEVIRVILDataset


# ============================================================
# Configuration
# ============================================================

FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"

INPUT_FRAMES = 12
TARGET_FRAMES = 12
IMAGE_SIZE = 128

MAX_SAMPLES = 100


# ============================================================
# Dataset
# ============================================================

dataset = SEVIRVILDataset(
    file_path=FILE,
    input_frames=INPUT_FRAMES,
    target_frames=TARGET_FRAMES,
    image_size=IMAGE_SIZE,
    max_samples=MAX_SAMPLES
)

print("Baseline samples:", len(dataset))


# ============================================================
# Persistence baseline
# ============================================================

criterion = nn.MSELoss()

total_loss = 0.0

for i in range(len(dataset)):

    past, future = dataset[i]

    # Last observed frame
    last_frame = past[-1:]

    # Repeat the last observed frame for all 12 future steps
    prediction = last_frame.repeat(TARGET_FRAMES, 1, 1, 1)

    loss = criterion(
        prediction,
        future
    )

    total_loss += loss.item()


average_mse = total_loss / len(dataset)


# ============================================================
# Result
# ============================================================

print("--------------------------------")
print("Persistence Baseline Results")
print("Average MSE:", average_mse)
print("--------------------------------")
