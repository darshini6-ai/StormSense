import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm import StormSenseConvLSTM


# ============================================================
# Configuration
# ============================================================

FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"

BATCH_SIZE = 2
EPOCHS = 5
MAX_SAMPLES = 500
LEARNING_RATE = 0.001

INPUT_FRAMES = 12
TARGET_FRAMES = 12
IMAGE_SIZE = 128

MODEL_PATH = "models/stormsense_convlstm_weighted.pth"


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
# Dataset
# ============================================================

dataset = SEVIRVILDataset(
    file_path=FILE,
    input_frames=INPUT_FRAMES,
    target_frames=TARGET_FRAMES,
    image_size=IMAGE_SIZE,
    max_samples=MAX_SAMPLES
)

print("Training samples:", len(dataset))


# ============================================================
# DataLoader
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# Model
# ============================================================

model = StormSenseConvLSTM(
    input_channels=1,
    hidden_channels=16,
    output_channels=1
)

model = model.to(DEVICE)


# ============================================================
# Storm-weighted MSE loss
# ============================================================

def storm_weighted_mse(prediction, target):

    # Give stronger VIL pixels more importance.
    #
    # Background:
    # target ≈ 0 → weight ≈ 1
    #
    # Strong storm:
    # target → 1 → weight approaches 6

    weight = 1.0 + 5.0 * target

    loss = weight * (prediction - target) ** 2

    return loss.mean()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0

    for batch_idx, (past, future) in enumerate(loader):

        past = past.to(DEVICE)
        future = future.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(
            past,
            future_steps=TARGET_FRAMES
        )

        loss = storm_weighted_mse(
            prediction,
            future
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Batch {batch_idx + 1}/{len(loader)} | "
            f"Loss: {loss.item():.6f}"
        )

    average_loss = total_loss / len(loader)

    print(
        f"\nEpoch {epoch + 1} complete | "
        f"Average Loss: {average_loss:.6f}\n"
    )


# ============================================================
# Save model
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("Model saved successfully.")
print("Saved to:", MODEL_PATH)
