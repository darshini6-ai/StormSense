import torch
from torch.utils.data import DataLoader

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3


# --------------------------------------------------
# Configuration
# --------------------------------------------------

FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"

BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 0.001
MAX_SAMPLES = 2000
IMAGE_SIZE = 128

INPUT_FRAMES = 12
TARGET_FRAMES = 12

TEACHER_FORCING_RATIO = 0.5

MODEL_FILE = "models/stormsense_convlstm_v5_spatialloss.pth"


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
    max_samples=MAX_SAMPLES
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Training samples:", len(dataset))


# --------------------------------------------------
# Model
# --------------------------------------------------

model = StormSenseConvLSTMv3(
    input_channels=1,
    hidden_channels=32
).to(device)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Spatial gradient loss
# --------------------------------------------------

def gradient_loss(prediction, target):

    pred_dx = prediction[:, :, :, :, 1:] - prediction[:, :, :, :, :-1]
    pred_dy = prediction[:, :, :, 1:, :] - prediction[:, :, :, :-1, :]

    target_dx = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
    target_dy = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]

    loss_x = torch.mean(
        torch.abs(pred_dx - target_dx)
    )

    loss_y = torch.mean(
        torch.abs(pred_dy - target_dy)
    )

    return loss_x + loss_y


# --------------------------------------------------
# Training
# --------------------------------------------------

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0
    total_mse = 0.0
    total_gradient = 0.0

    for past, future in loader:

        past = past.to(device)
        future = future.to(device)

        optimizer.zero_grad()

        prediction = model(
            past,
            future_frames=future,
            future_steps=TARGET_FRAMES,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO
        )

        # Standard pixel loss
        mse = torch.mean(
            (prediction - future) ** 2
        )

        # Spatial structure loss
        spatial = gradient_loss(
            prediction,
            future
        )

        # Combined loss
        loss = (
            0.8 * mse +
            0.2 * spatial
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_gradient += spatial.item()

    avg_loss = total_loss / len(loader)
    avg_mse = total_mse / len(loader)
    avg_gradient = total_gradient / len(loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {avg_loss:.6f} | "
        f"MSE: {avg_mse:.6f} | "
        f"Gradient: {avg_gradient:.6f}"
    )


# --------------------------------------------------
# Save model
# --------------------------------------------------

torch.save(
    model.state_dict(),
    MODEL_FILE
)

print("\nV5 model saved:")
print(MODEL_FILE)
