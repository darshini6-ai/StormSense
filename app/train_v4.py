import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3


FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"

INPUT_FRAMES = 12
TARGET_FRAMES = 12
IMAGE_SIZE = 128

BATCH_SIZE = 2
EPOCHS = 5
MAX_SAMPLES = 2000

LEARNING_RATE = 0.001
TEACHER_FORCING_RATIO = 0.5

MODEL_PATH = "models/stormsense_convlstm_v4_stormloss.pth"


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


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


model = StormSenseConvLSTMv3(
    input_channels=1,
    hidden_channels=32
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


def storm_focused_loss(prediction, target):
    """
    Combined loss:
    - ordinary MSE preserves overall field accuracy
    - higher-VIL pixels receive additional importance
    """

    mse = torch.mean(
        (prediction - target) ** 2
    )

    # Give progressively more weight to stronger VIL.
    storm_weight = 1.0 + 4.0 * target

    weighted_mse = torch.mean(
        storm_weight * (prediction - target) ** 2
    )

    # Combine the two objectives.
    loss = 0.5 * mse + 0.5 * weighted_mse

    return loss


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    samples_seen = 0

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

        loss = storm_focused_loss(
            prediction,
            future
        )

        loss.backward()

        optimizer.step()

        batch_size = past.size(0)

        running_loss += loss.item() * batch_size
        samples_seen += batch_size

    avg_loss = running_loss / samples_seen

    print(
        f"Epoch {epoch + 1}/{EPOCHS} - "
        f"Loss: {avg_loss:.6f}"
    )


torch.save(
    model.state_dict(),
    MODEL_PATH
)

print()
print("Saved:", MODEL_PATH)
