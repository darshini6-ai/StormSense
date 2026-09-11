import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3


FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"

BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 0.001
MAX_SAMPLES = 2000
IMAGE_SIZE = 128

TEACHER_FORCING_RATIO = 0.5


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", DEVICE)


dataset = SEVIRVILDataset(
    file_path=FILE,
    input_frames=12,
    target_frames=12,
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
    hidden_channels=32,
    output_channels=1
).to(DEVICE)


criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for past, future in loader:

        past = past.to(DEVICE)
        future = future.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(
            past,
            future_frames=future,
            future_steps=12,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO
        )

        loss = criterion(
            prediction,
            future
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.6f}"
    )


MODEL_FILE = "models/stormsense_convlstm_v3_multistep.pth"

torch.save(
    model.state_dict(),
    MODEL_FILE
)

print("Saved:", MODEL_FILE)
