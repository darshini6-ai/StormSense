import torch
import torch.nn as nn

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm import StormSenseConvLSTM


# ============================================================
# Configuration
# ============================================================

FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"

MODEL_PATH = "models/stormsense_convlstm_weighted.pth"

INPUT_FRAMES = 12
TARGET_FRAMES = 12
IMAGE_SIZE = 128

MAX_SAMPLES = 100


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

print("Evaluation samples:", len(dataset))


# ============================================================
# Load model
# ============================================================

model = StormSenseConvLSTM(
    input_channels=1,
    hidden_channels=16,
    output_channels=1
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# Evaluate
# ============================================================

criterion = nn.MSELoss()

total_loss = 0.0

with torch.no_grad():

    for i in range(len(dataset)):

        past, future = dataset[i]

        past = past.unsqueeze(0).to(DEVICE)
        future = future.unsqueeze(0).to(DEVICE)

        prediction = model(
            past,
            future_steps=TARGET_FRAMES
        )

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
print("ConvLSTM Evaluation Results")
print("Average MSE:", average_mse)
print("--------------------------------")
