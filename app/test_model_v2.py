import torch

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v2 import StormSenseConvLSTMv2


FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


dataset = SEVIRVILDataset(
    file_path=FILE,
    input_frames=12,
    target_frames=12,
    image_size=128,
    max_samples=1
)

past, future = dataset[0]

past = past.unsqueeze(0).to(DEVICE)
future = future.unsqueeze(0).to(DEVICE)


model = StormSenseConvLSTMv2(
    input_channels=1,
    hidden_channels=32,
    output_channels=1
).to(DEVICE)


model.eval()

with torch.no_grad():

    prediction = model(past)


print("Device:", DEVICE)

print("Input:", past.shape)

print("Target:", future[:, 0].shape)

print("Prediction:", prediction.shape)

print(
    "Prediction range:",
    prediction.min().item(),
    "to",
    prediction.max().item()
)
