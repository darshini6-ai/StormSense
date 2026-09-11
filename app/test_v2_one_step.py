import torch
import matplotlib.pyplot as plt

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v2 import StormSenseConvLSTMv2


FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
MODEL_FILE = "models/stormsense_convlstm_v2_onestep.pth"

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

model.load_state_dict(
    torch.load(MODEL_FILE, map_location=DEVICE)
)

model.eval()


with torch.no_grad():

    prediction = model(past)


actual_next = future[:, 0]
predicted_next = prediction


mse = torch.mean(
    (predicted_next - actual_next) ** 2
).item()


print("Device:", DEVICE)

print("Input:", past.shape)

print("Actual next:", actual_next.shape)

print("Prediction:", predicted_next.shape)

print(
    "Actual range:",
    actual_next.min().item(),
    "to",
    actual_next.max().item()
)

print(
    "Prediction range:",
    predicted_next.min().item(),
    "to",
    predicted_next.max().item()
)

print("V2 One-step MSE:", mse)


last_observed = past[0, -1, 0].cpu().numpy()
actual_image = actual_next[0, 0].cpu().numpy()
predicted_image = predicted_next[0, 0].cpu().numpy()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(last_observed, cmap="gray")
axes[0].set_title("Last observed")

axes[1].imshow(actual_image, cmap="gray")
axes[1].set_title("Actual next frame")

axes[2].imshow(predicted_image, cmap="gray")
axes[2].set_title("V2 predicted next frame")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output_file = "data/results/v2_one_step_test.png"

plt.savefig(
    output_file,
    dpi=150,
    bbox_inches="tight"
)

print("Saved:", output_file)
