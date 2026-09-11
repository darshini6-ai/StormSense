import torch
import matplotlib.pyplot as plt

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm import StormSenseConvLSTM


FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
MODEL_FILE = "models/stormsense_convlstm_sliding.pth"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Load one sliding-window sample
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


# Load model
model = StormSenseConvLSTM(
    input_channels=1,
    hidden_channels=16,
    output_channels=1
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_FILE, map_location=DEVICE)
)

model.eval()


with torch.no_grad():

    # Run the normal model, but request only ONE future frame
    prediction = model(
        past,
        future_steps=1
    )


print("Device:", DEVICE)
print("Past shape:", past.shape)
print("Actual next frame:", future[:, 0].shape)
print("Predicted next frame:", prediction[:, 0].shape)

print(
    "Actual range:",
    future[:, 0].min().item(),
    "to",
    future[:, 0].max().item()
)

print(
    "Prediction range:",
    prediction[:, 0].min().item(),
    "to",
    prediction[:, 0].max().item()
)


# One-step MSE
mse = torch.mean(
    (prediction[:, 0] - future[:, 0]) ** 2
).item()

print("One-step MSE:", mse)


# Visualization
last_observed = past[0, -1, 0].cpu().numpy()
actual_next = future[0, 0, 0].cpu().numpy()
predicted_next = prediction[0, 0, 0].cpu().numpy()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(last_observed, cmap="gray")
axes[0].set_title("Last observed")

axes[1].imshow(actual_next, cmap="gray")
axes[1].set_title("Actual next frame")

axes[2].imshow(predicted_next, cmap="gray")
axes[2].set_title("Predicted next frame")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output_file = "data/results/one_step_test.png"
plt.savefig(output_file, dpi=150)

print("Saved:", output_file)
