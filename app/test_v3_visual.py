import torch
import matplotlib.pyplot as plt

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3

FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
MODEL_PATH = "models/stormsense_convlstm_v3_multistep.pth"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

dataset = SEVIRVILDataset(
    FILE,
    input_frames=12,
    target_frames=12,
    image_size=128,
    max_samples=1
)

past, future = dataset[0]

past_batch = past.unsqueeze(0).to(device)

model = StormSenseConvLSTMv3(
    input_channels=1,
    hidden_channels=32
).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

with torch.no_grad():
    prediction = model(
        past_batch,
        future_frames=None,
        future_steps=12,
        teacher_forcing_ratio=0.0
    )

prediction = prediction[0, :, 0].cpu()
future = future[:, 0].cpu()
last_observed = past[-1, 0].cpu()

print()
print("===== V3 INTENSITY DIAGNOSTIC =====")

for t in range(12):

    actual_max = future[t].max().item()
    predicted_max = prediction[t].max().item()

    actual_mean = future[t].mean().item()
    predicted_mean = prediction[t].mean().item()

    minutes = (t + 1) * 5

    print(
        f"{minutes:2d} min | "
        f"Actual max: {actual_max:.3f} | "
        f"Pred max: {predicted_max:.3f} | "
        f"Actual mean: {actual_mean:.3f} | "
        f"Pred mean: {predicted_mean:.3f}"
    )

# Select forecast horizons
times = [0, 1, 2, 5, 11]

fig, axes = plt.subplots(3, len(times), figsize=(15, 8))

# Last observed
for i, t in enumerate(times):
    axes[0, i].imshow(
        last_observed,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[0, i].set_title("Last observed")
    axes[0, i].axis("off")

# Actual
for i, t in enumerate(times):
    axes[1, i].imshow(
        future[t],
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[1, i].set_title(f"Actual +{(t + 1) * 5} min")
    axes[1, i].axis("off")

# Prediction
for i, t in enumerate(times):
    axes[2, i].imshow(
        prediction[t],
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[2, i].set_title(f"V3 +{(t + 1) * 5} min")
    axes[2, i].axis("off")

plt.tight_layout()

output = "data/results/v3_multistep_diagnostic.png"
plt.savefig(output, dpi=150)

print()
print("Saved:", output)
