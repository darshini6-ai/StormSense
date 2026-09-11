import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

FILE_PATH = (
    "data/sevir/vil/"
    "SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
)

MODEL_PATH = (
    "models/"
    "stormsense_convlstm_v6_intensity.pth"
)


def intensity_weighted_mse(pred, target):
    """
    Give more importance to high-VIL pixels.

    The weight is bounded so that high-intensity pixels
    matter more without completely dominating the loss.
    """

    weight = 1.0 + 4.0 * target

    weight = torch.clamp(
        weight,
        min=1.0,
        max=5.0
    )

    return torch.mean(
        weight * (pred - target) ** 2
    )


def combined_loss(pred, target):

    mse = F.mse_loss(pred, target)

    weighted = intensity_weighted_mse(
        pred,
        target
    )

    loss = (
        0.8 * mse
        + 0.2 * weighted
    )

    return loss, mse, weighted


dataset = SEVIRVILDataset(
    FILE_PATH,
    max_samples=2000
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

model = StormSenseConvLSTMv3().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

print("StormSense V6 intensity-aware training")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Training samples: {len(dataset)}")

for epoch in range(epochs):

    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_weighted = 0.0

    for past, future in loader:

        past = past.to(DEVICE)
        future = future.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(
            past,
            future_frames=future,
            teacher_forcing_ratio=0.5
        )

        loss, mse, weighted = combined_loss(
            prediction,
            future
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_weighted += weighted.item()

    n = len(loader)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {total_loss / n:.6f} | "
        f"MSE: {total_mse / n:.6f} | "
        f"Weighted: {total_weighted / n:.6f}"
    )

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print()
print("=" * 60)
print(f"Model saved to: {MODEL_PATH}")
print("V6 training complete.")
