import torch
from torch.utils.data import DataLoader, Subset

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3


FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"

INPUT_FRAMES = 12
TARGET_FRAMES = 12
IMAGE_SIZE = 128

TRAIN_SAMPLES = 2000
TEST_SAMPLES = 200

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
    max_samples=TRAIN_SAMPLES + TEST_SAMPLES
)

test_dataset = Subset(
    dataset,
    range(TRAIN_SAMPLES, TRAIN_SAMPLES + TEST_SAMPLES)
)

loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False
)

print("Training samples:", TRAIN_SAMPLES)
print("Unseen test samples:", len(test_dataset))


model = StormSenseConvLSTMv3(
    input_channels=1,
    hidden_channels=32
).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()


total_v4_mse = 0.0
total_persistence_mse = 0.0
count = 0

horizon_errors_v4 = torch.zeros(TARGET_FRAMES)
horizon_errors_persistence = torch.zeros(TARGET_FRAMES)

peak_errors_v4 = torch.zeros(TARGET_FRAMES)
peak_errors_persistence = torch.zeros(TARGET_FRAMES)


with torch.no_grad():

    for past, future in loader:

        past = past.to(device)
        future = future.to(device)

        prediction = model(
            past,
            future_frames=None,
            future_steps=TARGET_FRAMES,
            teacher_forcing_ratio=0.0
        )

        # V4 overall MSE
        v4_error = (prediction - future) ** 2

        total_v4_mse += (
            v4_error.mean().item() * past.size(0)
        )

        # Persistence baseline
        last_frame = past[:, -1:, :, :, :]

        persistence = last_frame.repeat(
            1,
            TARGET_FRAMES,
            1,
            1,
            1
        )

        persistence_error = (
            persistence - future
        ) ** 2

        total_persistence_mse += (
            persistence_error.mean().item()
            * past.size(0)
        )

        # Horizon-wise metrics
        for t in range(TARGET_FRAMES):

            actual = future[:, t]
            pred = prediction[:, t]
            persist = persistence[:, t]

            horizon_errors_v4[t] += (
                (pred - actual) ** 2
            ).mean().item() * past.size(0)

            horizon_errors_persistence[t] += (
                (persist - actual) ** 2
            ).mean().item() * past.size(0)

            # Peak intensity error
            actual_peak = actual.amax(
                dim=(1, 2, 3)
            )

            predicted_peak = pred.amax(
                dim=(1, 2, 3)
            )

            persistence_peak = persist.amax(
                dim=(1, 2, 3)
            )

            peak_errors_v4[t] += (
                torch.abs(
                    predicted_peak - actual_peak
                ).mean().item()
                * past.size(0)
            )

            peak_errors_persistence[t] += (
                torch.abs(
                    persistence_peak - actual_peak
                ).mean().item()
                * past.size(0)
            )

        count += past.size(0)


v4_mse = total_v4_mse / count
persistence_mse = total_persistence_mse / count

horizon_errors_v4 /= count
horizon_errors_persistence /= count

peak_errors_v4 /= count
peak_errors_persistence /= count


print()
print("===== V4 MULTI-STEP EVALUATION =====")

print(f"V4 MSE:          {v4_mse:.6f}")
print(f"Persistence MSE: {persistence_mse:.6f}")

improvement = (
    (persistence_mse - v4_mse)
    / persistence_mse
) * 100

print(f"Improvement:     {improvement:.2f}%")


print()
print("===== HORIZON-WISE MSE =====")

for t in range(TARGET_FRAMES):

    minutes = (t + 1) * 5

    v4 = horizon_errors_v4[t].item()
    persistence = horizon_errors_persistence[t].item()

    improvement = (
        (persistence - v4)
        / persistence
    ) * 100

    print(
        f"{minutes:2d} min | "
        f"V4: {v4:.6f} | "
        f"Persistence: {persistence:.6f} | "
        f"Improvement: {improvement:.2f}%"
    )


print()
print("===== PEAK VIL ERROR =====")

for t in range(TARGET_FRAMES):

    minutes = (t + 1) * 5

    v4_peak = peak_errors_v4[t].item()
    persistence_peak = peak_errors_persistence[t].item()

    print(
        f"{minutes:2d} min | "
        f"V4 peak error: {v4_peak:.6f} | "
        f"Persistence peak error: {persistence_peak:.6f}"
    )
