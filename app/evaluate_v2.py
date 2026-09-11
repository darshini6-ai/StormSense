import torch
from torch.utils.data import DataLoader

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v2 import StormSenseConvLSTMv2


FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
MODEL_FILE = "models/stormsense_convlstm_v2_onestep.pth"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

TRAIN_SAMPLES = 2000
TEST_SAMPLES = 200
IMAGE_SIZE = 128


print("Device:", DEVICE)


# Load enough samples to reach the unseen test region
dataset = SEVIRVILDataset(
    file_path=FILE,
    input_frames=12,
    target_frames=12,
    image_size=IMAGE_SIZE,
    max_samples=TRAIN_SAMPLES + TEST_SAMPLES
)


# Select only samples AFTER the training region
test_indices = range(
    TRAIN_SAMPLES,
    TRAIN_SAMPLES + TEST_SAMPLES
)


model = StormSenseConvLSTMv2(
    input_channels=1,
    hidden_channels=32,
    output_channels=1
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_FILE, map_location=DEVICE)
)

model.eval()


total_v2_error = 0.0
total_baseline_error = 0.0
total_pixels = 0


with torch.no_grad():

    for index in test_indices:

        past, future = dataset[index]

        past = past.unsqueeze(0).to(DEVICE)
        target = future[0].unsqueeze(0).to(DEVICE)

        # StormSense V2
        prediction = model(past)

        # Persistence baseline
        baseline = past[:, -1]

        v2_error = torch.sum(
            (prediction - target) ** 2
        ).item()

        baseline_error = torch.sum(
            (baseline - target) ** 2
        ).item()

        total_v2_error += v2_error
        total_baseline_error += baseline_error

        total_pixels += target.numel()


v2_mse = total_v2_error / total_pixels
baseline_mse = total_baseline_error / total_pixels


print()
print("Training samples:", TRAIN_SAMPLES)
print("Unseen test samples:", TEST_SAMPLES)

print()
print("V2 unseen-test MSE:", v2_mse)
print("Persistence unseen-test MSE:", baseline_mse)


if v2_mse < baseline_mse:

    improvement = (
        (baseline_mse - v2_mse)
        / baseline_mse
    ) * 100

    print()
    print(
        f"V2 improves over persistence by "
        f"{improvement:.2f}%"
    )

else:

    difference = (
        (v2_mse - baseline_mse)
        / baseline_mse
    ) * 100

    print()
    print(
        f"V2 is {difference:.2f}% worse "
        f"than persistence"
    )
