import torch
import numpy as np

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

MODEL_PATH = "models/stormsense_convlstm_v3_multistep.pth"

dataset = SEVIRVILDataset(
    "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
)

past, future = dataset[0]

model = StormSenseConvLSTMv3().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)
model.eval()

input_tensor = past.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    prediction = model(input_tensor)

prediction = prediction.squeeze(0).cpu()

print("V3 spatial diagnostic")
print("=" * 60)

for i in range(12):

    pred = prediction[i, 0].numpy()
    actual = future[i, 0].numpy()

    pred_threshold = np.percentile(pred, 90)
    actual_threshold = np.percentile(actual, 90)

    pred_mask = pred >= pred_threshold
    actual_mask = actual >= actual_threshold

    pred_area = int(pred_mask.sum())
    actual_area = int(actual_mask.sum())

    pred_y, pred_x = np.where(pred_mask)
    actual_y, actual_x = np.where(actual_mask)

    print()
    print(f"{(i + 1) * 5:2d} min")
    print(
        f"Predicted: "
        f"max={pred.max():.4f}, "
        f"p90={pred_threshold:.4f}, "
        f"top10% area={pred_area}"
    )

    if len(pred_x) > 0:
        print(
            f"           centroid="
            f"({pred_x.mean():.1f}, {pred_y.mean():.1f})"
        )

    print(
        f"Actual:    "
        f"max={actual.max():.4f}, "
        f"p90={actual_threshold:.4f}, "
        f"top10% area={actual_area}"
    )

    if len(actual_x) > 0:
        print(
            f"           centroid="
            f"({actual_x.mean():.1f}, {actual_y.mean():.1f})"
        )

print()
print("=" * 60)
print("V3 spatial diagnostic complete.")
