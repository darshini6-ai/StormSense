import torch
import numpy as np

from app.sevir_dataset import SEVIRVILDataset
from app.storm_tracker import detect_storm_cells, track_cells
from models.convlstm_v3 import StormSenseConvLSTMv3


FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
MODEL_FILE = "models/stormsense_convlstm_v3_multistep.pth"

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

past_input = past.unsqueeze(0).to(device)

model = StormSenseConvLSTMv3(
    input_channels=1,
    hidden_channels=32
).to(device)

model.load_state_dict(
    torch.load(MODEL_FILE, map_location=device)
)

model.eval()

with torch.no_grad():
    prediction = model(
        past_input,
        future_frames=None,
        future_steps=12,
        teacher_forcing_ratio=0.0
    )

prediction = prediction[0, :, 0].cpu().numpy()
actual = future[:, 0].numpy()


def relative_mask(frame, percentile=90):
    threshold = np.percentile(frame, percentile)
    return frame >= threshold


print("\nRelative storm-region tracking")
print("Threshold: top 10% of each frame")
print("--------------------------------")

previous_pred = past[-1, 0].numpy()

for step in range(12):

    pred_frame = prediction[step]
    actual_frame = actual[step]

    pred_mask = relative_mask(pred_frame, percentile=90)
    actual_mask = relative_mask(actual_frame, percentile=90)

    pred_cells = detect_storm_cells(
        pred_frame,
        threshold=np.percentile(pred_frame, 90),
        min_area=5
    )

    actual_cells = detect_storm_cells(
        actual_frame,
        threshold=np.percentile(actual_frame, 90),
        min_area=5
    )

    minutes = (step + 1) * 5

    print(f"\n{minutes} min")
    print(
        f"  Prediction: "
        f"max={pred_frame.max():.3f}, "
        f"cells={len(pred_cells)}"
    )
    print(
        f"  Actual:     "
        f"max={actual_frame.max():.3f}, "
        f"cells={len(actual_cells)}"
    )

    if pred_cells:
        cell = max(pred_cells, key=lambda c: c["area"])

        print(
            f"  Predicted main cell: "
            f"x={cell['centroid_x']:.1f}, "
            f"y={cell['centroid_y']:.1f}, "
            f"area={cell['area']}"
        )

    if actual_cells:
        cell = max(actual_cells, key=lambda c: c["area"])

        print(
            f"  Actual main cell:     "
            f"x={cell['centroid_x']:.1f}, "
            f"y={cell['centroid_y']:.1f}, "
            f"area={cell['area']}"
        )

    if step > 0:
        tracks = track_cells(
            previous_pred,
            pred_frame,
            threshold=np.percentile(previous_pred, 90),
            min_area=5
        )

        if tracks:
            track = tracks[0]

            print(
                f"  Predicted movement: "
                f"dx={track['dx']:.2f}, "
                f"dy={track['dy']:.2f}, "
                f"distance={track['displacement_pixels']:.2f}px"
            )

    previous_pred = pred_frame

print("\nRelative tracking test complete.")
