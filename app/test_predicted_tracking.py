import torch

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3
from app.storm_tracker import detect_storm_cells, match_cells


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

print("V3 predicted storm tracking")
print("=" * 55)

previous_cells = None

for frame_index in range(12):

    frame = prediction[frame_index, 0].numpy()

    cells = detect_storm_cells(
        frame,
        threshold=0.05,
        min_area=20
    )

    print()
    print(
        f"Forecast frame {frame_index + 1}/12 "
        f"({(frame_index + 1) * 5} min)"
    )

    print(f"Detected cells: {len(cells)}")
    print(f"Maximum VIL: {frame.max():.4f}")

    if cells:

        strongest = max(
            cells,
            key=lambda x: x["max_vil"]
        )

        print(
            f"Strongest cell: "
            f"x={strongest['centroid_x']:.2f}, "
            f"y={strongest['centroid_y']:.2f}, "
            f"max VIL={strongest['max_vil']:.3f}, "
            f"area={strongest['area']}"
        )

    if previous_cells is not None and cells:

        matches = match_cells(
            previous_cells,
            cells,
            max_distance=15
        )

        print(f"Matched tracks: {len(matches)}")

        for match in matches:

            print(
                f"  Cell {match['from_id']} -> "
                f"{match['to_id']} | "
                f"dx={match['dx']:.2f}, "
                f"dy={match['dy']:.2f}, "
                f"distance={match['distance']:.2f}px"
            )

    previous_cells = cells

print()
print("=" * 55)
print("Predicted storm tracking test complete.")
