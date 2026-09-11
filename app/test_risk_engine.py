import torch
import numpy as np

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3
from app.risk_engine import (
    create_risk_mask,
    detect_storm_cells,
    calculate_risk_level
)

FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
MODEL_PATH = "models/stormsense_convlstm_v3_multistep.pth"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

# Load one real SEVIR sequence
dataset = SEVIRVILDataset(
    FILE,
    input_frames=12,
    target_frames=12,
    image_size=128,
    max_samples=1
)

past, future = dataset[0]

past = past.unsqueeze(0).to(device)

print("Input shape:", past.shape)

# Load trained V3
model = StormSenseConvLSTMv3(
    input_channels=1,
    hidden_channels=32
).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

# Generate 12 future frames
with torch.no_grad():

    prediction = model(
        past,
        future_frames=None,
        future_steps=12,
        teacher_forcing_ratio=0.0
    )

print("Prediction shape:", prediction.shape)

# Analyze each predicted frame
print()
print("===== STORM RISK FORECAST =====")

for t in range(12):

    frame = prediction[0, t, 0].cpu().numpy()

    risk_mask = create_risk_mask(
        frame,
        threshold=0.15
    )

    cells = detect_storm_cells(
        risk_mask,
        min_area=20
    )

    risk = calculate_risk_level(frame)

    minutes = (t + 1) * 5

    print()
    print(f"{minutes:2d} min:")
    print(f"  Risk level: {risk['level']}")
    print(f"  Max VIL:    {risk['max_vil']:.3f}")
    print(f"  Mean VIL:   {risk['mean_vil']:.3f}")
    print(f"  Storm cells: {len(cells)}")

    for cell in cells:
        print(
            f"    Cell {cell['id']}: "
            f"area={cell['area']} px, "
            f"center=({cell['centroid_x']:.1f}, "
            f"{cell['centroid_y']:.1f})"
        )
