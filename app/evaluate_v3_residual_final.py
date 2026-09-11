import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3
from models.convlstm_v3_residual import StormSenseConvLSTMv3Residual


FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"

V3_MODEL = "models/stormsense_convlstm_v3_multistep.pth"
RESIDUAL_MODEL = "models/stormsense_convlstm_v3_residual_event_disjoint.pth"

INPUT_FRAMES = 12
TARGET_FRAMES = 12
IMAGE_SIZE = 128

BATCH_SIZE = 2
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("=" * 75)
print("STORMSENSE - FINAL EVENT-DISJOINT V3 COMPARISON")
print("=" * 75)

print("Device:", device)


# ============================================================
# EVENT SPLIT
# ============================================================

with h5py.File(FILE, "r") as f:
    num_events = f["vil"].shape[0]

rng = np.random.default_rng(SEED)

event_indices = np.arange(num_events)
rng.shuffle(event_indices)

train_end = int(TRAIN_RATIO * num_events)
val_end = int((TRAIN_RATIO + VAL_RATIO) * num_events)

train_events = event_indices[:train_end]
val_events = event_indices[train_end:val_end]
test_events = event_indices[val_end:]

print("\nEvent split:")
print("Train events:", len(train_events))
print("Validation events:", len(val_events))
print("Test events:", len(test_events))


# ============================================================
# DATASET
# ============================================================

dataset = SEVIRVILDataset(
    FILE,
    input_frames=INPUT_FRAMES,
    target_frames=TARGET_FRAMES,
    image_size=IMAGE_SIZE,
    max_samples=None
)

windows_per_event = dataset.windows_per_sequence

print("Windows per event:", windows_per_event)


def make_window_indices(events):

    indices = []

    for event_id in events:

        start = int(event_id) * windows_per_event
        end = start + windows_per_event

        indices.extend(range(start, end))

    return indices


test_indices = make_window_indices(test_events)

print("Test windows:", len(test_indices))


# ============================================================
# DISJOINTNESS
# ============================================================

assert len(set(train_events) & set(test_events)) == 0
assert len(set(val_events) & set(test_events)) == 0

print("Event-disjointness check: PASS")


# ============================================================
# TEST LOADER
# ============================================================

test_dataset = Subset(
    dataset,
    test_indices
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD V3
# ============================================================

v3 = StormSenseConvLSTMv3(
    input_channels=1,
    hidden_channels=32,
    output_channels=1
).to(device)

v3.load_state_dict(
    torch.load(
        V3_MODEL,
        map_location=device
    )
)

v3.eval()


# ============================================================
# LOAD RESIDUAL V3
# ============================================================

residual = StormSenseConvLSTMv3Residual(
    input_channels=1,
    hidden_channels=32,
    output_channels=1
).to(device)

residual.load_state_dict(
    torch.load(
        RESIDUAL_MODEL,
        map_location=device
    )
)

residual.eval()

print("\nBoth models loaded.")


# ============================================================
# METRICS
# ============================================================

v3_error = np.zeros(
    TARGET_FRAMES,
    dtype=np.float64
)

residual_error = np.zeros(
    TARGET_FRAMES,
    dtype=np.float64
)

persistence_error = np.zeros(
    TARGET_FRAMES,
    dtype=np.float64
)

elements = np.zeros(
    TARGET_FRAMES,
    dtype=np.float64
)


# ============================================================
# EVALUATION
# ============================================================

print("\nRunning final test...")

with torch.no_grad():

    for batch_index, (past, future) in enumerate(
        test_loader,
        start=1
    ):

        past = past.to(device)
        future = future.to(device)

        pred_v3 = v3(
            past,
            future_frames=None,
            future_steps=TARGET_FRAMES,
            teacher_forcing_ratio=0.0
        )

        pred_residual = residual(
            past,
            future_frames=None,
            future_steps=TARGET_FRAMES,
            teacher_forcing_ratio=0.0
        )

        persistence = past[:, -1:].repeat(
            1,
            TARGET_FRAMES,
            1,
            1,
            1
        )

        for step in range(TARGET_FRAMES):

            v3_error[step] += (
                (pred_v3[:, step] - future[:, step]) ** 2
            ).sum().item()

            residual_error[step] += (
                (pred_residual[:, step] - future[:, step]) ** 2
            ).sum().item()

            persistence_error[step] += (
                (persistence[:, step] - future[:, step]) ** 2
            ).sum().item()

            elements[step] += future[:, step].numel()

        if (
            batch_index % 200 == 0
            or batch_index == len(test_loader)
        ):
            print(
                f"Test batch "
                f"{batch_index}/{len(test_loader)}"
            )


# ============================================================
# RESULTS
# ============================================================

v3_mse = v3_error / elements
residual_mse = residual_error / elements
persistence_mse = persistence_error / elements

overall_v3 = v3_mse.mean()
overall_residual = residual_mse.mean()
overall_persistence = persistence_mse.mean()

v3_improvement = (
    (overall_persistence - overall_v3)
    / overall_persistence
) * 100

residual_improvement = (
    (overall_persistence - overall_residual)
    / overall_persistence
) * 100


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 75)
print("FINAL EVENT-DISJOINT TEST RESULTS")
print("=" * 75)

print("Test events:", len(test_events))
print("Test windows:", len(test_indices))

print()
print("Overall:")
print("-------------------------------------------")

print(f"Original V3 MSE:       {overall_v3:.6f}")
print(f"Residual V3 MSE:       {overall_residual:.6f}")
print(f"Persistence MSE:       {overall_persistence:.6f}")

print(f"Original V3 improvement:   {v3_improvement:.2f}%")
print(f"Residual V3 improvement:   {residual_improvement:.2f}%")


print()
print("Horizon-wise:")
print("-------------------------------------------")

for step in range(TARGET_FRAMES):

    minutes = (step + 1) * 5

    v3_imp = (
        (persistence_mse[step] - v3_mse[step])
        / persistence_mse[step]
    ) * 100

    residual_imp = (
        (persistence_mse[step] - residual_mse[step])
        / persistence_mse[step]
    ) * 100

    print(
        f"{minutes:2d} min | "
        f"V3={v3_mse[step]:.6f} "
        f"({v3_imp:+.2f}%) | "
        f"Residual={residual_mse[step]:.6f} "
        f"({residual_imp:+.2f}%) | "
        f"Persistence={persistence_mse[step]:.6f}"
    )


print()
print("Final evaluation complete.")
