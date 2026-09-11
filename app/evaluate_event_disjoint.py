import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3


# ============================================================
# CONFIG
# ============================================================

FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
MODEL_FILE = "models/stormsense_convlstm_v3_multistep.pth"

INPUT_FRAMES = 12
TARGET_FRAMES = 12
IMAGE_SIZE = 128

BATCH_SIZE = 2
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 75)
print("STORMSENSE - EVENT-DISJOINT V3 EVALUATION")
print("=" * 75)

print("Device:", device)


# ============================================================
# FIND NUMBER OF EVENTS
# ============================================================

with h5py.File(FILE, "r") as f:
    num_events = f["vil"].shape[0]

print("Total events:", num_events)


# ============================================================
# EVENT-LEVEL SPLIT
# ============================================================

rng = np.random.default_rng(SEED)

event_indices = np.arange(num_events)
rng.shuffle(event_indices)

train_end = int(
    TRAIN_RATIO * num_events
)

val_end = int(
    (TRAIN_RATIO + VAL_RATIO)
    * num_events
)

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

print("\nWindows per event:", windows_per_event)


# ============================================================
# CONVERT EVENTS → WINDOW INDICES
# ============================================================

def make_window_indices(events):

    indices = []

    for event_id in events:

        start = (
            int(event_id)
            * windows_per_event
        )

        end = (
            start
            + windows_per_event
        )

        indices.extend(
            range(start, end)
        )

    return indices


train_indices = make_window_indices(
    train_events
)

val_indices = make_window_indices(
    val_events
)

test_indices = make_window_indices(
    test_events
)

print("\nWindow counts:")
print("Train windows:", len(train_indices))
print("Validation windows:", len(val_indices))
print("Test windows:", len(test_indices))


# ============================================================
# SANITY CHECK FOR EVENT DISJOINTNESS
# ============================================================

assert len(
    set(train_events)
    & set(val_events)
) == 0

assert len(
    set(train_events)
    & set(test_events)
) == 0

assert len(
    set(val_events)
    & set(test_events)
) == 0

print("\nEvent-disjointness check: PASS")


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
# MODEL
# ============================================================

model = StormSenseConvLSTMv3(
    input_channels=1,
    hidden_channels=32
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model.eval()

print("\nV3 model loaded.")


# ============================================================
# METRICS
# ============================================================

v3_squared_error = np.zeros(
    TARGET_FRAMES,
    dtype=np.float64
)

persistence_squared_error = np.zeros(
    TARGET_FRAMES,
    dtype=np.float64
)

horizon_elements = np.zeros(
    TARGET_FRAMES,
    dtype=np.float64
)


# ============================================================
# TEST
# ============================================================

print("\nRunning final event-disjoint test...")

with torch.no_grad():

    for batch_index, (past, future) in enumerate(
        test_loader,
        start=1
    ):

        past = past.to(device)
        future = future.to(device)

        # ----------------------------------------------------
        # V3
        # ----------------------------------------------------

        prediction = model(
            past,
            future_frames=None,
            future_steps=TARGET_FRAMES,
            teacher_forcing_ratio=0.0
        )

        # ----------------------------------------------------
        # Persistence
        # ----------------------------------------------------

        last_frame = past[:, -1:]

        persistence = last_frame.repeat(
            1,
            TARGET_FRAMES,
            1,
            1,
            1
        )

        # ----------------------------------------------------
        # Horizon errors
        # ----------------------------------------------------

        for step in range(TARGET_FRAMES):

            v3_error = (
                prediction[:, step]
                - future[:, step]
            ) ** 2

            persistence_error = (
                persistence[:, step]
                - future[:, step]
            ) ** 2

            v3_squared_error[step] += (
                v3_error.sum().item()
            )

            persistence_squared_error[step] += (
                persistence_error.sum().item()
            )

            horizon_elements[step] += (
                future[:, step].numel()
            )

        if (
            batch_index % 100 == 0
            or batch_index == len(test_loader)
        ):

            print(
                f"Test batch "
                f"{batch_index}/"
                f"{len(test_loader)}"
            )


# ============================================================
# RESULTS
# ============================================================

v3_mse = (
    v3_squared_error
    / horizon_elements
)

persistence_mse = (
    persistence_squared_error
    / horizon_elements
)

overall_v3 = v3_mse.mean()
overall_persistence = persistence_mse.mean()

improvement = (
    (overall_persistence - overall_v3)
    / overall_persistence
) * 100


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 75)
print("FINAL EVENT-DISJOINT TEST RESULTS")
print("=" * 75)

print("\nTest events:", len(test_events))
print("Test windows:", len(test_indices))

print("\nOverall:")
print("-------------------------------------------")
print(f"V3 MSE:          {overall_v3:.6f}")
print(f"Persistence MSE: {overall_persistence:.6f}")
print(f"Improvement:     {improvement:.2f}%")

print("\nHorizon-wise:")
print("-------------------------------------------")

for step in range(TARGET_FRAMES):

    minutes = (step + 1) * 5

    step_improvement = (
        (
            persistence_mse[step]
            - v3_mse[step]
        )
        / persistence_mse[step]
    ) * 100

    print(
        f"{minutes:2d} min | "
        f"V3={v3_mse[step]:.6f} | "
        f"Persistence={persistence_mse[step]:.6f} | "
        f"Improvement={step_improvement:.2f}%"
    )


# ============================================================
# SAVE
# ============================================================

np.savez(
    "event_disjoint_v3_results.npz",
    test_events=test_events,
    v3_mse=v3_mse,
    persistence_mse=persistence_mse,
    overall_v3=overall_v3,
    overall_persistence=overall_persistence,
    improvement=improvement
)

print("\nSaved:")
print("event_disjoint_v3_results.npz")

print("\nEvent-disjoint evaluation complete.")