import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3_residual import StormSenseConvLSTMv3Residual


# ============================================================
# CONFIG
# ============================================================

FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"

MODEL_FILE = (
    "models/stormsense_convlstm_v3_residual_event_disjoint.pth"
)

INPUT_FRAMES = 12
TARGET_FRAMES = 12
IMAGE_SIZE = 128

BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 0.001

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TEACHER_FORCING_RATIO = 0.5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("=" * 75)
print("STORMSENSE - RESIDUAL V3 EVENT-DISJOINT TRAINING")
print("=" * 75)

print("Device:", DEVICE)


# ============================================================
# FIND NUMBER OF EVENTS
# ============================================================

with h5py.File(FILE, "r") as f:
    num_events = f["vil"].shape[0]

print("Total events:", num_events)


# ============================================================
# EXACT SAME EVENT SPLIT
# ============================================================

rng = np.random.default_rng(SEED)

event_indices = np.arange(num_events)
rng.shuffle(event_indices)

train_end = int(
    TRAIN_RATIO * num_events
)

val_end = int(
    (TRAIN_RATIO + VAL_RATIO) * num_events
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
# EVENT -> WINDOW INDICES
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
# DISJOINTNESS CHECK
# ============================================================

assert len(
    set(train_events) & set(val_events)
) == 0

assert len(
    set(train_events) & set(test_events)
) == 0

assert len(
    set(val_events) & set(test_events)
) == 0

print("\nEvent-disjointness check: PASS")


# ============================================================
# TRAIN DATASET
# ============================================================

train_dataset = Subset(
    dataset,
    train_indices
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = StormSenseConvLSTMv3Residual(
    input_channels=1,
    hidden_channels=32,
    output_channels=1
).to(DEVICE)


criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")
print("Training windows:", len(train_dataset))
print("Batches per epoch:", len(train_loader))


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_index, (past, future) in enumerate(
        train_loader,
        start=1
    ):

        past = past.to(DEVICE)
        future = future.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(
            past,
            future_frames=future,
            future_steps=TARGET_FRAMES,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO
        )

        loss = criterion(
            prediction,
            future
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (
            batch_index % 500 == 0
            or batch_index == len(train_loader)
        ):

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_index}/{len(train_loader)} | "
                f"Loss {loss.item():.6f}"
            )

    average_loss = (
        total_loss
        / len(train_loader)
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Average Loss: {average_loss:.6f}"
    )


# ============================================================
# SAVE
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_FILE
)

print()
print("Saved:", MODEL_FILE)
print()
print("Residual V3 event-disjoint training complete.")
