import h5py
import numpy as np


FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"


def load_vil_sequence(index=0):
    """
    Load one VIL sequence from the SEVIR HDF5 file.
    """

    with h5py.File(FILE, "r") as f:
        vil = f["vil"][index]
        event_id = f["id"][index]

    event_id = event_id.decode("utf-8")

    return vil, event_id


def normalize_vil(vil):
    """
    Convert uint8 VIL values from 0-255 to 0-1.
    """

    return vil.astype(np.float32) / 255.0


def create_training_sample(
    index=0,
    input_frames=12,
    target_frames=12
):
    """
    Create one temporal training sample.

    Returns:
        past
        future
        event_id
    """

    vil, event_id = load_vil_sequence(index)

    vil = normalize_vil(vil)

    total_frames = input_frames + target_frames

    sequence = vil[:, :, :total_frames]

    # Convert:
    # (H, W, T)
    # to:
    # (T, H, W)

    sequence = np.transpose(
        sequence,
        (2, 0, 1)
    )

    past = sequence[:input_frames]

    future = sequence[
        input_frames:
        input_frames + target_frames
    ]

    return past, future, event_id
