import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from models.convlstm_v3 import StormSenseConvLSTMv3


# ============================================================
# Configuration
# ============================================================

FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
MODEL_PATH = "models/stormsense_convlstm_v3_multistep.pth"

INPUT_FRAMES = 12
FUTURE_FRAMES = 12
IMAGE_SIZE = 128


# ============================================================
# Device
# ============================================================

def get_device():
    """
    Select the best available PyTorch device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# Load SEVIR event
# ============================================================

def load_event(
    index=0,
    file_path=FILE,
    input_frames=INPUT_FRAMES,
    future_frames=FUTURE_FRAMES
):
    """
    Load one SEVIR VIL event.

    Returns:
        past_original:
            (input_frames, H, W)

        future_original:
            (future_frames, H, W)

        event_id:
            SEVIR event ID
    """

    with h5py.File(file_path, "r") as f:

        vil = f["vil"][index]
        event_id = f["id"][index]

    if isinstance(event_id, bytes):
        event_id = event_id.decode("utf-8")

    # Normalize uint8 VIL to [0, 1]
    vil = vil.astype(np.float32) / 255.0

    total_frames = input_frames + future_frames

    if vil.shape[2] < total_frames:
        raise ValueError(
            f"Event contains only {vil.shape[2]} frames, "
            f"but {total_frames} frames are required."
        )

    sequence = vil[:, :, :total_frames]

    # (H, W, T) -> (T, H, W)
    sequence = np.transpose(sequence, (2, 0, 1))

    past = sequence[:input_frames]

    future = sequence[
        input_frames:
        input_frames + future_frames
    ]

    return past, future, event_id


# ============================================================
# Preprocess for V3 model
# ============================================================

def prepare_input(
    past,
    image_size=IMAGE_SIZE
):
    """
    Convert:
        (T, H, W)

    into:
        (1, T, 1, image_size, image_size)
    """

    tensor = torch.from_numpy(
        past
    ).float()

    # (T, H, W)
    # ->
    # (T, 1, H, W)

    tensor = tensor.unsqueeze(1)

    # Resize spatial dimensions
    tensor = F.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False
    )

    # (T, 1, H, W)
    # ->
    # (1, T, 1, H, W)

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# Load V3 model
# ============================================================

def load_model(
    model_path=MODEL_PATH,
    device=None
):
    """
    Load the trained StormSense V3 ConvLSTM model.
    """

    if device is None:
        device = get_device()

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{model_path}"
        )

    model = StormSenseConvLSTMv3(
        input_channels=1,
        hidden_channels=32,
        output_channels=1
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    # Support both a raw state_dict and a checkpoint dictionary.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    model.eval()

    return model


# ============================================================
# Prediction
# ============================================================

def predict(
    index=0,
    file_path=FILE,
    model_path=MODEL_PATH,
    input_frames=INPUT_FRAMES,
    future_frames=FUTURE_FRAMES,
    image_size=IMAGE_SIZE,
    device=None
):
    """
    Generate a 12-frame autoregressive forecast.

    Returns a dictionary containing:

        event_id
        past
        actual_future
        prediction
        mse
        mae
        device

    Shapes:

        past:
            (12, 128, 128)

        actual_future:
            (12, 128, 128)

        prediction:
            (12, 128, 128)
    """

    if device is None:
        device = get_device()

    # --------------------------------------------------------
    # Load event
    # --------------------------------------------------------

    past_original, future_original, event_id = load_event(
        index=index,
        file_path=file_path,
        input_frames=input_frames,
        future_frames=future_frames
    )

    # --------------------------------------------------------
    # Prepare tensors
    # --------------------------------------------------------

    past = prepare_input(
        past_original,
        image_size=image_size
    )

    future = prepare_input(
        future_original,
        image_size=image_size
    )

    past = past.to(device)
    future = future.to(device)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        model_path=model_path,
        device=device
    )

    # --------------------------------------------------------
    # Generate forecast
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            past,
            future_steps=future_frames,
            teacher_forcing_ratio=0.0
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    mse = torch.mean(
        (prediction - future) ** 2
    ).item()

    mae = torch.mean(
        torch.abs(prediction - future)
    ).item()

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    prediction_np = (
        prediction[0, :, 0]
        .detach()
        .cpu()
        .numpy()
    )

    actual_np = (
        future[0, :, 0]
        .detach()
        .cpu()
        .numpy()
    )

    past_np = (
        past[0, :, 0]
        .detach()
        .cpu()
        .numpy()
    )

    return {
        "event_id": event_id,
        "past": past_np,
        "actual_future": actual_np,
        "prediction": prediction_np,
        "mse": mse,
        "mae": mae,
        "device": str(device)
    }


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("StormSense V3 Prediction Test")
    print("=" * 60)

    device = get_device()

    print("Device:", device)

    result = predict(
        index=0,
        device=device
    )

    print()
    print("Event ID:", result["event_id"])

    print(
        "Past shape:",
        result["past"].shape
    )

    print(
        "Actual future shape:",
        result["actual_future"].shape
    )

    print(
        "Prediction shape:",
        result["prediction"].shape
    )

    print()
    print("MSE:", result["mse"])
    print("MAE:", result["mae"])

    print()
    print(
        "Prediction range:",
        result["prediction"].min(),
        "to",
        result["prediction"].max()
    )

    print()
    print("V3 prediction completed successfully.")