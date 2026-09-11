import numpy as np
import torch

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3


FILE_PATH = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
MODEL_PATH = "models/stormsense_convlstm_v3_multistep.pth"

IMAGE_SIZE = 128
INPUT_FRAMES = 12
TARGET_FRAMES = 12

# Use a manageable sample first
START = 0
END = 128

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def load_model():
    model = StormSenseConvLSTMv3(hidden_channels=32)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


def main():

    print("=" * 70)
    print("StormSense V3 — INTENSITY / CORE DIAGNOSTIC")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Evaluation windows: {START}–{END - 1}")

    dataset = SEVIRVILDataset(
        FILE_PATH,
        input_frames=INPUT_FRAMES,
        target_frames=TARGET_FRAMES,
        image_size=IMAGE_SIZE
    )

    model = load_model()

    predicted_peaks = []
    actual_peaks = []
    peak_errors = []

    # Core threshold used by the previous V3 training/evaluation logic
    CORE_THRESHOLD = 0.20

    core_pixel_recalls = []
    core_pixel_precisions = []
    core_mses = []

    with torch.no_grad():

        for index in range(START, min(END, len(dataset))):

            past, future = dataset[index]

            past = past.unsqueeze(0).to(DEVICE)
            future = future.unsqueeze(0).to(DEVICE)

            prediction = model(
                past,
                future_frames=None,
                future_steps=TARGET_FRAMES,
                teacher_forcing_ratio=0.0
            )

            pred = prediction.squeeze(0).cpu().numpy()
            actual = future.squeeze(0).cpu().numpy()

            # --------------------------------------------------
            # Peak intensity
            # --------------------------------------------------

            pred_peak = pred.max(axis=(1, 2, 3))
            actual_peak = actual.max(axis=(1, 2, 3))

            predicted_peaks.append(pred_peak)
            actual_peaks.append(actual_peak)

            peak_errors.append(
                np.abs(pred_peak - actual_peak)
            )

            # --------------------------------------------------
            # Core-region metrics
            # --------------------------------------------------

            pred_core = pred >= CORE_THRESHOLD
            actual_core = actual >= CORE_THRESHOLD

            intersection = np.logical_and(
                pred_core,
                actual_core
            ).sum(axis=(1, 2, 3))

            actual_count = actual_core.sum(axis=(1, 2, 3))
            pred_count = pred_core.sum(axis=(1, 2, 3))

            recall = np.divide(
                intersection,
                actual_count,
                out=np.zeros_like(intersection, dtype=float),
                where=actual_count > 0
            )

            precision = np.divide(
                intersection,
                pred_count,
                out=np.zeros_like(intersection, dtype=float),
                where=pred_count > 0
            )

            core_mse = (
                (pred - actual) ** 2
            )

            # Only measure error inside actual core pixels
            actual_core_mask = actual_core

            core_error = np.divide(
                (core_mse * actual_core_mask).sum(axis=(1, 2, 3)),
                actual_count,
                out=np.zeros_like(actual_count, dtype=float),
                where=actual_count > 0
            )

            core_pixel_recalls.append(recall)
            core_pixel_precisions.append(precision)
            core_mses.append(core_error)

    predicted_peaks = np.array(predicted_peaks)
    actual_peaks = np.array(actual_peaks)
    peak_errors = np.array(peak_errors)

    core_pixel_recalls = np.array(core_pixel_recalls)
    core_pixel_precisions = np.array(core_pixel_precisions)
    core_mses = np.array(core_mses)

    print()
    print("=" * 70)
    print("OVERALL INTENSITY RESULTS")
    print("=" * 70)

    print(
        f"Mean predicted peak: {predicted_peaks.mean():.4f}"
    )

    print(
        f"Mean actual peak:    {actual_peaks.mean():.4f}"
    )

    print(
        f"Mean peak error:     {peak_errors.mean():.4f}"
    )

    print()
    print("=" * 70)
    print("HORIZON-WISE INTENSITY / CORE RESULTS")
    print("=" * 70)

    for h in range(TARGET_FRAMES):

        valid = actual_peaks[:, h] >= CORE_THRESHOLD

        if valid.any():

            recall = core_pixel_recalls[:, h][valid].mean()
            precision = core_pixel_precisions[:, h][valid].mean()
            core_mse = core_mses[:, h][valid].mean()

        else:

            recall = 0.0
            precision = 0.0
            core_mse = 0.0

        print(
            f"{(h + 1) * 5:2d} min | "
            f"Pred peak={predicted_peaks[:, h].mean():.4f} | "
            f"Actual peak={actual_peaks[:, h].mean():.4f} | "
            f"Peak err={peak_errors[:, h].mean():.4f} | "
            f"Core recall={recall:.4f} | "
            f"Core precision={precision:.4f} | "
            f"Core MSE={core_mse:.6f}"
        )

    print()
    print("Intensity/core diagnostic complete.")


if __name__ == "__main__":
    main()
