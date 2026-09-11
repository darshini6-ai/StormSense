import numpy as np
import torch

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3
from models.convlstm_v3_residual import StormSenseConvLSTMv3Residual


FILE_PATH = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"

V3_MODEL = "models/stormsense_convlstm_v3_multistep.pth"
RESIDUAL_MODEL = "models/stormsense_convlstm_v3_residual_event_disjoint.pth"

INPUT_FRAMES = 12
TARGET_FRAMES = 12
IMAGE_SIZE = 128

START = 0
END = 128

CORE_THRESHOLD = 0.20

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def load_v3():
    model = StormSenseConvLSTMv3(
        input_channels=1,
        hidden_channels=32,
        output_channels=1
    ).to(DEVICE)

    checkpoint = torch.load(
        V3_MODEL,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()

    return model


def load_residual():
    model = StormSenseConvLSTMv3Residual(
        input_channels=1,
        hidden_channels=32,
        output_channels=1
    ).to(DEVICE)

    checkpoint = torch.load(
        RESIDUAL_MODEL,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()

    return model


def calculate_metrics(pred, actual):

    pred_peak = pred.max(axis=(1, 2, 3))
    actual_peak = actual.max(axis=(1, 2, 3))

    peak_error = np.abs(
        pred_peak - actual_peak
    )

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

    squared_error = (
        pred - actual
    ) ** 2

    core_mse = np.divide(
        (squared_error * actual_core).sum(axis=(1, 2, 3)),
        actual_count,
        out=np.zeros_like(actual_count, dtype=float),
        where=actual_count > 0
    )

    return (
        pred_peak,
        actual_peak,
        peak_error,
        recall,
        precision,
        core_mse,
        actual_count
    )


def main():

    print("=" * 75)
    print("STORMSENSE V3 vs RESIDUAL V3")
    print("INTENSITY / CORE DIAGNOSTIC")
    print("=" * 75)

    print("Device:", DEVICE)
    print(f"Evaluation windows: {START}–{END - 1}")

    dataset = SEVIRVILDataset(
        FILE_PATH,
        input_frames=INPUT_FRAMES,
        target_frames=TARGET_FRAMES,
        image_size=IMAGE_SIZE
    )

    v3 = load_v3()
    residual = load_residual()

    print("V3 model loaded.")
    print("Residual V3 model loaded.")

    v3_peaks = []
    residual_peaks = []
    actual_peaks = []

    v3_peak_errors = []
    residual_peak_errors = []

    v3_recall = []
    residual_recall = []

    v3_precision = []
    residual_precision = []

    v3_core_mse = []
    residual_core_mse = []

    with torch.no_grad():

        for index in range(
            START,
            min(END, len(dataset))
        ):

            past, future = dataset[index]

            past = past.unsqueeze(0).to(DEVICE)
            future = future.unsqueeze(0).to(DEVICE)

            v3_prediction = v3(
                past,
                future_frames=None,
                future_steps=TARGET_FRAMES,
                teacher_forcing_ratio=0.0
            )

            residual_prediction = residual(
                past,
                future_frames=None,
                future_steps=TARGET_FRAMES,
                teacher_forcing_ratio=0.0
            )

            v3_np = (
                v3_prediction
                .squeeze(0)
                .cpu()
                .numpy()
            )

            residual_np = (
                residual_prediction
                .squeeze(0)
                .cpu()
                .numpy()
            )

            actual_np = (
                future
                .squeeze(0)
                .cpu()
                .numpy()
            )

            (
                vp,
                ap,
                ve,
                vr,
                vprec,
                vmse,
                _,
            ) = calculate_metrics(
                v3_np,
                actual_np
            )

            (
                rp,
                _,
                re,
                rr,
                rprec,
                rmse,
                _,
            ) = calculate_metrics(
                residual_np,
                actual_np
            )

            v3_peaks.append(vp)
            residual_peaks.append(rp)
            actual_peaks.append(ap)

            v3_peak_errors.append(ve)
            residual_peak_errors.append(re)

            v3_recall.append(vr)
            residual_recall.append(rr)

            v3_precision.append(vprec)
            residual_precision.append(rprec)

            v3_core_mse.append(vmse)
            residual_core_mse.append(rmse)

    v3_peaks = np.array(v3_peaks)
    residual_peaks = np.array(residual_peaks)
    actual_peaks = np.array(actual_peaks)

    v3_peak_errors = np.array(v3_peak_errors)
    residual_peak_errors = np.array(residual_peak_errors)

    v3_recall = np.array(v3_recall)
    residual_recall = np.array(residual_recall)

    v3_precision = np.array(v3_precision)
    residual_precision = np.array(residual_precision)

    v3_core_mse = np.array(v3_core_mse)
    residual_core_mse = np.array(residual_core_mse)

    print()
    print("=" * 75)
    print("OVERALL INTENSITY RESULTS")
    print("=" * 75)

    print(
        f"V3 mean peak:       {v3_peaks.mean():.4f}"
    )

    print(
        f"Residual mean peak: {residual_peaks.mean():.4f}"
    )

    print(
        f"Actual mean peak:   {actual_peaks.mean():.4f}"
    )

    print(
        f"V3 peak error:       {v3_peak_errors.mean():.4f}"
    )

    print(
        f"Residual peak error: {residual_peak_errors.mean():.4f}"
    )

    print()
    print("=" * 75)
    print("HORIZON-WISE INTENSITY / CORE RESULTS")
    print("=" * 75)

    for h in range(TARGET_FRAMES):

        valid_v3 = actual_peaks[:, h] >= CORE_THRESHOLD
        valid_residual = valid_v3

        v3_r = v3_recall[:, h][valid_v3].mean()
        r_r = residual_recall[:, h][valid_residual].mean()

        v3_p = v3_precision[:, h][valid_v3].mean()
        r_p = residual_precision[:, h][valid_residual].mean()

        v3_cmse = v3_core_mse[:, h][valid_v3].mean()
        r_cmse = residual_core_mse[:, h][valid_residual].mean()

        print(
            f"{(h + 1) * 5:2d} min | "
            f"V3 peak={v3_peaks[:, h].mean():.4f} | "
            f"Residual peak={residual_peaks[:, h].mean():.4f} | "
            f"Actual={actual_peaks[:, h].mean():.4f}"
        )

        print(
            f"         V3 err={v3_peak_errors[:, h].mean():.4f} | "
            f"Residual err={residual_peak_errors[:, h].mean():.4f}"
        )

        print(
            f"         V3 recall={v3_r:.4f} | "
            f"Residual recall={r_r:.4f}"
        )

        print(
            f"         V3 precision={v3_p:.4f} | "
            f"Residual precision={r_p:.4f}"
        )

        print(
            f"         V3 core MSE={v3_cmse:.6f} | "
            f"Residual core MSE={r_cmse:.6f}"
        )

    print()
    print("Intensity/core diagnostic complete.")


if __name__ == "__main__":
    main()
