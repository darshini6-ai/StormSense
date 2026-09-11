import torch
import numpy as np

from app.sevir_dataset import SEVIRVILDataset
from models.convlstm_v3 import StormSenseConvLSTMv3


FILE_PATH = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

START = 2000
END = 2200


def load_model(checkpoint):
    model = StormSenseConvLSTMv3(
        input_channels=1,
        hidden_channels=32
    ).to(DEVICE)

    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    return model


def evaluate(model, dataset, name):
    predictions = []
    actuals = []
    persistences = []

    with torch.no_grad():
        for i in range(START, END):
            past, future = dataset[i]

            past = past.unsqueeze(0).to(DEVICE)
            future = future.unsqueeze(0).to(DEVICE)

            prediction = model(
                past,
                future_frames=None,
                future_steps=12,
                teacher_forcing_ratio=0.0
            )

            persistence = past[:, -1:].repeat(1, 12, 1, 1, 1)

            predictions.append(
                ((prediction - future) ** 2).mean(dim=(2, 3, 4)).cpu().numpy()
            )

            actuals.append(
                future.cpu().numpy()
            )

            persistences.append(
                ((persistence - future) ** 2)
                .mean(dim=(2, 3, 4))
                .cpu()
                .numpy()
            )

    predictions = np.concatenate(predictions, axis=0)
    persistences = np.concatenate(persistences, axis=0)

    model_mse = predictions.mean(axis=0)
    persistence_mse = persistences.mean(axis=0)

    overall_model = model_mse.mean()
    overall_persistence = persistence_mse.mean()

    improvement = (
        (overall_persistence - overall_model)
        / overall_persistence
        * 100
    )

    print()
    print("=" * 65)
    print(name)
    print("=" * 65)

    print(f"Overall MSE:       {overall_model:.6f}")
    print(f"Persistence MSE:   {overall_persistence:.6f}")
    print(f"Improvement:       {improvement:.2f}%")

    print()
    print("Per-horizon performance:")
    print("-" * 65)

    for h in range(12):
        imp = (
            (persistence_mse[h] - model_mse[h])
            / persistence_mse[h]
            * 100
        )

        print(
            f"{(h + 1) * 5:2d} min | "
            f"Model: {model_mse[h]:.6f} | "
            f"Persistence: {persistence_mse[h]:.6f} | "
            f"Improvement: {imp:6.2f}%"
        )

    return model_mse


def evaluate_peak(model, dataset):
    errors = []
    predicted_peaks = []
    actual_peaks = []

    with torch.no_grad():
        for i in range(START, END):
            past, future = dataset[i]

            past = past.unsqueeze(0).to(DEVICE)
            future = future.unsqueeze(0).to(DEVICE)

            prediction = model(
                past,
                future_frames=None,
                future_steps=12,
                teacher_forcing_ratio=0.0
            )

            pred = prediction.squeeze(0).cpu().numpy()
            actual = future.squeeze(0).cpu().numpy()

            pred_peak = pred.max(axis=(1, 2, 3))
            actual_peak = actual.max(axis=(1, 2, 3))

            predicted_peaks.append(pred_peak)
            actual_peaks.append(actual_peak)

            errors.append(np.abs(pred_peak - actual_peak))

    predicted_peaks = np.array(predicted_peaks)
    actual_peaks = np.array(actual_peaks)
    errors = np.array(errors)

    print()
    print("=" * 65)
    print("PEAK VIL / INTENSITY DIAGNOSTIC")
    print("=" * 65)

    print(f"Mean predicted peak: {predicted_peaks.mean():.4f}")
    print(f"Mean actual peak:    {actual_peaks.mean():.4f}")
    print(f"Mean peak error:     {errors.mean():.4f}")

    print()
    print("Per-horizon peak comparison:")

    for h in range(12):
        print(
            f"{(h + 1) * 5:2d} min | "
            f"Pred peak: {predicted_peaks[:, h].mean():.4f} | "
            f"Actual peak: {actual_peaks[:, h].mean():.4f} | "
            f"Abs error: {errors[:, h].mean():.4f}"
        )


def main():

    print("StormSense V3 vs V6 evaluation")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    print(f"Evaluation windows: {START}–{END - 1}")

    dataset = SEVIRVILDataset(
        FILE_PATH,
        input_frames=12,
        target_frames=12,
        image_size=128
    )

    print(f"Dataset size: {len(dataset)}")

    print()
    print("Loading V3...")
    v3 = load_model(
        "models/stormsense_convlstm_v3_multistep.pth"
    )

    print("Loading V6...")
    v6 = load_model(
        "models/stormsense_convlstm_v6_intensity.pth"
    )

    evaluate(v3, dataset, "V3 RESULTS")

    evaluate(v6, dataset, "V6 RESULTS")

    print()
    print("=" * 65)
    print("V6 PEAK INTENSITY")
    print("=" * 65)

    evaluate_peak(v6, dataset)

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
