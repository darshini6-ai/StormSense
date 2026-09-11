import numpy as np

from app.sevir_dataset import SEVIRVILDataset


dataset = SEVIRVILDataset(
    "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
)

print("Sliding-window inspection")
print("=" * 60)

for index in [0, 1, 10, 20, 25]:

    past, future = dataset[index]

    past_np = past.numpy()[:, 0]
    future_np = future.numpy()[:, 0]

    print()
    print(f"WINDOW {index}")
    print("-" * 40)

    print("Past:")
    print(
        f"  min={past_np.min():.4f}, "
        f"mean={past_np.mean():.4f}, "
        f"max={past_np.max():.4f}"
    )

    print("Future:")
    print(
        f"  min={future_np.min():.4f}, "
        f"mean={future_np.mean():.4f}, "
        f"max={future_np.max():.4f}"
    )

    print("Future frame maxima:")

    for frame_index in range(12):

        frame = future_np[frame_index]

        print(
            f"  {(frame_index + 1) * 5:2d} min: "
            f"mean={frame.mean():.4f}, "
            f"max={frame.max():.4f}"
        )

print()
print("=" * 60)
print("Window inspection complete.")
